"""
Enhanced Training Pipeline for CBS Model
Includes curriculum learning, multi-task losses, checkpointing, and GPU optimization
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

import config
from cbs_model_enhanced import create_model
from waymo_tfexample_loader import get_data_loaders
from primitive_detector_v2 import PrimitiveDetector
import evaluate


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining trajectory, primitive, and confidence losses
    """
    
    def __init__(
        self,
        trajectory_weight: float = 1.0,
        primitive_weight: float = 0.5,
        confidence_weight: float = 0.3
    ):
        super().__init__()
        self.trajectory_weight = trajectory_weight
        self.primitive_weight = primitive_weight
        self.confidence_weight = confidence_weight
        
        # Loss functions
        self.trajectory_loss = nn.L1Loss(reduction='mean')
        self.primitive_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.confidence_loss = nn.MSELoss(reduction='mean')

        # EMTA configuration (with safe defaults)
        emta_cfg = config.TRAIN_CONFIG.get("emta", {}) if hasattr(config, "TRAIN_CONFIG") else {}
        self.emta_enabled = emta_cfg.get("enabled", True)
        self.initial_tolerance = emta_cfg.get("initial_tolerance", 4.0)
        self.final_tolerance = emta_cfg.get("final_tolerance", 2.5)
        self.lock_warmup_epochs = emta_cfg.get("lock_warmup_epochs", 5)
        self.fallback_scale = emta_cfg.get("fallback_scale", 1.5)
        self.coverage_margin = emta_cfg.get("coverage_margin", 3.0)
        self.coverage_weight = emta_cfg.get("coverage_weight", 0.1)
        self.coverage_timesteps = emta_cfg.get("coverage_timesteps", [-1])
        self.emta_temperature = emta_cfg.get("temperature", 2.0)
        self.confidence_rank_weight = emta_cfg.get("confidence_rank_weight", 0.0)

        # State updated per epoch
        self.current_epoch = 0
        self.total_epochs = 1
        self.current_match_tolerance = self.initial_tolerance
        self.current_fallback_tolerance = self.initial_tolerance * self.fallback_scale
        self.emta_active = False

    def update_epoch(self, epoch: int, total_epochs: int):
        """Update EMTA scheduling parameters for the current epoch."""
        self.current_epoch = max(epoch, 0)
        self.total_epochs = max(total_epochs, 1)
        if not self.emta_enabled:
            self.current_match_tolerance = self.final_tolerance
            self.current_fallback_tolerance = self.final_tolerance * self.fallback_scale
            self.emta_active = False
            return
        warmup = max(self.lock_warmup_epochs, 1)
        if self.current_epoch < warmup:
            progress = 0.0
        else:
            span = max(self.total_epochs - warmup, 1)
            progress = min((self.current_epoch - warmup) / span, 1.0)
        # Linear annealing from initial to final tolerance
        self.current_match_tolerance = self.initial_tolerance + progress * (self.final_tolerance - self.initial_tolerance)
        self.current_fallback_tolerance = self.current_match_tolerance * self.fallback_scale
        self.emta_active = self.current_epoch >= warmup
    
    def forward(
        self,
        predictions: dict,
        ground_truth: torch.Tensor,
        primitive_labels: torch.Tensor = None
    ) -> dict:
        """
        Compute multi-task loss
        
        Args:
            predictions: Model output dictionary
            ground_truth: [batch_size, future_length, 2]
            primitive_labels: [batch_size, num_primitives] optional
        
        Returns:
            Dictionary with individual and total losses
        """
        trajectories = predictions["trajectories"]  # [batch, modes, future, 2]
        confidences = predictions["confidences"]  # [batch, modes]
        
        # Trajectory loss: use minADE across all modes (better than winner-takes-all)
        batch_size, num_modes, future_length, _ = trajectories.shape
        
        # Compute ADE for each mode
        ade_per_mode = evaluate.compute_ade(trajectories, ground_truth)  # [batch, modes]
        
        # Ensure ade_per_mode is 2D
        if ade_per_mode.dim() == 1:
            ade_per_mode = ade_per_mode.unsqueeze(1)
            num_modes = 1
        
        # EMTA-style mode assignment with curriculum-aware activation
        mode_indices = torch.arange(num_modes, device=trajectories.device).unsqueeze(0).expand(batch_size, -1)
        default_idx = torch.full_like(mode_indices, num_modes)
        
        if self.emta_enabled and self.emta_active and num_modes > 1:
            match_tol = self.current_match_tolerance
            fallback_tol = self.current_fallback_tolerance
            primary_candidates = torch.where(ade_per_mode < match_tol, mode_indices, default_idx)
            primary_match_idx = primary_candidates.min(dim=1)[0]
            has_primary_match = primary_match_idx < num_modes
            
            fallback_candidates = torch.where(ade_per_mode < fallback_tol, mode_indices, default_idx)
            fallback_match_idx = fallback_candidates.min(dim=1)[0]
            has_fallback_match = fallback_match_idx < num_modes
        else:
            has_primary_match = torch.zeros(batch_size, dtype=torch.bool, device=trajectories.device)
            primary_match_idx = torch.full((batch_size,), num_modes, dtype=torch.long, device=trajectories.device)
            has_fallback_match = torch.zeros_like(has_primary_match)
            fallback_match_idx = torch.full_like(primary_match_idx, num_modes)
        
        argmin_idx = ade_per_mode.argmin(dim=1)
        best_mode_idx = torch.where(
            has_primary_match,
            primary_match_idx,
            torch.where(has_fallback_match, fallback_match_idx, argmin_idx)
        )
        
        batch_indices = torch.arange(batch_size, device=trajectories.device)
        best_trajectories = trajectories[batch_indices, best_mode_idx]  # [batch, future, 2]
        
        # L1 loss on best trajectories (primary loss)
        trajectory_loss_l1 = self.trajectory_loss(best_trajectories, ground_truth)
        
        # Also add L2 loss for smoother trajectories
        trajectory_loss_l2 = nn.MSELoss()(best_trajectories, ground_truth)
        
        # Combined trajectory loss (weighted)
        trajectory_loss = 0.7 * trajectory_loss_l1 + 0.3 * trajectory_loss_l2
        
        # Add FDE loss to improve final position accuracy
        # This helps reduce the high FDE (13m -> target 2-4m)
        fde_per_mode = evaluate.compute_fde(trajectories, ground_truth)  # [batch, modes]
        best_fde = fde_per_mode[batch_indices, best_mode_idx]  # [batch]
        fde_loss = best_fde.mean()
        
        # Add intermediate FDE losses at key timesteps to prevent divergence
        # Checkpoints at 25%, 50%, 75% of trajectory
        checkpoint_indices = [future_length // 4, future_length // 2, 3 * future_length // 4, future_length - 1]
        intermediate_fde_loss = torch.tensor(0.0, device=trajectories.device)
        for checkpoint_idx in checkpoint_indices:
            if checkpoint_idx < future_length:
                pred_pos = best_trajectories[:, checkpoint_idx, :]  # [batch, 2]
                gt_pos = ground_truth[:, checkpoint_idx, :]  # [batch, 2]
                intermediate_fde = torch.norm(pred_pos - gt_pos, dim=-1).mean()
                intermediate_fde_loss = intermediate_fde_loss + intermediate_fde
        intermediate_fde_loss = intermediate_fde_loss / len(checkpoint_indices)
        
        # Add smoothness loss: penalize large velocity/acceleration changes
        # This helps trajectories be more physically plausible and stable
        best_traj = best_trajectories  # [batch, future_length, 2]
        velocities = best_traj[:, 1:, :] - best_traj[:, :-1, :]  # [batch, future_length-1, 2]
        accelerations = velocities[:, 1:, :] - velocities[:, :-1, :]  # [batch, future_length-2, 2]
        
        # Penalize large accelerations (jerk) - but with lower weight to not interfere with accuracy
        smoothness_loss = torch.norm(accelerations, dim=-1).mean()
        
        # Rebalanced trajectory loss: emphasize FDE more strongly
        # FDE is 2-2.5x ADE, so we need to weight it more heavily
        # Increased FDE weight to 0.5 and intermediate to 0.25 to better control long-term divergence
        trajectory_loss = 0.2 * trajectory_loss + 0.5 * fde_loss + 0.25 * intermediate_fde_loss + 0.05 * smoothness_loss
        
        # Confidence loss: use ADE-based confidence (quality-aware)
        # Compute ADE for each mode (already computed above)
        # Convert ADE to confidence scores (lower ADE = higher confidence)
        # Use inverse ADE with temperature scaling
        ade_scores = -ade_per_mode  # Negative because lower ADE is better
        ade_confidence = F.softmax(ade_scores / max(self.emta_temperature, 1e-6), dim=-1)  # [batch, modes]
        
        # Build EMTA-aligned confidence targets
        primary_mask = has_primary_match.float().unsqueeze(1)
        fallback_mask = ((~has_primary_match) & has_fallback_match).float().unsqueeze(1)
        unmatched_mask = ((~has_primary_match) & (~has_fallback_match)).float().unsqueeze(1)
        one_hot_target = F.one_hot(best_mode_idx, num_classes=num_modes).float()
        fallback_target = 0.5 * one_hot_target + 0.5 * ade_confidence
        target_confidence = (
            primary_mask * one_hot_target +
            fallback_mask * fallback_target +
            unmatched_mask * ade_confidence
        )
        target_confidence = target_confidence / target_confidence.sum(dim=1, keepdim=True).clamp_min(1e-6)
        target_confidence = target_confidence.detach()
        
        # Combine predicted confidence with ADE-based targets
        if "confidence_logits" in predictions:
            confidence_logits = predictions["confidence_logits"]  # [batch, modes]
            predicted_confidence = F.softmax(confidence_logits, dim=-1)
            
            # KL divergence between targets and predictions
            kl_loss = -(target_confidence * torch.log(predicted_confidence + 1e-8)).sum(dim=1).mean()
            ce_loss = nn.CrossEntropyLoss()(confidence_logits, best_mode_idx)
            confidence_loss = 0.6 * kl_loss + 0.4 * ce_loss

            if self.confidence_rank_weight > 0 and num_modes > 1:
                best_conf = predicted_confidence[batch_indices, best_mode_idx].unsqueeze(1)
                others_conf = predicted_confidence
                margin = 0.1
                diff = best_conf - others_conf
                rank_mask = torch.ones_like(diff, dtype=torch.bool)
                rank_mask[batch_indices, best_mode_idx] = False
                if rank_mask.any():
                    rank_violation = F.relu(margin - diff)[rank_mask]
                    if rank_violation.numel() > 0:
                        confidence_loss = confidence_loss + self.confidence_rank_weight * rank_violation.mean()
        else:
            predicted_confidence = confidences / confidences.sum(dim=1, keepdim=True).clamp_min(1e-8)
            confidence_loss = -(target_confidence * torch.log(predicted_confidence + 1e-8)).sum(dim=1).mean()
        
        # Coverage penalty: ensure unmatched modes stay sufficiently distinct
        coverage_penalty = torch.tensor(0.0, device=trajectories.device)
        if num_modes > 1:
            penalties = []
            for step in self.coverage_timesteps:
                idx = int(step) if step >= 0 else future_length + int(step)
                idx = max(0, min(idx, future_length - 1))
                step_positions = trajectories[:, :, idx, :]  # [batch, modes, 2]
                best_pos = step_positions[batch_indices, best_mode_idx].unsqueeze(1)
                separation = torch.norm(step_positions - best_pos, dim=-1)
                margin_violation = torch.relu(self.coverage_margin - separation)
                matched_mask = F.one_hot(best_mode_idx, num_classes=num_modes).bool()
                if margin_violation.numel() > 0:
                    unmatched_losses = margin_violation.masked_fill(matched_mask, 0.0)
                    active = (~matched_mask) & (separation < self.coverage_margin)
                    if active.any():
                        penalties.append(unmatched_losses[active].mean())
            if penalties:
                coverage_penalty = torch.stack(penalties).mean()
        
        # Add diversity loss: encourage modes to be different
        # Compute pairwise distances between modes (optimized)
        mode_diversity = torch.tensor(0.0, device=trajectories.device)
        if num_modes > 1:
            # Compute all pairwise distances efficiently
            # trajectories: [batch, modes, future, 2]
            traj_flat = trajectories.view(batch_size, num_modes, -1)  # [batch, modes, future*2]
            
            # Compute pairwise L2 distances
            traj_norm = torch.norm(traj_flat, dim=-1, keepdim=True)  # [batch, modes, 1]
            pairwise_dist = torch.cdist(traj_flat, traj_flat, p=2)  # [batch, modes, modes]
            
            # Get upper triangle (excluding diagonal)
            mask = torch.triu(torch.ones(num_modes, num_modes, device=trajectories.device), diagonal=1).bool()
            pairwise_dist_upper = pairwise_dist[:, mask]  # [batch, num_pairs]
            
            # Penalize if modes are too similar (encourage at least 1m average difference)
            avg_pairwise_dist = pairwise_dist_upper.mean()
            mode_diversity = torch.relu(1.0 - avg_pairwise_dist)
        
        # Add small diversity regularization to total loss
        # This encourages modes to be diverse (important for multi-modal prediction)
        diversity_weight = 0.05  # Small weight to avoid interfering with main objectives
        
        # Primitive loss (if labels provided)
        primitive_loss = torch.tensor(0.0, device=trajectories.device)
        if primitive_labels is not None and "primitive_logits" in predictions:
            primitive_logits = predictions["primitive_logits"]  # [batch, num_primitives]
            primitive_loss = self.primitive_loss(primitive_logits, primitive_labels)
        
        # Total loss
        total_loss = (
            self.trajectory_weight * trajectory_loss +
            self.primitive_weight * primitive_loss +
            self.confidence_weight * confidence_loss +
            self.coverage_weight * coverage_penalty
        )
        
        return {
            "total_loss": total_loss,
            "trajectory_loss": trajectory_loss,
            "primitive_loss": primitive_loss,
            "confidence_loss": confidence_loss,
            "coverage_loss": coverage_penalty
        }


class CurriculumTrainer:
    """
    Implements curriculum learning across primitive complexity
    """
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Curriculum phases
        self.phase1_epochs = config.TRAIN_CONFIG["curriculum_phases"]["phase1_epochs"]
        self.phase2_epochs = config.TRAIN_CONFIG["curriculum_phases"]["phase2_epochs"]
        self.phase3_epochs = config.TRAIN_CONFIG["curriculum_phases"]["phase3_epochs"]
        
        # Primitive detector for labeling
        self.primitive_detector = PrimitiveDetector()
    
    def get_curriculum_phase(self, epoch: int) -> str:
        """Determine current curriculum phase"""
        if epoch < self.phase1_epochs:
            return "phase1"  # Single primitives
        elif epoch < self.phase1_epochs + self.phase2_epochs:
            return "phase2"  # Primitive combinations
        else:
            return "phase3"  # Novel combinations (testing)
    
    def filter_samples_by_phase(self, batch, phase: str) -> dict:
        """
        Filter or weight samples based on curriculum phase
        For now, we use all samples but could filter for single primitives in phase1
        """
        return batch  # Use all samples for now


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion: MultiTaskLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None,
    use_amp: bool = False,
    primitive_detector: PrimitiveDetector = None
) -> dict:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    loss_components = {
        "trajectory": 0.0,
        "primitive": 0.0,
        "confidence": 0.0,
        "coverage": 0.0
    }
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        history = batch['history'].to(device)
        future = batch['future'].to(device)
        
        # Detect primitives for multi-task learning
        primitive_labels = None
        if primitive_detector is not None:
            # Convert to numpy for primitive detection
            batch_size = history.shape[0]
            primitive_labels_list = []
            
            for i in range(batch_size):
                # Use current state for primitive detection
                current_traj = history[i].cpu().numpy()
                primitive_probs = primitive_detector.detect_primitives(current_traj)
                primitive_one_hot = primitive_detector.get_primitive_one_hot(primitive_probs)
                primitive_labels_list.append(primitive_one_hot)
            
            primitive_labels = torch.stack(primitive_labels_list).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                predictions = model(history, return_primitives=(primitive_labels is not None))
                loss_dict = criterion(predictions, future, primitive_labels)
            
            scaler.scale(loss_dict["total_loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN_CONFIG["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(history, return_primitives=(primitive_labels is not None))
            loss_dict = criterion(predictions, future, primitive_labels)
            
            loss_dict["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN_CONFIG["gradient_clip"])
            optimizer.step()
        
        # Accumulate losses
        total_loss += loss_dict["total_loss"].item()
        loss_components["trajectory"] += loss_dict["trajectory_loss"].item()
        loss_components["primitive"] += loss_dict["primitive_loss"].item()
        loss_components["confidence"] += loss_dict["confidence_loss"].item()
        loss_components["coverage"] += loss_dict.get("coverage_loss", 0.0)
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss_dict['total_loss'].item():.4f}",
            "traj": f"{loss_dict['trajectory_loss'].item():.4f}"
        })
    
    num_batches = len(train_loader)
    return {
        "total_loss": total_loss / num_batches,
        "trajectory_loss": loss_components["trajectory"] / num_batches,
        "primitive_loss": loss_components["primitive"] / num_batches,
        "confidence_loss": loss_components["confidence"] / num_batches,
        "coverage_loss": loss_components["coverage"] / num_batches
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    device: torch.device,
    primitive_detector: PrimitiveDetector = None
) -> dict:
    """Validate model"""
    model.eval()
    all_metrics = []
    
    for batch in tqdm(val_loader, desc="Validating"):
        history = batch['history'].to(device)
        future = batch['future'].to(device)
        
        # Forward pass
        predictions = model(history, return_primitives=False)
        
        # Evaluate
        metrics = evaluate.evaluate_batch(predictions, future)
        all_metrics.append(metrics)
    
    # Aggregate metrics
    aggregated = evaluate.aggregate_metrics(all_metrics)
    return aggregated


def train(
    model: nn.Module = None,
    num_epochs: int = None,
    resume_from: str = None,
    save_dir: Path = None
):
    """
    Main training function
    
    Args:
        model: CBS model (will create if None)
        num_epochs: Number of training epochs
        resume_from: Path to checkpoint to resume from
        save_dir: Directory to save checkpoints
    """
    # Setup
    device = config.get_device()
    if save_dir is None:
        save_dir = config.CHECKPOINTS_DIR
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model if needed
    if model is None:
        model = create_model(device)
    else:
        model = model.to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Data loaders
    print("Loading data...")
    train_loader, val_loader = get_data_loaders()
    
    # Loss and optimizer
    criterion = MultiTaskLoss(
        trajectory_weight=config.TRAIN_CONFIG["trajectory_loss_weight"],
        primitive_weight=config.TRAIN_CONFIG["primitive_loss_weight"],
        confidence_weight=config.TRAIN_CONFIG["confidence_loss_weight"]
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.TRAIN_CONFIG["learning_rate"],
        weight_decay=config.TRAIN_CONFIG["weight_decay"]
    )
    
    # Learning rate scheduler with warmup
    # Use cosine annealing with restarts for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-5
    )
    
    # Mixed precision training
    use_amp = config.GPU_CONFIG["mixed_precision"] and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    
    # Primitive detector
    primitive_detector = PrimitiveDetector()
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_ade = float('inf')
    
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_ade = checkpoint.get('best_val_ade', float('inf'))
        print(f"Resumed from epoch {start_epoch-1}, best_val_ade: {best_val_ade:.4f}")
    
    # TensorBoard writer
    log_dir = config.LOGS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Training loop
    num_epochs = num_epochs or config.TRAIN_CONFIG["num_epochs"]
    early_stopping_counter = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {device}, Mixed Precision: {use_amp}")
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
 
        # Update loss scheduling for current epoch
        if hasattr(criterion, "update_epoch"):
            criterion.update_epoch(epoch, num_epochs)

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, use_amp, primitive_detector
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device, primitive_detector)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['ade'])
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_metrics['total_loss'], epoch)
        writer.add_scalar('Loss/Train_Trajectory', train_metrics['trajectory_loss'], epoch)
        writer.add_scalar('Loss/Train_Primitive', train_metrics['primitive_loss'], epoch)
        writer.add_scalar('Loss/Train_Confidence', train_metrics['confidence_loss'], epoch)
        writer.add_scalar('Loss/Train_Coverage', train_metrics['coverage_loss'], epoch)
        writer.add_scalar('Metrics/Val_ADE', val_metrics['ade'], epoch)
        writer.add_scalar('Metrics/Val_FDE', val_metrics['fde'], epoch)
        writer.add_scalar('Metrics/Val_MissRate', val_metrics['miss_rate'], epoch)
        
        # Print metrics
        print(f"\nTrain Loss: {train_metrics['total_loss']:.4f}")
        evaluate.print_metrics(val_metrics, f"Validation (Epoch {epoch+1})")
        
        # Save checkpoint
        is_best = val_metrics['ade'] < best_val_ade
        
        if is_best:
            best_val_ade = val_metrics['ade']
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_ade': best_val_ade,
            'val_metrics': val_metrics,
            'config': config.MODEL_CONFIG
        }
        
        # Save regular checkpoint
        if (epoch + 1) % config.TRAIN_CONFIG["save_every"] == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
        
        # Early stopping
        if early_stopping_counter >= config.TRAIN_CONFIG["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    writer.close()
    print("\nTraining completed!")
    print(f"Best validation ADE: {best_val_ade:.4f}")


if __name__ == "__main__":
    # Run training
    train()

