"""
Evaluation metrics for trajectory prediction
Implements ADE, FDE, Miss Rate, and multi-modal evaluation
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


def compute_ade(predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Compute Average Displacement Error (ADE)
    
    Args:
        predicted: [batch_size, num_modes, future_length, 2] or [batch_size, future_length, 2]
        ground_truth: [batch_size, future_length, 2]
    
    Returns:
        ADE value(s): scalar or [batch_size, num_modes]
    """
    is_multi_modal = predicted.dim() == 4
    
    if is_multi_modal:
        # Multi-modal case: [batch_size, num_modes, future_length, 2]
        batch_size, num_modes, future_length, _ = predicted.shape
        predicted_flat = predicted.view(batch_size * num_modes, future_length, 2)
        ground_truth_expanded = ground_truth.unsqueeze(1).repeat(1, num_modes, 1, 1)
        ground_truth_flat = ground_truth_expanded.view(batch_size * num_modes, future_length, 2)
        
        # Compute L2 distance at each timestep
        displacement = torch.norm(predicted_flat - ground_truth_flat, dim=-1)  # [batch*modes, future_length]
        ade_flat = displacement.mean(dim=-1)  # Average over time: [batch*modes]
        
        # Reshape back for multi-modal
        ade = ade_flat.view(batch_size, num_modes)  # [batch, modes]
    else:
        # Single modal case: [batch_size, future_length, 2]
        # Compute L2 distance at each timestep
        displacement = torch.norm(predicted - ground_truth, dim=-1)  # [batch, future_length]
        ade = displacement.mean(dim=-1)  # Average over time: [batch]
    
    return ade


def compute_fde(predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Compute Final Displacement Error (FDE)
    
    Args:
        predicted: [batch_size, num_modes, future_length, 2] or [batch_size, future_length, 2]
        ground_truth: [batch_size, future_length, 2]
    
    Returns:
        FDE value(s): scalar or [batch_size, num_modes]
    """
    if predicted.dim() == 4:
        # Multi-modal case
        batch_size, num_modes, future_length, _ = predicted.shape
        predicted = predicted[:, :, -1, :]  # Final position [batch, modes, 2]
        ground_truth_final = ground_truth[:, -1, :].unsqueeze(1)  # [batch, 1, 2]
        ground_truth_final = ground_truth_final.repeat(1, num_modes, 1)  # [batch, modes, 2]
    else:
        predicted = predicted[:, -1, :]  # Final position [batch, 2]
        ground_truth_final = ground_truth[:, -1, :]  # [batch, 2]
    
    # Compute L2 distance at final timestep
    fde = torch.norm(predicted - ground_truth_final, dim=-1)
    
    return fde


def compute_miss_rate(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    threshold: float = 2.0,
    confidences: Optional[torch.Tensor] = None
) -> float:
    """
    Compute Miss Rate: fraction of predictions where best mode exceeds threshold
    
    Args:
        predicted: [batch_size, num_modes, future_length, 2]
        ground_truth: [batch_size, future_length, 2]
        threshold: Distance threshold in meters
        confidences: [batch_size, num_modes] optional mode confidences
    
    Returns:
        Miss rate (0-1)
    """
    if predicted.dim() != 4:
        raise ValueError("Miss rate requires multi-modal predictions")
    
    # Compute FDE for each mode
    fde_per_mode = compute_fde(predicted, ground_truth)  # [batch_size, num_modes]
    
    # Select best mode (either by confidence or by lowest FDE)
    if confidences is not None:
        best_mode_idx = confidences.argmax(dim=-1)  # [batch_size]
        best_fde = fde_per_mode.gather(1, best_mode_idx.unsqueeze(1)).squeeze(1)
    else:
        best_fde = fde_per_mode.min(dim=-1)[0]  # [batch_size]
    
    # Count misses
    misses = (best_fde > threshold).float()
    miss_rate = misses.mean().item()
    
    return miss_rate


def compute_multi_modal_metrics(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    confidences: torch.Tensor,
    threshold: float = 2.0
) -> Dict[str, float]:
    """
    Compute comprehensive multi-modal evaluation metrics
    
    Args:
        predicted: [batch_size, num_modes, future_length, 2]
        ground_truth: [batch_size, future_length, 2]
        confidences: [batch_size, num_modes]
        threshold: Miss rate threshold in meters
    
    Returns:
        Dictionary of metrics
    """
    # ADE per mode
    ade_per_mode = compute_ade(predicted, ground_truth)  # [batch_size, num_modes]
    
    # FDE per mode
    fde_per_mode = compute_fde(predicted, ground_truth)  # [batch_size, num_modes]
    
    # Best mode selection (by confidence)
    best_mode_idx = confidences.argmax(dim=-1)  # [batch_size]
    batch_indices = torch.arange(predicted.shape[0], device=predicted.device)
    best_ade = ade_per_mode[batch_indices, best_mode_idx]
    best_fde = fde_per_mode[batch_indices, best_mode_idx]
    
    # Min ADE/FDE (oracle - best possible)
    min_ade = ade_per_mode.min(dim=-1)[0]
    min_fde = fde_per_mode.min(dim=-1)[0]
    
    # Confidence-weighted metrics
    conf_weights = torch.softmax(confidences, dim=-1)  # [batch_size, num_modes]
    weighted_ade = (ade_per_mode * conf_weights).sum(dim=-1)
    weighted_fde = (fde_per_mode * conf_weights).sum(dim=-1)
    
    # Miss rate
    miss_rate = compute_miss_rate(predicted, ground_truth, threshold, confidences)
    
    # Aggregate over batch
    metrics = {
        "ade": best_ade.mean().item(),
        "fde": best_fde.mean().item(),
        "min_ade": min_ade.mean().item(),
        "min_fde": min_fde.mean().item(),
        "weighted_ade": weighted_ade.mean().item(),
        "weighted_fde": weighted_fde.mean().item(),
        "miss_rate": miss_rate,
    }
    
    return metrics


def evaluate_batch(
    model_output: Dict[str, torch.Tensor],
    ground_truth: torch.Tensor,
    threshold: float = 2.0
) -> Dict[str, float]:
    """
    Evaluate a batch of predictions
    
    Args:
        model_output: Dictionary with keys:
            - "trajectories": [batch_size, num_modes, future_length, 2]
            - "confidences": [batch_size, num_modes]
        ground_truth: [batch_size, future_length, 2]
        threshold: Miss rate threshold
    
    Returns:
        Dictionary of metrics
    """
    trajectories = model_output["trajectories"]
    confidences = model_output["confidences"]
    
    return compute_multi_modal_metrics(trajectories, ground_truth, confidences, threshold)


def aggregate_metrics(metrics_list: list) -> Dict[str, float]:
    """
    Aggregate metrics across multiple batches/scenarios
    
    Args:
        metrics_list: List of metric dictionaries
    
    Returns:
        Aggregated metrics
    """
    if not metrics_list:
        return {}
    
    aggregated = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        aggregated[key] = np.mean(values)
        aggregated[f"{key}_std"] = np.std(values)
    
    return aggregated


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for output
    """
    print(f"\n{prefix}Evaluation Metrics:")
    print("-" * 50)
    print(f"ADE:           {metrics.get('ade', 0):.4f} m")
    print(f"FDE:           {metrics.get('fde', 0):.4f} m")
    print(f"Min ADE:       {metrics.get('min_ade', 0):.4f} m")
    print(f"Min FDE:       {metrics.get('min_fde', 0):.4f} m")
    print(f"Weighted ADE:  {metrics.get('weighted_ade', 0):.4f} m")
    print(f"Weighted FDE:  {metrics.get('weighted_fde', 0):.4f} m")
    print(f"Miss Rate:     {metrics.get('miss_rate', 0):.4%}")
    if 'ade_std' in metrics:
        print(f"ADE Std:       {metrics.get('ade_std', 0):.4f} m")
        print(f"FDE Std:       {metrics.get('fde_std', 0):.4f} m")
    print("-" * 50)


if __name__ == "__main__":
    # Test evaluation functions
    batch_size = 4
    num_modes = 6
    future_length = 80
    
    # Generate dummy predictions
    predicted = torch.randn(batch_size, num_modes, future_length, 2)
    ground_truth = torch.randn(batch_size, future_length, 2)
    confidences = torch.softmax(torch.randn(batch_size, num_modes), dim=-1)
    
    model_output = {
        "trajectories": predicted,
        "confidences": confidences
    }
    
    metrics = evaluate_batch(model_output, ground_truth)
    print_metrics(metrics, "Test")

