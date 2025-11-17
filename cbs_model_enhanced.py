

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import config


class PrimitiveEncoder(nn.Module):
    """
    Specialized encoder for a single primitive type
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # LSTM encoder for temporal features
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        
        Returns:
            encoded: [batch_size, hidden_dim]
        """
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state
        final_hidden = h_n[-1]  # [batch_size, hidden_dim]
        
        # Project to primitive-specific representation
        encoded = self.projection(final_hidden)
        
        return encoded


class CompositionModule(nn.Module):
    """
    Learns to dynamically combine primitive encodings
    Enables zero-shot generalization to novel primitive combinations
    """
    
    def __init__(self, hidden_dim: int, num_primitives: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_primitives = num_primitives
        
        # Attention mechanism for primitive selection
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Composition weights network
        self.composition_net = nn.Sequential(
            nn.Linear(hidden_dim * num_primitives, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Learnable primitive embeddings (for attention)
        self.primitive_embeddings = nn.Parameter(
            torch.randn(1, num_primitives, hidden_dim)
        )
    
    def forward(self, primitive_encodings: torch.Tensor) -> torch.Tensor:
        """
        Compose primitive encodings into unified representation
        
        Args:
            primitive_encodings: [batch_size, num_primitives, hidden_dim]
        
        Returns:
            composed: [batch_size, hidden_dim]
        """
        batch_size = primitive_encodings.shape[0]
        
        # Expand primitive embeddings for batch
        primitive_keys = self.primitive_embeddings.expand(batch_size, -1, -1)
        
        # Use primitive encodings as queries
        queries = primitive_encodings
        
        # Attention: which primitives are most relevant?
        attended, attention_weights = self.attention(
            query=queries,
            key=primitive_keys,
            value=primitive_encodings
        )
        
        # Aggregate attended features
        attended_agg = attended.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Concatenate all primitive encodings
        primitive_concat = primitive_encodings.view(batch_size, -1)  # [batch_size, num_primitives * hidden_dim]
        
        # Learn composition
        composed = self.composition_net(primitive_concat)
        
        # Combine attention and composition
        final = composed + attended_agg
        
        return final, attention_weights


class MultiModalPredictionHead(nn.Module):
    """
    Generates multiple trajectory modes with confidences
    """
    
    def __init__(
        self,
        hidden_dim: int,
        future_length: int,
        num_modes: int,
        output_dim: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.future_length = future_length
        self.num_modes = num_modes
        self.output_dim = output_dim
        
        # Mode-specific decoders (deeper for better mode differentiation)
        self.mode_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_modes)
        ])
        
        # Trajectory generation (LSTM decoder for each mode)
        # Use initial hidden state from composed features
        self.trajectory_decoders = nn.ModuleList([
            nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout
            ) for _ in range(num_modes)
        ])
        
        # Initial hidden state projection for each mode
        self.initial_hidden_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim * 2) for _ in range(num_modes)  # For h0 and c0
        ])
        
        # Time embedding projection (shared across modes) - stronger encoding
        self.time_projection = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Feedback projection: map previous output back to LSTM input space
        self.feedback_projections = nn.ModuleList([
            nn.Linear(output_dim, hidden_dim // 2) for _ in range(num_modes)
        ])
        
        # Output projection for each mode (with residual connection)
        self.output_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in range(num_modes)
        ])
        
        # Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_modes)
        )
    
    def forward(self, composed_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate multi-modal trajectory predictions
        
        Args:
            composed_features: [batch_size, hidden_dim]
        
        Returns:
            Dictionary with:
                - trajectories: [batch_size, num_modes, future_length, output_dim]
                - confidences: [batch_size, num_modes]
        """
        batch_size = composed_features.shape[0]
        
        # Generate trajectory for each mode
        trajectories = []
        
        for mode_idx in range(self.num_modes):
            # Mode-specific encoding
            mode_features = self.mode_decoders[mode_idx](composed_features)
            
            # Initialize LSTM hidden state from composed features
            initial_hidden = self.initial_hidden_projections[mode_idx](composed_features)
            h0 = initial_hidden[:, :self.hidden_dim].unsqueeze(0).repeat(2, 1, 1)  # [2, batch, hidden]
            c0 = initial_hidden[:, self.hidden_dim:].unsqueeze(0).repeat(2, 1, 1)  # [2, batch, hidden]
            
            # Autoregressive decoding: use previous predictions as input
            # Initialize with zero position (origin in local coordinates)
            prev_output = torch.zeros(batch_size, self.output_dim, device=composed_features.device)
            lstm_outputs = []
            
            # Create time embeddings (stronger temporal signal)
            time_steps = torch.arange(self.future_length, device=composed_features.device, dtype=torch.float32)
            # Normalize time to [0, 1] range for better learning
            time_normalized = (time_steps / (self.future_length - 1)).unsqueeze(0).unsqueeze(-1)  # [1, future_length, 1]
            time_normalized = time_normalized.expand(batch_size, -1, -1)  # [batch, future_length, 1]
            
            # Process each timestep with autoregressive feedback
            hidden_state = (h0, c0)
            trajectory_outputs = []
            for t in range(self.future_length):
                # Time embedding for this timestep
                time_emb = self.time_projection(time_normalized[:, t:t+1, :])  # [batch, 1, hidden]
                
                # Feedback from previous output
                feedback = self.feedback_projections[mode_idx](prev_output)  # [batch, hidden//2]
                feedback = feedback.unsqueeze(1)  # [batch, 1, hidden//2]
                feedback_expanded = torch.zeros(batch_size, 1, self.hidden_dim, device=composed_features.device)
                feedback_expanded[:, :, :self.hidden_dim // 2] = feedback
                
                # Combine mode features, time embedding, and feedback
                mode_base = mode_features.unsqueeze(1)  # [batch, 1, hidden]
                lstm_input = mode_base + 0.5 * time_emb + 0.3 * feedback_expanded  # [batch, 1, hidden]
                
                # LSTM step
                lstm_out, hidden_state = self.trajectory_decoders[mode_idx](lstm_input, hidden_state)
                
                # Project to trajectory coordinates
                step_output = self.output_projections[mode_idx](lstm_out)  # [batch, 1, 2]
                trajectory_outputs.append(step_output)
                prev_output = step_output.squeeze(1)  # [batch, 2] - use for next timestep
            
            # Stack all outputs
            trajectory = torch.cat(trajectory_outputs, dim=1)  # [batch, future_length, 2]
            
            # Ensure first timestep starts at origin (already normalized to local coordinates)
            trajectory[:, 0, :] = 0.0
            
            trajectories.append(trajectory)
        
        # Stack trajectories
        trajectories = torch.stack(trajectories, dim=1)  # [batch_size, num_modes, future_length, output_dim]
        
        # Predict base confidences from composed features
        base_confidences = self.confidence_head(composed_features)  # [batch_size, num_modes]
        
        # Note: During training, confidence will be refined based on actual trajectory quality
        # via the loss function. Here we just provide initial predictions.
        # Apply temperature scaling for sharper confidence distribution
        temperature = 1.2  # Slightly reduced for sharper distribution
        confidences = F.softmax(base_confidences / temperature, dim=-1)
        
        return {
            "trajectories": trajectories,
            "confidences": confidences,
            "confidence_logits": base_confidences  # For loss computation
        }


class CBSModel(nn.Module):
    """
    Complete Compositional Behavior Synthesis Model
    """
    
    def __init__(
        self,
        input_dim: int = None,
        history_length: int = None,
        future_length: int = None,
        hidden_dim: int = None,
        num_modes: int = None,
        num_primitives: int = None,
        num_layers: int = None,
        dropout: float = None
    ):
        super().__init__()
        
        # Use config defaults if not specified
        cfg = config.MODEL_CONFIG
        self.input_dim = input_dim or cfg["input_dim"]
        self.history_length = history_length or cfg["history_length"]
        self.future_length = future_length or cfg["future_length"]
        self.hidden_dim = hidden_dim or cfg["hidden_dim"]
        self.num_modes = num_modes or cfg["num_modes"]
        self.num_primitives = num_primitives or cfg["num_primitives"]
        self.num_layers = num_layers or cfg["num_layers"]
        self.dropout = dropout or cfg["dropout"]
        
        # Input encoder (shared across primitives)
        self.input_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 4 specialized primitive encoders
        self.primitive_encoders = nn.ModuleList([
            PrimitiveEncoder(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout
            ) for _ in range(self.num_primitives)
        ])
        
        # Composition module
        self.composition_module = CompositionModule(
            hidden_dim=self.hidden_dim,
            num_primitives=self.num_primitives,
            dropout=self.dropout
        )
        
        # Multi-modal prediction head
        self.prediction_head = MultiModalPredictionHead(
            hidden_dim=self.hidden_dim,
            future_length=self.future_length,
            num_modes=self.num_modes,
            output_dim=2,  # x, y coordinates
            dropout=self.dropout
        )
        
        # Primitive classification head (for multi-task learning)
        self.primitive_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_primitives)
        )
    
    def forward(
        self,
        history: torch.Tensor,
        return_primitives: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            history: [batch_size, history_length, input_dim]
            return_primitives: Whether to return primitive predictions
        
        Returns:
            Dictionary with:
                - trajectories: [batch_size, num_modes, future_length, 2]
                - confidences: [batch_size, num_modes]
                - primitive_logits: [batch_size, num_primitives] (if return_primitives=True)
                - composition_weights: attention weights (if return_primitives=True)
        """
        batch_size = history.shape[0]
        
        # Encode input
        encoded_input = self.input_encoder(history)  # [batch_size, history_length, hidden_dim]
        
        # Encode with each primitive encoder
        primitive_encodings = []
        for encoder in self.primitive_encoders:
            primitive_encoding = encoder(encoded_input)  # [batch_size, hidden_dim]
            primitive_encodings.append(primitive_encoding)
        
        # Stack primitive encodings
        primitive_encodings = torch.stack(primitive_encodings, dim=1)  # [batch_size, num_primitives, hidden_dim]
        
        # Compose primitives
        composed_features, attention_weights = self.composition_module(primitive_encodings)
        
        # Generate predictions
        predictions = self.prediction_head(composed_features)
        
        result = {
            "trajectories": predictions["trajectories"],
            "confidences": predictions["confidences"]
        }
        
        # Include confidence logits if available
        if "confidence_logits" in predictions:
            result["confidence_logits"] = predictions["confidence_logits"]
        
        if return_primitives:
            # Predict primitive activations
            primitive_logits = self.primitive_classifier(composed_features)
            result["primitive_logits"] = primitive_logits
            result["composition_weights"] = attention_weights
        
        return result
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(device: torch.device = None) -> CBSModel:
    """Create and initialize CBS model"""
    if device is None:
        device = config.get_device()
    
    model = CBSModel()
    model = model.to(device)
    
    # Initialize weights
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            if 'output_projections' in name or 'output' in name.lower():
                # Initialize output layers with smaller weights for stability
                nn.init.xavier_uniform_(param, gain=0.1)
            else:
                nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)
    
    return model


if __name__ == "__main__":
    # Test model
    print("Testing CBS Model...")
    
    device = config.get_device()
    model = create_model(device)
    
    print(f"\nModel Parameters: {model.count_parameters():,}")
    print(f"Target: ~8,500,000")
    
    # Test forward pass
    batch_size = 4
    history = torch.randn(batch_size, config.MODEL_CONFIG["history_length"], config.MODEL_CONFIG["input_dim"]).to(device)
    
    print(f"\nInput shape: {history.shape}")
    
    with torch.no_grad():
        output = model(history, return_primitives=True)
    
    print(f"\nOutput shapes:")
    print(f"  Trajectories: {output['trajectories'].shape}")
    print(f"  Confidences: {output['confidences'].shape}")
    print(f"  Primitive logits: {output['primitive_logits'].shape}")
    print(f"  Composition weights: {output['composition_weights'].shape}")
    
    print("\nModel test completed successfully!")

