"""
Configuration file for CBS Trajectory Prediction System
Centralized configuration for model hyperparameters, data paths, and training settings
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data paths
DATA_ROOT = PROJECT_ROOT / "waymo_motion_data"
WAYMO_DATA_CACHE = PROJECT_ROOT / "waymo_data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [WAYMO_DATA_CACHE, OUTPUTS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model Architecture Parameters
MODEL_CONFIG = {
    "history_length": 10,      # 1 second at 10Hz
    "future_length": 80,        # 8 seconds at 10Hz
    "hidden_dim": 256,         # Increased from 128 for better capacity
    "num_modes": 6,            # Multi-modal predictions
    "num_primitives": 4,        # LF, LC, Y, T
    "input_dim": 7,             # [x, y, heading, vx, vy, ax, ay]
    "num_layers": 2,
    "dropout": 0.15,           # Slightly increased for regularization
}

# Training Configuration
TRAIN_CONFIG = {
    "batch_size": 32,
    "learning_rate": 5e-4,  # Reduced from 1e-3 for more stable training
    "weight_decay": 1e-4,
    "num_epochs": 100,
    "gradient_clip": 1.0,
    "early_stopping_patience": 15,  # Increased patience
    "save_every": 5,            # Save checkpoint every N epochs
    
    # Loss weights for multi-task learning
    "trajectory_loss_weight": 1.0,
    "primitive_loss_weight": 0.2,  # Reduced to focus more on trajectory
    "confidence_loss_weight": 0.5,  # Increased - critical for mode selection

    # EMTA-style mode diversification settings
    "emta": {
        "enabled": True,
        "initial_tolerance": 4.0,      # start coarse to allow exploration
        "final_tolerance": 2.0,        # tighten as training progresses
        "lock_warmup_epochs": 8,       # epochs before hard locking begins
        "fallback_scale": 1.6,         # fallback tolerance multiplier
        "coverage_margin": 3.0,       # desired minimum separation in meters
        "coverage_weight": 0.12,      # loss weight for coverage penalty
        "coverage_timesteps": [20, 40, 79],  # enforce separation along trajectory
        "temperature": 2.0,           # softness for ADE-derived confidence targets
        "confidence_rank_weight": 0.1 # weight for confidence ranking regularizer
    },
 
    # Curriculum learning phases
    "curriculum_phases": {
        "phase1_epochs": 20,    # Single primitives
        "phase2_epochs": 40,    # Primitive combinations
        "phase3_epochs": 40,    # Novel combinations (testing)
    },
}

# Data Configuration
DATA_CONFIG = {
    "num_agents": 128,
    "num_lanes": 215,           # Approximate, varies per scenario
    "total_timesteps": 91,      # 10 past + 1 current + 80 future
    "past_timesteps": 10,
    "current_timesteps": 1,
    "future_timesteps": 80,
    "fps": 10,                  # 10Hz data
    "cache_preprocessed": True,
    "num_workers": 4,           # DataLoader workers
    "pin_memory": True,         # For GPU transfer
}

# Evaluation Configuration
EVAL_CONFIG = {
    "miss_threshold": 2.0,      # meters for miss rate calculation
    "num_eval_scenarios": 50,
    "eval_batch_size": 16,
}

# GPU Configuration
GPU_CONFIG = {
    "device": "cuda",  # Will be determined by torch.cuda.is_available()
    "mixed_precision": True,    # Use FP16 for memory efficiency
    "max_memory_gb": 12,        # RTX 3080 Ti has 12GB VRAM
}

# Primitive Detection Configuration
PRIMITIVE_CONFIG = {
    "lane_following": {
        "lateral_threshold": 0.5,      # meters from lane center
        "heading_threshold": 0.2,      # radians
    },
    "lane_changing": {
        "lateral_velocity_threshold": 0.3,  # m/s
        "lane_crossing_threshold": 1.0,     # meters
    },
    "yielding": {
        "deceleration_threshold": -0.5,    # m/s^2
        "proximity_threshold": 5.0,         # meters
    },
    "turning": {
        "heading_rate_threshold": 0.1,     # rad/s
        "curvature_threshold": 0.05,        # 1/m
    },
}

# Visualization Configuration
VIZ_CONFIG = {
    "video_fps": 10,
    "figure_size": (18, 6),     # 3 panels: 6x6 each
    "dpi": 100,
    "trail_length": 20,          # Number of past positions to show
    "agent_size": 2.0,
}

# TNT Baseline Configuration (if implemented)
TNT_CONFIG = {
    "weights_path": PROJECT_ROOT / "tnt_baseline" / "TNT" / "TNT" / "best_TNT.pth",
    "enabled": False,            # Set to True if TNT baseline is implemented
}

def get_device():
    """Get the computation device (CUDA if available, else CPU)"""
    import torch
    if torch.cuda.is_available() and GPU_CONFIG["device"] == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")

def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("CBS Trajectory Prediction System Configuration")
    print("=" * 60)
    print(f"\nModel Config:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")
    print(f"\nTraining Config:")
    for key, value in TRAIN_CONFIG.items():
        if key != "curriculum_phases":
            print(f"  {key}: {value}")
    print(f"\nData Config:")
    for key, value in DATA_CONFIG.items():
        print(f"  {key}: {value}")
    print(f"\nDevice: {get_device()}")
    print("=" * 60)

if __name__ == "__main__":
    print_config()

