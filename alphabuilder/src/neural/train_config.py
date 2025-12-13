"""
Training Configuration for AlphaBuilder v3.1.

Centralized configuration dataclasses for different training environments.
"""
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class TrainConfig:
    """Base training configuration."""
    
    # Model architecture
    use_swin: bool = False
    feature_size: int = 24
    in_channels: int = 7
    out_channels: int = 2
    
    # Training hyperparameters
    batch_size: int = 8
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    
    # DataLoader settings
    num_workers: int = 4
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Logging
    log_every_n_steps: int = 100
    save_every_n_epochs: int = 5
    use_tensorboard: bool = True
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Paths
    db_path: str = "data/warm_up_data/merged_warmup_data.db"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Validation split
    val_split: float = 0.1
    
    # Mixed precision
    use_amp: bool = True
    
    def __post_init__(self):
        """Create directories if needed."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


# Pre-defined configurations for different environments
CONFIGS = {
    'local_iris': TrainConfig(
        use_swin=False,
        feature_size=24,
        batch_size=1,  # Optimal for 64x32x32 tensors on Iris Xe
        epochs=30,
        num_workers=0,  # SQLite lazy-loading works best with single worker
        prefetch_factor=2,
        use_amp=False,  # XPU uses different AMP path
        log_every_n_steps=50,
        db_path="data/warm_up_data/warmup_data.db",
    ),
    
    'kaggle_t4_simple': TrainConfig(
        use_swin=False,
        feature_size=24,
        batch_size=64,
        epochs=100,
        num_workers=4,
        prefetch_factor=4,
        use_amp=True,
    ),
    
    'kaggle_t4_swin': TrainConfig(
        use_swin=True,
        feature_size=24,
        batch_size=8,
        epochs=100,
        num_workers=4,
        prefetch_factor=4,
        use_amp=True,
    ),
    
    'kaggle_t4_dual_simple': TrainConfig(
        use_swin=False,
        feature_size=24,
        batch_size=128,  # 64 per GPU
        epochs=100,
        num_workers=4,
        prefetch_factor=4,
        use_amp=True,
    ),
    
    'kaggle_t4_dual_swin': TrainConfig(
        use_swin=True,
        feature_size=24,
        batch_size=16,  # 8 per GPU
        epochs=100,
        num_workers=4,
        prefetch_factor=4,
        use_amp=True,
    ),
}


def get_config(name: str) -> TrainConfig:
    """Get a pre-defined configuration by name."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]
