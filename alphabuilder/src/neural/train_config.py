"""
Training Configuration for AlphaBuilder v3.3.

Centralized configuration dataclasses for local runs and Kaggle T4 x2 training.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    """Base training configuration for AlphaBuilder v3.3."""

    # Model architecture
    use_swin: bool = False
    feature_size: int = 24
    in_channels: int = 7
    out_channels: int = 2

    # Training hyperparameters
    batch_size: int = 8  # Per-GPU batch size under DDP (or global under single-process)
    epochs: int = 40
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    grad_clip_norm: float = 1.0

    # Loss weights (calibrated v3.3)
    lambda_policy: float = 1.0
    lambda_value: float = 2.0
    lambda_bp: float = 0.50

    # DataLoader & Symmetry settings
    num_workers: int = 4
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    enable_rot90_collate: bool = True

    # Logging & Checkpointing
    log_every_n_steps: int = 100
    save_every_n_epochs: int = 5
    use_tensorboard: bool = True

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Paths & Split
    db_path: str = "data/warm_up_data/warmup_v3_3_kaggle.db"
    split_json: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    val_split: float = 0.1

    # Mixed precision
    use_amp: bool = True

    def __post_init__(self):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


CONFIGS = {
    "local_iris": TrainConfig(
        use_swin=False,
        feature_size=24,
        batch_size=4,
        epochs=20,
        learning_rate=3e-4,
        num_workers=0,
        prefetch_factor=2,
        pin_memory=False,
        persistent_workers=False,
        use_amp=False,
        log_every_n_steps=50,
    ),
    "kaggle_t4_simple": TrainConfig(
        use_swin=False,
        feature_size=24,
        batch_size=64,
        epochs=40,
        learning_rate=3e-4,
        warmup_epochs=3,
        num_workers=4,
        prefetch_factor=4,
        use_amp=True,
    ),
    "kaggle_t4_swin": TrainConfig(
        use_swin=True,
        feature_size=24,
        batch_size=8,
        epochs=40,
        learning_rate=2e-4,
        warmup_epochs=3,
        num_workers=4,
        prefetch_factor=4,
        use_amp=True,
    ),
    "kaggle_t4_dual_simple": TrainConfig(
        use_swin=False,
        feature_size=24,
        batch_size=64,  # Per GPU (128 global under 2-rank DDP)
        epochs=40,
        learning_rate=3e-4,
        warmup_epochs=3,
        num_workers=4,
        prefetch_factor=4,
        use_amp=True,
    ),
    "kaggle_t4_dual_swin": TrainConfig(
        use_swin=True,
        feature_size=24,
        batch_size=8,   # Per GPU (16 global under 2-rank DDP)
        epochs=40,
        learning_rate=2e-4,
        warmup_epochs=3,
        num_workers=4,
        prefetch_factor=4,
        use_amp=True,
    ),
}


def get_config(name: str) -> TrainConfig:
    """Return a fresh copy-like TrainConfig preset by name."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    cfg = CONFIGS[name]
    return TrainConfig(**vars(cfg))
