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
    architecture: str = "simple"
    use_swin: bool = False
    feature_size: int = 24
    embed_dim: int = 192
    transformer_depth: int = 5
    num_heads: int = 6
    in_channels: int = 7
    out_channels: int = 2

    # Training hyperparameters
    batch_size: int = 8  # Per-GPU batch size under DDP (or global under single-process)
    epochs: int = 40
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    grad_clip_norm: float = 1.0

    # Loss weights (calibrated v3.3 + v4.0 factored physics)
    lambda_policy: float = 1.0
    lambda_value: float = 2.0
    lambda_bp: float = 0.50
    lambda_log_c: float = 0.0
    lambda_conn: float = 0.0
    lambda_sed: float = 0.0

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
    max_wall_hours: Optional[float] = None

    def __post_init__(self):
        if self.use_swin and self.architecture == "simple":
            self.architecture = "swin"
        if self.architecture == "swin":
            self.use_swin = True
        elif self.architecture == "phys_uvit":
            self.use_swin = False
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
    "local_phys_uvit": TrainConfig(
        architecture="phys_uvit",
        use_swin=False,
        feature_size=32,
        embed_dim=192,
        transformer_depth=5,
        num_heads=6,
        batch_size=4,
        epochs=20,
        learning_rate=3e-4,
        weight_decay=1e-2,
        warmup_epochs=2,
        lambda_log_c=0.20,
        lambda_conn=0.25,
        lambda_sed=0.0,
        num_workers=0,
        prefetch_factor=2,
        pin_memory=False,
        persistent_workers=False,
        use_amp=False,
        log_every_n_steps=50,
    ),
    "kaggle_t4_phys_uvit": TrainConfig(
        architecture="phys_uvit",
        use_swin=False,
        feature_size=32,
        embed_dim=192,
        transformer_depth=5,
        num_heads=6,
        batch_size=64,
        epochs=30,
        learning_rate=3e-4,
        weight_decay=1e-2,
        warmup_epochs=3,
        lambda_log_c=0.20,
        lambda_conn=0.25,
        lambda_sed=0.0,
        num_workers=4,
        prefetch_factor=4,
        use_amp=True,
    ),
    "kaggle_t4_dual_phys_uvit": TrainConfig(
        architecture="phys_uvit",
        use_swin=False,
        feature_size=32,
        embed_dim=192,
        transformer_depth=5,
        num_heads=6,
        batch_size=64,  # Per GPU (128 global under 2-rank DDP)
        epochs=30,
        learning_rate=3e-4,
        weight_decay=1e-2,
        warmup_epochs=3,
        lambda_log_c=0.20,
        lambda_conn=0.25,
        lambda_sed=0.0,
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
