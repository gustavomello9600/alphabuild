"""
Training Pipeline for AlphaBuilder v3.1.

Robust training script with:
- Automatic hardware detection
- Multi-GPU support via DataParallel
- Efficient data loading with prefetching
- TensorBoard logging
- Checkpoint save/restore
- Early stopping
"""
import argparse
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.neural.model import AlphaBuilderV31
from alphabuilder.src.neural.dataset import TopologyDatasetV31
from alphabuilder.src.neural.trainer import policy_loss, weighted_value_loss, LAMBDA_POLICY
from alphabuilder.src.neural.train_config import TrainConfig, get_config, CONFIGS


def detect_environment() -> Tuple[str, torch.device]:
    """
    Detect hardware and return appropriate config name and device.
    
    Returns:
        (config_name, device) tuple
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0).lower()
        
        if 't4' in gpu_name or 'tesla' in gpu_name:
            if gpu_count >= 2:
                return 'kaggle_t4_dual_swin', torch.device('cuda')
            else:
                return 'kaggle_t4_swin', torch.device('cuda')
        else:
            # Generic CUDA device
            return 'kaggle_t4_swin', torch.device('cuda')
    
    # Check for Intel integrated GPU (via PyTorch XPU or fallback to CPU)
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return 'local_iris', torch.device('xpu')
    except:
        pass
    
    # CPU fallback
    return 'local_iris', torch.device('cpu')


def create_dataloaders(
    config: TrainConfig,
    db_path: str
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.
    
    Args:
        config: Training configuration
        db_path: Path to SQLite database
        
    Returns:
        (train_loader, val_loader) tuple
    """
    print(f"\n📂 Loading dataset from: {db_path}")
    
    # Load full dataset
    full_dataset = TopologyDatasetV31(
        db_path=Path(db_path),
        augment=True,
        preload_to_ram=True
    )
    
    print(f"   Total samples: {len(full_dataset):,}")
    print(f"   Memory usage: {full_dataset.get_memory_usage_mb():.1f} MB")
    
    # Split into train/val
    val_size = int(len(full_dataset) * config.val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Disable augmentation for validation
    # Note: This requires a wrapper since random_split returns Subset
    
    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Val samples: {len(val_dataset):,}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        drop_last=True,  # Avoid small batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
    )
    
    batches_per_epoch = len(train_loader)
    print(f"   Batches per epoch: {batches_per_epoch}")
    
    return train_loader, val_loader


def create_model(config: TrainConfig, device: torch.device) -> nn.Module:
    """
    Create and configure model.
    
    Args:
        config: Training configuration
        device: Target device
        
    Returns:
        Configured model on device
    """
    print(f"\n🧠 Creating model...")
    print(f"   Architecture: {'Swin-UNETR' if config.use_swin else 'SimpleBackbone'}")
    print(f"   Feature size: {config.feature_size}")
    
    model = AlphaBuilderV31(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        feature_size=config.feature_size,
        use_swin=config.use_swin
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"   Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    return model


def optimize_for_xpu(model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    """
    Apply Intel IPEX optimizations for XPU devices.
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        device: Target device
        
    Returns:
        (optimized_model, optimized_optimizer) tuple
    """
    if device.type != 'xpu':
        return model, optimizer
    
    try:
        import intel_extension_for_pytorch as ipex
        print(f"   Applying IPEX optimizations for XPU...")
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
        print(f"   ✓ IPEX optimization applied")
    except ImportError:
        print(f"   ⚠ IPEX not available, running without optimizations")
    except Exception as e:
        print(f"   ⚠ IPEX optimization failed: {e}")
    
    return model, optimizer


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    use_amp: bool,
    log_every: int = 100
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Target device
        scaler: GradScaler for mixed precision
        use_amp: Whether to use automatic mixed precision
        log_every: Log progress every N steps
        
    Returns:
        Dictionary with loss metrics
    """
    model.train()
    
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        state = batch['state'].to(device, non_blocking=True)
        target_policy = batch['policy'].to(device, non_blocking=True)
        target_value = batch['value'].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with optional AMP
        if use_amp and scaler is not None:
            with autocast():
                pred_policy, pred_value = model(state)
                v_loss = weighted_value_loss(pred_value, target_value)
                p_loss = policy_loss(pred_policy, target_policy)
                loss = v_loss + LAMBDA_POLICY * p_loss
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_policy, pred_value = model(state)
            v_loss = weighted_value_loss(pred_value, target_value)
            p_loss = policy_loss(pred_policy, target_policy)
            loss = v_loss + LAMBDA_POLICY * p_loss
            
            loss.backward()
            optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_policy_loss += p_loss.item()
        total_value_loss += v_loss.item()
        num_batches += 1
        
        # Update progress bar
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'p_loss': f"{p_loss.item():.4f}",
                'v_loss': f"{v_loss.item():.4f}",
            })
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'policy_loss': total_policy_loss / max(num_batches, 1),
        'value_loss': total_value_loss / max(num_batches, 1),
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Args:
        model: Neural network model
        dataloader: Validation data loader
        device: Target device
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Dictionary with loss metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            state = batch['state'].to(device, non_blocking=True)
            target_policy = batch['policy'].to(device, non_blocking=True)
            target_value = batch['value'].to(device, non_blocking=True)
            
            if use_amp:
                with autocast():
                    pred_policy, pred_value = model(state)
                    v_loss = weighted_value_loss(pred_value, target_value)
                    p_loss = policy_loss(pred_policy, target_policy)
                    loss = v_loss + LAMBDA_POLICY * p_loss
            else:
                pred_policy, pred_value = model(state)
                v_loss = weighted_value_loss(pred_value, target_value)
                p_loss = policy_loss(pred_policy, target_policy)
                loss = v_loss + LAMBDA_POLICY * p_loss
            
            total_loss += loss.item()
            total_policy_loss += p_loss.item()
            total_value_loss += v_loss.item()
            num_batches += 1
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'policy_loss': total_policy_loss / max(num_batches, 1),
        'value_loss': total_value_loss / max(num_batches, 1),
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: TrainConfig,
    path: Path
):
    """Save training checkpoint."""
    # Handle DataParallel wrapper
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': {
            'use_swin': config.use_swin,
            'feature_size': config.feature_size,
            'in_channels': config.in_channels,
            'out_channels': config.out_channels,
        }
    }
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> int:
    """
    Load training checkpoint.
    
    Returns:
        Starting epoch number
    """
    checkpoint = torch.load(path, map_location='cpu')
    
    # Handle DataParallel wrapper
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'] + 1


def main(
    config_name: Optional[str] = None,
    epochs: Optional[int] = None,
    db_path: Optional[str] = None,
    resume: Optional[str] = None,
):
    """
    Main training function.
    
    Args:
        config_name: Name of pre-defined config (auto-detected if None)
        epochs: Override number of epochs
        db_path: Override database path
        resume: Path to checkpoint to resume from
    """
    print("=" * 60)
    print("🏗️  AlphaBuilder v3.1 - Training Pipeline")
    print("=" * 60)
    
    # Detect environment
    detected_config, device = detect_environment()
    config_name = config_name or detected_config
    
    print(f"\n⚙️  Configuration: {config_name}")
    print(f"   Device: {device}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Load config
    config = get_config(config_name)
    
    # Override values if provided
    if epochs is not None:
        config.epochs = epochs
    if db_path is not None:
        config.db_path = db_path
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config_name}_{timestamp}"
    
    # Create log directory
    log_dir = Path(config.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard setup
    writer = None
    if config.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir)
            print(f"   TensorBoard: {log_dir}")
        except ImportError:
            print("   TensorBoard: Not available")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, config.db_path)
    
    # Create model
    model = create_model(config, device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Apply IPEX optimizations for XPU
    model, optimizer = optimize_for_xpu(model, optimizer, device)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * 0.01
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.use_amp and device.type == 'cuda' else None
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume:
        print(f"\n📂 Resuming from: {resume}")
        start_epoch = load_checkpoint(Path(resume), model, optimizer)
        print(f"   Resuming from epoch {start_epoch}")
    
    # Early stopping state
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    print(f"\n🚀 Starting training: {config.epochs} epochs")
    print("-" * 60)
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            scaler, config.use_amp, config.log_every_n_steps
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, config.use_amp)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - epoch_start
        samples_per_sec = len(train_loader.dataset) / epoch_time
        
        # Log to console
        print(f"\nEpoch {epoch + 1}/{config.epochs} ({epoch_time:.1f}s, {samples_per_sec:.0f} samples/sec)")
        print(f"  Train | Loss: {train_metrics['loss']:.4f} | Policy: {train_metrics['policy_loss']:.4f} | Value: {train_metrics['value_loss']:.4f}")
        print(f"  Val   | Loss: {val_metrics['loss']:.4f} | Policy: {val_metrics['policy_loss']:.4f} | Value: {val_metrics['value_loss']:.4f}")
        print(f"  LR: {current_lr:.2e}")
        
        # Log to TensorBoard
        if writer:
            global_step = (epoch + 1) * len(train_loader)
            writer.add_scalar('loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('loss/train_policy', train_metrics['policy_loss'], epoch)
            writer.add_scalar('loss/train_value', train_metrics['value_loss'], epoch)
            writer.add_scalar('loss/val_policy', val_metrics['policy_loss'], epoch)
            writer.add_scalar('loss/val_value', val_metrics['value_loss'], epoch)
            writer.add_scalar('lr', current_lr, epoch)
            writer.add_scalar('throughput/samples_per_sec', samples_per_sec, epoch)
        
        # Save checkpoints
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_metrics, config,
                checkpoint_dir / "best_model.pt"
            )
            print(f"  ✓ New best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Save periodic checkpoints
        if (epoch + 1) % config.save_every_n_epochs == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics, config,
                checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            )
        
        # Save last model (always)
        save_checkpoint(
            model, optimizer, epoch, val_metrics, config,
            checkpoint_dir / "last_model.pt"
        )
        
        # Early stopping check
        if patience_counter >= config.patience:
            print(f"\n⚠️  Early stopping triggered (patience={config.patience})")
            break
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Checkpoints saved to: {checkpoint_dir}")
    if writer:
        print(f"   TensorBoard logs: {log_dir}")
    print("=" * 60)
    
    if writer:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaBuilder v3.1 Training")
    parser.add_argument(
        "--config", type=str, default=None,
        help=f"Config preset: {list(CONFIGS.keys())}"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Override database path"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint path"
    )
    
    args = parser.parse_args()
    
    main(
        config_name=args.config,
        epochs=args.epochs,
        db_path=args.db_path,
        resume=args.resume,
    )
