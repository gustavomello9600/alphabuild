"""
AlphaBuilder v3.1 - Kaggle Training Script
==========================================

This script is designed to run on Kaggle with GPU T4 x2.
Upload this as a Kaggle notebook and enable GPU accelerator.

Expected runtime: ~5 hours for 30 epochs (SimpleBackbone)
"""

# ============================================================================
# CELL 1: Environment Setup
# ============================================================================

import subprocess
import sys
import os

print("="*60)
print("🚀 AlphaBuilder v3.1 - Kaggle Training")
print("="*60)

# Fix NumPy version compatibility (data was serialized with NumPy 2.x)
print("\n📦 Upgrading NumPy for compatibility...")
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "numpy", "-q"], check=True)

# Check GPU
import torch
print(f"\n📊 Hardware Detection:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"       Memory: {props.total_memory / 1024**3:.1f} GB")

# ============================================================================
# CELL 2: Clone Repository and Install Dependencies
# ============================================================================

REPO_URL = "https://github.com/gustavomello9600/alphabuild.git"
REPO_DIR = "/kaggle/working/alphabuild"

if not os.path.exists(REPO_DIR):
    print(f"\n📥 Cloning repository...")
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    print(f"\n📥 Updating repository...")
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

# Add to path
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

print(f"   Working directory: {os.getcwd()}")

# ============================================================================
# CELL 3: Download Training Data
# ============================================================================

# Option A: From Kaggle Dataset (recommended)
# Add the dataset "gustavomello9600/alphabuilder-warmup-data" to your notebook

DATA_PATH = "/kaggle/input/alphabuilder-warmup-data/warmup_data.db"

# Option B: From Google Drive (backup)
if not os.path.exists(DATA_PATH):
    print("\n⚠️ Dataset not found in Kaggle input.")
    print("   Please add the dataset: gustavomello9600/alphabuilder-warmup-data")
    print("   Or upload warmup_data.db manually.")
    
    # Try local path as fallback
    DATA_PATH = "/kaggle/working/alphabuild/data/warm_up_data/warmup_data.db"
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

print(f"\n📂 Training data: {DATA_PATH}")
print(f"   Size: {os.path.getsize(DATA_PATH) / 1024**2:.1f} MB")

# ============================================================================
# CELL 4: Configure Training
# ============================================================================

# Training configuration
CONFIG = {
    'use_swin': False,          # False = SimpleBackbone, True = Swin-UNETR
    'feature_size': 24,
    'batch_size': 32,           # Per GPU (64 total with 2 GPUs)
    'epochs': 30,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_workers': 4,
    'use_amp': True,            # Mixed precision
    'val_split': 0.1,
    'patience': 10,             # Early stopping
    'save_every': 5,            # Save checkpoint every N epochs
}

print(f"\n⚙️ Training Configuration:")
for k, v in CONFIG.items():
    print(f"   {k}: {v}")

# ============================================================================
# CELL 5: Import and Setup
# ============================================================================

import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

from alphabuilder.src.neural.model import AlphaBuilderV31
from alphabuilder.src.neural.dataset import TopologyDatasetV31
from alphabuilder.src.neural.trainer import policy_loss, weighted_value_loss, LAMBDA_POLICY

# ============================================================================
# CELL 6: Create DataLoaders
# ============================================================================

print(f"\n📂 Loading dataset...")

full_dataset = TopologyDatasetV31(
    db_path=Path(DATA_PATH),
    augment=True,
    preload_to_ram=True  # Kaggle has enough RAM
)

print(f"   Total samples: {len(full_dataset):,}")

# Split train/val
val_size = int(len(full_dataset) * CONFIG['val_split'])
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"   Train samples: {len(train_dataset):,}")
print(f"   Val samples: {len(val_dataset):,}")

# NOTE: DataLoaders are created fresh each epoch to manage memory
# See training loop for DataLoader configuration

print(f"   DataLoaders will be created per-epoch for memory management")

# ============================================================================
# CELL 7: Create Model
# ============================================================================

print(f"\n🧠 Creating model...")

device = torch.device('cuda')

model = AlphaBuilderV31(
    in_channels=7,
    out_channels=2,
    feature_size=CONFIG['feature_size'],
    use_swin=CONFIG['use_swin']
)

total_params = sum(p.numel() for p in model.parameters())
print(f"   Architecture: {'Swin-UNETR' if CONFIG['use_swin'] else 'SimpleBackbone'}")
print(f"   Total parameters: {total_params:,}")

# Multi-GPU
if torch.cuda.device_count() > 1:
    print(f"   Using DataParallel on {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay']
)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=CONFIG['epochs'],
    eta_min=CONFIG['learning_rate'] * 0.01
)

# Mixed precision
scaler = GradScaler() if CONFIG['use_amp'] else None

# ============================================================================
# CELL 8: Training Functions
# ============================================================================

def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    total_p_loss = 0
    total_v_loss = 0
    n_batches = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for batch in pbar:
        state = batch['state'].to(device, non_blocking=True)
        target_policy = batch['policy'].to(device, non_blocking=True)
        target_value = batch['value'].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with autocast():
                pred_policy, pred_value = model(state)
                v_loss = weighted_value_loss(pred_value, target_value)
                p_loss = policy_loss(pred_policy, target_policy)
                loss = v_loss + LAMBDA_POLICY * p_loss
            
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
        
        total_loss += loss.item()
        total_p_loss += p_loss.item()
        total_v_loss += v_loss.item()
        n_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return {
        'loss': total_loss / n_batches,
        'policy_loss': total_p_loss / n_batches,
        'value_loss': total_v_loss / n_batches,
    }


def validate_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    total_p_loss = 0
    total_v_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            state = batch['state'].to(device, non_blocking=True)
            target_policy = batch['policy'].to(device, non_blocking=True)
            target_value = batch['value'].to(device, non_blocking=True)
            
            with autocast():
                pred_policy, pred_value = model(state)
                v_loss = weighted_value_loss(pred_value, target_value)
                p_loss = policy_loss(pred_policy, target_policy)
                loss = v_loss + LAMBDA_POLICY * p_loss
            
            total_loss += loss.item()
            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()
            n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'policy_loss': total_p_loss / n_batches,
        'value_loss': total_v_loss / n_batches,
    }

# ============================================================================
# CELL 9: Training Loop
# ============================================================================

def get_memory_info():
    """Get current GPU and RAM memory usage."""
    import psutil
    ram_percent = psutil.virtual_memory().percent
    ram_used = psutil.virtual_memory().used / 1024**3
    
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        return f"RAM: {ram_used:.1f}GB ({ram_percent:.0f}%) | GPU: {gpu_allocated:.1f}GB alloc / {gpu_reserved:.1f}GB reserved"
    return f"RAM: {ram_used:.1f}GB ({ram_percent:.0f}%)"

print(f"\n🚀 Starting training: {CONFIG['epochs']} epochs")
print("-" * 60)
print(f"📊 Initial memory: {get_memory_info()}")

# Checkpoint directory
checkpoint_dir = Path("/kaggle/working/checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Tracking
best_val_loss = float('inf')
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'lr': []}

# DataLoader config
batch_size = CONFIG['batch_size'] * max(1, torch.cuda.device_count())
num_workers = CONFIG['num_workers']

training_start = time.time()

for epoch in range(CONFIG['epochs']):
    epoch_start = time.time()
    
    # === CREATE TRAIN DATALOADER (fresh each epoch) ===
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # Train
    train_metrics = train_epoch(model, train_loader, optimizer, scaler, device)
    
    # === CLEANUP TRAIN LOADER ===
    del train_loader
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  📊 After train cleanup: {get_memory_info()}")
    
    # === CREATE VAL DATALOADER (fresh) ===
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Validate
    val_metrics = validate_epoch(model, val_loader, device)
    
    # === CLEANUP VAL LOADER ===
    del val_loader
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  📊 After val cleanup: {get_memory_info()}")
    
    # Update scheduler
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    
    # Track
    history['train_loss'].append(train_metrics['loss'])
    history['val_loss'].append(val_metrics['loss'])
    history['lr'].append(current_lr)
    
    epoch_time = time.time() - epoch_start
    samples_per_sec = len(train_dataset) / epoch_time
    
    # Log
    print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']} ({epoch_time:.1f}s, {samples_per_sec:.0f} samples/s)")
    print(f"  Train | Loss: {train_metrics['loss']:.4f} | P: {train_metrics['policy_loss']:.4f} | V: {train_metrics['value_loss']:.4f}")
    print(f"  Val   | Loss: {val_metrics['loss']:.4f} | P: {val_metrics['policy_loss']:.4f} | V: {val_metrics['value_loss']:.4f}")
    print(f"  LR: {current_lr:.2e}")
    
    # Save best
    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        patience_counter = 0
        
        # Memory cleanup before checkpoint
        torch.cuda.empty_cache()
        print(f"  📊 Before checkpoint: {get_memory_info()}")
        
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'config': CONFIG,
        }, checkpoint_dir / "best_model.pt")
        print(f"  ✓ New best model saved (val_loss: {best_val_loss:.4f})")
        del model_state  # Free memory immediately
    else:
        patience_counter += 1
    
    # Save periodic
    if (epoch + 1) % CONFIG['save_every'] == 0:
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
            'config': CONFIG,
        }, checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt")
    
    # Early stopping
    if patience_counter >= CONFIG['patience']:
        print(f"\n⚠️ Early stopping triggered (patience={CONFIG['patience']})")
        break

# ============================================================================
# CELL 10: Final Summary
# ============================================================================

total_time = time.time() - training_start

print("\n" + "=" * 60)
print("🎉 Training Complete!")
print("=" * 60)
print(f"   Total time: {total_time / 3600:.1f} hours")
print(f"   Best validation loss: {best_val_loss:.4f}")
print(f"   Checkpoints saved to: {checkpoint_dir}")

# List saved files
print(f"\n📁 Saved files:")
for f in sorted(checkpoint_dir.glob("*.pt")):
    print(f"   {f.name} ({f.stat().st_size / 1024**2:.1f} MB)")

# Plot training history
try:
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history['lr'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('LR Schedule')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(checkpoint_dir / "training_history.png", dpi=150)
    plt.show()
    print(f"\n📊 Training history saved to: {checkpoint_dir / 'training_history.png'}")
except Exception as e:
    print(f"\n⚠️ Could not plot history: {e}")

print("\n✅ Done! Download checkpoints from /kaggle/working/checkpoints/")
