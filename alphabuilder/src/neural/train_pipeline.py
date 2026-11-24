"""
Training Pipeline for AlphaBuilder v3.3.

Supports:
- Strict episode-level train/val split (`split_v3_3.json` or deterministic fallback) with augment=False on validation
- Full D4 x Z2 = 16x symmetry augmentation (8x flips in dataset + 2x rot90 in train collate_fn)
- Multi-head training: masked KL/CE policy + Huber value + Blueprint BCE+Dice
- Multi-GPU DDP via `torchrun` (`LOCAL_RANK` / `WORLD_SIZE`) as well as single-GPU/CPU execution
- PyTorch AMP (`torch.amp.autocast` / `GradScaler`), LR warmup + cosine annealing, grad clipping
- Console, TensorBoard, and `metrics_history.json` logging
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.neural.dataset import (  # noqa: E402
    TopologyDatasetV31,
    detect_schema_version,
    make_symmetry_collate_fn,
)
from alphabuilder.src.neural.model import AlphaBuilderV31, VALID_ARCHITECTURES, build_model  # noqa: E402
from alphabuilder.src.neural.train_config import CONFIGS, TrainConfig, get_config  # noqa: E402
from alphabuilder.src.neural.trainer import (  # noqa: E402
    blueprint_loss,
    compute_backbone_grad_ratio,
    compute_blueprint_dice,
    compute_connectivity_accuracy,
    compute_log_c_mae,
    compute_policy_top1_accuracy,
    compute_total_loss,
    policy_loss,
)


def _setup_distributed() -> Tuple[bool, int, int, int]:
    """Detect and initialize torch.distributed when launched via torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if world_size > 1 and local_rank >= 0:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def detect_environment(local_rank: int = 0, is_distributed: bool = False) -> Tuple[str, torch.device]:
    """Detect hardware and return appropriate preset name and torch.device."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(local_rank).lower()
        dev = torch.device("cuda", local_rank) if is_distributed else torch.device("cuda")
        if "t4" in gpu_name or "tesla" in gpu_name:
            preset = "kaggle_t4_dual_swin" if gpu_count >= 2 else "kaggle_t4_swin"
            return preset, dev
        return ("kaggle_t4_dual_swin" if gpu_count >= 2 else "kaggle_t4_swin"), dev

    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "local_iris", torch.device("xpu")
    except Exception:
        pass

    return "local_iris", torch.device("cpu")


def _resolve_episode_split(
    db_path: Path,
    val_split: float,
    split_json: Optional[str],
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Load split_v3_3.json or build deterministic episode-level split."""
    if split_json:
        data = json.loads(Path(split_json).read_text(encoding="utf-8"))
        train_ids = [str(e) for e in data["train_episode_ids"]]
        val_ids = [str(e) for e in data["val_episode_ids"]]
        if set(train_ids) & set(val_ids):
            raise RuntimeError(f"Episode overlap found in {split_json}")
        return train_ids, val_ids

    schema = detect_schema_version(db_path)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    if schema == "v2":
        rows = conn.execute("SELECT DISTINCT episode_id FROM episodes ORDER BY episode_id").fetchall()
    else:
        rows = conn.execute("SELECT DISTINCT episode_id FROM training_data ORDER BY episode_id").fetchall()
    conn.close()

    ep_ids = [str(r[0]) for r in rows]
    rng = random.Random(seed)
    rng.shuffle(ep_ids)

    if len(ep_ids) >= 2:
        n_val = max(1, int(round(len(ep_ids) * val_split)))
        n_val = min(n_val, len(ep_ids) - 1)
        val_ids = ep_ids[:n_val]
        train_ids = ep_ids[n_val:]
    else:
        train_ids = ep_ids
        val_ids = []

    return train_ids, val_ids


def create_dataloaders(
    config: TrainConfig,
    db_path: str,
    is_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    is_main_process: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create episode-split training (augment=True) and validation (augment=False) DataLoaders.
    """
    db_file = Path(db_path)
    if is_main_process:
        print(f"\nLoading dataset from: {db_file}")

    train_ids, val_ids = _resolve_episode_split(
        db_file, val_split=config.val_split, split_json=config.split_json
    )

    train_dataset = TopologyDatasetV31(
        db_path=db_file,
        augment=True,
        preload_to_ram=True,
        episode_ids=train_ids,
    )
    val_dataset = (
        TopologyDatasetV31(
            db_path=db_file,
            augment=False,
            preload_to_ram=True,
            episode_ids=val_ids,
        )
        if len(val_ids) > 0
        else None
    )

    if is_main_process:
        print(f"   Episodes -> train: {len(train_ids)}, val: {len(val_ids)}")
        print(f"   Train samples: {len(train_dataset):,} ({train_dataset.get_memory_usage_mb():.1f} MB RAM)")
        if val_dataset is not None:
            print(f"   Val samples:   {len(val_dataset):,} ({val_dataset.get_memory_usage_mb():.1f} MB RAM)")

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        if is_distributed
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if (is_distributed and val_dataset is not None)
        else None
    )

    use_workers = config.num_workers > 0
    pin_mem = config.pin_memory and torch.cuda.is_available()
    drop_last = len(train_dataset) >= config.batch_size * max(1, world_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=make_symmetry_collate_fn(enable_rot90=config.enable_rot90_collate),
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor if use_workers else None,
        pin_memory=pin_mem,
        persistent_workers=config.persistent_workers if use_workers else False,
        drop_last=drop_last,
    )

    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor if use_workers else None,
            pin_memory=pin_mem,
            persistent_workers=config.persistent_workers if use_workers else False,
        )

    if is_main_process:
        print(f"   Train batches/epoch (per rank): {len(train_loader)}")

    return train_loader, val_loader


def create_model(
    config: TrainConfig,
    device: torch.device,
    is_distributed: bool = False,
    local_rank: int = 0,
    is_main_process: bool = True,
) -> nn.Module:
    """Create configured architecture (SimpleBackbone, Swin-UNETR, or Phys-UViT 3D) and wrap in DDP."""
    model = build_model(
        architecture=config.architecture,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        feature_size=config.feature_size,
        use_swin=config.use_swin,
        embed_dim=config.embed_dim,
        transformer_depth=config.transformer_depth,
        num_heads=config.num_heads,
    ).to(device)

    # Freeze unused auxiliary SED head when lambda_sed <= 0 to keep DDP graphs 100% utilized
    if config.lambda_sed <= 0.0 and hasattr(model, "sed_head"):
        for p in model.sed_head.parameters():
            p.requires_grad_(False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process:
        if config.architecture == "phys_uvit":
            arch = (
                f"Phys-UViT 3D (feature_size={config.feature_size}, "
                f"embed_dim={config.embed_dim}, depth={config.transformer_depth}, heads={config.num_heads})"
            )
        elif config.use_swin or config.architecture == "swin":
            arch = f"Swin-UNETR (feature_size={config.feature_size})"
        else:
            arch = f"SimpleBackbone (feature_size={config.feature_size})"
        print(f"\nModel: {arch}")
        print(f"   Parameters: {total_params:,} (trainable: {trainable_params:,})")

    if is_distributed:
        device_ids = [local_rank] if device.type == "cuda" else None
        output_device = local_rank if device.type == "cuda" else None
        model = DDP(
            model,
            device_ids=device_ids,
            output_device=output_device,
            find_unused_parameters=(config.lambda_bp <= 0.0),
        )
        if is_main_process:
            print(f"   Wrapped in DistributedDataParallel (world_size={dist.get_world_size()})")
    elif device.type == "cuda" and torch.cuda.device_count() > 1:
        if is_main_process:
            print(f"   Wrapped in DataParallel ({torch.cuda.device_count()} GPUs)")
        model = nn.DataParallel(model)

    return model


def _reduce_sums(sums: List[float], device: torch.device, is_distributed: bool) -> List[float]:
    if not is_distributed:
        return sums
    tensor = torch.tensor(sums, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    config: TrainConfig,
    is_distributed: bool = False,
    is_main_process: bool = True,
) -> Dict[str, float]:
    """Train one epoch with AMP, grad clipping, and optional backbone grad-norm check."""
    model.train()
    use_amp = bool(config.use_amp and device.type == "cuda")

    sum_loss = 0.0
    sum_core = 0.0
    sum_pol = 0.0
    sum_val = 0.0
    sum_bp = 0.0
    sum_log_c = 0.0
    sum_conn = 0.0
    n_samples = 0
    grad_ratio: Optional[float] = None
    is_phys_uvit = config.architecture == "phys_uvit"

    pbar = tqdm(dataloader, desc="Train", leave=False, disable=not is_main_process)
    for batch_idx, batch in enumerate(pbar):
        state = batch["state"].to(device, non_blocking=True)
        target_policy = batch["policy"].to(device, non_blocking=True)
        target_value = batch["value"].to(device, non_blocking=True)
        legal_mask = batch["legal_mask"].to(device, non_blocking=True) if "legal_mask" in batch else None
        policy_mask = batch["policy_mask"].to(device, non_blocking=True) if "policy_mask" in batch else None
        target_bp = batch["blueprint"].to(device, non_blocking=True) if "blueprint" in batch else None
        bp_mask = batch["blueprint_mask"].to(device, non_blocking=True) if "blueprint_mask" in batch else None
        target_log_c = batch["log_compliance"].to(device, non_blocking=True) if "log_compliance" in batch else None
        comp_mask = batch["compliance_mask"].to(device, non_blocking=True) if "compliance_mask" in batch else None
        target_conn = batch["conn_target"].to(device, non_blocking=True) if "conn_target" in batch else None
        conn_mask = batch["conn_mask"].to(device, non_blocking=True) if "conn_mask" in batch else None

        bs = state.shape[0]
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            if is_phys_uvit:
                pred_policy, pred_value, aux = model(
                    state, return_aux=True, compute_sed=(config.lambda_sed > 0.0)
                )
                pred_bp = aux["blueprint"] if (config.lambda_bp > 0.0 and target_bp is not None) else None
                pred_log_c = aux["log_compliance"]
                pred_conn_logit = aux["conn_logit"]
            else:
                outputs = model(state, return_blueprint=(config.lambda_bp > 0.0 and target_bp is not None))
                if len(outputs) == 3:
                    pred_policy, pred_value, pred_bp = outputs
                else:
                    pred_policy, pred_value = outputs
                    pred_bp = None
                pred_log_c = None
                pred_conn_logit = None

            loss, metrics = compute_total_loss(
                pred_policy=pred_policy,
                pred_value=pred_value,
                target_policy=target_policy,
                target_value=target_value,
                legal_mask=legal_mask,
                policy_mask=policy_mask,
                pred_blueprint=pred_bp,
                target_blueprint=target_bp,
                blueprint_mask=bp_mask,
                pred_log_c=pred_log_c,
                target_log_c=target_log_c,
                compliance_mask=comp_mask,
                pred_conn_logit=pred_conn_logit,
                target_conn=target_conn,
                conn_mask=conn_mask,
                lambda_policy=config.lambda_policy,
                lambda_value=config.lambda_value,
                lambda_bp=config.lambda_bp,
                lambda_log_c=config.lambda_log_c,
                lambda_conn=config.lambda_conn,
            )

        # Sample gradient norm ratio once at batch 0 on main process (unscaled float32 losses)
        if batch_idx == 0 and is_main_process and pred_bp is not None and bp_mask is not None and (bp_mask > 0.5).any():
            try:
                p_diag = policy_loss(pred_policy, target_policy, legal_mask=legal_mask, policy_mask=policy_mask)
                bp_diag = blueprint_loss(pred_bp, target_bp, blueprint_mask=bp_mask)
                grad_ratio = compute_backbone_grad_ratio(model, p_diag, bp_diag, lambda_bp=config.lambda_bp)
            except Exception:
                grad_ratio = None

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if config.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

        sum_loss += metrics["loss"] * bs
        sum_core += metrics["core_loss"] * bs
        sum_pol += metrics["policy_loss"] * bs
        sum_val += metrics["value_loss"] * bs
        sum_bp += metrics["blueprint_loss"] * bs
        sum_log_c += metrics["log_c_loss"] * bs
        sum_conn += metrics["conn_loss"] * bs
        n_samples += bs

        if is_main_process and batch_idx % 10 == 0:
            pbar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                pol=f"{metrics['policy_loss']:.4f}",
                val=f"{metrics['value_loss']:.4f}",
                bp=f"{metrics['blueprint_loss']:.4f}",
            )

    sum_loss, sum_core, sum_pol, sum_val, sum_bp, sum_log_c, sum_conn, total_n = _reduce_sums(
        [sum_loss, sum_core, sum_pol, sum_val, sum_bp, sum_log_c, sum_conn, float(n_samples)],
        device,
        is_distributed,
    )
    denom = max(total_n, 1.0)
    out = {
        "loss": sum_loss / denom,
        "core_loss": sum_core / denom,
        "policy_loss": sum_pol / denom,
        "value_loss": sum_val / denom,
        "blueprint_loss": sum_bp / denom,
    }
    if config.lambda_log_c > 0.0:
        out["log_c_loss"] = sum_log_c / denom
    if config.lambda_conn > 0.0:
        out["conn_loss"] = sum_conn / denom
    if grad_ratio is not None:
        out["bp_pol_grad_ratio"] = grad_ratio
    return out


def validate_epoch(
    model: nn.Module,
    dataloader: Optional[DataLoader],
    device: torch.device,
    config: TrainConfig,
    is_distributed: bool = False,
    is_main_process: bool = True,
) -> Dict[str, float]:
    """Validate for one epoch with detailed policy top-1, value MAE, and blueprint Dice."""
    if dataloader is None:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "blueprint_loss": 0.0}

    model.eval()
    use_amp = bool(config.use_amp and device.type == "cuda")

    sum_loss = 0.0
    sum_core = 0.0
    sum_pol = 0.0
    sum_val = 0.0
    sum_bp = 0.0
    sum_log_c = 0.0
    sum_conn = 0.0
    sum_mae = 0.0
    sum_top1 = 0.0
    cnt_top1 = 0.0
    sum_dice = 0.0
    cnt_dice = 0.0
    sum_lc_mae = 0.0
    cnt_lc = 0.0
    sum_conn_acc = 0.0
    cnt_conn = 0.0
    n_samples = 0
    is_phys_uvit = config.architecture == "phys_uvit"

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val", leave=False, disable=not is_main_process):
            state = batch["state"].to(device, non_blocking=True)
            target_policy = batch["policy"].to(device, non_blocking=True)
            target_value = batch["value"].to(device, non_blocking=True)
            legal_mask = batch["legal_mask"].to(device, non_blocking=True) if "legal_mask" in batch else None
            policy_mask = batch["policy_mask"].to(device, non_blocking=True) if "policy_mask" in batch else None
            target_bp = batch["blueprint"].to(device, non_blocking=True) if "blueprint" in batch else None
            bp_mask = batch["blueprint_mask"].to(device, non_blocking=True) if "blueprint_mask" in batch else None
            target_log_c = batch["log_compliance"].to(device, non_blocking=True) if "log_compliance" in batch else None
            comp_mask = batch["compliance_mask"].to(device, non_blocking=True) if "compliance_mask" in batch else None
            target_conn = batch["conn_target"].to(device, non_blocking=True) if "conn_target" in batch else None
            conn_mask = batch["conn_mask"].to(device, non_blocking=True) if "conn_mask" in batch else None

            bs = state.shape[0]
            with torch.amp.autocast("cuda", enabled=use_amp):
                if is_phys_uvit:
                    pred_policy, pred_value, aux = model(
                        state, return_aux=True, compute_sed=False
                    )
                    pred_bp = aux["blueprint"] if (config.lambda_bp > 0.0 and target_bp is not None) else None
                    pred_log_c = aux["log_compliance"]
                    pred_conn_logit = aux["conn_logit"]
                else:
                    outputs = model(state, return_blueprint=(config.lambda_bp > 0.0 and target_bp is not None))
                    if len(outputs) == 3:
                        pred_policy, pred_value, pred_bp = outputs
                    else:
                        pred_policy, pred_value = outputs
                        pred_bp = None
                    pred_log_c = None
                    pred_conn_logit = None

                _, metrics = compute_total_loss(
                    pred_policy=pred_policy,
                    pred_value=pred_value,
                    target_policy=target_policy,
                    target_value=target_value,
                    legal_mask=legal_mask,
                    policy_mask=policy_mask,
                    pred_blueprint=pred_bp,
                    target_blueprint=target_bp,
                    blueprint_mask=bp_mask,
                    pred_log_c=pred_log_c,
                    target_log_c=target_log_c,
                    compliance_mask=comp_mask,
                    pred_conn_logit=pred_conn_logit,
                    target_conn=target_conn,
                    conn_mask=conn_mask,
                    lambda_policy=config.lambda_policy,
                    lambda_value=config.lambda_value,
                    lambda_bp=config.lambda_bp,
                    lambda_log_c=config.lambda_log_c,
                    lambda_conn=config.lambda_conn,
                )

            sum_loss += metrics["loss"] * bs
            sum_core += metrics["core_loss"] * bs
            sum_pol += metrics["policy_loss"] * bs
            sum_val += metrics["value_loss"] * bs
            sum_bp += metrics["blueprint_loss"] * bs
            sum_log_c += metrics["log_c_loss"] * bs
            sum_conn += metrics["conn_loss"] * bs
            sum_mae += float((pred_value.float() - target_value.float()).abs().sum().item())
            n_samples += bs

            top1 = compute_policy_top1_accuracy(pred_policy, target_policy, legal_mask=legal_mask, policy_mask=policy_mask)
            if top1 is not None:
                valid_p_cnt = float((policy_mask > 0.5).sum().item()) if policy_mask is not None else float(bs)
                sum_top1 += top1 * max(1.0, valid_p_cnt)
                cnt_top1 += max(1.0, valid_p_cnt)

            if pred_bp is not None and target_bp is not None:
                dice = compute_blueprint_dice(pred_bp, target_bp, blueprint_mask=bp_mask)
                if dice is not None:
                    valid_b_cnt = float((bp_mask > 0.5).sum().item()) if bp_mask is not None else float(bs)
                    sum_dice += dice * max(1.0, valid_b_cnt)
                    cnt_dice += max(1.0, valid_b_cnt)

            if pred_log_c is not None and target_log_c is not None:
                lc_mae = compute_log_c_mae(pred_log_c, target_log_c, compliance_mask=comp_mask)
                if lc_mae is not None:
                    valid_c_cnt = float((comp_mask > 0.5).sum().item()) if comp_mask is not None else float(bs)
                    sum_lc_mae += lc_mae * max(1.0, valid_c_cnt)
                    cnt_lc += max(1.0, valid_c_cnt)

            if pred_conn_logit is not None and target_conn is not None:
                c_acc = compute_connectivity_accuracy(pred_conn_logit, target_conn, conn_mask=conn_mask)
                if c_acc is not None:
                    valid_k_cnt = float((conn_mask > 0.5).sum().item()) if conn_mask is not None else float(bs)
                    sum_conn_acc += c_acc * max(1.0, valid_k_cnt)
                    cnt_conn += max(1.0, valid_k_cnt)

    (
        sum_loss,
        sum_core,
        sum_pol,
        sum_val,
        sum_bp,
        sum_log_c,
        sum_conn,
        sum_mae,
        sum_top1,
        cnt_top1,
        sum_dice,
        cnt_dice,
        sum_lc_mae,
        cnt_lc,
        sum_conn_acc,
        cnt_conn,
        total_n,
    ) = _reduce_sums(
        [
            sum_loss,
            sum_core,
            sum_pol,
            sum_val,
            sum_bp,
            sum_log_c,
            sum_conn,
            sum_mae,
            sum_top1,
            cnt_top1,
            sum_dice,
            cnt_dice,
            sum_lc_mae,
            cnt_lc,
            sum_conn_acc,
            cnt_conn,
            float(n_samples),
        ],
        device,
        is_distributed,
    )
    denom = max(total_n, 1.0)
    out = {
        "loss": sum_loss / denom,
        "core_loss": sum_core / denom,
        "policy_loss": sum_pol / denom,
        "value_loss": sum_val / denom,
        "blueprint_loss": sum_bp / denom,
        "value_mae": sum_mae / denom,
    }
    if config.lambda_log_c > 0.0:
        out["log_c_loss"] = sum_log_c / denom
    if config.lambda_conn > 0.0:
        out["conn_loss"] = sum_conn / denom
    if cnt_top1 > 0:
        out["policy_top1_acc"] = sum_top1 / cnt_top1
    if cnt_dice > 0:
        out["blueprint_dice"] = sum_dice / cnt_dice
    if cnt_lc > 0:
        out["log_c_mae"] = sum_lc_mae / cnt_lc
    if cnt_conn > 0:
        out["conn_acc"] = sum_conn_acc / cnt_conn
    return out


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: TrainConfig,
    path: Path,
) -> None:
    """Save training checkpoint compatible with AlphaBuilderInference."""
    raw_model = model.module if hasattr(model, "module") else model
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "val_loss": float(metrics.get("loss", float("inf"))),
        "config": {
            "architecture": config.architecture,
            "use_swin": config.use_swin,
            "feature_size": config.feature_size,
            "embed_dim": config.embed_dim,
            "transformer_depth": config.transformer_depth,
            "num_heads": config.num_heads,
            "in_channels": config.in_channels,
            "out_channels": config.out_channels,
            "lambda_bp": config.lambda_bp,
            "lambda_log_c": config.lambda_log_c,
            "lambda_conn": config.lambda_conn,
            "lambda_sed": config.lambda_sed,
        },
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """Load checkpoint and return next epoch index."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    target_model = model.module if hasattr(model, "module") else model
    raw_sd = checkpoint["model_state_dict"]
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in raw_sd.items()}
    target_model.load_state_dict(state_dict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["epoch"]) + 1


def _build_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig):
    warmup_ep = max(0, min(config.warmup_epochs, max(0, config.epochs - 1)))
    if warmup_ep > 0 and config.epochs > 1:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_ep)
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.epochs - warmup_ep),
            eta_min=config.learning_rate * 0.01,
        )
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_ep])
    return CosineAnnealingLR(optimizer, T_max=max(1, config.epochs), eta_min=config.learning_rate * 0.01)


def main(
    config_name: Optional[str] = None,
    epochs: Optional[int] = None,
    db_path: Optional[str] = None,
    split_json: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    lambda_bp: Optional[float] = None,
    use_swin: Optional[bool] = None,
    architecture: Optional[str] = None,
    num_workers: Optional[int] = None,
    resume: Optional[str] = None,
) -> None:
    is_distributed, rank, local_rank, world_size = _setup_distributed()
    is_main_process = rank == 0

    try:
        detected_config, device = detect_environment(local_rank=local_rank, is_distributed=is_distributed)
        config_name = config_name or detected_config
        config = get_config(config_name)

        if epochs is not None:
            config.epochs = epochs
        if db_path is not None:
            config.db_path = db_path
        if split_json is not None:
            config.split_json = split_json
        if checkpoint_dir is not None:
            config.checkpoint_dir = checkpoint_dir
        if log_dir is not None:
            config.log_dir = log_dir
        if batch_size is not None:
            config.batch_size = batch_size
        if lr is not None:
            config.learning_rate = lr
        if lambda_bp is not None:
            config.lambda_bp = lambda_bp
        if architecture is not None:
            config.architecture = architecture
            config.use_swin = architecture == "swin"
        elif use_swin is not None:
            config.use_swin = use_swin
            config.architecture = "swin" if use_swin else "simple"
        if num_workers is not None:
            config.num_workers = num_workers

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        if is_main_process:
            print("=" * 64)
            print(
                f"AlphaBuilder Training Pipeline | config={config_name} | arch={config.architecture} | device={device}"
            )
            print(f"  per_gpu_bs={config.batch_size} | world_size={world_size} | global_bs={config.batch_size * world_size}")
            print(
                f"  loss_weights: pol={config.lambda_policy}, val={config.lambda_value}, "
                f"bp={config.lambda_bp}, logC={config.lambda_log_c}, conn={config.lambda_conn}"
            )
            print("=" * 64)

        writer = None
        if is_main_process and config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                run_name = f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                tb_dir = Path(config.log_dir) / run_name
                tb_dir.mkdir(parents=True, exist_ok=True)
                writer = SummaryWriter(tb_dir)
            except ImportError:
                writer = None

        train_loader, val_loader = create_dataloaders(
            config,
            config.db_path,
            is_distributed=is_distributed,
            rank=rank,
            world_size=world_size,
            is_main_process=is_main_process,
        )

        model = create_model(
            config,
            device,
            is_distributed=is_distributed,
            local_rank=local_rank,
            is_main_process=is_main_process,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = _build_scheduler(optimizer, config)
        use_amp = bool(config.use_amp and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

        start_epoch = 0
        if resume:
            if is_main_process:
                print(f"Resuming from checkpoint: {resume}")
            start_epoch = load_checkpoint(Path(resume), model, optimizer)

        best_val_loss = float("inf")
        patience_counter = 0
        history: List[dict] = []
        ckpt_dir = Path(config.checkpoint_dir)

        for epoch in range(start_epoch, config.epochs):
            epoch_start = time.time()
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                scaler,
                config,
                is_distributed=is_distributed,
                is_main_process=is_main_process,
            )
            val_metrics = validate_epoch(
                model,
                val_loader,
                device,
                config,
                is_distributed=is_distributed,
                is_main_process=is_main_process,
            )

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start
            samples_per_sec = len(train_loader.dataset) / max(epoch_time, 1e-6)

            stop_signal = False
            if is_main_process:
                print(
                    f"\nEpoch {epoch + 1}/{config.epochs} ({epoch_time:.1f}s, {samples_per_sec:.0f} samples/s, lr={current_lr:.2e})"
                )
                print(
                    f"  Train | Loss={train_metrics['loss']:.4f} Pol={train_metrics['policy_loss']:.4f} "
                    f"Val={train_metrics['value_loss']:.4f} BP={train_metrics['blueprint_loss']:.4f}"
                    + (
                        f" grad_ratio={train_metrics['bp_pol_grad_ratio']:.3f}"
                        if "bp_pol_grad_ratio" in train_metrics
                        else ""
                    )
                )
                if val_loader is not None:
                    extra_val = ""
                    if "log_c_mae" in val_metrics:
                        extra_val += f" logC_MAE={val_metrics['log_c_mae']:.4f}"
                    if "conn_acc" in val_metrics:
                        extra_val += f" Conn={val_metrics['conn_acc']:.3f}"
                    print(
                        f"  Val   | Loss={val_metrics['loss']:.4f} Core={val_metrics.get('core_loss', val_metrics['loss']):.4f} "
                        f"Pol={val_metrics['policy_loss']:.4f} Val={val_metrics['value_loss']:.4f} "
                        f"BP={val_metrics['blueprint_loss']:.4f} MAE={val_metrics.get('value_mae', 0.0):.4f} "
                        f"Top1={val_metrics.get('policy_top1_acc', 0.0):.3f} "
                        f"Dice={val_metrics.get('blueprint_dice', 0.0):.3f}{extra_val}"
                    )

                history.append(
                    {
                        "epoch": epoch + 1,
                        "lr": current_lr,
                        "elapsed_sec": round(epoch_time, 2),
                        "samples_per_sec": round(samples_per_sec, 1),
                        "train": train_metrics,
                        "val": val_metrics,
                    }
                )
                (ckpt_dir / "metrics_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

                if writer is not None:
                    for k, v in train_metrics.items():
                        writer.add_scalar(f"train/{k}", v, epoch)
                    for k, v in val_metrics.items():
                        writer.add_scalar(f"val/{k}", v, epoch)
                    writer.add_scalar("lr", current_lr, epoch)

                monitor_loss = val_metrics["loss"] if val_loader is not None else train_metrics["loss"]
                if monitor_loss < best_val_loss - config.min_delta:
                    best_val_loss = monitor_loss
                    patience_counter = 0
                    save_checkpoint(model, optimizer, epoch, val_metrics, config, ckpt_dir / "best_model.pt")
                    print(f"  -> New best checkpoint saved ({best_val_loss:.4f})")
                else:
                    patience_counter += 1

                if (epoch + 1) % config.save_every_n_epochs == 0:
                    save_checkpoint(model, optimizer, epoch, val_metrics, config, ckpt_dir / f"checkpoint_epoch_{epoch + 1}.pt")
                save_checkpoint(model, optimizer, epoch, val_metrics, config, ckpt_dir / "last_model.pt")

                if patience_counter >= config.patience:
                    print(f"\nEarly stopping triggered (patience={config.patience})")
                    stop_signal = True

            if is_distributed:
                stop_tensor = torch.tensor([1 if stop_signal else 0], dtype=torch.int32, device=device)
                dist.broadcast(stop_tensor, src=0)
                stop_signal = bool(stop_tensor.item())

            if stop_signal:
                break

        if writer is not None:
            writer.close()

        if is_main_process:
            print(f"\nTraining complete! Best monitored loss: {best_val_loss:.4f} | dir={ckpt_dir}")
    finally:
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaBuilder v3.3 Training Pipeline")
    parser.add_argument("--config", type=str, default=None, help=f"Preset: {list(CONFIGS.keys())}")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--split-json", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda-bp", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--arch", type=str, default=None, choices=list(VALID_ARCHITECTURES))
    parser.add_argument("--use-swin", dest="use_swin", action="store_true", default=None)
    parser.add_argument("--no-swin", dest="use_swin", action="store_false")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    main(
        config_name=args.config,
        epochs=args.epochs,
        db_path=args.db_path,
        split_json=args.split_json,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_bp=args.lambda_bp,
        use_swin=args.use_swin,
        architecture=args.arch,
        num_workers=args.num_workers,
        resume=args.resume,
    )
