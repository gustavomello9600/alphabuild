#!/usr/bin/env python3
"""
Mixed Replay-Buffer DDP Fine-Tuning Step for AlphaBuilder v4.0 Stage 6 Self-Play.

Combines:
- Accumulated Self-Play `game_steps` + counterfactual `fem_probes` (policy w/ advantage gate,
  real FEM `value_target`, `log10(C)`, connectivity, optional `domain_mask` & `sed_map`)
- Warmup anchor subset (`warmup_v3_3_kaggle.db` + `split_v3_3.json`) preserving Blueprint geometry
- `BucketedResolutionBatchSampler` + `D4 x Z2 = 16x` symmetries for Tier-3 multi-resolution
- Dual validation tracking (`val_selfplay` + `val_warmup`) to verify OOD calibration without forgetting
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphabuilder.src.neural.dataset import TopologyDatasetV31, make_symmetry_collate_fn
from alphabuilder.src.neural.model import build_model
from alphabuilder.src.neural.selfplay_dataset import (
    BucketedResolutionBatchSampler,
    MixedReplayDataset,
    SelfPlayDataset,
    collect_game_ids_from_dbs,
)
from alphabuilder.src.neural.trainer import (
    compute_blueprint_dice,
    compute_connectivity_accuracy,
    compute_log_c_mae,
    compute_policy_top1_accuracy,
    compute_total_loss,
)


def _setup_distributed() -> Tuple[bool, int, int, int]:
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


def _detect_device(local_rank: int, is_distributed: bool, requested: str = "auto") -> torch.device:
    if requested != "auto":
        if requested == "cuda" and is_distributed:
            return torch.device("cuda", local_rank)
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank) if is_distributed else torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def _make_pbar(iterable, desc: str, enabled: bool):
    is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
    return tqdm(
        iterable,
        desc=desc,
        leave=False,
        disable=not enabled,
        mininterval=2.0 if is_tty else 45.0,
        maxinterval=90.0,
        ncols=100,
        dynamic_ncols=False,
        ascii=True,
    )


def _reduce_sums(values: List[float], device: torch.device, is_distributed: bool) -> List[float]:
    if not is_distributed:
        return values
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def _normalize_eval_batch(batch: dict) -> dict:
    _, D, H, W = batch["state"].shape[1:]
    B = batch["state"].shape[0]
    if "domain_mask" not in batch:
        batch["domain_mask"] = torch.ones((B, 1, D, H, W), dtype=torch.float32)
    if "sed_target" not in batch:
        batch["sed_target"] = torch.zeros((B, 1, D, H, W), dtype=torch.float32)
    if "sed_mask" not in batch:
        batch["sed_mask"] = torch.zeros((B,), dtype=torch.float32)
    if "op_target" not in batch:
        batch["op_target"] = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32).expand(B, -1).clone()
    if "op_mask" not in batch:
        batch["op_mask"] = torch.zeros((B,), dtype=torch.float32)
    return batch


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    is_distributed: bool,
    is_main: bool,
    use_amp: bool,
    lambda_policy: float,
    lambda_value: float,
    lambda_bp: float,
    lambda_log_c: float,
    lambda_conn: float,
    lambda_sed: float,
    lambda_op: float,
    grad_clip_norm: float,
    tag: str,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    want_sed = lambda_sed > 0.0

    sum_loss = 0.0
    sum_pol = 0.0
    sum_val = 0.0
    sum_bp = 0.0
    sum_lc = 0.0
    sum_conn = 0.0
    sum_sed = 0.0
    sum_mae = 0.0
    sum_top1 = 0.0
    cnt_top1 = 0.0
    sum_dice = 0.0
    cnt_dice = 0.0
    sum_lc_mae = 0.0
    cnt_lc = 0.0
    sum_conn_acc = 0.0
    cnt_conn = 0.0
    n_samples = 0.0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in _make_pbar(loader, desc=tag, enabled=is_main):
            batch = _normalize_eval_batch(batch)
            state = batch["state"].to(device, non_blocking=True)
            dm = batch["domain_mask"].to(device, non_blocking=True)
            t_pol = batch["policy"].to(device, non_blocking=True)
            t_val = batch["value"].to(device, non_blocking=True)
            legal_mask = batch["legal_mask"].to(device, non_blocking=True)
            pol_mask = batch["policy_mask"].to(device, non_blocking=True)
            t_bp = batch["blueprint"].to(device, non_blocking=True)
            bp_mask = batch["blueprint_mask"].to(device, non_blocking=True)
            t_lc = batch["log_compliance"].to(device, non_blocking=True)
            lc_mask = batch["compliance_mask"].to(device, non_blocking=True)
            t_conn = batch["conn_target"].to(device, non_blocking=True)
            conn_mask = batch["conn_mask"].to(device, non_blocking=True)
            t_sed = batch["sed_target"].to(device, non_blocking=True)
            sed_mask = batch["sed_mask"].to(device, non_blocking=True)
            t_op = batch["op_target"].to(device, non_blocking=True)
            op_mask = batch["op_mask"].to(device, non_blocking=True)

            bs = float(state.shape[0])
            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_pol, pred_val, aux = model(
                    state,
                    domain_mask=dm,
                    return_aux=True,
                    compute_sed=want_sed,
                )
                loss, m = compute_total_loss(
                    pred_policy=pred_pol,
                    pred_value=pred_val,
                    target_policy=t_pol,
                    target_value=t_val,
                    legal_mask=legal_mask,
                    policy_mask=pol_mask,
                    pred_blueprint=aux["blueprint"] if lambda_bp > 0.0 else None,
                    target_blueprint=t_bp,
                    blueprint_mask=bp_mask,
                    pred_log_c=aux["log_compliance"],
                    target_log_c=t_lc,
                    compliance_mask=lc_mask,
                    pred_conn_logit=aux["conn_logit"],
                    target_conn=t_conn,
                    conn_mask=conn_mask,
                    pred_sed_u=aux.get("sed_u") if want_sed else None,
                    target_sed=t_sed,
                    sed_mask=sed_mask,
                    pred_op_logits=aux.get("op_logits"),
                    target_op=t_op,
                    op_mask=op_mask,
                    lambda_policy=lambda_policy,
                    lambda_value=lambda_value,
                    lambda_bp=lambda_bp,
                    lambda_log_c=lambda_log_c,
                    lambda_conn=lambda_conn,
                    lambda_sed=lambda_sed,
                    lambda_op=lambda_op,
                )

            if is_train and optimizer is not None:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip_norm > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

            sum_loss += m["loss"] * bs
            sum_pol += m["policy_loss"] * bs
            sum_val += m["value_loss"] * bs
            sum_bp += m["blueprint_loss"] * bs
            sum_lc += m["log_c_loss"] * bs
            sum_conn += m["conn_loss"] * bs
            sum_sed += m.get("sed_loss", 0.0) * bs
            sum_mae += float((pred_val.float() - t_val.float()).abs().sum().item())
            n_samples += bs

            if not is_train:
                top1 = compute_policy_top1_accuracy(pred_pol, t_pol, legal_mask=legal_mask, policy_mask=pol_mask)
                if top1 is not None:
                    cnt = float((pol_mask > 0.5).sum().item())
                    sum_top1 += top1 * max(1.0, cnt)
                    cnt_top1 += max(1.0, cnt)

                if aux.get("blueprint") is not None:
                    dice = compute_blueprint_dice(aux["blueprint"], t_bp, blueprint_mask=bp_mask)
                    if dice is not None:
                        cnt = float((bp_mask > 0.5).sum().item())
                        sum_dice += dice * max(1.0, cnt)
                        cnt_dice += max(1.0, cnt)

                lc_mae = compute_log_c_mae(aux["log_compliance"], t_lc, compliance_mask=lc_mask)
                if lc_mae is not None:
                    cnt = float((lc_mask > 0.5).sum().item())
                    sum_lc_mae += lc_mae * max(1.0, cnt)
                    cnt_lc += max(1.0, cnt)

                c_acc = compute_connectivity_accuracy(aux["conn_logit"], t_conn, conn_mask=conn_mask)
                if c_acc is not None:
                    cnt = float((conn_mask > 0.5).sum().item())
                    sum_conn_acc += c_acc * max(1.0, cnt)
                    cnt_conn += max(1.0, cnt)

    red = _reduce_sums(
        [
            sum_loss, sum_pol, sum_val, sum_bp, sum_lc, sum_conn, sum_sed,
            sum_mae, sum_top1, cnt_top1, sum_dice, cnt_dice, sum_lc_mae, cnt_lc, sum_conn_acc, cnt_conn, n_samples,
        ],
        device,
        is_distributed,
    )
    (
        sum_loss, sum_pol, sum_val, sum_bp, sum_lc, sum_conn, sum_sed,
        sum_mae, sum_top1, cnt_top1, sum_dice, cnt_dice, sum_lc_mae, cnt_lc, sum_conn_acc, cnt_conn, tot_n,
    ) = red
    denom = max(tot_n, 1.0)
    out = {
        "loss": sum_loss / denom,
        "policy_loss": sum_pol / denom,
        "value_loss": sum_val / denom,
        "blueprint_loss": sum_bp / denom,
        "log_c_loss": sum_lc / denom,
        "conn_loss": sum_conn / denom,
        "sed_loss": sum_sed / denom,
        "value_mae": sum_mae / denom,
    }
    if cnt_top1 > 0:
        out["policy_top1_acc"] = sum_top1 / cnt_top1
    if cnt_dice > 0:
        out["blueprint_dice"] = sum_dice / cnt_dice
    if cnt_lc > 0:
        out["log_c_mae"] = sum_lc_mae / cnt_lc
    if cnt_conn > 0:
        out["conn_acc"] = sum_conn_acc / cnt_conn
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6 Self-Play Mixed Replay Fine-Tuning Step.")
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--selfplay-dbs", type=str, required=True, help="Comma-separated self-play SQLite DBs")
    parser.add_argument("--warmup-db", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--selfplay-fraction", type=float, default=0.60)
    parser.add_argument("--min-epoch-samples", type=int, default=12000)
    parser.add_argument("--lambda-policy", type=float, default=1.0)
    parser.add_argument("--lambda-value", type=float, default=2.0)
    parser.add_argument("--lambda-bp", type=float, default=0.25)
    parser.add_argument("--lambda-log-c", type=float, default=0.35)
    parser.add_argument("--lambda-conn", type=float, default=0.25)
    parser.add_argument("--lambda-sed", type=float, default=0.15)
    parser.add_argument("--lambda-op", type=float, default=0.20)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    is_distributed, rank, local_rank, world_size = _setup_distributed()
    is_main = rank == 0
    device = _detect_device(local_rank, is_distributed, args.device)
    use_amp = bool(device.type == "cuda")

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sp_paths = [Path(p.strip()) for p in args.selfplay_dbs.split(",") if p.strip()]
    all_games = collect_game_ids_from_dbs(sp_paths)
    if not all_games:
        raise RuntimeError(f"No self-play games found across {sp_paths}")

    rng = random.Random(args.seed)
    shuffled_games = list(all_games)
    rng.shuffle(shuffled_games)
    if len(shuffled_games) >= 4:
        n_val_g = max(1, int(round(0.15 * len(shuffled_games))))
        val_games = shuffled_games[:n_val_g]
        train_games = shuffled_games[n_val_g:]
    else:
        train_games = shuffled_games
        val_games = []

    sp_train = SelfPlayDataset(sp_paths, allowed_game_ids=train_games, augment=True, include_probes=True)
    sp_val = SelfPlayDataset(sp_paths, allowed_game_ids=val_games, augment=False, include_probes=True) if val_games else None

    wu_train = TopologyDatasetV31(
        args.warmup_db,
        augment=True,
        preload_to_ram=True,
        split_json=args.split_json,
        split="train" if args.split_json else None,
    )
    wu_val_full = (
        TopologyDatasetV31(
            args.warmup_db,
            augment=False,
            preload_to_ram=True,
            split_json=args.split_json,
            split="val",
        )
        if args.split_json
        else None
    )
    if wu_val_full is not None and len(wu_val_full) > 4096:
        val_rng = np.random.default_rng(args.seed)
        sub_idx = np.sort(val_rng.choice(len(wu_val_full), size=4096, replace=False)).tolist()
        wu_val = Subset(wu_val_full, sub_idx)
    else:
        wu_val = wu_val_full

    if args.num_workers >= 0:
        n_workers = args.num_workers
    elif device.type == "cuda":
        n_workers = 2
    else:
        n_workers = 0

    if is_main:
        print("=" * 64)
        print(f"Stage 6 Self-Play Retraining | init={args.init_checkpoint.name} | device={device} (ranks={world_size})")
        print(f"  Self-Play games: {len(all_games)} (train={len(train_games)}, val={len(val_games)})")
        print(f"  Self-Play samples: train={len(sp_train):,}, val={len(sp_val) if sp_val else 0:,} ({sp_train.get_memory_usage_mb():.1f} MB)")
        print(f"  Warmup anchor: train={len(wu_train):,}, val_sub={len(wu_val) if wu_val else 0:,}")
        print(f"  Mix={args.selfplay_fraction:.0%}/{1-args.selfplay_fraction:.0%} | lr={args.lr:.2e} | epochs={args.epochs} | lambda_sed={args.lambda_sed}")
        print("=" * 64)

    ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    arch = str(cfg.get("architecture", "phys_uvit"))
    model = build_model(
        architecture=arch,
        in_channels=int(cfg.get("in_channels", 7)),
        out_channels=int(cfg.get("out_channels", 2)),
        feature_size=int(cfg.get("feature_size", 32)),
        use_swin=bool(cfg.get("use_swin", False)),
        embed_dim=int(cfg.get("embed_dim", 192)),
        transformer_depth=int(cfg.get("transformer_depth", 5)),
        num_heads=int(cfg.get("num_heads", 6)),
    )
    raw_sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    clean_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in raw_sd.items()}
    incompat = model.load_state_dict(clean_sd, strict=False)
    if incompat.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {incompat.unexpected_keys}")
    disallowed_missing = [k for k in incompat.missing_keys if not k.startswith("op_head.")]
    if disallowed_missing:
        raise RuntimeError(f"Missing checkpoint keys: {disallowed_missing}")
    if args.lambda_sed > 0.0 and hasattr(model, "sed_head"):
        for p in model.sed_head.parameters():
            p.requires_grad_(True)
    model = model.to(device)

    if is_distributed:
        dev_ids = [local_rank] if device.type == "cuda" else None
        out_dev = local_rank if device.type == "cuda" else None
        model = DDP(model, device_ids=dev_ids, output_device=out_dev, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.epochs > 1:
        warmup_sch = LinearLR(optimizer, start_factor=0.2, total_iters=1)
        cos_sch = CosineAnnealingLR(optimizer, T_max=args.epochs - 1, eta_min=args.lr * 0.1)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sch, cos_sch], milestones=[1])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=1, eta_min=args.lr * 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    train_ds = MixedReplayDataset(
        selfplay_ds=sp_train,
        warmup_ds=wu_train,
        selfplay_fraction=args.selfplay_fraction,
        min_epoch_samples=args.min_epoch_samples,
        seed=args.seed,
    )
    train_sampler = BucketedResolutionBatchSampler(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
    )
    collate_fn = make_symmetry_collate_fn(enable_rot90=True)

    val_sp_loader = None
    if sp_val is not None and len(sp_val) > 0:
        val_sp_sampler = BucketedResolutionBatchSampler(
            sp_val, batch_size=args.batch_size, shuffle=False, rank=rank, world_size=world_size, seed=args.seed
        )
        val_sp_loader = DataLoader(sp_val, batch_sampler=val_sp_sampler, num_workers=n_workers, pin_memory=(device.type == "cuda"))

    val_wu_loader = None
    if wu_val is not None and len(wu_val) > 0:
        val_wu_sampler = BucketedResolutionBatchSampler(
            wu_val, batch_size=args.batch_size, shuffle=False, rank=rank, world_size=world_size, seed=args.seed
        )
        val_wu_loader = DataLoader(wu_val, batch_sampler=val_wu_sampler, num_workers=n_workers, pin_memory=(device.type == "cuda"))

    best_score = float("inf")
    history: List[dict] = []

    def _save(path: Path, ep_idx: int, metrics: dict) -> None:
        raw_m = model.module if hasattr(model, "module") else model
        payload = {
            "epoch": ep_idx,
            "model_state_dict": raw_m.state_dict(),
            "metrics": metrics,
            "val_loss": float(metrics.get("combined_val_loss", metrics.get("loss", float("inf")))),
            "config": {
                **cfg,
                "architecture": arch,
                "lambda_bp": args.lambda_bp,
                "lambda_log_c": args.lambda_log_c,
                "lambda_conn": args.lambda_conn,
                "lambda_sed": args.lambda_sed,
                "lambda_op": args.lambda_op,
            },
        }
        torch.save(payload, path)

    try:
        for epoch in range(args.epochs):
            t0 = time.time()
            train_ds.set_epoch(epoch)
            train_sampler.set_epoch(epoch)
            train_loader = DataLoader(
                train_ds,
                batch_sampler=train_sampler,
                collate_fn=collate_fn,
                num_workers=n_workers,
                pin_memory=(device.type == "cuda"),
            )
            tr_m = _run_epoch(
                model, train_loader, optimizer, scaler, device, is_distributed, is_main, use_amp,
                args.lambda_policy, args.lambda_value, args.lambda_bp, args.lambda_log_c, args.lambda_conn, args.lambda_sed, args.lambda_op,
                1.0, f"SP-Train {epoch+1}/{args.epochs}",
            )
            val_sp_m = (
                _run_epoch(
                    model, val_sp_loader, None, None, device, is_distributed, is_main, use_amp,
                    args.lambda_policy, args.lambda_value, 0.0, args.lambda_log_c, args.lambda_conn, args.lambda_sed, args.lambda_op,
                    0.0, "Val-SP",
                )
                if val_sp_loader is not None
                else None
            )
            val_wu_m = (
                _run_epoch(
                    model, val_wu_loader, None, None, device, is_distributed, is_main, use_amp,
                    args.lambda_policy, args.lambda_value, args.lambda_bp, args.lambda_log_c, args.lambda_conn, 0.0, 0.0,
                    0.0, "Val-WU",
                )
                if val_wu_loader is not None
                else None
            )

            scheduler.step()
            cur_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            if val_sp_m is not None and val_wu_m is not None:
                comb_loss = 0.70 * val_sp_m["loss"] + 0.30 * val_wu_m["loss"]
            elif val_sp_m is not None:
                comb_loss = val_sp_m["loss"]
            elif val_wu_m is not None:
                comb_loss = val_wu_m["loss"]
            else:
                comb_loss = tr_m["loss"]

            if is_main:
                sp_str = (
                    f"ValSP={val_sp_m['loss']:.4f}(MAE={val_sp_m['value_mae']:.3f},logC={val_sp_m.get('log_c_mae',0.0):.3f})"
                    if val_sp_m
                    else "ValSP=---"
                )
                wu_str = (
                    f"ValWU={val_wu_m['loss']:.4f}(Dice={val_wu_m.get('blueprint_dice',0.0):.3f})"
                    if val_wu_m
                    else "ValWU=---"
                )
                print(
                    f"Ep {epoch+1}/{args.epochs} ({elapsed:.1f}s, lr={cur_lr:.2e}) | "
                    f"Train={tr_m['loss']:.4f} | {sp_str} | {wu_str} | Comb={comb_loss:.4f}",
                    flush=True,
                )
                rec = {
                    "epoch": epoch + 1,
                    "lr": cur_lr,
                    "elapsed_sec": round(elapsed, 2),
                    "combined_val_loss": float(comb_loss),
                    "train": tr_m,
                    "val_selfplay": val_sp_m,
                    "val_warmup": val_wu_m,
                }
                history.append(rec)
                (args.out_dir / "train_metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
                _save(args.out_dir / "last_model.pt", epoch, rec)
                if comb_loss < best_score:
                    best_score = comb_loss
                    _save(args.out_dir / "best_model.pt", epoch, rec)
    finally:
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
