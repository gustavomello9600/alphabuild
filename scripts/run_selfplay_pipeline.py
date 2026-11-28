#!/usr/bin/env python3
"""
Iterative Multi-Generation Self-Play Pipeline Orchestrator for AlphaBuilder v4.0 (Stage 6).

For each generation g = start_gen .. end_gen:
  1. Adaptive 4-Tier Curriculum Sampling + Multi-GPU Episode Generation (`runner.py` + `fast_amg`)
  2. Shard DB merge (`merge_selfplay_dbs`) + telemetry summary + Phase-1 adaptive pacing update
  3. Mixed Replay DDP Fine-Tuning (`scripts/train_selfplay_step.py` over recent Self-Play + Warmup Anchor)
  4. Deterministic MCTS Evaluation (pure MCTS, zero probes/SED!) on `S1-S6` + `G1-G4` OOD suite
  5. Incremental persistence of `selfplay_evolution_history.json` + wall-clock guard for Kaggle
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphabuilder.src.logic.selfplay import initialize_selfplay_db, merge_selfplay_dbs
from alphabuilder.src.logic.selfplay.runner import EpisodeConfig, run_episode_with_stats
from alphabuilder.src.logic.selfplay.scenarios_stage6 import (
    adjust_tier_weights,
    get_base_tier_weights,
)
from alphabuilder.src.neural.inference import AlphaBuilderInference


def _run_worker_shard(args: argparse.Namespace) -> None:
    """Worker entrypoint generating a subset of self-play episodes into `--shard-db`."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tier_weights = tuple(float(x) for x in args.tier_weights.split(",")) if args.tier_weights else None
    initialize_selfplay_db(args.shard_db)
    model = AlphaBuilderInference(args.checkpoint, device=args.device)

    stats_list: List[Dict[str, Any]] = []
    for i in range(args.episodes):
        ep_cfg = EpisodeConfig(
            max_steps=600,
            max_phase1_steps=args.max_phase1_steps,
            max_phase2_steps=args.max_phase2_steps,
            generation=args.generation,
            num_simulations_p1=args.num_simulations,
            num_simulations_p2=args.num_simulations,
            p2_action_batch_size=args.p2_batch_early,
            p2_action_batch_size_late=args.p2_batch_late,
            p2_min_remove_candidates=16,
            p2_remove_prior_floor=0.25,
            p2_remove_probe_prob=args.probe_prob,
            fem_backend=args.fem_backend,
            record_displacement_map=False,
            record_sed_map=True,
            use_v41_mcts=not args.legacy_mcts,
            use_v42_mcts=bool(args.v42_mcts and not args.legacy_mcts),
            use_best_of_episode_targets=not args.legacy_mcts,
            tier_weights=tier_weights,  # type: ignore[arg-type]
            seed=args.seed * 1000 + i,
            verbose=False,
        )
        _, _, _, st = run_episode_with_stats(model, args.shard_db, ep_cfg)
        stats_list.append(st)
        print(
            f"  [Worker seed={args.seed}] ep {i+1}/{args.episodes} | "
            f"T{st['tier']} ({st['family']}) {st['end_reason']} | "
            f"conn={st['connected_final']} P1={st['phase1_steps']} P2={st['phase2_steps']} "
            f"V={st['final_volume']*100:.2f}% C={st['final_compliance']} "
            f"Score={st['final_score']:+.3f} Best={st.get('best_score', st['final_score']):+.3f} "
            f"REM={st['p2_rem_actions']} reuse={st.get('tree_reuse_hits', 0)} dag={st.get('dag_transpositions', 0)} "
            f"probes={st['probes_adopted']}/{st['probes_run']} rej={st.get('trust_rejects', 0)} cyc={st.get('cycle_avoided', 0)} "
            f"({st['elapsed_sec']:.1f}s)",
            flush=True,
        )
        gc.collect()

    args.shard_stats.write_text(json.dumps(stats_list, indent=2), encoding="utf-8")


def _summarize_generation_stats(stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not stats_list:
        return {"episodes": 0, "tier_conn_rates": {}}
    by_tier: Dict[int, List[bool]] = {}
    scores: List[float] = []
    comps: List[float] = []
    vols: List[float] = []
    adds = 0
    rems = 0
    p_run = 0
    p_adopt = 0
    rej_tot = 0
    cyc_tot = 0
    elapseds: List[float] = []
    for s in stats_list:
        by_tier.setdefault(int(s["tier"]), []).append(bool(s["reached_phase2"] and s["connected_final"]))
        scores.append(float(s["final_score"]))
        if s["final_compliance"] is not None:
            comps.append(float(s["final_compliance"]))
        vols.append(float(s["final_volume"]))
        adds += int(s.get("p2_add_actions", 0))
        rems += int(s.get("p2_rem_actions", 0))
        p_run += int(s.get("probes_run", 0))
        p_adopt += int(s.get("probes_adopted", 0))
        rej_tot += int(s.get("trust_rejects", 0))
        cyc_tot += int(s.get("cycle_avoided", 0))
        elapseds.append(float(s.get("elapsed_sec", 0.0)))

    tier_rates = {str(t): float(np.mean(vals)) for t, vals in sorted(by_tier.items())}
    tot_actions = max(1, adds + rems)
    return {
        "episodes": len(stats_list),
        "conn_rate": float(np.mean([s["connected_final"] for s in stats_list])),
        "tier_conn_rates": tier_rates,
        "mean_score": float(np.mean(scores)),
        "mean_compliance": float(np.mean(comps)) if comps else None,
        "mean_volume": float(np.mean(vols)),
        "p2_add_total": adds,
        "p2_rem_total": rems,
        "p2_rem_fraction": float(rems / tot_actions),
        "probes_run": p_run,
        "probes_adopted": p_adopt,
        "total_trust_rejects": rej_tot,
        "total_cycle_avoided": cyc_tot,
        "mean_best_score": float(np.mean([s.get("best_score", s["final_score"]) for s in stats_list])),
        "stop_fraction": float(np.mean([1.0 if s.get("end_reason") == "MCTS_STOP" else 0.0 for s in stats_list])),
        "mean_tree_reuse_hits": float(np.mean([s.get("tree_reuse_hits", 0) for s in stats_list])),
        "total_dag_transpositions": int(sum(int(s.get("dag_transpositions", 0)) for s in stats_list)),
        "mean_episode_sec": float(np.mean(elapseds)),
    }


def _summarize_eval_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    eps = payload.get("episodes", [])
    if not eps:
        return {}
    scores = [float(e["final_score"]) for e in eps]
    comps = [float(e["final_compliance"]) for e in eps if e.get("final_compliance") is not None]
    vols = [float(e["final_volume"]) for e in eps]
    hollows = [float(e["geometry"]["hollow_ratio_xy"]) for e in eps]
    inertias = [float(e["geometry"]["inertia_y_norm"]) for e in eps]
    val_maes = [float(e["value_calibration_p2"]["mae"]) for e in eps if e.get("value_calibration_p2", {}).get("mae") is not None]
    val_bias = [float(e["value_calibration_p2"]["bias_net_minus_fem"]) for e in eps if e.get("value_calibration_p2", {}).get("bias_net_minus_fem") is not None]
    lc_maes = [float(e["physics_aux_diagnostics"]["log_c_mae"]) for e in eps if (e.get("physics_aux_diagnostics") or {}).get("log_c_mae") is not None]
    lc_bias = [float(e["physics_aux_diagnostics"]["log_c_bias_pred_minus_true"]) for e in eps if (e.get("physics_aux_diagnostics") or {}).get("log_c_bias_pred_minus_true") is not None]
    rems = sum(int(e["mcts_diagnostics"]["p2_rem_actions_total"]) for e in eps)
    adds = sum(int(e["mcts_diagnostics"]["p2_add_actions_total"]) for e in eps)
    return {
        "n_scenarios": len(eps),
        "n_connected": sum(1 for e in eps if e.get("connected_final")),
        "mean_score": float(np.mean(scores)),
        "mean_compliance": float(np.mean(comps)) if comps else None,
        "mean_volume": float(np.mean(vols)),
        "mean_hollow_xy": float(np.mean(hollows)),
        "mean_inertia_y": float(np.mean(inertias)),
        "p2_add_total": adds,
        "p2_rem_total": rems,
        "value_ood_mae": float(np.mean(val_maes)) if val_maes else None,
        "value_ood_bias": float(np.mean(val_bias)) if val_bias else None,
        "log_c_ood_mae": float(np.mean(lc_maes)) if lc_maes else None,
        "log_c_ood_bias": float(np.mean(lc_bias)) if lc_bias else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6 Iterative Self-Play Pipeline Orchestrator.")
    parser.add_argument("--_worker-shard", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shard-db", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shard-stats", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--generation", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--episodes", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--tier-weights", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--probe-prob", type=float, default=0.50, help=argparse.SUPPRESS)
    parser.add_argument("--p2-batch-early", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument("--p2-batch-late", type=int, default=8, help=argparse.SUPPRESS)

    parser.add_argument("--mode", type=str, default="full", choices=["smoke", "full"])
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--warmup-db", type=Path, default=None)
    parser.add_argument("--split-json", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=Path("data/stage6_work"))
    parser.add_argument("--start-gen", type=int, default=None)
    parser.add_argument("--end-gen", type=int, default=None)
    parser.add_argument("--episodes-per-gen", type=int, default=None)
    parser.add_argument("--train-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--selfplay-fraction", type=float, default=0.60)
    parser.add_argument("--num-simulations", type=int, default=None)
    parser.add_argument("--max-phase1-steps", type=int, default=100)
    parser.add_argument("--max-phase2-steps", type=int, default=None)
    parser.add_argument("--fem-backend", type=str, default="fast_amg", choices=["auto", "dolfinx", "fast_amg"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--prior-selfplay-dbs", type=str, default="")
    parser.add_argument("--eval-extended", action="store_true", default=True)
    parser.add_argument("--eval-ood", action="store_true", default=True)
    parser.add_argument("--legacy-mcts", action="store_true", default=False, help="Disable v4.1 MCTS-DAG/macro/lifecycle upgrades")
    parser.add_argument("--v42-mcts", action="store_true", default=False, help="Enable Stage 6 v4.2 volume-aware expansion + trust-region upgrades")
    parser.add_argument("--lambda-op", type=float, default=0.20)
    parser.add_argument("--max-wall-hours", type=float, default=10.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args._worker_shard:
        _run_worker_shard(args)
        return

    if args.init_checkpoint is None or args.warmup_db is None:
        raise ValueError("--init-checkpoint and --warmup-db are required in orchestrator mode.")

    if args.mode == "smoke":
        start_gen = args.start_gen or 1
        end_gen = args.end_gen or 1
        eps_per_gen = args.episodes_per_gen or 8
        train_epochs = args.train_epochs or 2
        num_sims = args.num_simulations or 64
        max_p2 = args.max_phase2_steps or 32
    else:
        start_gen = args.start_gen or 1
        end_gen = args.end_gen or 5
        eps_per_gen = args.episodes_per_gen or 64
        train_epochs = args.train_epochs or 6
        num_sims = args.num_simulations or 160
        max_p2 = args.max_phase2_steps or 64

    args.max_phase1_steps = args.max_phase1_steps
    args.work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.work_dir / "checkpoints"
    eval_dir = args.work_dir / "eval"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    run_t0 = time.time()
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_workers = max(1, n_gpus)
    eval_dev = args.device if args.device != "auto" else ("cuda" if n_gpus > 0 else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"))

    history_path = args.work_dir / "selfplay_evolution_history.json"
    history: List[Dict[str, Any]] = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    cumulative_gpu_hours = float(history[-1]["cumulative_gpu_hours"]) if history else 0.0
    prev_tier_rates: Optional[Dict[int, float]] = None
    if history and history[-1].get("generation_stats", {}).get("tier_conn_rates"):
        prev_tier_rates = {int(k): float(v) for k, v in history[-1]["generation_stats"]["tier_conn_rates"].items()}

    sp_dbs_window: List[Path] = [Path(p.strip()) for p in args.prior_selfplay_dbs.split(",") if p.strip() and Path(p.strip()).exists()]
    current_ckpt = args.init_checkpoint

    print("=" * 72)
    print(f"AlphaBuilder Stage 6 Self-Play Orchestrator ({args.mode.upper()})")
    print(f"  Generations: {start_gen} -> {end_gen} | Episodes/gen: {eps_per_gen} | GPUs: {n_gpus}")
    print(f"  Init ckpt:   {current_ckpt}")
    print(f"  FEM backend: {args.fem_backend} | Sims: {num_sims} | P2 budget: {max_p2} steps")
    print("=" * 72, flush=True)

    for gen in range(start_gen, end_gen + 1):
        elapsed_h = (time.time() - run_t0) / 3600.0
        if history and elapsed_h >= args.max_wall_hours:
            print(f"\n[WALL GUARD] Reached {elapsed_h:.2f}h >= {args.max_wall_hours:.2f}h. Stopping cleanly before Gen {gen}.", flush=True)
            break

        gen_t0 = time.time()
        base_tw = get_base_tier_weights(gen)
        tw = adjust_tier_weights(base_tw, prev_tier_rates, min_conn_rate=0.65)
        tw_str = ",".join(f"{w:.4f}" for w in tw)
        print(f"\n{'#'*72}\n# GENERATION {gen}/{end_gen} | Tier weights (T0..T3): {tw_str}\n{'#'*72}", flush=True)

        # 1. Parallel Episode Generation across available GPUs
        gen_db = args.work_dir / f"selfplay_gen{gen}.db"
        shard_dir = args.work_dir / f"shards_gen{gen}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        procs = []
        shard_dbs: List[Path] = []
        shard_stats_paths: List[Path] = []

        per_worker = eps_per_gen // n_workers
        rem = eps_per_gen % n_workers
        for r in range(n_workers):
            cnt = per_worker + (1 if r < rem else 0)
            if cnt == 0:
                continue
            sdb = shard_dir / f"shard_{r}.db"
            sst = shard_dir / f"stats_{r}.json"
            shard_dbs.append(sdb)
            shard_stats_paths.append(sst)
            env = os.environ.copy()
            if n_gpus > 0:
                env["CUDA_VISIBLE_DEVICES"] = str(r)
            cmd = [
                sys.executable,
                __file__,
                "--_worker-shard",
                "--shard-db", str(sdb),
                "--shard-stats", str(sst),
                "--checkpoint", str(current_ckpt),
                "--generation", str(gen),
                "--episodes", str(cnt),
                "--tier-weights", tw_str,
                "--num-simulations", str(num_sims),
                "--max-phase1-steps", str(args.max_phase1_steps),
                "--max-phase2-steps", str(max_p2),
                "--fem-backend", args.fem_backend,
                "--device", "cuda" if n_gpus > 0 else eval_dev,
                "--seed", str(args.seed + gen * 100 + r),
            ]
            if args.legacy_mcts:
                cmd.append("--legacy-mcts")
            elif args.v42_mcts:
                cmd.append("--v42-mcts")
            procs.append(subprocess.Popen(cmd, env=env))

        for p in procs:
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"Generation {gen} worker shard exited with code {rc}")

        if gen_db.exists():
            gen_db.unlink()
        n_g, n_s, n_p = merge_selfplay_dbs(shard_dbs, gen_db)
        all_ep_stats: List[Dict[str, Any]] = []
        for sst in shard_stats_paths:
            if sst.exists():
                all_ep_stats.extend(json.loads(sst.read_text(encoding="utf-8")))
        gen_stats = _summarize_generation_stats(all_ep_stats)
        prev_tier_rates = {int(k): float(v) for k, v in gen_stats.get("tier_conn_rates", {}).items()}
        sp_dbs_window.append(gen_db)
        recent_dbs = sp_dbs_window[-5:]  # Sliding window of last 5 generations

        print(
            f"[GEN {gen}] Merged {n_g} games, {n_s} steps, {n_p} fem_probes | "
            f"conn={gen_stats['conn_rate']:.1%} REM={gen_stats['p2_rem_fraction']:.1%} "
            f"probes_adopted={gen_stats['probes_adopted']}/{gen_stats['probes_run']}",
            flush=True,
        )

        # 2. Mixed Replay DDP Retraining
        train_out = args.work_dir / f"train_gen{gen}"
        if n_gpus > 1:
            train_cmd = ["torchrun", "--standalone", f"--nproc_per_node={n_gpus}", str(REPO_ROOT / "scripts" / "train_selfplay_step.py")]
        else:
            train_cmd = [sys.executable, str(REPO_ROOT / "scripts" / "train_selfplay_step.py")]
        train_cmd.extend([
            "--init-checkpoint", str(current_ckpt),
            "--selfplay-dbs", ",".join(str(p) for p in recent_dbs),
            "--warmup-db", str(args.warmup_db),
            "--out-dir", str(train_out),
            "--epochs", str(train_epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--selfplay-fraction", str(args.selfplay_fraction),
            "--lambda-op", str(0.0 if args.legacy_mcts else args.lambda_op),
            "--seed", str(args.seed + gen),
        ])
        if args.split_json is not None:
            train_cmd.extend(["--split-json", str(args.split_json)])
        if n_gpus == 0:
            train_cmd.extend(["--device", eval_dev])
        subprocess.run(train_cmd, check=True)

        gen_ckpt = ckpt_dir / f"phys_uvit_selfplay_gen{gen}.pt"
        shutil.copy2(train_out / "best_model.pt", gen_ckpt)
        current_ckpt = gen_ckpt
        train_hist = json.loads((train_out / "train_metrics.json").read_text(encoding="utf-8"))

        # 3. Deterministic Pure-MCTS Evaluation (S1-S6 canonical + extended + G1-G4 OOD)
        eval_db = eval_dir / f"eval_gen{gen}.db"
        eval_can_json = eval_dir / f"eval_gen{gen}_canonical.json"
        can_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_inference_scenarios.py"),
            "--checkpoint", str(gen_ckpt),
            "--prefix", f"sp_gen{gen}",
            "--checkpoint-version", f"selfplay-gen{gen}",
            "--db-path", str(eval_db),
            "--json-out", str(eval_can_json),
            "--device", eval_dev,
            "--num-simulations", str(num_sims),
            "--max-phase1-steps", "80",
            "--max-phase2-steps", "30",
            "--suite", "canonical",
            "--fem-backend", args.fem_backend,
        ]
        if not args.legacy_mcts:
            can_cmd.append("--v42-mcts" if args.v42_mcts else "--v41-mcts")
        subprocess.run(can_cmd, check=True)
        can_summary = _summarize_eval_payload(json.loads(eval_can_json.read_text(encoding="utf-8")))

        ext_summary = None
        if args.eval_extended and max_p2 > 30:
            eval_ext_json = eval_dir / f"eval_gen{gen}_extended.json"
            ext_cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "evaluate_inference_scenarios.py"),
                "--checkpoint", str(gen_ckpt),
                "--prefix", f"sp_gen{gen}_ext",
                "--checkpoint-version", f"selfplay-gen{gen}-ext",
                "--db-path", str(eval_db),
                "--json-out", str(eval_ext_json),
                "--device", eval_dev,
                "--num-simulations", str(num_sims),
                "--max-phase1-steps", "80",
                "--max-phase2-steps", str(max_p2),
                "--p2-stepped-batch",
                "--suite", "canonical",
                "--fem-backend", args.fem_backend,
            ]
            if not args.legacy_mcts:
                ext_cmd.append("--v42-mcts" if args.v42_mcts else "--v41-mcts")
            subprocess.run(ext_cmd, check=True)
            ext_summary = _summarize_eval_payload(json.loads(eval_ext_json.read_text(encoding="utf-8")))

        ood_summary = None
        if args.eval_ood:
            eval_ood_json = eval_dir / f"eval_gen{gen}_ood.json"
            ood_cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "evaluate_inference_scenarios.py"),
                "--checkpoint", str(gen_ckpt),
                "--prefix", f"sp_gen{gen}_ood",
                "--checkpoint-version", f"selfplay-gen{gen}-ood",
                "--db-path", str(eval_db),
                "--json-out", str(eval_ood_json),
                "--device", eval_dev,
                "--num-simulations", str(num_sims),
                "--max-phase1-steps", "80",
                "--max-phase2-steps", str(max_p2),
                "--p2-stepped-batch",
                "--suite", "ood",
                "--fem-backend", args.fem_backend,
            ]
            if not args.legacy_mcts:
                ood_cmd.append("--v42-mcts" if args.v42_mcts else "--v41-mcts")
            subprocess.run(ood_cmd, check=True)
            ood_summary = _summarize_eval_payload(json.loads(eval_ood_json.read_text(encoding="utf-8")))

        gen_sec = time.time() - gen_t0
        gen_gpu_h = (gen_sec / 3600.0) * max(1, n_gpus)
        cumulative_gpu_hours += gen_gpu_h

        rec = {
            "generation": gen,
            "checkpoint": str(gen_ckpt),
            "tier_weights": list(tw),
            "elapsed_sec": round(gen_sec, 1),
            "gen_gpu_hours": round(gen_gpu_h, 3),
            "cumulative_gpu_hours": round(cumulative_gpu_hours, 3),
            "generation_stats": gen_stats,
            "train_final": train_hist[-1] if train_hist else None,
            "eval_canonical_30x8": can_summary,
            "eval_canonical_extended": ext_summary,
            "eval_ood_g1_g4": ood_summary,
        }
        history.append(rec)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"[GEN {gen} COMPLETE in {gen_sec/60:.1f}m] "
            f"S1-S6(30x8) Score={can_summary.get('mean_score', 0):+.3f} C={can_summary.get('mean_compliance')} "
            f"logC_MAE={can_summary.get('log_c_ood_mae')} | "
            f"ExtScore={ext_summary.get('mean_score') if ext_summary else '---'} | "
            f"OOD={ood_summary.get('mean_score') if ood_summary else '---'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
