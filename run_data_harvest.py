#!/usr/bin/env python3
"""
Data Harvest Script for AlphaBuilder v3.3 (The Data Factory).

Implements the v3.3 Specification (Bloco A):
1. Fixed Bezier / Seeded Cantilever generators anchored at load_config['x'].
2. Phase 1 geodesic-frontier slicing (step 0..N-1) with discounted rewards.
3. SIMP Optimization via FEniTop respecting FULL_CLAMP and RAIL_XY BCs.
4. Phase 2 sliding-window (k=5) joint Add/Remove policy targets with protection.
5. Unified v3.3 reward targets + raw physics scalars + atomic SQLite persistence.
"""

import sys
import time
import socket
import argparse
import random
import subprocess
import numpy as np
from pathlib import Path
import gc
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.core.physics_model import PhysicalProperties
from alphabuilder.src.logic.storage import (
    initialize_database,
    Phase,
    EpisodeInfo,
    StepRecord,
    deterministic_episode_id,
    get_completed_seeds,
    save_complete_episode,
)
from alphabuilder.src.utils.logger import TrainingLogger

from alphabuilder.src.logic.harvest.config import (
    SIMPConfig,
    SCHEMA_VERSION,
    PHASE1_NUM_SLICES,
    PHASE1_GEO_BETA,
    PHASE1_GAMMA,
    REFINEMENT_WINDOW_SIZE,
    REFINEMENT_NOISE_THRESHOLD,
    REFINEMENT_SHARPNESS,
)
from alphabuilder.src.logic.harvest.generators import (
    generate_random_load_config,
    generate_bezier_structure,
    generate_seeded_cantilever,
    generate_bc_masks,
    get_load_region_bounds,
)
from alphabuilder.src.logic.harvest.optimization import run_fenitop_optimization
from alphabuilder.src.logic.harvest.processing import (
    check_connectivity,
    generate_phase1_slices,
    generate_refinement_targets,
)
from alphabuilder.src.logic.selfplay.reward import (
    calculate_reward,
    calculate_fem_score,
    calculate_compliance_score,
    calculate_volume_bonus,
)


def _detect_code_version() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return f"{out}+{SCHEMA_VERSION}"
    except Exception:
        pass
    return SCHEMA_VERSION


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaBuilder v3.3 Data Harvest")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--db-path", type=str, default="data/training_data.db")
    parser.add_argument("--resolution", type=str, default="64x32x8")
    parser.add_argument("--seed-offset", type=int, default=0, help="Offset for random seed")
    parser.add_argument("--strategy", type=str, choices=['BEZIER', 'FULL_DOMAIN'], help="Force specific strategy")
    parser.add_argument("--bc-type", type=str, choices=['FULL_CLAMP', 'RAIL_XY'], default=None,
                        help="Optional override for boundary condition type")
    parser.add_argument("--max-iter", type=int, default=120, help="Max SIMP iterations (default 120, use 30 for tests)")
    parser.add_argument("--window-size", type=int, default=REFINEMENT_WINDOW_SIZE,
                        help="Sliding window k for Phase 2 targets (default 5)")
    parser.add_argument("--shard-id", type=str, default=None, help="Shard identifier for distributed harvest")
    parser.add_argument("--worker-id", type=str, default=None, help="Worker node identifier")
    parser.add_argument("--resume-shard", action="store_true",
                        help="Skip seeds already committed in --db-path")
    parser.add_argument("--save-csv-log", action="store_true",
                        help="Write per-episode FEniTop debug CSV logs (disabled by default)")
    parser.add_argument("--code-version", type=str, default=None, help="Code/schema version tag")
    return parser.parse_args()


def main():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    is_main_rank = (comm.rank == 0)

    args = parse_args()
    db_path = Path(args.db_path)
    resolution = tuple(int(x) for x in args.resolution.split("x"))
    nx, ny, nz = resolution
    worker_id = args.worker_id or socket.gethostname()
    code_version = args.code_version or _detect_code_version()

    completed_seeds = set()
    if is_main_rank:
        initialize_database(db_path)
        if args.resume_shard:
            completed_seeds = get_completed_seeds(db_path)
    completed_seeds = comm.bcast(completed_seeds, root=0)

    logger = None
    if is_main_rank:
        logger = TrainingLogger(
            str(db_path.parent / "logs"),
            "harvest_v33.csv",
            ["episode", "seed", "duration", "compliance", "vol_frac", "max_disp", "phase1_samples", "phase2_samples", "status"],
        )

    props = PhysicalProperties()

    if is_main_rank:
        print(f"Starting Harvest v3.3: {args.episodes} episodes | seed_offset={args.seed_offset} | db={db_path}")
        episode_iterator = tqdm(
            range(args.episodes),
            desc="Episodes",
            unit="ep",
            disable=not sys.stderr.isatty(),
        )
    else:
        episode_iterator = range(args.episodes)

    for i in episode_iterator:
        seed = args.seed_offset + i
        if args.resume_shard and seed in completed_seeds:
            if is_main_rank:
                print(f"Episode {i} (seed={seed}): already committed in {db_path.name}, skipping.", flush=True)
            continue

        start_time = time.time()

        # Set identical seed across all MPI ranks
        np.random.seed(seed)
        random.seed(seed)

        episode_id = deterministic_episode_id(seed, resolution, schema_tag=SCHEMA_VERSION)

        # Strategy selection: 20% FULL_DOMAIN, 80% BEZIER
        if args.strategy:
            strategy = args.strategy
        else:
            strategy = 'FULL_DOMAIN' if random.random() < 0.2 else 'BEZIER'

        load_config = generate_random_load_config(resolution)
        if args.bc_type:
            load_config['bc_type'] = args.bc_type

        if strategy == 'FULL_DOMAIN':
            v_constructed = generate_seeded_cantilever(resolution, load_config)
            target_vol = 0.15
        else:
            v_constructed = generate_bezier_structure(resolution, load_config)
            target_vol = 0.10

        bc_type_episode = load_config.get('bc_type', 'FULL_CLAMP')
        mask_x, mask_y, mask_z = generate_bc_masks(resolution, bc_type_episode)
        bc_masks = np.stack([mask_x, mask_y, mask_z], axis=0)

        fx = np.zeros(resolution, dtype=np.float32)
        fy = np.zeros(resolution, dtype=np.float32)
        fz = np.zeros(resolution, dtype=np.float32)
        lx, y_min, y_max, z_min, z_max = get_load_region_bounds(resolution, load_config)
        fy[lx, y_min:y_max, z_min:z_max] = -1.0
        forces = np.stack([fx, fy, fz], axis=0)

        if is_main_rank:
            print(
                f"Episode {i} (seed={seed}): Strategy={strategy}, Target V={target_vol:.3f}, "
                f"BC={bc_type_episode}, load_x={lx}",
                flush=True,
            )

        simp_config = SIMPConfig(
            vol_frac=target_vol,
            max_iter=args.max_iter,
            r_min=1.5,
            adaptive_penal=True,
            load_config=load_config,
            debug_log_path=(f"debug_simp_log_ep{i}.csv" if args.save_csv_log else None),
        )

        # Run FEniTop Optimization (all MPI ranks participate)
        try:
            simp_history = run_fenitop_optimization(
                resolution,
                props,
                simp_config,
                initial_density=v_constructed,
                strategy=strategy,
                record_displacement_map=False,
            )
        except Exception as e:
            if is_main_rank:
                print(f"Episode {i} (seed={seed}): FEniTop failed with error: {e}")
                import traceback
                traceback.print_exc()
            continue

        if not is_main_rank:
            continue

        if not simp_history:
            print(f"Episode {i} (seed={seed}): SIMP failed (no history). Skipping.")
            continue

        final_state = simp_history[-1]
        s_final_compliance = float(final_state['compliance'])
        s_final_vol = float(final_state['vol_frac'])
        s_final_disp = float(final_state.get('max_displacement', 0.0))

        is_final_connected, _ = check_connectivity(
            final_state['density_map'], 0.5, load_config, strict_load_coverage=True
        )
        s_final_reward = calculate_reward(
            s_final_compliance,
            s_final_vol,
            is_valid=is_final_connected,
            max_displacement=s_final_disp,
        )

        if not is_final_connected:
            is_connected_low, _ = check_connectivity(
                final_state['density_map'], 0.1, load_config, strict_load_coverage=True
            )
            if not is_connected_low:
                print(f"Episode {i} (seed={seed}): WARNING: Final structure DISCONNECTED (R={s_final_reward:.3f}).")
            else:
                print(f"Episode {i} (seed={seed}): Final structure connected only at thresh=0.1 (R={s_final_reward:.3f}).")
        else:
            print(f"Episode {i} (seed={seed}): Final structure CONNECTED (R={s_final_reward:.3f}).")

        # Phase 1: Geodesic-frontier slicing (step 0..N-1)
        if strategy == 'FULL_DOMAIN':
            phase1_records = []
        else:
            phase1_records = generate_phase1_slices(
                v_constructed,
                s_final_reward,
                num_steps=PHASE1_NUM_SLICES,
                bc_masks=(mask_x, mask_y, mask_z),
                beta_geo=PHASE1_GEO_BETA,
                gamma=PHASE1_GAMMA,
                include_empty_step=True,
                target_mode="geodesic_boundary",
            )
            # Re-index sequentially so Phase 1 is strictly 0..len-1 even if a degenerate slice is skipped
            for seq_idx, rec in enumerate(phase1_records):
                rec['step'] = seq_idx

        # Phase 2: SIMP History including final converged frame simp_history[-1]
        phase2_records = []
        start_step = 0
        for t, frame in enumerate(simp_history):
            if frame.get('beta', 1) > 1:
                start_step = t
                break

        total_p2 = len(simp_history) - start_step
        print(f"  Phase 2: start_step={start_step}, frames={total_p2}, window_k={args.window_size}")

        for record_idx, t in enumerate(range(start_step, len(simp_history))):
            current_frame = simp_history[t]
            curr_dens = current_frame['density_map']
            is_last_frame = (t == len(simp_history) - 1)

            # Adaptive thresholding for binary input state
            best_binary = None
            for thresh in (0.5, 0.4, 0.3, 0.2, 0.1):
                is_conn_thresh, binary_mask = check_connectivity(
                    curr_dens, thresh, load_config, strict_load_coverage=True
                )
                if is_conn_thresh:
                    best_binary = binary_mask.astype(np.float32)
                    break

            if best_binary is None:
                best_binary = (curr_dens > 0.1).astype(np.float32)

            input_state = best_binary

            if is_last_frame:
                target_add = np.zeros(resolution, dtype=np.float32)
                target_remove = np.zeros(resolution, dtype=np.float32)
            else:
                next_idx = min(t + args.window_size, len(simp_history) - 1)
                next_dens = simp_history[next_idx]['density_map']
                target_add, target_remove = generate_refinement_targets(
                    curr_dens,
                    next_dens,
                    input_state,
                    bc_masks=(mask_x, mask_y, mask_z),
                    forces=forces,
                    noise_threshold=REFINEMENT_NOISE_THRESHOLD,
                    normalization="joint",
                    sharpness=REFINEMENT_SHARPNESS,
                )

            current_compliance = float(current_frame['compliance'])
            current_vol = float(current_frame['vol_frac'])
            current_max_disp = float(current_frame.get('max_displacement', 0.0))
            is_conn_step, _ = check_connectivity(
                input_state, 0.5, load_config, strict_load_coverage=True
            )
            r_fem = calculate_reward(
                current_compliance,
                current_vol,
                is_valid=is_conn_step,
                max_displacement=current_max_disp,
            )
            lam = float(record_idx) / float(max(1, total_p2 - 1))
            target_value = float((1.0 - lam) * r_fem + lam * s_final_reward)

            phase2_records.append({
                "phase": Phase.REFINEMENT,
                "step": len(phase1_records) + record_idx,
                "input_state": input_state,
                "rho_cont": curr_dens,
                "target_add": target_add,
                "target_remove": target_remove,
                "target_value": target_value,
                "compliance": current_compliance,
                "vol_frac": current_vol,
                "max_displacement": current_max_disp,
                "beta": int(current_frame.get('beta', 1)),
                "is_connected": is_conn_step,
                "is_final_step": is_last_frame,
                "reward_components": {
                    "c_score": calculate_compliance_score(current_compliance),
                    "v_bonus": calculate_volume_bonus(current_vol),
                    "fem_score": calculate_fem_score(current_compliance, current_vol),
                    "r_fem": r_fem,
                    "r_final": s_final_reward,
                    "mixing_factor": lam,
                },
            })

        duration = time.time() - start_time

        episode_info = EpisodeInfo(
            episode_id=episode_id,
            bc_masks=bc_masks,
            forces=forces,
            load_config=load_config,
            bc_type=bc_type_episode,
            strategy=strategy,
            resolution=resolution,
            final_compliance=s_final_compliance,
            final_volume=s_final_vol,
            seed=seed,
            shard_id=args.shard_id,
            worker_id=worker_id,
            code_version=code_version,
            v_constructed=v_constructed,
            final_max_displacement=s_final_disp,
            final_is_connected=is_final_connected,
            simp_iters=len(simp_history),
            duration_sec=duration,
        )

        step_records: list[StepRecord] = []
        for rec in phase1_records:
            p1_dens = rec['input_state']
            p1_conn, _ = check_connectivity(p1_dens, 0.5, load_config, strict_load_coverage=True)
            step_records.append(StepRecord(
                episode_id=episode_id,
                step=rec['step'],
                phase=Phase.GROWTH,
                density=p1_dens,
                policy_add=rec['target_add'],
                policy_remove=rec['target_remove'],
                fitness_score=rec['target_value'],
                is_final_step=False,
                is_connected=p1_conn,
                displacement_map=None,
                reward_components={"r_final": s_final_reward},
                rho_cont=None,
                compliance=None,
                vol_frac=float(rec['current_vol']),
                max_displacement=None,
                beta=None,
            ))

        for rec in phase2_records:
            step_records.append(StepRecord(
                episode_id=episode_id,
                step=rec['step'],
                phase=Phase.REFINEMENT,
                density=rec['input_state'],
                policy_add=rec['target_add'],
                policy_remove=rec['target_remove'],
                fitness_score=rec['target_value'],
                is_final_step=rec['is_final_step'],
                is_connected=rec['is_connected'],
                displacement_map=None,
                reward_components=rec['reward_components'],
                rho_cont=rec['rho_cont'],
                compliance=rec['compliance'],
                vol_frac=rec['vol_frac'],
                max_displacement=rec['max_displacement'],
                beta=rec['beta'],
            ))

        if not step_records:
            print(f"Episode {i} (seed={seed}): WARNING: 0 step records produced. Skipping DB write.")
            continue

        save_complete_episode(db_path, episode_info, step_records)

        logger.log({
            "episode": i,
            "seed": seed,
            "duration": duration,
            "compliance": s_final_compliance,
            "vol_frac": s_final_vol,
            "max_disp": s_final_disp,
            "phase1_samples": len(phase1_records),
            "phase2_samples": len(phase2_records),
            "status": "CONNECTED" if is_final_connected else "DISCONNECTED",
        })

        print(
            f"Ep {i+1}/{args.episodes} (seed={seed}): C={s_final_compliance:.2f}, "
            f"V={s_final_vol:.3f}, Umax={s_final_disp:.2f}, "
            f"Samples={len(step_records)}, Time={duration:.1f}s",
            flush=True,
        )

        gc.collect()


if __name__ == "__main__":
    main()
