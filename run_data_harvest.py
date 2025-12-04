#!/usr/bin/env python3
"""
Data Harvest Script for AlphaBuilder v3.1 (The Data Factory).

Implements the v3.1 Specification:
1. Generator v2 (Bézier + Rectangular Sections) for Phase 1 Ground Truth.
2. 50-Step Slicing for Phase 1 Training Data.
3. SIMP Optimization for Phase 2 Ground Truth and Training Data.
4. Instance Normalization compatible data structures.
5. Log-Squash Value Normalization (Spec 4.2).
6. Variable Boundary Conditions (full clamp, roller, rail).
7. 7-Channel Input Tensor with separate BC masks.
"""

import sys
import time
import argparse
import random
import uuid
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
    save_episode,
    save_step,
    update_episode_final,
    EpisodeInfo,
    StepRecord
)
from alphabuilder.src.utils.logger import TrainingLogger

# Import from new modules
from alphabuilder.src.logic.harvest.config import SIMPConfig
from alphabuilder.src.logic.harvest.generators import (
    generate_random_load_config,
    generate_bezier_structure,
    generate_seeded_cantilever,
    generate_bc_masks
)
from alphabuilder.src.logic.harvest.optimization import run_fenitop_optimization
from alphabuilder.src.logic.harvest.processing import (
    compute_normalized_value,
    check_connectivity,
    generate_phase1_slices,
    generate_refinement_targets
)

def parse_args():
    parser = argparse.ArgumentParser(description="AlphaBuilder v2.0 Data Harvest")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--db-path", type=str, default="data/training_data.db")
    parser.add_argument("--resolution", type=str, default="64x32x8")
    parser.add_argument("--seed-offset", type=int, default=0, help="Offset for random seed")
    parser.add_argument("--strategy", type=str, choices=['BEZIER', 'FULL_DOMAIN'], help="Force specific strategy")
    parser.add_argument("--max-iter", type=int, default=120, help="Max SIMP iterations (default 120, use 30 for tests)")
    return parser.parse_args()

def main():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    is_main_rank = (comm.rank == 0)
    
    args = parse_args()
    
    # Setup (only main rank does IO)
    db_path = Path(args.db_path)
    if is_main_rank:
        initialize_database(db_path)
    
    # Resolution
    resolution = [int(x) for x in args.resolution.split("x")]
    nx, ny, nz = resolution
    
    # Logger (only main rank)
    logger = None
    if is_main_rank:
        logger = TrainingLogger(str(db_path.parent / "logs"), "harvest_v2.csv", 
                               ["episode", "duration", "compliance", "vol_frac", "phase1_samples", "phase2_samples"])
    
    # Physics Props
    props = PhysicalProperties()
    
    if is_main_rank:
        print(f"Starting Harvest: {args.episodes} episodes")
        episode_iterator = tqdm(range(args.episodes), desc="Episodes", unit="ep")
    else:
        episode_iterator = range(args.episodes)
    
    for i in episode_iterator:
        start_time = time.time()
        
        # Set seed (all ranks must use same seed for consistency)
        seed = args.seed_offset + i
        np.random.seed(seed)
        random.seed(seed)
        
        episode_id = str(uuid.uuid4())
        
        # Strategy Selection: 20% Full Domain, 80% Bezier
        if args.strategy:
            strategy = args.strategy
        else:
            strategy = 'FULL_DOMAIN' if random.random() < 0.2 else 'BEZIER'
        
        # 1. Generate Initial Structure (all ranks generate same structure due to same seed)
        load_config = generate_random_load_config(resolution)
        
        if strategy == 'FULL_DOMAIN':
            v_constructed = generate_seeded_cantilever(resolution, load_config)
        else:
            v_constructed = generate_bezier_structure(resolution, load_config)
        
        # Target volumes
        if strategy == 'FULL_DOMAIN':
            target_vol = 0.15
        else:
            target_vol = 0.10
        
        bc_type_episode = load_config.get('bc_type', 'FULL_CLAMP')
        if is_main_rank:
            print(f"Episode {i}: Strategy={strategy}, Target V={target_vol:.3f}, BC={bc_type_episode}", flush=True)

            # Create and Save EpisodeInfo (v2 Schema)
            mask_x, mask_y, mask_z = generate_bc_masks(tuple(resolution), bc_type_episode)
            bc_masks = np.stack([mask_x, mask_y, mask_z], axis=0)
            
            fx = np.zeros(resolution, dtype=np.float32)
            fy = np.zeros(resolution, dtype=np.float32)
            fz = np.zeros(resolution, dtype=np.float32)
            
            ly = load_config['y']
            lz_center = (load_config['z_start'] + load_config['z_end']) / 2.0
            load_half_width = 1.0
            
            y_min = max(0, int(ly - load_half_width))
            y_max = min(ny, int(ly + load_half_width) + 1)
            z_min = max(0, int(lz_center - load_half_width))
            z_max = min(nz, int(lz_center + load_half_width) + 1)
            
            fy[nx-1, y_min:y_max, z_min:z_max] = -1.0
            forces = np.stack([fx, fy, fz], axis=0)
            
            episode_info = EpisodeInfo(
                episode_id=episode_id,
                bc_masks=bc_masks,
                forces=forces,
                load_config=load_config,
                bc_type=bc_type_episode,
                strategy=strategy,
                resolution=tuple(resolution),
                final_compliance=None, # Updated later
                final_volume=None      # Updated later
            )
            save_episode(db_path, episode_info)

        simp_config = SIMPConfig(
            vol_frac=target_vol,
            max_iter=args.max_iter,
            r_min=1.5,
            adaptive_penal=True, 
            load_config=load_config,
            debug_log_path=f"debug_simp_log_ep{i}.csv" 
        )
        
        # Run FEniTop Optimization
        try:
            simp_history = run_fenitop_optimization(
                resolution, props, simp_config, 
                initial_density=v_constructed,
                strategy=strategy
            )
        except Exception as e:
            if is_main_rank:
                print(f"Episode {i}: FEniTop failed with error: {e}")
                import traceback
                traceback.print_exc()
            continue
        
        # Only main rank has the full history
        if not is_main_rank:
            continue
        
        if not simp_history:
            print(f"Episode {i}: SIMP failed (no history). Skipping.")
            continue
            
        final_state = simp_history[-1]
        s_final_compliance = final_state['compliance']
        s_final_vol = final_state['vol_frac']
        
        s_final_value = compute_normalized_value(s_final_compliance, s_final_vol)
        
        # 4. Check Connectivity of Final State
        is_connected, _ = check_connectivity(final_state['density_map'], 0.5, load_config)
        
        if not is_connected:
            is_connected_low, _ = check_connectivity(final_state['density_map'], 0.1, load_config)
            if not is_connected_low:
                print(f"Episode {i}: WARNING: Final structure DISCONNECTED. Proceeding with save as requested.")
                logger.log({
                    "episode": i,
                    "duration": 0,
                    "compliance": s_final_compliance,
                    "vol_frac": s_final_vol,
                    "phase1_samples": 0,
                    "phase2_samples": 0,
                    "status": "DISCONNECTED_SAVED"
                })
            else:
                 print(f"Episode {i}: Final structure connected only at low threshold (0.1). Saving with warning.")
        else:
            print(f"Episode {i}: Final structure CONNECTED. Proceeding to save.")

        # 5. Generate Phase 1 Records (Slicing)
        if strategy == 'FULL_DOMAIN':
            phase1_records = []
        else:
            phase1_records = generate_phase1_slices(v_constructed, s_final_value)
        
        # 6. Generate Phase 2 Records (SIMP History)
        phase2_records = []
        
        # Find first step where beta > 1
        start_step = 0
        for t, frame in enumerate(simp_history):
            if frame.get('beta', 1) > 1:
                start_step = t
                break
        
        # NEW STRATEGY: Compare t with t+1 (Next Step)
        print(f"  Phase 2: Skipping first {start_step} steps (beta=1), using Next-Step Difference Strategy")
        
        record_idx = 0
        # Iterate until the second to last step
        for t in range(start_step, len(simp_history) - 1):
            current_frame = simp_history[t]
            next_frame = simp_history[t + 1]
            
            curr_dens = current_frame['density_map']
            next_dens = next_frame['density_map']
            
            # Adaptive Thresholding for Input State
            best_binary = None
            found_threshold = 0.5
            for thresh in [0.5, 0.4, 0.3, 0.2, 0.1]:
                is_conn, binary_mask = check_connectivity(curr_dens, thresh, load_config)
                if is_conn:
                    best_binary = binary_mask.astype(np.float32)
                    found_threshold = thresh
                    break
            
            if best_binary is None:
                best_binary = (curr_dens > 0.1).astype(np.float32)
            
            input_state = best_binary
            
            # Generate Targets using NEW Logic
            target_add, target_remove = generate_refinement_targets(curr_dens, next_dens, input_state)
            
            # Compute normalized value
            current_compliance = float(current_frame['compliance'])
            current_vol = float(np.mean(curr_dens))
            target_value = compute_normalized_value(current_compliance, current_vol)
            
            phase2_records.append({
                "phase": Phase.REFINEMENT,
                "step": len(phase1_records) + record_idx,
                "input_state": input_state,
                "target_add": target_add,
                "target_remove": target_remove,
                "target_value": target_value,
                "threshold": found_threshold,
                "beta": current_frame.get('beta', 1),
                "current_compliance": current_compliance,
                "current_vol": current_vol
            })
            record_idx += 1
            
        # 7. Save to DB (v2 Schema)
        def save_batch(records, phase_enum, is_final_list=None):
            for idx, rec in enumerate(records):
                density = rec['input_state']
                
                policy_add = rec['target_add'].astype(np.float32)
                policy_remove = rec['target_remove'].astype(np.float32)
                
                is_final = is_final_list[idx] if is_final_list else False
                
                is_conn = False
                if phase_enum == Phase.REFINEMENT:
                    is_conn, _ = check_connectivity(density, 0.5, load_config)
                
                step_record = StepRecord(
                    episode_id=episode_id,
                    step=rec['step'],
                    phase=phase_enum,
                    density=density,
                    policy_add=policy_add,
                    policy_remove=policy_remove,
                    fitness_score=rec['target_value'],
                    is_final_step=is_final,
                    is_connected=is_conn
                )
                save_step(db_path, step_record)

        # Phase 1
        phase1_final = [False] * len(phase1_records)
        save_batch(phase1_records, Phase.GROWTH, phase1_final)
        
        # Phase 2
        phase2_final = [False] * len(phase2_records)
        if phase2_final:
            phase2_final[-1] = True
        save_batch(phase2_records, Phase.REFINEMENT, phase2_final)
        
        # Update Episode
        if is_main_rank:
            update_episode_final(db_path, episode_id, s_final_compliance, s_final_vol)
        
        duration = time.time() - start_time
        logger.log({
            "episode": i,
            "duration": duration,
            "compliance": s_final_compliance,
            "vol_frac": s_final_vol,
            "phase1_samples": len(phase1_records),
            "phase2_samples": len(phase2_records),
            "status": "SUCCESS"
        })
        
        print(f"Ep {i+1}/{args.episodes}: C={s_final_compliance:.2f}, V={s_final_vol:.2f}, Samples={len(phase1_records)+len(phase2_records)}, Time={duration:.1f}s")
        
        gc.collect()

if __name__ == "__main__":
    main()
