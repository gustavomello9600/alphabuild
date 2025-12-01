#!/usr/bin/env python3
"""
Data Harvest Script for AlphaBuilder v2.0 (The Data Factory).

Implements the v2.0 Specification:
1. Generator v2 (Bézier + Rectangular Sections) for Phase 1 Ground Truth.
2. 50-Step Slicing for Phase 1 Training Data.
3. SIMP Optimization for Phase 2 Ground Truth and Training Data.
4. Instance Normalization compatible data structures.
"""

import sys
import time
import argparse
import random
import uuid
import numpy as np
from pathlib import Path
from datetime import datetime
import gc
from typing import List, Tuple, Dict, Any
import scipy.ndimage

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.core.physics_model import (
    initialize_cantilever_context,
    PhysicalProperties
)
from alphabuilder.src.logic.simp_generator import (
    run_simp_optimization_3d, 
    SIMPConfig
)
from alphabuilder.src.logic.storage import (
    initialize_database, 
    get_episode_count, 
    TrainingRecord, 
    Phase, 
    serialize_state, 
    save_record, 
    generate_episode_id
)
from alphabuilder.src.utils.logger import TrainingLogger
from alphabuilder.src.core.tensor_utils import build_input_tensor

# --- Generator v2: Bézier & Rectangular Sections ---

def quadratic_bezier(p0, p1, p2, t):
    """Calculate point on quadratic Bézier curve at t."""
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

def generate_bezier_structure(
    resolution: Tuple[int, int, int],
    load_config: Dict[str, Any]
) -> np.ndarray:
    """
    Generate a procedural structure using Bézier curves with rectangular sections.
    """
    nx, ny, nz = resolution
    # Safe Background: 0.15 to allow sensitivity flow (Vanishing Gradient Fix)
    voxel_grid = np.full(resolution, 0.15, dtype=np.float32)
    
    # 1. Determine Number of Curves
    # Increased to ensure higher initial volume (Erosion Target)
    num_curves = random.randint(3, 5)
    
    # Load Point (Target)
    # Load is a line in Z. We target the center of that line.
    target_x = load_config['x']
    target_y = load_config['y']
    target_z_center = (load_config['z_start'] + load_config['z_end']) / 2.0
    
    # Adjust target to be slightly below the load point to ensure the "top" of the bar hits the load
    # But simpler: just target the load point and ensure the section is large enough.
    p2 = np.array([target_x, target_y, target_z_center])
    
    for _ in range(num_curves):
        # 2. Start Point (Wall X=0)
        # Random Y and Z on the wall
        start_y = random.uniform(0, ny-1)
        start_z = random.uniform(0, nz-1)
        p0 = np.array([0.0, start_y, start_z])
        
        # 3. Control Point (Intermediate)
        # Random point in the volume, biased towards center?
        # Add noise as per spec.
        # Let's pick a midpoint and add gaussian noise.
        midpoint = (p0 + p2) / 2.0
        noise = np.random.normal(0, 5.0, size=3) # Sigma=5 voxels
        p1 = midpoint + noise
        # Clip to bounds
        p1[0] = np.clip(p1[0], 0, nx-1)
        p1[1] = np.clip(p1[1], 0, ny-1)
        p1[2] = np.clip(p1[2], 0, nz-1)
        
        # 4. Rasterize Curve
        num_steps = 100
        t_values = np.linspace(0, 1, num_steps)
        
        # Section dimensions (Linear Interpolation)
        # Ensure minimum width/height is 4 as requested
        # Section dimensions (Linear Interpolation)
        # Increased dimensions to ensure initial volume > 30%
        w_base = random.uniform(8, 14)
        h_base = random.uniform(12, 24)
        w_tip = random.uniform(6, 10) 
        h_tip = random.uniform(6, 10)
        
        for t in t_values:
            # Current point
            p = quadratic_bezier(p0, p1, p2, t)
            cx, cy, cz = p
            
            # Current section size
            w_curr = w_base + (w_tip - w_base) * t
            h_curr = h_base + (h_tip - h_base) * t
            
            # Adjust cy so that the top of the section aligns with the target Y at the end
            # At t=1, we want cy + h_tip/2 approx target_y? 
            # Actually, the load is at target_y. We want the material to be UNDER the load?
            # Or the load is applied TO the material.
            # Let's assume the load is at (target_x, target_y, ...).
            # We want the material to exist at that location.
            # The current logic centers the box at (cx, cy, cz).
            # If p2 = (target_x, target_y, ...), then at t=1, center is at target_y.
            # So material extends from target_y - h/2 to target_y + h/2.
            # This covers the load point.
            
            # However, user said: "receber a carga em seus voxels superiores" (receive load on its top voxels).
            # This implies the load should be at the TOP surface.
            # So at t=1, we want (cy + h_curr/2) == target_y.
            # So cy should be target_y - h_curr/2.
            # But p2 is the target center.
            # Let's shift the curve vertically by offset.
            
            # Calculate offset to align top surface with curve path
            # We want the curve p(t) to represent the TOP surface? 
            # Or just shift the box down?
            # Let's shift the box down relative to the curve point.
            # Box Y range: [cy - h_curr, cy] (Top is cy)
            
            # Let's modify the box limits definition:
            y_max = int(cy) # Top is at curve
            y_min = int(cy - h_curr) # Bottom extends down
            
            # But wait, p0 is at (0, start_y, start_z). start_y is random.
            # If we shift the box, we shift the start too.
            # Let's keep center alignment but ensure p2 is at the top?
            # If p2 is (target_x, target_y, ...), and we want target_y to be the top.
            # Then the curve should end at target_y.
            # And the box should be below it.
            
            # So:
            y_max = int(cy + 1) # Ensure we cover the line
            y_min = int(cy - h_curr + 1)
            
            z_min = int(cz - w_curr / 2)
            z_max = int(cz + w_curr / 2)
            x_curr = int(cx)
            
            # Clip
            y_min = max(0, y_min)
            y_max = min(ny, y_max)
            z_min = max(0, z_min)
            z_max = min(nz, z_max)
            x_curr = max(0, min(nx-1, x_curr))
            
            # Fill voxels
            # We fill a small slice in X to ensure continuity?
            # Or just the point?
            # To ensure connectivity, we might need to fill x_prev to x_curr.
            # But with 100 steps for 64 voxels, we are likely fine.
            # Let's fill a small block in X too (thickness 1 or 2)
            x_min = max(0, x_curr)
            x_max = min(nx, x_curr + 2)
            
            voxel_grid[x_min:x_max, y_min:y_max, z_min:z_max] = 1.0
            
    # Ensure Wall Connection (X=0) is solid where curves start?
    # The spec says "connecting the wall". The curves start at X=0.
    # We might want to enforce a base plate or just trust the curves.
    # Let's trust the curves but ensure at least some voxels at X=0 are 1.
    mid_y, mid_z = ny//2, nz//2
    voxel_grid[0:2, mid_y-4:mid_y+4, mid_z-4:mid_z+4] = 1.0

    # Load Anchor: Force solid material at load application point
    # This prevents immediate disconnection at the load singularity
    lx, ly = load_config['x'], load_config['y']
    lz_s, lz_e = load_config['z_start'], load_config['z_end']
    
    # Clip coordinates
    lx = min(lx, nx-1)
    ly = min(ly, ny-1)
    lz_s = max(0, lz_s)
    lz_e = min(nz, lz_e)
    
    # Create a 3x3x(Z) anchor block
    x_s, x_e = max(0, lx-1), min(nx, lx+2)
    y_s, y_e = max(0, ly-1), min(ny, ly+2)
    
    voxel_grid[x_s:x_e, y_s:y_e, lz_s:lz_e] = 1.0
        
    return voxel_grid

def generate_seeded_cantilever(
    resolution: Tuple[int, int, int],
    load_config: Dict[str, Any]
) -> np.ndarray:
    """
    Generate a seeded initialization for Full Domain strategy.
    Creates a gray background with a solid bar connecting load to support.
    """
    nx, ny, nz = resolution
    
    # 1. Gray Background (0.35)
    grid = np.full(resolution, 0.35, dtype=np.float32)
    
    # 2. Define Seed Bar (Load -> Support)
    # Point A: Load
    p1 = np.array([load_config['x'], load_config['y'], (load_config['z_start'] + load_config['z_end']) / 2])
    
    # Point B: Support Center (X=0)
    p0 = np.array([0.0, ny / 2.0, nz / 2.0])
    
    # Rasterize Line
    num_steps = int(np.linalg.norm(p1 - p0)) * 2
    t_values = np.linspace(0, 1, num_steps)
    
    thickness = 2 # Radius of the bar
    
    for t in t_values:
        p = p0 + (p1 - p0) * t
        cx, cy, cz = int(p[0]), int(p[1]), int(p[2])
        
        # Draw sphere/box at p
        x_min, x_max = max(0, cx-thickness), min(nx, cx+thickness+1)
        y_min, y_max = max(0, cy-thickness), min(ny, cy+thickness+1)
        z_min, z_max = max(0, cz-thickness), min(nz, cz+thickness+1)
        
        grid[x_min:x_max, y_min:y_max, z_min:z_max] = 1.0
        
    return grid

# --- Phase 1: Slicing ---

def generate_phase1_slices(
    final_mask: np.ndarray, 
    target_value: float,
    num_steps: int = 50
) -> List[Dict[str, Any]]:
    """
    Slice the constructed structure into growth steps.
    """
    # 1. Distance Transform from Base (X=0)
    # We want distance from X=0 within the mask.
    # Invert mask: 0 is structure, 1 is void (for CDT)
    # Wait, cdt calculates distance to nearest zero.
    # So we want X=0 face to be zero, and everything else non-zero?
    # No, we want geodesic distance INSIDE the structure.
    # So we treat void as "walls" (infinity).
    
    # Create a grid where structure is 1, void is 0.
    structure = final_mask > 0.5
    
    # We can use a graph traversal (BFS) to compute geodesic distance from X=0 voxels.
    # Initialize distances to infinity
    dists = np.full(final_mask.shape, -1, dtype=np.int32)
    
    # Queue for BFS
    queue = []
    
    # Add all structure voxels at X=0 to queue
    starts = np.where(structure[0, :, :])
    for y, z in zip(starts[0], starts[1]):
        dists[0, y, z] = 0
        queue.append((0, y, z))
        
    # BFS
    head = 0
    while head < len(queue):
        x, y, z = queue[head]
        head += 1
        
        current_dist = dists[x, y, z]
        
        # Neighbors (6-connectivity)
        for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            nx, ny, nz = x+dx, y+dy, z+dz
            
            if 0 <= nx < final_mask.shape[0] and \
               0 <= ny < final_mask.shape[1] and \
               0 <= nz < final_mask.shape[2]:
                
                if structure[nx, ny, nz] and dists[nx, ny, nz] == -1:
                    dists[nx, ny, nz] = current_dist + 1
                    queue.append((nx, ny, nz))
                    
    # Max distance
    max_dist = np.max(dists)
    if max_dist <= 0:
        # Should not happen if connected
        return []
        
    # Generate 50 slices
    records = []
    
    # If max_dist < 50, we just output all steps.
    # If max_dist > 50, we sample.
    
    # We want cumulative percentages: 2%, 4%, ... 100%
    # Percentage of what? Distance? Or Volume?
    # Spec says: "Calcular distância geodésica... Fatiar em 50 passos cumulativos".
    # Usually implies distance thresholds.
    
    for i in range(1, num_steps + 1):
        percent = i / num_steps
        threshold = int(max_dist * percent)
        
        # Input State: Voxels with dist <= threshold
        input_mask = (dists <= threshold) & (dists != -1)
        input_grid = input_mask.astype(np.float32)
        
        # Target Policy (Add): The FULL final mask
        # Policy: Add where final_mask has material but input_grid doesn't
        # Remove where input_grid has material but final_mask doesn't (shouldn't happen in growth)
        
        target_add = np.where((final_mask > 0.5) & (input_grid < 0.5), 1.0, 0.0)
        target_remove = np.zeros_like(target_add) # Growth only adds
        
        # Mask: Ensure we don't propose adding where material already exists
        # This is already handled by the (input_grid < 0.5) condition above.
        
        records.append({
            "phase": Phase.GROWTH, # Map Phase 1 to GROWTH
            "step": i,
            "input_state": input_grid,
            "target_add": target_add,
            "target_remove": target_remove,
            "target_add": target_add,
            "target_remove": target_remove,
            "target_value": target_value,
            "current_vol": float(np.mean(input_grid)),
            "current_compliance": None # Undefined for Phase 1
        })
        
    return records

# --- Main Script ---

def generate_random_load_config(resolution):
    nx, ny, nz = resolution
    # x in [nx/2, nx-1], y in [0, ny-1], z in [0, nz-5]
    load_x = random.randint(nx//2, nx-1)
    load_y = random.randint(0, ny-1)
    load_z_start = random.randint(0, nz-5)
    load_z_end = load_z_start + 4
    
    return {
        'x': load_x,
        'y': load_y,
        'z_start': load_z_start,
        'z_end': load_z_end,
        'magnitude': -100.0 # Increased load as requested
    }

def parse_args():
    parser = argparse.ArgumentParser(description="AlphaBuilder v2.0 Data Harvest")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--db-path", type=str, default="data/training_data.db")
    parser.add_argument("--resolution", type=str, default="64x32x8")
    parser.add_argument("--seed-offset", type=int, default=0, help="Offset for random seed")
    parser.add_argument("--strategy", type=str, choices=['BEZIER', 'FULL_DOMAIN'], help="Force specific strategy")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup
    db_path = Path(args.db_path)
    initialize_database(db_path)
    
    # Resolution
    resolution = [int(x) for x in args.resolution.split("x")]
    nx, ny, nz = resolution
    
    # Logger
    logger = TrainingLogger(str(db_path.parent / "logs"), "harvest_v2.csv", 
                           ["episode", "duration", "compliance", "vol_frac", "phase1_samples", "phase2_samples"])
    
    # Physics Props
    props = PhysicalProperties()
    
    # Initialize Context
    print("Initializing FEM Context...")
    ctx = initialize_cantilever_context(resolution, props)
    
    print(f"Starting Harvest: {args.episodes} episodes")
    
    for i in range(args.episodes):
        start_time = time.time()
        episode_id = generate_episode_id()
        
        # Set seed
        seed = args.seed_offset + i
        np.random.seed(seed)
        random.seed(seed)
        
        episode_id = str(uuid.uuid4())
        
        # Strategy Selection: 20% Full Domain, 80% Bezier
        if args.strategy:
            strategy = args.strategy
        else:
            strategy = 'FULL_DOMAIN' if random.random() < 0.2 else 'BEZIER'
        
        # 1. Generate Initial Structure
        load_config = generate_random_load_config(resolution)
        
        if strategy == 'FULL_DOMAIN':
            # Hybrid Seeding: Gray background + Connection Bar
            v_constructed = generate_seeded_cantilever(resolution, load_config)
        else:
            v_constructed = generate_bezier_structure(resolution, load_config)
        
        # 2. Run SIMP (Phase 2 Ground Truth)
        # We run this FIRST to get S_final (Oracle Value)
        # Spec: "Calcular Score Final (S_final)... Este S_final será usado como Target Value para TODOS os registros"
        
        # Force Erosion: Target Volume = 40% (0.4) for FULL_DOMAIN to match MBB validation
        if strategy == 'FULL_DOMAIN':
            target_vol = 0.4
        else:
            initial_vol = np.mean(v_constructed)
            target_vol = initial_vol * 0.75
            # Ensure target is not too small (min 15%)
            target_vol = max(0.15, target_vol)
        
        print(f"Episode {i}: Strategy={strategy}, Target V={target_vol:.3f}")

        simp_config = SIMPConfig(
            vol_frac=target_vol,
            max_iter=60, 
            r_min=2.5,   # Recommended 2.5 in Spec v2.1
            adaptive_penal=True, 
            load_config=load_config,
            debug_log_path=f"debug_simp_log_ep{i}.csv" 
        )
        
        # Run SIMP starting from V_constructed
        simp_history = run_simp_optimization_3d(
            ctx, props, simp_config, resolution, 
            initial_density=v_constructed
        )
        
        if not simp_history:
            print(f"Episode {i}: SIMP failed. Skipping.")
            continue
            
        final_state = simp_history[-1]
        s_final_compliance = final_state['compliance']
        s_final_vol = np.mean(final_state['binary_map'])
        
        # Calculate Value Score (Tanh normalized)
        # Formula: tanh( (-log(C) - alpha*V - mu) / sigma )
        # We need constants. Let's use placeholders or raw compliance for now?
        # The spec gives the formula but not the constants.
        # Let's store raw compliance in metadata and a heuristic value in 'fitness_score'.
        # Heuristic: 1 / (C * V + epsilon) normalized?
        # Let's use a simple inversion for now: 10 / compliance (clamped)
        # Or just use the formula with estimated constants.
        # mu approx 2.0, sigma approx 1.0, alpha approx 0.0?
        # Let's just use 1/Compliance for now as the 'fitness_score' field in DB expects float.
        # The training loop can do the normalization.
        s_final_value = 1.0 / (s_final_compliance + 1e-9)
        
        # 4. Check Connectivity of Final State
        # Helper for Connectivity Check
        def check_connectivity(density_grid, threshold, load_cfg):
            # Binarize
            binary = density_grid > threshold
            
            # Label connected components
            labeled, n_components = scipy.ndimage.label(binary)
            
            if n_components == 0:
                return False, binary
                
            # Check if Support (X=0) and Load Region are in the same component
            # Support: Any voxel at X=0
            support_labels = np.unique(labeled[0, :, :])
            support_labels = support_labels[support_labels > 0] # Exclude background 0
            
            if len(support_labels) == 0:
                return False, binary
                
            # Load Region
            lx, ly, lz_s, lz_e = load_cfg['x'], load_cfg['y'], load_cfg['z_start'], load_cfg['z_end']
            # Clip
            lx = min(lx, nx-1)
            ly = min(ly, ny-1)
            lz_s = max(0, lz_s)
            lz_e = min(nz, lz_e)
            
            # Load region slice
            load_slice = labeled[lx-2:lx+2, ly-2:ly+2, lz_s:lz_e] # Check a small volume around load
            load_labels = np.unique(load_slice)
            load_labels = load_labels[load_labels > 0]
            
            common = np.intersect1d(support_labels, load_labels)
            return len(common) > 0, binary

        # Check Final State Connectivity
        is_connected, _ = check_connectivity(final_state['density_map'], 0.5, load_config)
        
        if not is_connected:
            # Try lower threshold?
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
                # continue # Proceed anyway
            else:
                 print(f"Episode {i}: Final structure connected only at low threshold (0.1). Saving with warning.")
        
        else:
            print(f"Episode {i}: Final structure CONNECTED. Proceeding to save.")

        # 5. Generate Phase 1 Records (Slicing)
        # Skip Phase 1 for FULL_DOMAIN as it starts from solid block
        if strategy == 'FULL_DOMAIN':
            phase1_records = []
        else:
            phase1_records = generate_phase1_slices(v_constructed, s_final_value)
        
        # 6. Generate Phase 2 Records (SIMP History)
        phase2_records = []
        
        # Filter: Skip first 20 steps if adaptive_penal is used (Policy Pollution Prevention)
        # The first 20 steps use p=1 (convex), which is not representative of the final topology optimization problem.
        start_step = 20 if simp_config.adaptive_penal else 0
        
        for t in range(start_step, len(simp_history) - 1):
            current_frame = simp_history[t]
            next_frame = simp_history[t+1]
            
            curr_dens = current_frame['density_map']
            next_dens = next_frame['density_map']
            
            # Adaptive Thresholding (Smart Saving)
            # Find the highest threshold that maintains connectivity
            best_binary = None
            found_threshold = 0.5
            
            # Try thresholds from 0.5 down to 0.1
            for thresh in [0.5, 0.4, 0.3, 0.2, 0.1]:
                is_conn, binary_mask = check_connectivity(curr_dens, thresh, load_config)
                if is_conn:
                    best_binary = binary_mask.astype(np.float32)
                    found_threshold = thresh
                    break
            
            if best_binary is None:
                # Fallback
                best_binary = (curr_dens > 0.1).astype(np.float32)
            
            # Use the Connected Binary Mask as Input State
            input_state = best_binary
            
            diff = curr_dens - next_dens
            target_remove = np.maximum(0, diff)
            target_add = np.maximum(0, -diff)
            
            phase2_records.append({
                "phase": Phase.REFINEMENT,
                "step": len(phase1_records) + t,
                "input_state": input_state, # Saved as Connected Binary
                "target_add": target_add,
                "target_remove": target_remove,
                "target_value": 1.0 / (float(current_frame['compliance']) + 1e-9),
                "threshold": found_threshold,
                "current_compliance": float(current_frame['compliance']),
                "current_vol": float(np.mean(curr_dens))
            })
            
        # 7. Save to DB
        # Common function to save
        def save_batch(records, phase_enum):
            for rec in records:
                # Build Input Tensor (5 Channels)
                # 0: Density
                # 1: Mask (Support) -> X=0
                # 2,3,4: Forces
                
                # Support Mask
                mask = np.zeros(resolution, dtype=np.float32)
                mask[0, :, :] = 1.0
                
                # Force Channels
                fx = np.zeros(resolution, dtype=np.float32)
                fy = np.zeros(resolution, dtype=np.float32)
                fz = np.zeros(resolution, dtype=np.float32)
                
                # Apply Load to Force Channels
                # Load is distributed line
                # We can just paint the load region
                # Load Config: x, y, z_start, z_end
                lx, ly, lz_s, lz_e = load_config['x'], load_config['y'], load_config['z_start'], load_config['z_end']
                # Clip
                lx = min(lx, nx-1)
                ly = min(ly, ny-1)
                lz_s = max(0, lz_s)
                lz_e = min(nz, lz_e)
                
                # Force is -Y direction (magnitude)
                fy[lx, ly, lz_s:lz_e] = load_config.get('magnitude', -1.0)
                
                # Input State
                # rec['input_state'] is the density channel
                density = rec['input_state']
                
                # Stack
                input_tensor = np.stack([density, mask, fx, fy, fz], axis=0)
                
                # Target Policy
                # Stack Add/Remove
                policy_tensor = np.stack([rec['target_add'], rec['target_remove']], axis=0)
                
                db_record = TrainingRecord(
                    episode_id=episode_id,
                    step=rec['step'],
                    phase=phase_enum,
                    state_blob=serialize_state(input_tensor),
                    policy_blob=serialize_state(policy_tensor),
                    fitness_score=rec['target_value'],
                    valid_fem=True,
                    metadata={
                        "compliance": rec.get('current_compliance', s_final_compliance),
                        "vol_frac": rec.get('current_vol', s_final_vol),
                        "load_config": load_config,
                        "phase": "GROWTH" if phase_enum == Phase.GROWTH else "REFINEMENT"
                    }
                )
                save_record(db_path, db_record)

        save_batch(phase1_records, Phase.GROWTH) # Map to GROWTH (Phase 1)
        save_batch(phase2_records, Phase.REFINEMENT) # Map to REFINEMENT (Phase 2)
        
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
