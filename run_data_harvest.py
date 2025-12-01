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

# --- FEniTop Integration (Based on fenitop/scripts/beam_3d.py) ---
from alphabuilder.src.logic.fenitop.topopt import topopt
from dolfinx.mesh import create_box, CellType
from dolfinx.fem import functionspace
from mpi4py import MPI
import basix.ufl

def run_fenitop_optimization(
    resolution: Tuple[int, int, int],
    props: PhysicalProperties,
    simp_config: SIMPConfig,
    initial_density: np.ndarray = None,
    strategy: str = 'BEZIER'
) -> List[Dict[str, Any]]:
    """
    Run Topology Optimization using FEniTop Core.
    
    Based directly on fenitop/scripts/beam_3d.py reference implementation.
    Cantilever beam: fixed at X=0, surface traction at X=Lx (free end).
    
    Physical model (from beam_3d.py):
    - E = 100, nu = 0.25
    - Surface traction at free end
    - Full clamped support at X=0
    
    Args:
        resolution: (nx, ny, nz) voxel grid dimensions
        props: Physical properties (unused - we use beam_3d.py values)
        simp_config: SIMP configuration with load_config, vol_frac, etc.
        initial_density: Optional initial density grid (nx, ny, nz)
        strategy: 'BEZIER' or 'FULL_DOMAIN' - affects optimization params
    
    Returns:
        List of step dictionaries with density_map, compliance, vol_frac, beta
    """
    comm = MPI.COMM_WORLD
    nx, ny, nz = resolution
    
    # Physical dimensions (1:1 mapping with voxels)
    Lx, Ly, Lz = float(nx), float(ny), float(nz)
    
    # --- Mesh Creation (following beam_3d.py pattern) ---
    mesh = create_box(comm, [[0, 0, 0], [Lx, Ly, Lz]], [nx, ny, nz], CellType.hexahedron)
    
    if comm.rank == 0:
        mesh_serial = create_box(MPI.COMM_SELF, [[0, 0, 0], [Lx, Ly, Lz]], [nx, ny, nz], CellType.hexahedron)
    else:
        mesh_serial = None
    
    # --- Grid Mapper for DOF -> Voxel conversion ---
    grid_mapper = None
    if comm.rank == 0 and mesh_serial is not None:
        element = basix.ufl.element("Lagrange", mesh_serial.topology.cell_name(), 1)
        V_serial = functionspace(mesh_serial, element)
        coords = V_serial.tabulate_dof_coordinates()
        
        x_idx = np.rint(coords[:, 0]).astype(int)
        y_idx = np.rint(coords[:, 1]).astype(int)
        z_idx = np.rint(coords[:, 2]).astype(int)
        
        x_idx = np.clip(x_idx, 0, nx)
        y_idx = np.clip(y_idx, 0, ny)
        z_idx = np.clip(z_idx, 0, nz)
        
        grid_mapper = (x_idx, y_idx, z_idx)
    
    # --- Load Configuration (beam_3d.py style: surface traction at free end) ---
    # Load is applied at X=Lx face (free end), centered in Y and Z
    load_cfg = simp_config.load_config
    
    # Load region: 2x2 patch at X=Lx face
    load_y_center = float(load_cfg['y'])
    load_z_center = (float(load_cfg['z_start']) + float(load_cfg['z_end'])) / 2.0
    load_half_width = 1.0  # Half-width of 2x2 load region
    
    # --- FEM Parameters (beam_3d.py style) ---
    fem = {
        "mesh": mesh,
        "mesh_serial": mesh_serial,
        # Physical properties from beam_3d.py
        "young's modulus": 100,
        "poisson's ratio": 0.25,
        # Displacement BC: Fix ALL DOFs at X=0 (full clamped wall)
        "disp_bc": lambda x: np.isclose(x[0], 0),
        # Traction BC: Surface traction at X=Lx (free end), force in -Y direction
        # Using >= and <= with 0.5 tolerance to ensure nodes are captured
        "traction_bcs": [[
            (0, -2.0, 0),  # Force vector (Fx, Fy, Fz) - same magnitude as beam_3d.py
            lambda x, _Lx=Lx, _yc=load_y_center, _zc=load_z_center, _hw=load_half_width: (
                np.isclose(x[0], _Lx) &  # At free end face
                (x[1] >= _yc - _hw - 0.5) & (x[1] <= _yc + _hw + 0.5) &
                (x[2] >= _zc - _hw - 0.5) & (x[2] <= _zc + _hw + 0.5)
            )
        ]],
        "body_force": (0, 0, 0),
        "quadrature_degree": 2,
        "petsc_options": {
            "ksp_type": "cg",
            "pc_type": "gamg",
        },
    }
    
    # --- Optimization Parameters (beam_3d.py style, tuned for speed) ---
    if strategy == 'FULL_DOMAIN':
        max_iter = min(simp_config.max_iter, 120)
        beta_interval = 30
        move_limit = 0.02
    else:  # BEZIER - starts closer to solution
        max_iter = min(simp_config.max_iter, 100)
        beta_interval = 25
        move_limit = 0.02
    
    # Filter radius: ~1.5-2 elements for good filtering (beam_3d.py uses 0.6 for finer mesh)
    filter_r = max(simp_config.r_min, 1.2)
    
    opt = {
        "max_iter": max_iter,
        "opt_tol": 1e-4,
        "vol_frac": simp_config.vol_frac,
        "solid_zone": lambda x: np.full(x.shape[1], False),
        "void_zone": lambda x: np.full(x.shape[1], False),
        "penalty": 3.0,
        "epsilon": 1e-6,
        "filter_radius": filter_r,
        "beta_interval": beta_interval,
        "beta_max": 128,
        "use_oc": True,  # OC optimizer (as in beam_3d.py)
        "move": move_limit,
        "opt_compliance": True,
    }
    
    # --- History Recording ---
    history = []
    
    def record_step(data):
        if comm.rank == 0:
            rho_flat = data['density']
            
            # Reconstruct 3D voxel grid from flat DOF array
            if grid_mapper is None:
                rho_3d = np.full((nx, ny, nz), simp_config.vol_frac, dtype=np.float32)
            elif len(rho_flat) != len(grid_mapper[0]):
                print(f"  WARNING: Size mismatch in record_step")
                rho_3d = np.full((nx, ny, nz), simp_config.vol_frac, dtype=np.float32)
            else:
                rho_nodal = np.zeros((nx+1, ny+1, nz+1), dtype=np.float32)
                rho_nodal[grid_mapper[0], grid_mapper[1], grid_mapper[2]] = rho_flat
                rho_3d = rho_nodal[:-1, :-1, :-1]

            history.append({
                'step': data['iter'],
                'density_map': rho_3d,
                'compliance': float(data['compliance']),
                'vol_frac': float(data['vol_frac']),
                'beta': data.get('beta', 1)  # Track beta for filtering later
            })
            
            if data['iter'] % 10 == 0:
                print(f"  Step {data['iter']:3d}: C={data['compliance']:.4f} V={data['vol_frac']:.4f} β={data.get('beta', 1)}")

    # --- Run Optimization ---
    if comm.rank == 0:
        print(f"  FEniTop: iter={max_iter}, V={simp_config.vol_frac:.2f}, r={filter_r:.1f}, strategy={strategy}")
    topopt(fem, opt, initial_density=initial_density, callback=record_step)
    
    return history

def quadratic_bezier(p0, p1, p2, t):
    """Calculate point on quadratic Bézier curve at t."""
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

def generate_bezier_structure(
    resolution: Tuple[int, int, int],
    load_config: Dict[str, Any]
) -> np.ndarray:
    """
    Generate a procedural structure using Bézier curves with rectangular sections.
    Uses triangular distribution biased toward smaller section sizes.
    """
    nx, ny, nz = resolution
    # Safe Background: 0.15 to allow sensitivity flow (Vanishing Gradient Fix)
    voxel_grid = np.full(resolution, 0.15, dtype=np.float32)
    
    # 1. Determine Number of Curves
    num_curves = random.randint(2, 4)
    
    # Load Point (Target)
    target_x = load_config['x']
    target_y = load_config['y']
    target_z_center = (load_config['z_start'] + load_config['z_end']) / 2.0
    
    p2 = np.array([target_x, target_y, target_z_center])
    
    # Track final section positions for anchor placement
    final_positions = []
    final_sections = []
    
    for _ in range(num_curves):
        # 2. Start Point (Wall X=0)
        start_y = random.uniform(0, ny-1)
        start_z = random.uniform(0, nz-1)
        p0 = np.array([0.0, start_y, start_z])
        
        # 3. Control Point (Intermediate) - reduced noise for less curvature
        midpoint = (p0 + p2) / 2.0
        noise = np.random.normal(0, 3.0, size=3)  # Reduced from 5.0
        p1 = midpoint + noise
        p1[0] = np.clip(p1[0], 0, nx-1)
        p1[1] = np.clip(p1[1], 0, ny-1)
        p1[2] = np.clip(p1[2], 0, nz-1)
        
        # 4. Rasterize Curve
        num_steps = 100
        t_values = np.linspace(0, 1, num_steps)
        
        # Section dimensions with BIAS toward smaller values using triangular distribution
        # triangular(low, mode, high) - mode near low creates bias toward smaller values
        # Constraints: 2 <= w_f <= w_i <= 8 and 2 <= h_f <= h_i <= 32
        w_tip = random.triangular(2, 2.5, 6)    # Bias toward 2-3
        w_base = random.triangular(w_tip, w_tip + 0.5, 8)  # Slightly larger than tip
        h_tip = random.triangular(2, 3, 12)     # Bias toward 2-4, max reduced to 12
        h_base = random.triangular(h_tip, h_tip + 2, min(h_tip + 10, 20))  # Controlled growth
        
        for i, t in enumerate(t_values):
            # Current point on curve
            p = quadratic_bezier(p0, p1, p2, t)
            cx, cy, cz = p
            
            # Current section size (interpolated)
            w_curr = w_base + (w_tip - w_base) * t
            h_curr = h_base + (h_tip - h_base) * t
            
            # Section centered on curve point
            y_min = int(cy - h_curr / 2)
            y_max = int(cy + h_curr / 2)
            z_min = int(cz - w_curr / 2)
            z_max = int(cz + w_curr / 2)
            x_curr = int(cx)
            
            # Clip to bounds
            y_min = max(0, y_min)
            y_max = min(ny, y_max)
            z_min = max(0, z_min)
            z_max = min(nz, z_max)
            x_curr = max(0, min(nx-1, x_curr))
            
            # Fill voxels (thin slice in X for connectivity)
            x_min = max(0, x_curr)
            x_max = min(nx, x_curr + 2)
            
            voxel_grid[x_min:x_max, y_min:y_max, z_min:z_max] = 1.0
            
            # Track final position (last step of each curve)
            if i == len(t_values) - 1:
                final_positions.append((cx, cy, cz))
                final_sections.append((w_tip, h_tip))
            
    # Ensure Wall Connection (X=0) - small base plate
    mid_y, mid_z = ny//2, nz//2
    voxel_grid[0:2, mid_y-3:mid_y+3, mid_z-2:mid_z+2] = 1.0

    # Load Anchor: Positioned at the CENTER of final Bezier sections
    # Use average of final curve endpoints to determine anchor position
    if final_positions:
        avg_y = np.mean([p[1] for p in final_positions])
        avg_z = np.mean([p[2] for p in final_positions])
        max_w = max([s[0] for s in final_sections])
        max_h = max([s[1] for s in final_sections])
    else:
        avg_y = load_config['y']
        avg_z = (load_config['z_start'] + load_config['z_end']) / 2.0
        max_w, max_h = 4, 6
    
    # Anchor covers the average final section position
    # Extends back 3 voxels for connectivity, sized to cover final section
    anchor_depth = 3
    anchor_hw = max(2, int(max_w / 2) + 1)  # Half-width in Z
    anchor_hh = max(2, int(max_h / 2) + 1)  # Half-height in Y
    
    x_s = max(0, nx - anchor_depth)
    x_e = nx
    y_s = max(0, int(avg_y - anchor_hh))
    y_e = min(ny, int(avg_y + anchor_hh))
    z_s = max(0, int(avg_z - anchor_hw))
    z_e = min(nz, int(avg_z + anchor_hw))
    
    voxel_grid[x_s:x_e, y_s:y_e, z_s:z_e] = 1.0
        
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
    ly = load_config['y']
    lz_center = (load_config['z_start'] + load_config['z_end']) / 2
    
    # Point A: Load (at free end)
    p1 = np.array([nx - 1, ly, lz_center])
    
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
    
    # 3. Load Anchor at free end (ensures connection with 2x2 load region)
    anchor_depth = 4
    anchor_margin = 3
    
    x_s = max(0, nx - anchor_depth)
    x_e = nx
    y_s = max(0, int(ly - anchor_margin))
    y_e = min(ny, int(ly + anchor_margin + 1))
    z_s = max(0, int(lz_center - anchor_margin))
    z_e = min(nz, int(lz_center + anchor_margin + 1))
    
    grid[x_s:x_e, y_s:y_e, z_s:z_e] = 1.0
        
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
    """
    Generate load configuration for cantilever beam.
    
    Load is applied as surface traction at X=nx (free end face),
    similar to beam_3d.py from FEniTop. The load region is a 
    centered patch on the free end face.
    """
    nx, ny, nz = resolution
    
    # Load always at X=nx-1 (free end face)
    load_x = nx - 1
    
    # Center the load region in Y and Z with some randomness
    # Similar to beam_3d.py where load is centered on the end face
    load_y = ny // 2 + random.randint(-ny//4, ny//4)
    load_y = max(2, min(ny-3, load_y))  # Keep within bounds
    
    load_z_center = nz // 2 + random.randint(-nz//4, nz//4)
    load_z_center = max(2, min(nz-3, load_z_center))
    
    # 2x2 load region
    half_width = 1
    load_z_start = load_z_center - half_width
    load_z_end = load_z_center + half_width
    
    return {
        'x': load_x,
        'y': load_y,
        'z_start': load_z_start,
        'z_end': load_z_end,
        # Magnitude now handled by FEniTop config (-2.0 as in beam_3d.py)
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
    
    for i in range(args.episodes):
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
        
        if is_main_rank:
            print(f"Episode {i}: Strategy={strategy}, Target V={target_vol:.3f}")

        simp_config = SIMPConfig(
            vol_frac=target_vol,
            max_iter=120,  # More iterations for lower volume targets
            r_min=1.5,     # Filter radius ~1.5 elements
            adaptive_penal=True, 
            load_config=load_config,
            debug_log_path=f"debug_simp_log_ep{i}.csv" 
        )
        
        # Run FEniTop Optimization (ALL ranks must participate in MPI collective operations)
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
        
        # Only main rank has the full history (callback only runs on rank 0)
        if not is_main_rank:
            continue  # Other ranks skip the rest of episode processing
        
        if not simp_history:
            print(f"Episode {i}: SIMP failed (no history). Skipping.")
            continue
            
        final_state = simp_history[-1]
        s_final_compliance = final_state['compliance']
        # s_final_vol = np.mean(final_state['binary_map']) # FEniTop returns continuous density
        s_final_vol = final_state['vol_frac'] # Use the volume fraction from FEniTop
        
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
        
        # Filter: Skip steps where beta=1 (pure density manipulation phase)
        # FEniTop uses Heaviside continuation: beta starts at 1 (soft projection)
        # and increases over iterations. We only save steps after beta > 1
        # when real topology decisions are being made.
        
        # Find first step where beta > 1
        start_step = 0
        for t, frame in enumerate(simp_history):
            if frame.get('beta', 1) > 1:
                start_step = t
                break
        
        print(f"  Phase 2: Skipping first {start_step} steps (beta=1 phase), saving from step {start_step}")
        
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
                "step": len(phase1_records) + (t - start_step),  # Renumber from 0
                "input_state": input_state,
                "target_add": target_add,
                "target_remove": target_remove,
                "target_value": 1.0 / (float(current_frame['compliance']) + 1e-9),
                "threshold": found_threshold,
                "beta": current_frame.get('beta', 1),
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
                
                # Support Mask (full clamped wall at X=0)
                mask = np.zeros(resolution, dtype=np.float32)
                mask[0, :, :] = 1.0
                
                # Force Channels
                fx = np.zeros(resolution, dtype=np.float32)
                fy = np.zeros(resolution, dtype=np.float32)
                fz = np.zeros(resolution, dtype=np.float32)
                
                # Apply Load to Force Channels
                # Load is surface traction at X=Lx (free end), 2x2 patch
                ly = load_config['y']
                lz_center = (load_config['z_start'] + load_config['z_end']) / 2.0
                load_half_width = 1.0  # 2x2 load region
                
                # Mark load region at X=nx-1 (free end face)
                y_min = max(0, int(ly - load_half_width))
                y_max = min(ny, int(ly + load_half_width) + 1)
                z_min = max(0, int(lz_center - load_half_width))
                z_max = min(nz, int(lz_center + load_half_width) + 1)
                
                # Force is -Y direction (traction = -2.0, normalized to -1 for NN input)
                fy[nx-1, y_min:y_max, z_min:z_max] = -1.0
                
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
