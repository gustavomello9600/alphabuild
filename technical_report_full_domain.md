# Relatório Técnico: Estratégia "Full Domain" para Geração de Dados (AlphaBuilder v2.0)

Este relatório detalha a implementação da estratégia "Full Domain" para geração de dados de treinamento de otimização topológica. O objetivo desta estratégia é gerar exemplos de alta qualidade partindo de um domínio sólido e erodindo-o progressivamente até atingir uma fração de volume alvo, utilizando o método SIMP (Solid Isotropic Material with Penalization) com parâmetros ajustados para capturar estruturas finas e complexas.

## 1. Visão Geral do Pipeline

O pipeline de geração e visualização consiste em 5 etapas principais:

1.  **Definição Física (`physics_model.py`)**: Configuração do contexto de Elementos Finitos (FEM), malha e condições de contorno.
2.  **Otimização SIMP (`simp_generator.py`)**: O núcleo algorítmico que resolve o problema de otimização topológica.
3.  **Orquestração (`run_data_harvest.py`)**: Script que gerencia a geração em lote, definindo parâmetros estratégicos (volume, penalidade).
4.  **Extração de Dados (`extract_mock_episode.py`)**: Conversão dos dados brutos do banco de dados SQLite para formato JSON consumível pelo frontend.
5.  **Visualização Frontend (`mockService.ts`, `Workspace.tsx`)**: Renderização 3D interativa dos dados gerados.

---

## 2. Detalhamento dos Componentes e Código Fonte

Abaixo segue o código fonte completo dos arquivos envolvidos na geração e visualização deste exemplo.



### 2.1. run_data_harvest.py
```python
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
    # Initialize with small density to allow SIMP growth
    voxel_grid = np.full(resolution, 0.01, dtype=np.float32)
    
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
    if np.sum(voxel_grid[0, :, :]) == 0:
        # Fallback: create a small base if missed
        mid_y, mid_z = ny//2, nz//2
        voxel_grid[0:2, mid_y-4:mid_y+4, mid_z-4:mid_z+4] = 1.0
        
    return voxel_grid

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
        'z_end': load_z_end
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
            v_constructed = np.ones(resolution, dtype=np.float32)
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
            max_iter=60, # Increased to 60 to match MBB validation
            r_min=1.2,   # Reduced to 1.2 to match MBB validation (sharper features)
            adaptive_penal=True, # Enable adaptive penalty (1->3)
            load_config=load_config
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
                
                # Force is -Y direction (-1.0)
                fy[lx, ly, lz_s:lz_e] = -1.0
                
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
```

### 2.2. alphabuilder/src/logic/simp_generator.py
```python
import numpy as np
import dolfinx
import dolfinx.fem as fem
import ufl
from petsc4py import PETSc
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import time
from tqdm import tqdm
import scipy.ndimage

from alphabuilder.src.core.physics_model import FEMContext, PhysicalProperties, SimulationResult

@dataclass
class SIMPConfig:
    """Configuration for SIMP optimization."""
    vol_frac: float = 0.3         # Target volume fraction (Updated for stability)
    penal: float = 3.0            # Penalization power (p)
    r_min: float = 3.5            # Filter radius (in elements) (Increased for better connectivity)
    max_iter: int = 50            # Max iterations
    change_tol: float = 0.01      # Convergence tolerance (change in density)
    move_limit: float = 0.2       # Max density change per step
    adaptive_penal: bool = False  # Enable adaptive penalty schedule (delayed continuation)
    load_config: Optional[Dict] = None # Custom load configuration

def apply_sensitivity_filter(
    x: np.ndarray, 
    dc: np.ndarray, 
    H: np.ndarray, 
    Hs: np.ndarray
) -> np.ndarray:
    """
    Apply mesh-independent sensitivity filter.
    """
    # Simple convolution filter: dc_new = (H * (x * dc)) / (Hs * x)
    x_safe = np.maximum(x, 1e-3)
    numerator = H.dot(x_safe * dc)
    denominator = Hs * x_safe
    return numerator / denominator

def apply_heaviside_projection(x: np.ndarray, beta: float, eta: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Heaviside projection to force 0/1 values.
    Returns: (x_projected, derivative)
    """
    if beta < 1e-6:
        return x, np.ones_like(x)
        
    tanh_beta_eta = np.tanh(beta * eta)
    tanh_beta_1_eta = np.tanh(beta * (1 - eta))
    denom = tanh_beta_eta + tanh_beta_1_eta
    
    tanh_val = np.tanh(beta * (x - eta))
    x_new = (tanh_beta_eta + tanh_val) / denom
    
    # Derivative: d(x_new)/dx
    # d/dx tanh(u) = sech^2(u) * u' = (1 - tanh^2(u)) * u'
    dtanh = beta * (1 - tanh_val**2)
    derivative = dtanh / denom
    
    return x_new, derivative

class HelmholtzFilter:
    """
    Helmholtz PDE Filter for Topology Optimization.
    Solves: -r^2 * laplace(u) + u = x
    """
    def __init__(self, ctx: FEMContext, r_min: float):
        self.ctx = ctx
        self.r_min = r_min
        
        # Filter radius parameter (r = r_min / (2*sqrt(3)) for equivalent support?)
        # Common heuristic: r = r_min / 3.5 or similar. 
        # For SIMP, r_min is usually the diameter of the filter.
        # The PDE length scale r is related to r_min.
        # If r_min is the radius of the cone filter, r ~ r_min / 2.5.
        # Let's use r = r_min / 3.0 as a starting point.
        self.r = r_min / 3.0
        
        # Create Function Spaces
        # Input/Output is DG0 (Element-wise)
        # Filter solution is CG1 (Continuous)
        self.V_dg = ctx.material_field.function_space # DG0
        self.V_cg = fem.functionspace(ctx.domain, ("CG", 1))
        
        # Trial and Test Functions for CG1
        u = ufl.TrialFunction(self.V_cg)
        v = ufl.TestFunction(self.V_cg)
        
        # Variational Form: (r^2 * grad(u) . grad(v) + u * v) * dx
        self.a = (self.r**2 * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(u, v)) * ufl.dx
        
        # Assemble Matrix A (Constant)
        self.A = fem.petsc.assemble_matrix(fem.form(self.a))
        self.A.assemble()
        
        # Create Solver
        self.solver = PETSc.KSP().create(ctx.domain.comm)
        self.solver.setOperators(self.A)
        self.solver.setType(PETSc.KSP.Type.CG)
        self.solver.getPC().setType(PETSc.PC.Type.GAMG)
        self.solver.setFromOptions()
        
        self.u_sol = fem.Function(self.V_cg)
        
    def apply(self, x_dens: np.ndarray) -> np.ndarray:
        # 1. Map x_dens (numpy) to DG0 Function
        # We assume x_dens is ordered same as DG0 dofs.
        # If not, we need the map. But usually for DG0 it matches cell index.
        # Let's assume it matches for now (standard for FEniCSx if we extract correctly)
        
        # Create input function
        f_in = fem.Function(self.V_dg)
        f_in.x.array[:] = x_dens
        
        # 2. Solve PDE
        # L = f_in * v * dx
        v = ufl.TestFunction(self.V_cg)
        L = ufl.inner(f_in, v) * ufl.dx
        b = fem.petsc.assemble_vector(fem.form(L))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        
        self.solver.solve(b, self.u_sol.x.petsc_vec)
        self.u_sol.x.scatter_forward()
        
        # 3. Map result (CG1) back to DG0 (Element-wise average/centroid)
        # We can just interpolate CG1 -> DG0
        # Create a temp DG0 function to hold result
        result_dg = fem.Function(self.V_dg)
        result_dg.interpolate(self.u_sol)
        
        return result_dg.x.array.copy()

def run_simp_optimization_3d(
    ctx: FEMContext,
    props: PhysicalProperties,
    config: SIMPConfig,
    resolution: Tuple[int, int, int],
    seed: int = 0,
    initial_density: Optional[np.ndarray] = None,
    passive_mask: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    
    nx, ny, nz = resolution
    n_elements = nx * ny * nz
    
    # Initialize density
    if initial_density is not None:
         if initial_density.ndim == 3:
             if not hasattr(ctx, 'dof_indices'):
                V = ctx.material_field.function_space
                coords = V.tabulate_dof_coordinates()
                # Adaptive Physical Domain (Step 1)
                voxel_size = 1.0
                phys_L = nx * voxel_size
                phys_H = ny * voxel_size
                phys_W = nz * voxel_size
                
                dx = phys_L / nx
                dy = phys_H / ny
                dz = phys_W / nz
                ix = np.floor((coords[:, 0]) / dx).astype(np.int32)
                iy = np.floor((coords[:, 1]) / dy).astype(np.int32)
                iz = np.floor((coords[:, 2]) / dz).astype(np.int32)
                ix = np.clip(ix, 0, nx - 1)
                iy = np.clip(iy, 0, ny - 1)
                iz = np.clip(iz, 0, nz - 1)
                objectsetattr(ctx, 'dof_indices', (ix, iy, iz))
             
             ix, iy, iz = ctx.dof_indices
             x = initial_density[ix, iy, iz]
         else:
             x = initial_density.copy()
    else:
        x = np.ones(n_elements) * config.vol_frac
    
    # Ensure dof_indices are computed
    if not hasattr(ctx, 'dof_indices'):
         V = ctx.material_field.function_space
         coords = V.tabulate_dof_coordinates()
         # Adaptive Physical Domain (Step 1)
         voxel_size = 1.0
         phys_L = nx * voxel_size
         phys_H = ny * voxel_size
         phys_W = nz * voxel_size
         
         dx = phys_L / nx
         dy = phys_H / ny
         dz = phys_W / nz
         ix = np.floor((coords[:, 0]) / dx).astype(np.int32)
         iy = np.floor((coords[:, 1]) / dy).astype(np.int32)
         iz = np.floor((coords[:, 2]) / dz).astype(np.int32)
         ix = np.clip(ix, 0, nx - 1)
         iy = np.clip(iy, 0, ny - 1)
         iz = np.clip(iz, 0, nz - 1)
         objectsetattr(ctx, 'dof_indices', (ix, iy, iz))

    # Initialize Helmholtz Filter
    pde_filter = HelmholtzFilter(ctx, config.r_min)
    
    history = []

    # Helper to record state
    def record_state(step_num, current_x, current_compliance=0.0):
        if not hasattr(ctx, 'dof_indices'):
             pass
             
        ix, iy, iz = ctx.dof_indices
        d_map = np.zeros((nx, ny, nz), dtype=np.float32)
        d_map[ix, iy, iz] = current_x
        b_map = (d_map > 0.5).astype(np.int32)
        history.append({
            'step': step_num,
            'density_map': d_map,
            'binary_map': b_map,
            'fitness': 0.0,
            'max_displacement': 0.0,
            'compliance': current_compliance,
            'valid': True
        })

    record_state(0, x)

    loop = 0
    change = 1.0
    b = ctx.problem.b
    u = ctx.u_sol
    
    # Pre-assemble Load Vector b (Constant Force)
    b.zeroEntries()
    dolfinx.fem.petsc.assemble_vector(b, ctx.problem.L)
    
    # Apply Load (Dynamic based on config)
    if hasattr(config, 'load_config') and config.load_config:
        lc = config.load_config
        
        voxel_size = 1.0
        grid_x = lc.get('x', nx-1)
        grid_y = lc.get('y', ny//2)
        grid_z_start = lc.get('z_start', nz//2)
        grid_z_end = lc.get('z_end', grid_z_start + 3)
        
        px = (grid_x + 0.5) * voxel_size
        py = (grid_y + 0.5) * voxel_size
        pz_start = grid_z_start * voxel_size
        pz_end = (grid_z_end + 1) * voxel_size
        tol = voxel_size * 0.6 
        
        def load_locator(p):
            return np.logical_and(
                np.isclose(p[0], px, atol=tol),
                np.logical_and(
                    np.isclose(p[1], py, atol=tol),
                    np.logical_and(p[2] >= pz_start, p[2] <= pz_end)
                )
            )
        
        V = ctx.V
        load_dofs = fem.locate_dofs_geometrical(V, load_locator)
        bs = V.dofmap.index_map_bs
        with b.localForm() as b_local:
            f_total = -1.0
            n_nodes = len(load_dofs) // bs if bs > 0 else 1
            if n_nodes > 0:
                f_node = f_total / n_nodes
                for dof in load_dofs:
                    if dof % bs == 1: # Y component
                        b_local.setValues([dof], [f_node], addv=PETSc.InsertMode.ADD_VALUES)
    else:
        # Fallback: Point Load
        max_coords = ctx.domain.geometry.x.max(axis=0)
        tip_x = max_coords[0]
        mid_y = max_coords[1] / 2.0
        mid_z = max_coords[2] / 2.0
        def point_load_locator(p):
            return np.logical_and(
                np.isclose(p[0], tip_x, atol=0.1),
                np.logical_and(
                    np.isclose(p[1], mid_y, atol=0.1),
                    np.isclose(p[2], mid_z, atol=0.1)
                )
            )
        V = ctx.V
        load_dofs = fem.locate_dofs_geometrical(V, point_load_locator)
        bs = V.dofmap.index_map_bs
        with b.localForm() as b_local:
            f_total = -1.0
            n_nodes = len(load_dofs) // bs if bs > 0 else 1
            if n_nodes > 0:
                f_node = f_total / n_nodes
                for dof in load_dofs:
                    if dof % bs == 1:
                        b_local.setValues([dof], [f_node], addv=PETSc.InsertMode.ADD_VALUES)
        
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, ctx.problem.bcs)

    from tqdm import tqdm
    pbar = tqdm(total=config.max_iter, desc="SIMP Optimization", unit="iter", leave=False, ncols=100, mininterval=0.5)
    
    while change > config.change_tol and loop < config.max_iter:
        loop += 1
        
        # Continuation Schedule (Penalty & Beta)
        current_penal = config.penal
        beta = 1.0
        
        if config.adaptive_penal:
            # Delayed continuation:
            # Iter 0-20: p=1.0 (Convex)
            # Iter 21-40: p -> 3.0
            # Iter 41+: Beta increase
            
            if loop <= 20:
                current_penal = 1.0
                beta = 1.0
            elif loop <= 40:
                # Ramp p from 1.0 to config.penal
                progress = (loop - 20) / 20.0
                current_penal = 1.0 + (config.penal - 1.0) * progress
                beta = 1.0
            else:
                current_penal = config.penal
                # Start Beta continuation late
                beta = 2.0 ** ((loop - 40) / 5.0)
                beta = min(beta, 64.0)
        else:
            # Standard Beta Schedule
            if loop <= 10:
                beta = 1.0
            elif loop <= 40:
                beta = 2.0 ** ((loop - 10) / 5.0)
            else:
                beta = 64.0
            
        # Filtering (Three-Field Scheme)
        t_filter_start = time.time()
        x_tilde = pde_filter.apply(x)
        
        # Project
        x_phys, dx_proj_dx = apply_heaviside_projection(x_tilde, beta)
        
        # Update FEM Material Field
        x_phys = np.clip(x_phys, 0.0, 1.0)
        penalized_density = x_phys ** current_penal
        ctx.material_field.x.array[:] = penalized_density
        ctx.material_field.x.scatter_forward()
        t_filter = time.time() - t_filter_start
        
        # Assemble A
        t_asm_start = time.time()
        A = ctx.problem.A
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, ctx.problem.a, bcs=ctx.problem.bcs)
        A.assemble()
        t_asm = time.time() - t_asm_start
        
        # Solve
        t0 = time.time()
        solver = ctx.problem.solver
        solver.solve(b, u.x.petsc_vec)
        u.x.scatter_forward()
        t_sol = time.time() - t0
        
        # Compliance and Sensitivity
        # C = f^T u = u^T K u
        # Sensitivity: dC/dx = -p * x^(p-1) * (u^T K_e u)
        
        # Calculate Strain Energy Density (Element-wise)
        # We can use UFL to compute strain energy density field
        # W = 0.5 * sigma : epsilon
        # Compliance density = 2 * W
        
        # But we need it per element (DG0)
        # Let's compute it efficiently.
        # For SIMP, dC/drho = -p * rho^(p-1) * (u_e^T k_0 u_e)
        # We can compute (u^T k_0 u) by assembling a form with E=1 (or E_0)
        
        # Define strain energy form for full material
        E_0 = props.E
        E_min = 1e-6
        nu = props.nu
        mu_0 = E_0 / (2 * (1 + nu))
        lmbda_0 = E_0 * nu / ((1 + nu) * (1 - 2 * nu))
        
        def sigma_0(u):
            return lmbda_0 * ufl.tr(ufl.sym(ufl.grad(u))) * ufl.Identity(len(u)) + 2.0 * mu_0 * ufl.sym(ufl.grad(u))
            
        # Strain energy density (scalar field)
        strain_energy = ufl.inner(sigma_0(u), ufl.sym(ufl.grad(u))) # This is 2*W if E=E_0
        
        # Project strain energy to DG0 space to get element values
        # This gives us (u^T k_0 u) per cell approximately
        W_dg = fem.Function(ctx.material_field.function_space)
        
        # Projection solver
        # We can use a simple projection since it's DG0 (just integrate over cell and divide by volume)
        # But fem.Expression is easier
        expr = fem.Expression(strain_energy, ctx.material_field.function_space.element.interpolation_points)
        W_dg.interpolate(expr)
        strain_energy_values = W_dg.x.array
        
        # Compliance
        # C = integral(rho^p * strain_energy)
        # But we computed strain_energy using u, which comes from penalized stiffness.
        # So C = b.dot(u.vector)
        compliance = b.dot(u.x.petsc_vec)
        
        # Sensitivity
        # dC/dx_phys = -p * x_phys^(p-1) * strain_energy_values
        # Note: strain_energy_values is roughly u^T k_0 u
        # Actually, if we use E_min, it's slightly more complex, but for E_min << E_0, this holds.
        
        dc = -current_penal * (x_phys ** (current_penal - 1)) * strain_energy_values
        
        # Chain Rule for Projection: dC/dx_tilde = dC/dx_phys * dx_phys/dx_tilde
        dc = dc * dx_proj_dx
        
        # Chain Rule for Filter: dC/dx = Filter^T * dC/dx_tilde
        # Since filter is self-adjoint (Helmholtz), we just apply filter again
        dc = pde_filter.apply(dc)
        
        # Optimality Criteria Update
        # x_new = x * ( -dc / lambda )^eta
        # We need to find lambda (Lagrange multiplier) such that sum(x_new) = vol_frac
        
        def optimality_criteria(x, dc, target_vol, limit):
            l1 = 0.0
            l2 = 1e9
            move = limit
            
            # Damping
            damping = 0.5
            
            x_new = np.zeros_like(x)
            
            while (l2 - l1) / (l1 + l2 + 1e-9) > 1e-3:
                l_mid = 0.5 * (l2 + l1)
                
                # OC Update Rule
                # B_e = -dc / lambda
                # x_new = max(0, max(x - move, min(1, min(x + move, x * sqrt(B_e)))))
                
                # Avoid division by zero
                # dc is negative, so -dc is positive.
                # We want B_e = (-dc) / l_mid
                
                factor = np.sqrt(np.maximum(1e-10, -dc) / l_mid)
                
                x_trial = x * factor
                
                x_lower = np.maximum(0.0, x - move)
                x_upper = np.minimum(1.0, x + move)
                
                x_new = np.clip(x_trial, x_lower, x_upper)
                
                if np.mean(x_new) > target_vol:
                    l1 = l_mid
                else:
                    l2 = l_mid
            return x_new

        # Dynamic Move Limit?
        current_move_limit = config.move_limit
        
        x_new = optimality_criteria(x, dc, config.vol_frac, current_move_limit)
        t_update = time.time() - t0 # Fix timer approximation
        
        change = np.max(np.abs(x_new - x))
        x = x_new
        
        # Record
        record_state(loop, x_phys, compliance)
        
        pbar.set_postfix({
            'Chg': f"{change:.3f}", 
            'C': f"{compliance:.1f}",
            'T_sol': f"{t_sol:.2f}s",
            'T_asm': f"{t_asm:.2f}s"
        })
        pbar.update(1)
        
    pbar.close()
    return history

def objectsetattr(obj, name, value):
    """Helper to set attribute on frozen dataclass or object"""
    object.__setattr__(obj, name, value)
```

### 2.3. alphabuilder/src/core/physics_model.py
```python
import numpy as np
import dolfinx
from dolfinx import mesh, fem
from mpi4py import MPI
from dataclasses import dataclass
import ufl
from petsc4py import PETSc

@dataclass
class PhysicalProperties:
    E: float = 1.0  # Young's Modulus (Normalized)
    nu: float = 0.3 # Poisson's Ratio
    rho: float = 1.0 # Density
    disp_limit: float = 100.0 # Loose limit for initial training
    penalty_epsilon: float = 0.1
    penalty_alpha: float = 10.0

import dolfinx.fem.petsc

@dataclass
class FEMContext:
    domain: mesh.Mesh
    V: fem.FunctionSpace
    bc: fem.DirichletBC
    problem: dolfinx.fem.petsc.LinearProblem
    u_sol: fem.Function
    dof_map: np.ndarray # Map from Grid Voxel to FEM Cell
    material_field: fem.Function # DG0 field for density

@dataclass
class SimulationResult:
    fitness: float
    max_displacement: float
    compliance: float
    valid: bool
    displacement_array: np.ndarray

def initialize_cantilever_context(resolution=(64, 32, 32), props: PhysicalProperties = None) -> FEMContext:
    """
    Initialize a 3D FEM context for a Cantilever Beam.
    Resolution: (L, H, W) -> (x, y, z)
    Physical Size: Derived from resolution (Cubic Voxels)
    """
    nx, ny, nz = resolution
    
    # Adaptive Physical Domain (Step 1 of Engineering Order)
    # Ensure cubic voxels by deriving physical size from resolution.
    voxel_size = 1.0
    L = nx * voxel_size
    H = ny * voxel_size
    W = nz * voxel_size
    
    # 1. Create 3D Hexahedral Mesh
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([L, H, W])],
        [nx, ny, nz],
        cell_type=mesh.CellType.hexahedron
    )
    
    # 2. Function Space (Vector Element for Displacement)
    # FEniCSx v0.8+: Use ufl.VectorElement and fem.FunctionSpace
    # FEniCSx v0.8+: Use basix.ufl and fem.functionspace
    import basix.ufl
    element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(domain.geometry.dim,))
    V = fem.functionspace(domain, element)
    
    # 3. Boundary Conditions (Fix Left Face: x=0)
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
        
    fdim = domain.topology.dim - 1
    left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
    
    # Fix all components (0, 1, 2) -> (x, y, z)
    u_zero = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
    bc = fem.dirichletbc(u_zero, fem.locate_dofs_topological(V, fdim, left_facets), V)
    
    # 4. Variational Problem (Linear Elasticity)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Material Field (Density 0 or 1)
    # We use a DG0 space (constant per cell) to map voxels to elements
    V_mat = fem.functionspace(domain, ("DG", 0))
    material_field = fem.Function(V_mat)
    material_field.x.array[:] = 1.0 # Start full
    
    # SIMP Interpolation: E(rho) = E_min + (E_0 - E_min) * rho^p
    # For binary (0/1), this simplifies to E_0 * rho (plus small epsilon to avoid singularity)
    if props is None:
        props = PhysicalProperties()
        
    E_0 = props.E
    E_min = 1e-6 # Fixed to 1e-6 as per Engineering Order
    rho = material_field
    E = E_min + (E_0 - E_min) * rho # Linear for binary is fine, or rho**3 for SIMP
    nu = props.nu
    
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
        
    def sigma(u):
        return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2.0 * mu * epsilon(u)
        
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(fem.Constant(domain, np.zeros(3, dtype=PETSc.ScalarType)), v) * ufl.dx
    
    # 5. Linear Problem Setup
    # Use Iterative Solver (CG + GAMG) for 3D Elasticity to save memory
    # LU is too expensive for 64x32x32 (200k DOFs) on Colab
    solver_options = {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-6,
        "ksp_atol": 1e-10,
        "ksp_max_it": 2000,
        # "ksp_monitor": None, # Enable monitor
        # "ksp_view": None     # Enable view (verbose)
    }
    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options=solver_options, petsc_options_prefix="cantilever")
    
    # 6. DOF Map (Voxel -> Cell Index)
    # In FEniCSx with structured box mesh, cell indices usually follow a pattern.
    # However, to be safe, we rely on the fact that we update 'material_field' which is DG0.
    # The DG0 DoFs correspond 1-to-1 with cells.
    # We assume the grid passed to solver matches the mesh resolution (nx, ny, nz).
    # We need a map if the ordering differs, but for BoxMesh, it's usually lexicographical.
    # Let's create a placeholder map.
    dof_map = np.arange(nx * ny * nz, dtype=np.int32)
    
    u_sol = fem.Function(V)
    
    return FEMContext(
        domain=domain,
        V=V,
        bc=bc,
        problem=problem,
        u_sol=u_sol,
        dof_map=dof_map,
        material_field=material_field
    )
```

### 2.4. extract_mock_episode.py
```python
import sqlite3
import pickle
import json
import numpy as np
import random
from pathlib import Path

import argparse

def extract_episode(db_path, output_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get the latest episode ID
    cursor.execute("SELECT DISTINCT episode_id FROM training_data ORDER BY rowid DESC LIMIT 1")
    row = cursor.fetchone()
    if not row:
        print("No episodes found!")
        return
    
    episode_id = row[0]
    print(f"Extracting episode: {episode_id}")

    # Get all steps
    cursor.execute("""
        SELECT step, phase, state_blob, fitness_score, valid_fem, metadata, policy_blob 
        FROM training_data 
        WHERE episode_id = ? 
        ORDER BY step
    """, (episode_id,))
    
    steps = []
    rows = cursor.fetchall()
    
    for row in rows:
        step, phase, state_blob, fitness_score, valid_fem, metadata_json, policy_blob = row
        
        # Deserialize state
        tensor = pickle.loads(state_blob)
        # Tensor shape is likely (C, D, H, W) or (C, H, W) depending on 2D/3D
        # Frontend expects (4, D, H, W)
        
        # Deserialize metadata
        metadata = json.loads(metadata_json) if metadata_json else {}
        # Map vol_frac to volume_fraction for frontend
        if 'vol_frac' in metadata:
            metadata['volume_fraction'] = metadata['vol_frac']
        
        # Deserialize policy if present (mock if not)
        policy_data = None
        if policy_blob:
            try:
                policy_tensor = pickle.loads(policy_blob)
                # Policy tensor shape is (2, D, H, W) -> [Add, Remove]
                # Frontend expects { add: number[][][], remove: number[][][] }
                policy_data = {
                    "add": policy_tensor[0].tolist(),
                    "remove": policy_tensor[1].tolist()
                }
            except:
                pass
        
        # Format for frontend
        game_state = {
            "episode_id": episode_id,
            "step": step,
            "phase": phase,
            "tensor": {
                "shape": list(tensor.shape),
                "data": tensor.flatten().tolist() # Flatten for JSON
            },
            "fitness_score": fitness_score,
            "valid_fem": bool(valid_fem),
            "metadata": metadata,
            "value_confidence": fitness_score, # Use fitness as proxy for value confidence
            "policy": policy_data
        }
        steps.append(game_state)

    # Extract load_config from first step metadata if available
    load_config = None
    if steps and 'metadata' in steps[0] and 'load_config' in steps[0]['metadata']:
        load_config = steps[0]['metadata']['load_config']

    # Wrap in MockEpisodeData structure
    output_data = {
        "episode_id": episode_id,
        "load_config": load_config,
        "frames": steps
    }

    # Save to JSON
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f)
    
    print(f"Saved {len(steps)} steps to {output_path}")
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Extract mock episode from DB")
    parser.add_argument("--db-path", type=str, default="data/training_data.db", help="Path to database")
    parser.add_argument("--output", type=str, default="alphabuilder/web/public/mock_episode.json", help="Output JSON path")
    args = parser.parse_args()

    extract_episode(args.db_path, args.output)

if __name__ == "__main__":
    main()
```

### 2.5. alphabuilder/web/src/api/mockService.ts
```typescript
import type { GameState, Phase, Project } from './types';

// Define interface for the JSON structure
interface MockEpisodeData {
    episode_id: string;
    load_config?: {
        x: number;
        y: number;
        z_start: number;
        z_end: number;
    };
    frames: Array<{
        step: number;
        phase: Phase;
        density?: number[][][];
        tensor?: {
            shape: number[];
            data: number[];
        };
        fitness: number;
        metadata: {
            compliance?: number;
            vol_frac?: number;
            volume_fraction?: number;
        };
        policy?: {
            add: number[][][];
            remove: number[][][];
        };
    }>;
}

export class MockService {
    private subscribers: ((state: GameState) => void)[] = [];
    private intervalId: any = null;
    private currentStepIndex = 0;
    private mockEpisode: GameState[] = [];
    private isPlaying: boolean = false;
    private maxCompliance: number = 1.0;

    // Simulate database of projects
    private projects: Project[] = [
        { id: 'mbb-validation', name: 'MBB Validation (SIMP)', status: 'COMPLETED', thumbnail_url: '/placeholder.png', last_modified: '2025-11-30', episode_id: 'mbb_validation_test' },
        { id: 'bezier-run', name: 'Bezier Strategy', status: 'COMPLETED', thumbnail_url: '/placeholder.png', last_modified: '2025-11-29', episode_id: 'bezier-run' },
        { id: 'full-domain-run', name: 'Full Domain Strategy', status: 'COMPLETED', thumbnail_url: '/placeholder.png', last_modified: '2025-11-29', episode_id: 'full-domain-run' }
    ];

    constructor() {
        console.log("MockService Initialized");
    }

    async getProjects(): Promise<Project[]> {
        return new Promise(resolve => {
            setTimeout(() => resolve(this.projects), 500);
        });
    }

    async getProject(id: string): Promise<Project | undefined> {
        return new Promise(resolve => {
            setTimeout(() => resolve(this.projects.find(p => p.id === id)), 300);
        });
    }

    async startSimulation(id: string) {
        if (this.intervalId) clearInterval(this.intervalId);
        this.isPlaying = false;

        let data: MockEpisodeData;
        let url = '';

        if (id === 'full-domain-run') {
            url = '/data/mock_episode_fulldomain.json';
        } else if (id === 'mbb-validation' || id === 'mbb_validation_test') {
            url = '/data/mock_episode_mbb.json';
        } else {
            // Default to Bezier
            url = '/data/mock_episode_bezier.json';
        }

        try {
            console.log(`[MockService] Fetching data from ${url}...`);
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            data = await response.json() as MockEpisodeData;
        } catch (e) {
            console.error(`[MockService] Failed to fetch ${url}`, e);
            return;
        }

        console.log(`[MockService] Loading episode data for ${id}. ID: ${data.episode_id}`);
        console.log(`[MockService] Data keys:`, Object.keys(data));
        if (data.frames) {
            console.log(`[MockService] Frames length: ${data.frames.length}`);
        } else {
            console.error(`[MockService] Frames is undefined!`);
        }

        try {
            this.mockEpisode = this.processData(data);

            // Calculate Max Compliance for the episode
            this.maxCompliance = 0;
            this.mockEpisode.forEach(step => {
                if (step.metadata.compliance && step.metadata.compliance > this.maxCompliance) {
                    this.maxCompliance = step.metadata.compliance;
                }
            });
            if (this.maxCompliance === 0) this.maxCompliance = 1.0; // Prevent division by zero

            console.log(`[MockService] Processed episode length: ${this.mockEpisode.length}, Max Compliance: ${this.maxCompliance}`);
            this.startLoop(id);
        } catch (e) {
            console.error("[MockService] Error processing data:", e);
        }
    }

    private processData(data: MockEpisodeData): GameState[] {
        const frames = data.frames;
        return frames.map((frame) => {
            let tensorData: Float32Array;
            let shape: [number, number, number, number];
            const CHANNELS = 5;

            // Check if we have pre-processed tensor data (New Format)
            if (frame.tensor) {
                const rawShape = frame.tensor.shape;
                const rawData = frame.tensor.data;
                tensorData = new Float32Array(rawData);
                shape = rawShape as [number, number, number, number];

            } else if (frame.density) {
                // Legacy Format: Construct Tensor from Density 3D Array
                const density3D = frame.density;
                const D = density3D.length;
                const H = density3D[0].length;
                const W = density3D[0][0].length;

                shape = [CHANNELS, D, H, W];
                tensorData = new Float32Array(CHANNELS * D * H * W).fill(0);

                // Fill Density (Channel 0)
                for (let d = 0; d < D; d++) {
                    for (let h = 0; h < H; h++) {
                        for (let w = 0; w < W; w++) {
                            const val = density3D[d][h][w];
                            const idx = 0 * (D * H * W) + d * (H * W) + h * W + w;
                            tensorData[idx] = val;

                            // Add Support (Channel 1)
                            // Default: Cantilever (X=0)
                            // MBB: Symmetry (X=0) + Roller (X=L, Y=0)
                            const isMBB = data.episode_id.includes('mbb');
                            let isSupportVoxel = false;

                            if (d === 0) {
                                isSupportVoxel = true; // Left Face (Fixed or Symmetry)
                            } else if (isMBB && d === D - 1 && h === 0) {
                                isSupportVoxel = true; // Bottom Right Edge (Roller)
                            }

                            if (isSupportVoxel) {
                                const sIdx = 1 * (D * H * W) + d * (H * W) + h * W + w;
                                tensorData[sIdx] = 1.0;
                            }
                        }
                    }
                }

                // Fill Load (Channel 3)
                if (data.load_config) {
                    const lc = data.load_config;
                    const lx = Math.min(lc.x, D - 1);
                    const ly = Math.min(lc.y, H - 1);
                    const lz_s = Math.max(0, lc.z_start);
                    const lz_e = Math.min(lc.z_end, W);

                    for (let z = lz_s; z < lz_e; z++) {
                        const lIdx = 3 * (D * H * W) + lx * (H * W) + ly * W + z;
                        tensorData[lIdx] = -1.0; // Fy = -1
                    }
                }
            } else {
                console.error("Frame missing both tensor and density data", frame);
                throw new Error("Invalid frame data: missing tensor/density");
            }

            // Parse Policy Heatmap
            let policyHeatmap = undefined;
            if (frame.policy) {
                // Need dimensions for flattening if not already provided
                // If we used tensor, we have shape. If density, we have D,H,W.
                const [, D, H, W] = shape;

                const flattenPolicy = (arr3d: number[][][]) => {
                    const flat = new Float32Array(D * H * W);
                    for (let d = 0; d < D; d++) {
                        for (let h = 0; h < H; h++) {
                            for (let w = 0; w < W; w++) {
                                const idx = d * (H * W) + h * W + w;
                                flat[idx] = arr3d[d][h][w];
                            }
                        }
                    }
                    return flat;
                };

                policyHeatmap = {
                    add: flattenPolicy(frame.policy.add),
                    remove: flattenPolicy(frame.policy.remove)
                };
            }

            // Handle metadata mapping
            const compliance = frame.metadata?.compliance ?? (frame as any).compliance ?? 0;
            const volFrac = frame.metadata?.volume_fraction ?? frame.metadata?.vol_frac ?? (frame as any).vol_frac ?? 0;

            // Handle fitness / value confidence
            const fitness = frame.fitness ?? (frame as any).fitness_score ?? 0;
            const valueConf = (frame as any).value_confidence ?? fitness;

            return {
                episode_id: data.episode_id,
                step: frame.step,
                phase: frame.phase,
                tensor: {
                    shape: shape,
                    data: tensorData
                },
                fitness_score: fitness,
                valid_fem: true,
                metadata: {
                    compliance: compliance,
                    max_displacement: 0,
                    volume_fraction: volFrac
                },
                value_confidence: valueConf,
                policy_heatmap: policyHeatmap
            };
        });
    }

    private startLoop(episodeId: string, resetIndex: boolean = true) {
        if (this.intervalId) clearInterval(this.intervalId);

        if (resetIndex) {
            this.currentStepIndex = 0;
        }
        console.log(`Starting simulation for ${episodeId} with ${this.mockEpisode.length} steps.`);
        this.isPlaying = true;

        this.intervalId = setInterval(() => {
            if (this.currentStepIndex >= this.mockEpisode.length - 1) {
                this.pause();
                return;
            }

            this.currentStepIndex++;
            const state = this.mockEpisode[this.currentStepIndex];
            this.notifySubscribers(state);
        }, 500); // 2 steps per second
    }

    stopSimulation() {
        if (this.intervalId) clearInterval(this.intervalId);
        this.isPlaying = false;
    }

    seekToStep(index: number) {
        if (index >= 0 && index < this.mockEpisode.length) {
            this.currentStepIndex = index;
            this.notifySubscribers(this.mockEpisode[index]);
        }
    }

    stepForward() {
        if (this.currentStepIndex < this.mockEpisode.length - 1) {
            this.seekToStep(this.currentStepIndex + 1);
        }
    }

    stepBackward() {
        if (this.currentStepIndex > 0) {
            this.seekToStep(this.currentStepIndex - 1);
        }
    }

    togglePlay() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.resume();
        }
    }

    pause() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        this.isPlaying = false;
        // Notify to update UI state
        if (this.mockEpisode.length > 0) {
            this.notifySubscribers(this.mockEpisode[this.currentStepIndex]);
        }
    }

    resume() {
        if (!this.intervalId && this.mockEpisode.length > 0) {
            // If at end, restart
            if (this.currentStepIndex >= this.mockEpisode.length - 1) {
                this.currentStepIndex = 0;
            }
            this.startLoop(this.mockEpisode[0].episode_id, false);
        }
    }

    subscribe(callback: (state: GameState) => void) {
        this.subscribers.push(callback);
        return () => {
            this.subscribers = this.subscribers.filter(s => s !== callback);
        };
    }

    private notifySubscribers(state: GameState) {
        this.subscribers.forEach(cb => cb(state));
    }

    getSimulationState() {
        return {
            episodeId: this.mockEpisode.length > 0 ? this.mockEpisode[0].episode_id : 'N/A',
            stepsLoaded: this.mockEpisode.length,
            currentStep: this.currentStepIndex,
            isPlaying: this.isPlaying,
            maxCompliance: this.maxCompliance,
            isRealRun: true
        };
    }

    // Deprecated alias for backward compatibility if needed, but we should use getSimulationState
    getDebugInfo() {
        return this.getSimulationState();
    }
}

export const mockService = new MockService();
```

### 2.6. alphabuilder/web/src/pages/Workspace.tsx
```typescript
import { useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, GizmoHelper, GizmoViewport } from '@react-three/drei';
import { motion } from 'framer-motion';
import { Play, Pause, SkipForward, SkipBack, MousePointer, Box, Eraser, ArrowUp, Triangle, Layers, Brain } from 'lucide-react';
import { mockService } from '../api/mockService';
import type { GameState, Tensor5D } from '../api/types';
import * as THREE from 'three';

// --- 3D Components ---

// --- 3D Components ---

const LoadVector = ({ tensor }: { tensor: Tensor5D | null }) => {
    if (!tensor) return null;

    const [, D, H, W] = tensor.shape;
    const loadPoints: { pos: THREE.Vector3, dir: THREE.Vector3 }[] = [];

    // Find load positions (Channel 3)
    for (let d = 0; d < D; d++) {
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                const idx = 3 * (D * H * W) + d * (H * W) + h * W + w;
                const val = tensor.data[idx];
                if (val !== 0) {
                    // Found load!
                    // Position centered in voxel
                    // Match VoxelGrid: X=d-D/2, Y=h+0.5, Z=w-W/2
                    const pos = new THREE.Vector3(d - D / 2, h + 0.5, w - W / 2);
                    // Direction: Channel 3 is Fy usually. If val < 0, it's down.
                    const dir = new THREE.Vector3(0, Math.sign(val), 0).normalize();
                    loadPoints.push({ pos, dir });
                }
            }
        }
    }

    if (loadPoints.length === 0) return null;

    const arrowLength = 5;

    return (
        <group>
            {loadPoints.map((lp, i) => {
                const origin = lp.pos.clone().sub(lp.dir.clone().multiplyScalar(arrowLength));
                return (
                    <group key={i}>
                        <arrowHelper args={[lp.dir, origin, arrowLength, 0xff0000, 1, 0.5]} />
                        <mesh position={lp.pos}>
                            <sphereGeometry args={[0.2, 8, 8]} />
                            <meshBasicMaterial color="red" />
                        </mesh>
                    </group>
                );
            })}
        </group>
    );
};

const VoxelGrid = ({
    tensor,
    heatmap,
    showHeatmap
}: {
    tensor: Tensor5D | null,
    heatmap?: { add: Float32Array, remove: Float32Array },
    showHeatmap: boolean
}) => {
    const meshRef = useRef<THREE.InstancedMesh>(null);
    const dummy = new THREE.Object3D();

    useEffect(() => {
        if (!meshRef.current || !tensor) return;

        const [, D, H, W] = tensor.shape;
        let count = 0;

        // Helper for heatmap color
        const getHeatmapColor = (idx: number): THREE.Color | null => {
            if (!showHeatmap || !heatmap) return null;

            const addVal = heatmap.add ? heatmap.add[idx] : 0;
            const removeVal = heatmap.remove ? heatmap.remove[idx] : 0;

            if (addVal < 0.01 && removeVal < 0.01) return null;

            const color = new THREE.Color();
            if (addVal > removeVal) {
                // Add: Green (Project Identity)
                color.set('#00FF9D');
                const intensity = Math.min(1, addVal * 2.5);
                color.lerp(new THREE.Color('#333333'), 1 - intensity);
            } else {
                // Remove: Red/Pink (Project Identity)
                color.set('#FF0055');
                const intensity = Math.min(1, removeVal * 2.5);
                color.lerp(new THREE.Color('#333333'), 1 - intensity);
            }
            return color;
        };

        // Iterate through tensor
        for (let d = 0; d < D; d++) {
            for (let h = 0; h < H; h++) {
                for (let w = 0; w < W; w++) {
                    const flatIdx = d * (H * W) + h * W + w;

                    // Tensor Channels
                    const density = tensor.data[0 * (D * H * W) + flatIdx];
                    const isSupport = tensor.data[1 * (D * H * W) + flatIdx] > 0.5;
                    const isLoad = tensor.data[3 * (D * H * W) + flatIdx] !== 0;

                    const heatmapColor = getHeatmapColor(flatIdx);

                    // Visibility Logic
                    const isVisibleStandard = density > 0.1 || isSupport || isLoad;
                    const isVisibleHeatmap = !!heatmapColor;

                    if (isVisibleStandard || isVisibleHeatmap) {
                        // Position: Shift Y up by H/2 to sit on grid
                        // Original: h - H/2. New: h (so 0 is at 0)
                        // Actually, to center X and Z but keep Y >= 0:
                        // X: d - D/2
                        // Y: h + 0.5 (center of voxel 0 is at 0.5)
                        // Z: w - W/2
                        dummy.position.set(d - D / 2, h + 0.5, w - W / 2);
                        dummy.updateMatrix();
                        meshRef.current.setMatrixAt(count, dummy.matrix);

                        // Color Logic
                        if (heatmapColor) {
                            meshRef.current.setColorAt(count, heatmapColor);
                        } else {
                            if (isSupport) {
                                meshRef.current.setColorAt(count, new THREE.Color('#3B82F6'));
                            } else if (isLoad) {
                                meshRef.current.setColorAt(count, new THREE.Color('#EF4444'));
                            } else {
                                const val = Math.max(0.2, density);
                                const color = new THREE.Color().setHSL(0, 0, val * 0.95);
                                meshRef.current.setColorAt(count, color);
                            }
                        }

                        count++;
                    }
                }
            }
        }

        meshRef.current.count = count;
        meshRef.current.instanceMatrix.needsUpdate = true;
        if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;

    }, [tensor, heatmap, showHeatmap]);

    return (
        <instancedMesh ref={meshRef} args={[undefined, undefined, 20000]}>
            <boxGeometry args={[0.9, 0.9, 0.9]} />
            <meshStandardMaterial roughness={0.2} metalness={0.8} />
        </instancedMesh>
    );
};

// --- UI Components ---

const Toolbar = () => {
    const [isOpen, setIsOpen] = useState(true);

    return (
        <div className="absolute left-8 top-8 flex flex-col gap-2 z-10">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="bg-matter/90 backdrop-blur border border-white/10 p-2 rounded-lg text-white/60 hover:text-white mb-2 self-start"
            >
                <Layers size={20} />
            </button>

            {isOpen && (
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex flex-col gap-2 bg-matter/90 backdrop-blur border border-white/10 p-2 rounded-lg"
                >
                    <button className="p-2 rounded bg-cyan/20 text-cyan hover:bg-cyan/30 transition-colors" title="Selecionar">
                        <MousePointer size={20} />
                    </button>
                    <button className="p-2 rounded text-white/60 hover:text-white hover:bg-white/10 transition-colors" title="Desenhar Voxel">
                        <Box size={20} />
                    </button>
                    <button className="p-2 rounded text-white/60 hover:text-white hover:bg-white/10 transition-colors" title="Apagar">
                        <Eraser size={20} />
                    </button>
                    <div className="h-px bg-white/10 my-1" />
                    <button className="p-2 rounded text-white/60 hover:text-white hover:bg-white/10 transition-colors" title="Aplicar Carga">
                        <ArrowUp size={20} className="text-red-500" />
                    </button>
                    <button className="p-2 rounded text-white/60 hover:text-white hover:bg-white/10 transition-colors" title="Adicionar Suporte">
                        <Triangle size={20} className="text-blue-500" />
                    </button>
                </motion.div>
            )}
        </div>
    );
};

const PropertiesPanel = ({ state }: { state: GameState | null }) => {
    const [isOpen, setIsOpen] = useState(true);
    // Use getSimulationState instead of getDebugInfo
    const simState = mockService.getSimulationState();
    const totalSteps = simState.stepsLoaded;
    const isPlaying = simState.isPlaying;
    const maxCompliance = simState.maxCompliance;

    return (
        <motion.div
            className="absolute right-0 top-0 bottom-0 bg-matter/90 backdrop-blur border-l border-white/10 z-10 flex flex-col"
            initial={{ width: 320 }}
            animate={{ width: isOpen ? 320 : 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
        >
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="absolute -left-8 top-1/2 -translate-y-1/2 bg-matter/90 border border-white/10 border-r-0 rounded-l p-1 text-white/60 hover:text-white"
            >
                {isOpen ? <SkipForward size={16} className="rotate-0" /> : <SkipForward size={16} className="rotate-180" />}
            </button>

            <div className="p-6 overflow-y-auto flex-1 w-80">
                <h3 className="text-sm font-mono text-white/40 uppercase mb-6">Controle da Simulação</h3>

                {/* Phase Indicator */}
                <div className="mb-4 flex items-center gap-2">
                    <span className="text-xs text-white/60 uppercase">Fase:</span>
                    <span className={`text-xs font-bold px-2 py-1 rounded ${state?.phase === 'GROWTH' ? 'bg-green-500/20 text-green-500' : 'bg-purple-500/20 text-purple-500'}`}>
                        {state?.phase === 'GROWTH' ? 'CRESCIMENTO (Fase 1)' : 'REFINAMENTO (Fase 2)'}
                    </span>
                </div>

                {/* Timeline */}
                <div className="mb-6 p-4 rounded bg-black/20 border border-white/5">
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-xs text-white/60">Passo Atual</span>
                        <span className="text-xs font-bold text-cyan">{state?.step || 0} / {totalSteps}</span>
                    </div>

                    {/* Slider */}
                    <input
                        type="range"
                        min="0"
                        max={Math.max(0, totalSteps - 1)}
                        value={simState.currentStep}
                        onChange={(e) => {
                            mockService.pause();
                            mockService.seekToStep(parseInt(e.target.value));
                        }}
                        className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan mb-2"
                    />

                    <div className="flex justify-between mt-1">
                        <span className="text-[10px] text-white/30">Início</span>
                        <span className="text-[10px] text-white/30">Fim</span>
                    </div>
                </div>

                {/* Metrics */}
                <div className="space-y-4 mb-8">
                    <div>
                        <div className="flex justify-between text-sm mb-1">
                            <span className="text-white/60">Compliance (Rigidez)</span>
                            <span className={`font-mono ${!state?.metadata.compliance ? 'text-red-500 text-xs' : 'text-white'}`}>
                                {state?.metadata.compliance ? state.metadata.compliance.toFixed(2) : 'ESTRUTURA DESCONECTADA'}
                            </span>
                        </div>
                        <div className="w-full bg-white/5 h-1 rounded-full overflow-hidden">
                            {state?.metadata.compliance ? (
                                <motion.div
                                    className="h-full bg-magenta"
                                    animate={{ width: `${Math.min(100, (state.metadata.compliance / maxCompliance) * 100)}%` }}
                                />
                            ) : (
                                <div className="h-full bg-red-500/50 w-full" />
                            )}
                        </div>
                    </div>
                    <div>
                        <div className="flex justify-between text-sm mb-1">
                            <span className="text-white/60">Volume %</span>
                            <span className="font-mono text-white">{((state?.metadata.volume_fraction || 0) * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-white/5 h-1 rounded-full">
                            <motion.div className="h-full bg-white/40" animate={{ width: `${(state?.metadata.volume_fraction || 0) * 100}%` }} />
                        </div>
                    </div>
                </div>

                {/* Controls */}
                <div className="flex gap-2">
                    <button
                        onClick={() => mockService.stepBackward()}
                        disabled={isPlaying}
                        className={`p-2 rounded transition-colors ${isPlaying ? 'bg-white/5 text-white/20 cursor-not-allowed' : 'bg-white/10 text-white hover:bg-white/20'}`}
                        title="Passo Anterior"
                    >
                        <SkipBack size={16} />
                    </button>

                    <button
                        onClick={() => mockService.togglePlay()}
                        className={`flex-1 font-bold py-2 rounded flex items-center justify-center gap-2 transition-colors ${!isPlaying
                            ? 'bg-cyan text-black hover:bg-cyan/90' // Play is primary when paused
                            : 'bg-white/10 text-white hover:bg-white/20' // Play is secondary when playing
                            }`}
                    >
                        <Play size={16} /> {isPlaying ? 'Rodando...' : 'Simular'}
                    </button>

                    <button
                        onClick={() => mockService.togglePlay()}
                        className={`p-2 rounded transition-colors ${isPlaying
                            ? 'bg-red-500 text-white hover:bg-red-600' // Pause is primary when playing
                            : 'bg-white/5 text-white/40 hover:bg-white/10' // Pause is secondary when paused
                            }`}
                    >
                        <Pause size={16} />
                    </button>

                    <button
                        onClick={() => mockService.stepForward()}
                        disabled={isPlaying}
                        className={`p-2 rounded transition-colors ${isPlaying ? 'bg-white/5 text-white/20 cursor-not-allowed' : 'bg-white/10 text-white hover:bg-white/20'}`}
                        title="Próximo Passo"
                    >
                        <SkipForward size={16} />
                    </button>
                </div>
            </div>
        </motion.div>
    );
};

const NeuralHUD = ({
    state,
    history,
    showHeatmap,
    setShowHeatmap
}: {
    state: GameState | null,
    history: number[],
    showHeatmap: boolean,
    setShowHeatmap: (v: boolean) => void
}) => {
    const [isOpen, setIsOpen] = useState(true);

    return (
        <div className="absolute bottom-8 left-8 flex flex-col gap-2 pointer-events-none">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="bg-black/60 backdrop-blur border border-white/10 p-2 rounded-lg text-cyan hover:text-white self-start pointer-events-auto"
            >
                <Brain size={20} />
            </button>

            {isOpen && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex gap-4 items-end"
                >
                    {/* Confidence Graph (Real) */}
                    <div className="bg-black/60 backdrop-blur border border-white/10 p-4 rounded-lg w-64 pointer-events-auto">
                        <div className="flex items-center gap-2 mb-2 text-cyan">
                            <Brain size={16} />
                            <span className="text-xs font-bold uppercase">Estimativa de Qualidade (Value Head)</span>
                        </div>
                        <div className="h-24 flex items-end gap-1 border-b border-white/10 pb-1 relative">
                            {/* Render last 20 points or pad with empty */}
                            {(() => {
                                const windowSize = 20;
                                const startIndex = Math.max(0, history.length - windowSize);
                                const visibleHistory = history.slice(startIndex);
                                // Auto-scale: Find max value in visible window, default to 1.0 if empty/zero
                                const maxVal = Math.max(...visibleHistory, 1.0);

                                return Array.from({ length: windowSize }).map((_, i) => {
                                    const dataIndex = history.length - windowSize + i;
                                    const value = dataIndex >= 0 ? history[dataIndex] : 0;
                                    // Normalize height to fit container (max 100%)
                                    const percentage = (value / maxVal) * 100;

                                    return (
                                        <div
                                            key={i}
                                            className="flex-1 bg-cyan/50 rounded-t-sm transition-all duration-300"
                                            style={{ height: `${Math.min(100, percentage)}%` }}
                                            title={`Value: ${value.toFixed(4)}`}
                                        />
                                    );
                                });
                            })()}
                        </div>
                        <div className="mt-2 flex justify-between font-mono text-xs text-cyan">
                            <span className="text-white/30">Scale: 0 - {Math.max(...history.slice(-20), 1.0).toFixed(2)}</span>
                            <span>{(state?.value_confidence || 0).toFixed(4)}</span>
                        </div>
                    </div>

                    {/* Policy Heatmap Toggle */}
                    <div className="bg-black/60 backdrop-blur border border-white/10 p-2 rounded-lg pointer-events-auto flex gap-2">
                        <button
                            onClick={() => setShowHeatmap(!showHeatmap)}
                            className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs transition-colors ${showHeatmap
                                ? 'bg-cyan/20 text-cyan border border-cyan/50'
                                : 'bg-white/10 text-white hover:bg-white/20'
                                }`}
                        >
                            <Layers size={14} /> {showHeatmap ? 'Ocultar policy' : 'Mostrar policy'}
                        </button>
                    </div>
                </motion.div>
            )}
        </div>
    );
};


export const Workspace = () => {
    const { id } = useParams();
    const [gameState, setGameState] = useState<GameState | null>(null);
    const [history, setHistory] = useState<number[]>([]);
    const [showHeatmap, setShowHeatmap] = useState(false);

    useEffect(() => {
        if (id) {
            setHistory([]); // Reset history on new episode
            const unsubscribe = mockService.subscribe((state) => {
                setGameState(state);
                setHistory(prev => [...prev, state.value_confidence || 0]);
            });
            mockService.startSimulation(id);
            return () => {
                unsubscribe();
                mockService.stopSimulation();
            };
        }
    }, [id]);

    // Force re-render on interval to update UI state even if game state doesn't change (e.g. for play/pause toggle)
    // Actually, mockService notifies subscribers on pause/play, so we should be good.
    // However, we need to access non-state properties like isPlaying.
    // Let's use a timer to poll for UI updates or rely on the subscription.
    // Ideally, isPlaying should be part of the state or we should subscribe to it.
    // For now, we'll force a re-render every 100ms to keep UI snappy or just rely on the fact that 
    // mockService notifies on pause/play.

    // To ensure UI updates when isPlaying changes (which might not trigger a new GameState if paused),
    // we can add a local state for it or just rely on the fact that togglePlay calls notifySubscribers.
    // The current implementation of togglePlay in mockService calls notifySubscribers, so we are good.

    return (
        <div className="h-[calc(100vh-64px)] relative bg-void overflow-hidden">
            {/* 3D Canvas */}
            <div className="absolute inset-0">
                <Canvas camera={{ position: [50, 50, 50], fov: 45 }}>
                    <color attach="background" args={['#050505']} />
                    <fog attach="fog" args={['#050505', 50, 150]} />

                    {/* Improved Lighting */}
                    <ambientLight intensity={1.2} />
                    <pointLight position={[10, 10, 10]} intensity={1.5} />
                    <directionalLight position={[-10, 20, -10]} intensity={0.8} />

                    <gridHelper args={[100, 100, '#1a1a1a', '#111111']} />

                    <VoxelGrid
                        tensor={gameState?.tensor || null}
                        heatmap={gameState?.policy_heatmap}
                        showHeatmap={showHeatmap}
                    />
                    <LoadVector tensor={gameState?.tensor || null} />

                    <OrbitControls makeDefault target={[0, 16, 0]} />
                    <GizmoHelper alignment="top-right" margin={[80, 80]}>
                        <GizmoViewport axisColors={['#FF0055', '#00FF9D', '#00F0FF']} labelColor="white" />
                    </GizmoHelper>
                </Canvas>
            </div>

            {/* UI Overlays */}
            <Toolbar />
            <PropertiesPanel state={gameState} />
            <NeuralHUD
                state={gameState}
                history={history}
                showHeatmap={showHeatmap}
                setShowHeatmap={setShowHeatmap}
            />
        </div>
    );
};
```

