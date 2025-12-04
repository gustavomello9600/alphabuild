"""
Data Processing and Policy Target Generation.
"""
import numpy as np
import scipy.ndimage
from typing import List, Dict, Any

from alphabuilder.src.logic.storage import Phase
from alphabuilder.src.logic.harvest.config import (
    LOG_SQUASH_ALPHA, LOG_SQUASH_MU, LOG_SQUASH_SIGMA, LOG_SQUASH_EPSILON
)

def compute_normalized_value(compliance: float, vol_frac: float) -> float:
    """
    Compute normalized value score using Log-Squash formula (Spec 4.2).
    """
    s_raw = -np.log(compliance + LOG_SQUASH_EPSILON) - LOG_SQUASH_ALPHA * vol_frac
    normalized = np.tanh((s_raw - LOG_SQUASH_MU) / LOG_SQUASH_SIGMA)
    return float(normalized)

def check_connectivity(density_grid: np.ndarray, threshold: float, load_cfg: Dict[str, Any]) -> tuple:
    """
    Check if the structure connects the support (X=0) to the load region.
    """
    nx, ny, nz = density_grid.shape
    
    # Binarize
    binary = density_grid > threshold
    
    # Label connected components
    labeled, n_components = scipy.ndimage.label(binary)
    
    if n_components == 0:
        return False, binary
        
    # Check if Support (X=0) and Load Region are in the same component
    support_labels = np.unique(labeled[0, :, :])
    support_labels = support_labels[support_labels > 0]
    
    if len(support_labels) == 0:
        return False, binary
        
    # Load Region
    lx, ly, lz_s, lz_e = load_cfg['x'], load_cfg['y'], load_cfg['z_start'], load_cfg['z_end']
    lx = min(lx, nx-1)
    ly = min(ly, ny-1)
    lz_s = max(0, lz_s)
    lz_e = min(nz, lz_e)
    
    load_slice = labeled[lx-2:lx+2, ly-2:ly+2, lz_s:lz_e]
    load_labels = np.unique(load_slice)
    load_labels = load_labels[load_labels > 0]
    
    common = np.intersect1d(support_labels, load_labels)
    return len(common) > 0, binary

def generate_phase1_slices(
    final_mask: np.ndarray, 
    target_value: float,
    num_steps: int = 50
) -> List[Dict[str, Any]]:
    """
    Slice the constructed structure into growth steps.
    """
    structure = final_mask > 0.5
    dists = np.full(final_mask.shape, -1, dtype=np.int32)
    
    queue = []
    starts = np.where(structure[0, :, :])
    for y, z in zip(starts[0], starts[1]):
        dists[0, y, z] = 0
        queue.append((0, y, z))
        
    head = 0
    while head < len(queue):
        x, y, z = queue[head]
        head += 1
        current_dist = dists[x, y, z]
        
        for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < final_mask.shape[0] and \
               0 <= ny < final_mask.shape[1] and \
               0 <= nz < final_mask.shape[2]:
                if structure[nx, ny, nz] and dists[nx, ny, nz] == -1:
                    dists[nx, ny, nz] = current_dist + 1
                    queue.append((nx, ny, nz))
                    
    max_dist = np.max(dists)
    if max_dist <= 0:
        return []
        
    records = []
    for i in range(1, num_steps + 1):
        percent = i / num_steps
        threshold = int(max_dist * percent)
        
        input_mask = (dists <= threshold) & (dists != -1)
        input_grid = input_mask.astype(np.float32)
        
        target_add = np.where((final_mask > 0.5) & (input_grid < 0.5), 1.0, 0.0)
        target_remove = np.zeros_like(target_add)
        
        records.append({
            "phase": Phase.GROWTH,
            "step": i,
            "input_state": input_grid,
            "target_add": target_add,
            "target_remove": target_remove,
            "target_value": target_value,
            "current_vol": float(np.mean(input_grid)),
            "current_compliance": None
        })
        
    return records

def compute_boundary_mask(binary_density: np.ndarray) -> np.ndarray:
    """
    Compute boundary mask: voxels that are 0 (void) but have at least one 1 (solid) neighbor.
    Args:
        binary_density: Binary numpy array (0 or 1)
    """
    # Ensure binary input
    solid = (binary_density > 0.5).astype(np.int32)
    
    # Dilate solid to find neighbors
    # Structure for dilation: 6-connectivity
    struct = scipy.ndimage.generate_binary_structure(3, 1)
    dilated = scipy.ndimage.binary_dilation(solid, structure=struct).astype(np.int32)
    
    # Boundary = Dilated - Solid (i.e., neighbors that are not solid themselves)
    boundary = dilated - solid
    
    return boundary.astype(np.float32)

def compute_filled_mask(binary_density: np.ndarray) -> np.ndarray:
    """
    Compute filled mask: voxels that have material (1).
    Args:
        binary_density: Binary numpy array (0 or 1)
    """
    return (binary_density > 0.5).astype(np.float32)

def generate_refinement_targets(
    current_density: np.ndarray,
    next_density: np.ndarray,
    current_binary_mask: np.ndarray
) -> tuple:
    """
    Generate policy targets for Refinement phase (Phase 2).
    
    Logic:
    1. Calculate difference: diff = next - current
    2. Add Channel: max(0, diff) * boundary_mask(current_binary)
    3. Remove Channel: max(0, -diff) * filled_mask(current_binary)
    4. Normalize: Max-Scaling (val / max(val))
    """
    # 1. Difference
    diff = next_density - current_density
    
    # 2. Raw Targets
    raw_add = np.maximum(0, diff)
    raw_remove = np.maximum(0, -diff)
    
    # 3. Masks (using binary input state)
    # Boundary mask for ADD: 0 voxels with neighbor 1
    boundary_mask = compute_boundary_mask(current_binary_mask)
    
    # Filled mask for REMOVE: 1 voxels
    filled_mask = compute_filled_mask(current_binary_mask)
    
    # 4. Apply Masks
    target_add = raw_add * boundary_mask
    target_remove = raw_remove * filled_mask
    
    # 5. Normalization (Max-Scaling)
    # Scale so the maximum value is 1.0 (if there is any signal)
    # This helps MCTS identify the "best" actions regardless of absolute magnitude
    
    max_add = np.max(target_add)
    if max_add > 1e-8:
        target_add = target_add / max_add
        
    max_remove = np.max(target_remove)
    if max_remove > 1e-8:
        target_remove = target_remove / max_remove
    
    return target_add, target_remove
