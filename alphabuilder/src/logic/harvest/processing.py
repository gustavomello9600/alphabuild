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

def compute_boundary_mask(density: np.ndarray) -> np.ndarray:
    """
    Compute boundary mask: voxels that are 0 (void) but have at least one 1 (solid) neighbor.
    """
    # Binary structure (assuming density is already somewhat binary or we threshold it)
    # For SIMP, density is continuous. We treat < 0.5 as void, >= 0.5 as solid for mask purposes?
    # Or should we use the continuous density?
    # The user said "Apply mask of boundary (0, with neighbor 1)". This implies binary logic.
    # We will use threshold 0.1 to be safe (anything with some material is "solid" for neighbor check)
    
    solid = (density > 0.1).astype(np.int32)
    
    # Dilate solid to find neighbors
    # Structure for dilation: 6-connectivity
    struct = scipy.ndimage.generate_binary_structure(3, 1)
    dilated = scipy.ndimage.binary_dilation(solid, structure=struct).astype(np.int32)
    
    # Boundary = Dilated - Solid (i.e., neighbors that are not solid themselves)
    boundary = dilated - solid
    
    # Also, we only want to ADD where density is low.
    # So we multiply by (1 - density) to soften the mask?
    # User said "Apply mask... (0, with neighbor 1)".
    # We'll return a binary mask where density is low (<0.1) AND it has a neighbor.
    
    return boundary.astype(np.float32)

def compute_filled_mask(density: np.ndarray) -> np.ndarray:
    """
    Compute filled mask: voxels that have material (1).
    """
    # User said "material filled (1)".
    # We'll use a threshold.
    return (density > 0.1).astype(np.float32)

def generate_refinement_targets(
    current_density: np.ndarray,
    next_density: np.ndarray
) -> tuple:
    """
    Generate policy targets for Refinement phase (Phase 2).
    
    Logic:
    1. Calculate difference: diff = next - current
    2. Add Channel: max(0, diff) * boundary_mask
    3. Remove Channel: max(0, -diff) * filled_mask
    4. Normalize to [0, 1]
    """
    # 1. Difference
    diff = next_density - current_density
    
    # 2. Raw Targets
    raw_add = np.maximum(0, diff)
    raw_remove = np.maximum(0, -diff)
    
    # 3. Masks
    # Boundary mask for ADD: 0 voxels with neighbor 1
    # We use current_density to determine boundary
    boundary_mask = compute_boundary_mask(current_density)
    
    # Filled mask for REMOVE: 1 voxels
    filled_mask = compute_filled_mask(current_density)
    
    # 4. Apply Masks
    target_add = raw_add * boundary_mask
    target_remove = raw_remove * filled_mask
    
    # 5. Normalization
    # Since diff is already in range [0, 1] (density difference),
    # and we want "normalized values 0 to 1", we are good.
    # However, SIMP steps are small.
    # If we want to emphasize the signal, we could scale it.
    # But "normalized" usually implies [0, 1].
    # We will clip to ensure it stays in range (though it should be).
    
    target_add = np.clip(target_add, 0.0, 1.0)
    target_remove = np.clip(target_remove, 0.0, 1.0)
    
    return target_add, target_remove
