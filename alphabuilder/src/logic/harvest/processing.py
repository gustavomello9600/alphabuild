"""
Data Processing and Policy Target Generation (v3.1 / v3.3).
"""
import numpy as np
import scipy.ndimage
from typing import List, Dict, Any, Optional, Tuple

from alphabuilder.src.logic.storage import Phase
from alphabuilder.src.logic.harvest.config import (
    LOG_SQUASH_ALPHA, LOG_SQUASH_MU, LOG_SQUASH_SIGMA, LOG_SQUASH_EPSILON
)
from alphabuilder.src.logic.harvest.generators import get_load_region_bounds
from alphabuilder.src.logic.mcts.legal_moves import (
    STRUCTURE_3D,
    get_legal_add_moves,
    get_legal_remove_moves,
    get_protected_regions,
)


def compute_normalized_value(compliance: float, vol_frac: float) -> float:
    """
    [DEPRECATED v3.1] Compute normalized value score using Log-Squash tanh formula.
    Kept only for backward-compatibility with legacy tests. Use reward.calculate_reward in v3.3.
    """
    s_raw = -np.log(compliance + LOG_SQUASH_EPSILON) - LOG_SQUASH_ALPHA * vol_frac
    normalized = np.tanh((s_raw - LOG_SQUASH_MU) / LOG_SQUASH_SIGMA)
    return float(normalized)


def check_connectivity(
    density_grid: np.ndarray,
    threshold: float,
    load_cfg: Dict[str, Any],
    strict_load_coverage: bool = False,
) -> tuple:
    """
    Check if the structure connects the support (X=0) to the load region (6-connected).

    Args:
        density_grid: 3D density array.
        threshold: Binarization threshold.
        load_cfg: Load configuration dictionary.
        strict_load_coverage: When True, requires all load patch voxels at
            (lx, y_min:y_max, z_min:z_max) to belong to a single support-connected component
            (matching reward.check_structure_connectivity).
    """
    binary = density_grid > threshold
    labeled, n_components = scipy.ndimage.label(binary, structure=STRUCTURE_3D)

    if n_components == 0:
        return False, binary

    support_labels = np.unique(labeled[0, :, :])
    support_labels = support_labels[support_labels > 0]
    if len(support_labels) == 0:
        return False, binary

    lx, y_min, y_max, z_min, z_max = get_load_region_bounds(density_grid.shape, load_cfg)

    if strict_load_coverage:
        load_slice = labeled[lx, y_min:y_max, z_min:z_max]
        present_labels = np.unique(load_slice)
        present_labels = present_labels[present_labels > 0]
        if len(present_labels) != 1:
            return False, binary
        if present_labels[0] not in support_labels:
            return False, binary
        expected_vol = (y_max - y_min) * (z_max - z_min)
        if int(np.sum(load_slice > 0)) < expected_vol:
            return False, binary
        return True, binary

    nx, ny, nz = density_grid.shape
    load_slice = labeled[max(0, lx - 2):min(nx, lx + 2), y_min:y_max, z_min:z_max]
    load_labels = np.unique(load_slice)
    load_labels = load_labels[load_labels > 0]
    common = np.intersect1d(support_labels, load_labels)
    return bool(len(common) > 0), binary


def _compute_geodesic_distances_from_support(structure: np.ndarray) -> np.ndarray:
    """BFS 6-connected geodesic distance from X=0 support plane within solid structure."""
    dists = np.full(structure.shape, -1, dtype=np.int32)
    queue = []
    starts = np.where(structure[0, :, :])
    for y, z in zip(starts[0], starts[1]):
        dists[0, y, z] = 0
        queue.append((0, int(y), int(z)))

    head = 0
    sx, sy, sz = structure.shape
    while head < len(queue):
        x, y, z = queue[head]
        head += 1
        current_dist = dists[x, y, z]
        for dx, dy, dz in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz:
                if structure[nx, ny, nz] and dists[nx, ny, nz] == -1:
                    dists[nx, ny, nz] = current_dist + 1
                    queue.append((nx, ny, nz))
    return dists


def generate_phase1_slices(
    final_mask: np.ndarray,
    target_value: float,
    num_steps: int = 50,
    bc_masks: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    beta_geo: float = 0.5,
    gamma: float = 0.99,
    include_empty_step: bool = False,
    target_mode: str = "legacy",
) -> List[Dict[str, Any]]:
    """
    Slice the constructed structure into Phase 1 (GROWTH) training steps.

    - target_mode='legacy': original v3.1 behavior (steps 1..num_steps, full remaining blueprint).
    - target_mode='geodesic_boundary': v3.3 behavior aligned with MCTS legal moves.
      Targets lie strictly on legal_add(input_grid) ∩ V_constructed with geodesic exponential
      weights normalized to sum 1.0, and values discounted by gamma**(num_steps - i).
    """
    structure = final_mask > 0.5
    dists = _compute_geodesic_distances_from_support(structure)
    max_dist = int(np.max(dists))
    if max_dist <= 0:
        return []

    records = []
    start_idx = 0 if include_empty_step else 1
    end_idx = num_steps if include_empty_step else (num_steps + 1)

    for i in range(start_idx, end_idx):
        if i == 0:
            input_grid = np.zeros(final_mask.shape, dtype=np.float32)
        else:
            percent = i / float(num_steps)
            threshold = int(max_dist * percent)
            if target_mode == "geodesic_boundary":
                threshold = min(max_dist - 1, threshold)
            input_mask = (dists <= threshold) & (dists != -1)
            input_grid = input_mask.astype(np.float32)

        if target_mode == "geodesic_boundary":
            valid_add = get_legal_add_moves(
                input_grid,
                threshold=0.5,
                include_supports=True,
                bc_masks=bc_masks,
            )
            frontier = (valid_add > 0.5) & structure & (dists >= 0)
            if not np.any(frontier):
                continue

            frontier_dists = dists[frontier].astype(np.float32)
            d_min = float(np.min(frontier_dists))
            weights = np.exp(-float(beta_geo) * (frontier_dists - d_min)).astype(np.float32)
            w_sum = float(np.sum(weights))
            if w_sum <= 1e-12:
                continue

            target_add = np.zeros(final_mask.shape, dtype=np.float32)
            target_add[frontier] = weights / w_sum
            target_remove = np.zeros_like(target_add)
            step_value = float((gamma ** (num_steps - i)) * target_value)
        else:
            target_add = np.where((final_mask > 0.5) & (input_grid < 0.5), 1.0, 0.0).astype(np.float32)
            target_remove = np.zeros_like(target_add)
            step_value = float(target_value)

        records.append({
            "phase": Phase.GROWTH,
            "step": i,
            "input_state": input_grid,
            "target_add": target_add,
            "target_remove": target_remove,
            "target_value": step_value,
            "current_vol": float(np.mean(input_grid)),
            "current_compliance": None,
        })

    return records


def compute_boundary_mask(binary_density: np.ndarray) -> np.ndarray:
    """
    Compute boundary mask: voxels that are 0 (void) but have at least one 1 (solid) neighbor.
    """
    solid = (binary_density > 0.5).astype(np.int32)
    dilated = scipy.ndimage.binary_dilation(solid, structure=STRUCTURE_3D).astype(np.int32)
    boundary = dilated - solid
    return boundary.astype(np.float32)


def compute_filled_mask(binary_density: np.ndarray) -> np.ndarray:
    """
    Compute filled mask: voxels that have material (1).
    """
    return (binary_density > 0.5).astype(np.float32)


def generate_refinement_targets(
    current_density: np.ndarray,
    next_density: np.ndarray,
    current_binary_mask: np.ndarray,
    bc_masks: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    forces: Optional[np.ndarray] = None,
    noise_threshold: float = 0.0,
    normalization: str = "max",
    sharpness: float = 1.0,
) -> tuple:
    """
    Generate policy targets for Refinement phase (Phase 2).

    - normalization='max': legacy per-channel max-scaling (default for backward compat).
    - normalization='joint': v3.3 joint Add+Remove probability distribution over legal moves
      (with support/load protection and noise thresholding), summing to 1.0 (or 0.0 if stationary).
    """
    diff = next_density - current_density

    raw_add = np.maximum(0.0, diff - noise_threshold)
    raw_remove = np.maximum(0.0, -diff - noise_threshold)

    if bc_masks is not None:
        boundary_mask = get_legal_add_moves(
            current_binary_mask, threshold=0.5, include_supports=True, bc_masks=bc_masks
        )
    else:
        boundary_mask = compute_boundary_mask(current_binary_mask)

    filled_mask = get_legal_remove_moves(current_binary_mask, threshold=0.5)
    if bc_masks is not None and forces is not None:
        force_mag = np.linalg.norm(forces, axis=0) if forces.ndim == 4 else forces
        protected = get_protected_regions(bc_masks, force_mag)
        filled_mask = filled_mask * (1.0 - protected)

    target_add = (raw_add * boundary_mask).astype(np.float32)
    target_remove = (raw_remove * filled_mask).astype(np.float32)

    if normalization == "joint":
        if sharpness != 1.0:
            target_add = np.power(target_add, sharpness)
            target_remove = np.power(target_remove, sharpness)
        total = float(np.sum(target_add) + np.sum(target_remove))
        if total > 1e-8:
            target_add = (target_add / total).astype(np.float32)
            target_remove = (target_remove / total).astype(np.float32)
        else:
            target_add.fill(0.0)
            target_remove.fill(0.0)
    else:
        max_add = float(np.max(target_add))
        if max_add > 1e-8:
            target_add = (target_add / max_add).astype(np.float32)

        max_remove = float(np.max(target_remove))
        if max_remove > 1e-8:
            target_remove = (target_remove / max_remove).astype(np.float32)

    return target_add, target_remove
