"""
Legal move generation and topological safety filters for AlphaBuilder MCTS.

Supports v3.3 defaults and v4.1 upgrades:
- Bidirectional Phase-1 seeding (support neighborhood + optional load patch seeds)
- Surface-only removal mode to avoid wasteful internal bubbles
- Post-batch BFS articulation filtering (`filter_disconnect_safe_batch`)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import ndimage


# 6-connectivity (faces only, not corners/edges)
STRUCTURE_3D = ndimage.generate_binary_structure(3, 1)


def get_protected_regions(
    bc_masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
    force_magnitude: np.ndarray,
) -> np.ndarray:
    """
    Get regions that should be protected from removal:
    1. Boundary condition locations (supports)
    2. Force application points
    """
    mask_x, mask_y, mask_z = bc_masks
    bc_protected = (mask_x + mask_y + mask_z) > 0
    force_protected = force_magnitude > 0
    return (bc_protected | force_protected).astype(np.float32)


def get_legal_add_moves(
    density: np.ndarray,
    threshold: float = 0.5,
    include_supports: bool = True,
    bc_masks: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    domain_mask: Optional[np.ndarray] = None,
    forces: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    include_load_seeds: bool = False,
) -> np.ndarray:
    """
    Get valid positions for adding material:
    1. 6-neighbor boundary of existing material: dilate(grid) - grid
    2. Support locations (dilated by 1 voxel) so structures can initiate/anchor
    3. Optional load patch seeds (`include_load_seeds=True`) for bidirectional Phase-1 growth
    """
    solid = density > threshold
    dilated = ndimage.binary_dilation(solid, structure=STRUCTURE_3D)
    boundary = dilated.astype(np.float32) - solid.astype(np.float32)
    valid_add = (boundary > 0).astype(np.float32)

    if include_supports:
        support_mask = np.zeros_like(density, dtype=np.float32)
        if bc_masks is not None:
            mask_x, mask_y, mask_z = bc_masks
            raw_support = (mask_x > 0.5) | (mask_y > 0.5) | (mask_z > 0.5)
            dilated_support = ndimage.binary_dilation(raw_support, structure=STRUCTURE_3D)
            support_mask = dilated_support.astype(np.float32)
        else:
            support_mask[0, :, :] = 1.0
            if density.shape[0] > 1:
                support_mask[1, :, :] = 1.0

        support_valid = support_mask * (1.0 - solid.astype(np.float32))
        valid_add = np.maximum(valid_add, support_valid)

    if include_load_seeds and forces is not None:
        force_mag = np.abs(forces[0]) + np.abs(forces[1]) + np.abs(forces[2])
        load_raw = force_mag > 1e-6
        if np.any(load_raw):
            dilated_load = ndimage.binary_dilation(load_raw, structure=STRUCTURE_3D) | load_raw
            load_valid = dilated_load.astype(np.float32) * (1.0 - solid.astype(np.float32))
            valid_add = np.maximum(valid_add, load_valid)

    if domain_mask is not None:
        valid_add = valid_add * (domain_mask > 0.5).astype(np.float32)

    return valid_add


def get_legal_remove_moves(
    density: np.ndarray,
    threshold: float = 0.5,
    bc_masks: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    forces: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    surface_only: bool = False,
) -> np.ndarray:
    """
    Get valid positions for removing material.

    When `surface_only=True`, restricts candidates to exterior/cavity surface voxels
    (6-adjacent to void or domain border), falling back safely to all non-protected
    solid voxels only if no non-protected surface voxel exists.
    """
    solid = density > threshold
    valid_remove = solid.copy()

    if bc_masks is not None or forces is not None:
        zeros = np.zeros_like(density, dtype=np.float32)
        bc_tuple = bc_masks if bc_masks is not None else (zeros, zeros, zeros)
        if forces is not None:
            force_mag = np.abs(forces[0]) + np.abs(forces[1]) + np.abs(forces[2])
        else:
            force_mag = zeros
        protected = get_protected_regions(bc_tuple, force_mag) > 0.5
        valid_remove = valid_remove & (~protected)

    if surface_only and np.any(valid_remove):
        interior = ndimage.binary_erosion(solid, structure=STRUCTURE_3D, border_value=0)
        surface_valid = valid_remove & (~interior)
        if np.any(surface_valid):
            valid_remove = surface_valid

    return valid_remove.astype(np.float32)


def get_legal_moves(
    density: np.ndarray,
    threshold: float = 0.5,
    bc_masks: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    forces: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    domain_mask: Optional[np.ndarray] = None,
    surface_only_remove: bool = False,
    include_load_seeds: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get all legal moves `(valid_add, valid_remove)` for current state."""
    valid_add = get_legal_add_moves(
        density,
        threshold,
        bc_masks=bc_masks,
        domain_mask=domain_mask,
        forces=forces,
        include_load_seeds=include_load_seeds,
    )
    valid_remove = get_legal_remove_moves(
        density,
        threshold,
        bc_masks=bc_masks,
        forces=forces,
        surface_only=surface_only_remove,
    )
    return valid_add, valid_remove


def filter_disconnect_safe_batch(
    density: np.ndarray,
    actions: List[Tuple[int, int, int, int]],
    load_config: Dict[str, Any],
    bc_masks: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    forces: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    threshold: float = 0.5,
) -> List[Tuple[int, int, int, int]]:
    """
    Prevent multi-voxel REMOVE batches from jointly severing load-to-support connectivity.

    Applies a fast single-BFS batch check first; only if committing all edits together
    disconnects an already-connected structure does it fall back to sequential incremental
    articulation filtering (ADDs first, then safe REMOVEs).
    """
    if not actions or not any(ch == 1 for ch, _, _, _ in actions):
        return list(actions)

    from ..selfplay.reward import analyze_structure_islands

    has_forces = (
        forces is not None
        and float(np.abs(forces[0]).sum() + np.abs(forces[1]).sum() + np.abs(forces[2]).sum()) > 1e-6
    )
    forces_arg = forces if has_forces else None

    base_isl = analyze_structure_islands(
        density, load_config, threshold=threshold, bc_masks=bc_masks, forces=forces_arg
    )
    if not base_isl["is_connected"]:
        return list(actions)

    cand = density.copy()
    for ch, x, y, z in actions:
        cand[x, y, z] = 1.0 if ch == 0 else 0.0
    post_isl = analyze_structure_islands(
        cand, load_config, threshold=threshold, bc_masks=bc_masks, forces=forces_arg
    )
    if post_isl["is_connected"]:
        return list(actions)

    # Fallback: commit ADDs first so new bridges reinforce before carving REMOVEs
    safe_density = density.copy()
    safe_actions: List[Tuple[int, int, int, int]] = []
    add_acts = [a for a in actions if a[0] == 0]
    rem_acts = [a for a in actions if a[0] == 1]

    for ch, x, y, z in add_acts:
        safe_density[x, y, z] = 1.0
        safe_actions.append((ch, x, y, z))

    for ch, x, y, z in rem_acts:
        if safe_density[x, y, z] <= threshold:
            continue
        safe_density[x, y, z] = 0.0
        check_isl = analyze_structure_islands(
            safe_density, load_config, threshold=threshold, bc_masks=bc_masks, forces=forces_arg
        )
        if check_isl["is_connected"]:
            safe_actions.append((ch, x, y, z))
        else:
            safe_density[x, y, z] = 1.0

    return safe_actions


def apply_action_mask(
    policy_add: np.ndarray,
    policy_remove: np.ndarray,
    valid_add: np.ndarray,
    valid_remove: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Set invalid action logits to `-inf` so they receive 0 probability under softmax."""
    masked_add = np.where(valid_add > 0, policy_add, -np.inf)
    masked_remove = np.where(valid_remove > 0, policy_remove, -np.inf)
    return masked_add, masked_remove


def masked_softmax(
    policy_add: np.ndarray,
    policy_remove: np.ndarray,
    valid_add: np.ndarray,
    valid_remove: np.ndarray,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute joint softmax over legal add and remove actions."""
    temp = max(float(temperature), 1e-6)
    masked_add, masked_remove = apply_action_mask(
        policy_add / temp, policy_remove / temp, valid_add, valid_remove
    )

    flat_add = masked_add.flatten()
    flat_remove = masked_remove.flatten()
    combined = np.concatenate([flat_add, flat_remove])

    finite_mask = np.isfinite(combined)
    if not np.any(finite_mask):
        return np.zeros_like(policy_add), np.zeros_like(policy_remove)

    max_val = np.max(combined[finite_mask])
    exp_combined = np.exp(combined - max_val)
    exp_combined[~finite_mask] = 0.0

    total = exp_combined.sum()
    if total < 1e-10:
        probs = np.zeros_like(combined)
    else:
        probs = exp_combined / total

    spatial_size = flat_add.size
    prob_add = probs[:spatial_size].reshape(policy_add.shape)
    prob_remove = probs[spatial_size:].reshape(policy_remove.shape)
    return prob_add, prob_remove


def count_legal_moves(
    density: np.ndarray,
    threshold: float = 0.5,
    bc_masks: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    forces: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[int, int]:
    """Count total legal add/remove moves available."""
    valid_add, valid_remove = get_legal_moves(density, threshold, bc_masks=bc_masks, forces=forces)
    return int(valid_add.sum()), int(valid_remove.sum())


def is_terminal_state(
    density: np.ndarray,
    max_volume_fraction: float = 0.3,
    min_volume_fraction: float = 0.01,
    threshold: float = 0.5,
    bc_masks: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    forces: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    domain_mask: Optional[np.ndarray] = None,
) -> Tuple[bool, str]:
    """Check if MCTS simulation leaf state is terminal."""
    solid = density > threshold
    vol_fraction = solid.sum() / solid.size

    if vol_fraction > max_volume_fraction:
        return True, "max_volume_exceeded"

    if vol_fraction < min_volume_fraction:
        return True, "structure_too_small"

    valid_add, valid_remove = get_legal_moves(
        density, threshold, bc_masks=bc_masks, forces=forces, domain_mask=domain_mask
    )
    if valid_add.sum() == 0 and valid_remove.sum() == 0:
        return True, "no_legal_moves"

    return False, ""
