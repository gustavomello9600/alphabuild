"""
Legal move generation for AlphaBuilder MCTS.

Generates valid action masks for add/remove operations:
- Add: Only at boundary (dilate(grid) - grid)
- Remove: Only existing material (grid)

Reference: AlphaBuilder v3.1 MCTS Specification Section 2 (Expansion)
"""

import numpy as np
from scipy import ndimage
from typing import Tuple


# Pre-computed 3D structuring element for morphological operations
# 6-connectivity (faces only, not corners/edges)
STRUCTURE_3D = ndimage.generate_binary_structure(3, 1)


def get_legal_add_moves(
    density: np.ndarray, 
    threshold: float = 0.5,
    include_supports: bool = True
) -> np.ndarray:
    """
    Get valid positions for adding material.
    
    Add actions are valid at:
    1. Boundary of existing material: dilate(grid) - grid
    2. Support locations (X=0 plane) - ensures structure can grow from supports
    
    This ensures:
    - We only add adjacent to existing structure (connectivity)
    - We don't add where material already exists
    - We can always place at supports to initiate/reinforce connection
    
    Args:
        density: Current density grid (D, H, W) with values in [0, 1]
        threshold: Density threshold for "solid" material (default 0.5)
        include_supports: If True, include X=0 plane as valid add positions
        
    Returns:
        Binary mask (D, H, W) where 1 indicates valid add position
    """
    # Binarize current grid
    solid = density > threshold
    
    # Dilate to get envelope + solid
    dilated = ndimage.binary_dilation(solid, structure=STRUCTURE_3D)
    
    # Boundary = dilated minus solid (can't add where already solid)
    boundary = dilated.astype(np.float32) - solid.astype(np.float32)
    valid_add = (boundary > 0).astype(np.float32)
    
    # Add support plane (X=0) as valid, excluding already-solid voxels
    if include_supports:
        support_mask = np.zeros_like(density)
        support_mask[0, :, :] = 1.0  # X=0 plane
        # Only where not already solid
        support_valid = support_mask * (1.0 - solid.astype(np.float32))
        valid_add = np.maximum(valid_add, support_valid)
    
    return valid_add


def get_legal_remove_moves(density: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Get valid positions for removing material.
    
    Remove actions are valid only where material exists:
    valid_remove = grid > threshold
    
    Args:
        density: Current density grid (D, H, W) with values in [0, 1]
        threshold: Density threshold for "solid" material (default 0.5)
        
    Returns:
        Binary mask (D, H, W) where 1 indicates valid remove position
    """
    return (density > threshold).astype(np.float32)


def get_legal_moves(
    density: np.ndarray, 
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get all legal moves (add and remove) for current state.
    
    Args:
        density: Current density grid (D, H, W) with values in [0, 1]
        threshold: Density threshold for "solid" material (default 0.5)
        
    Returns:
        Tuple of (valid_add, valid_remove) binary masks
    """
    valid_add = get_legal_add_moves(density, threshold)
    valid_remove = get_legal_remove_moves(density, threshold)
    
    return valid_add, valid_remove


def apply_action_mask(
    policy_add: np.ndarray,
    policy_remove: np.ndarray,
    valid_add: np.ndarray,
    valid_remove: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply legal move masks to policy logits.
    
    Sets invalid actions to -inf so they get zero probability after softmax.
    
    L_masked = L + (1 - Mask) * (-inf)
    
    Args:
        policy_add: Add action logits (D, H, W)
        policy_remove: Remove action logits (D, H, W)
        valid_add: Binary mask of valid add positions
        valid_remove: Binary mask of valid remove positions
        
    Returns:
        Tuple of (masked_add, masked_remove) logits
    """
    # Apply masks: invalid positions get -inf
    masked_add = np.where(valid_add > 0, policy_add, -np.inf)
    masked_remove = np.where(valid_remove > 0, policy_remove, -np.inf)
    
    return masked_add, masked_remove


def masked_softmax(
    policy_add: np.ndarray,
    policy_remove: np.ndarray,
    valid_add: np.ndarray,
    valid_remove: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute softmax over valid actions only.
    
    Combines add and remove into single distribution, applies mask,
    and returns normalized probabilities.
    
    Args:
        policy_add: Add action logits (D, H, W)
        policy_remove: Remove action logits (D, H, W)
        valid_add: Binary mask of valid add positions
        valid_remove: Binary mask of valid remove positions
        
    Returns:
        Tuple of (prob_add, prob_remove) normalized probability distributions
    """
    # Apply masks
    masked_add, masked_remove = apply_action_mask(
        policy_add, policy_remove, valid_add, valid_remove
    )
    
    # Flatten and combine for joint softmax
    flat_add = masked_add.flatten()
    flat_remove = masked_remove.flatten()
    combined = np.concatenate([flat_add, flat_remove])
    
    # Stable softmax (subtract max for numerical stability)
    finite_mask = np.isfinite(combined)
    if not np.any(finite_mask):
        # No valid actions - return zeros
        zeros_add = np.zeros_like(policy_add)
        zeros_remove = np.zeros_like(policy_remove)
        return zeros_add, zeros_remove
    
    max_val = np.max(combined[finite_mask])
    exp_combined = np.exp(combined - max_val)
    exp_combined[~finite_mask] = 0  # Zero out masked positions
    
    # Normalize
    total = exp_combined.sum()
    if total < 1e-10:
        probs = np.zeros_like(combined)
    else:
        probs = exp_combined / total
    
    # Split back to add/remove
    spatial_size = flat_add.size
    prob_add = probs[:spatial_size].reshape(policy_add.shape)
    prob_remove = probs[spatial_size:].reshape(policy_remove.shape)
    
    return prob_add, prob_remove


def count_legal_moves(
    density: np.ndarray,
    threshold: float = 0.5
) -> Tuple[int, int]:
    """
    Count total legal moves available.
    
    Args:
        density: Current density grid
        threshold: Density threshold
        
    Returns:
        Tuple of (num_add, num_remove) legal moves
    """
    valid_add, valid_remove = get_legal_moves(density, threshold)
    return int(valid_add.sum()), int(valid_remove.sum())


def is_terminal_state(
    density: np.ndarray,
    max_volume_fraction: float = 0.3,
    min_volume_fraction: float = 0.01,
    threshold: float = 0.5
) -> Tuple[bool, str]:
    """
    Check if state is terminal (no more meaningful moves).
    
    Terminal conditions:
    1. Volume fraction too high (>max_volume_fraction)
    2. Volume fraction too low (<min_volume_fraction)
    3. No legal moves available
    
    Args:
        density: Current density grid
        max_volume_fraction: Maximum allowed volume fraction
        min_volume_fraction: Minimum allowed volume fraction
        threshold: Density threshold
        
    Returns:
        Tuple of (is_terminal, reason)
    """
    solid = density > threshold
    vol_fraction = solid.sum() / solid.size
    
    if vol_fraction > max_volume_fraction:
        return True, "max_volume_exceeded"
    
    if vol_fraction < min_volume_fraction:
        return True, "structure_too_small"
    
    valid_add, valid_remove = get_legal_moves(density, threshold)
    if valid_add.sum() == 0 and valid_remove.sum() == 0:
        return True, "no_legal_moves"
    
    return False, ""


def get_protected_regions(
    bc_masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
    force_magnitude: np.ndarray
) -> np.ndarray:
    """
    Get regions that should be protected from removal.
    
    Protected regions include:
    1. Boundary condition locations (supports)
    2. Force application points
    
    Args:
        bc_masks: Tuple of (mask_x, mask_y, mask_z) boundary condition masks
        force_magnitude: Magnitude of forces at each voxel
        
    Returns:
        Binary mask where 1 indicates protected (cannot remove)
    """
    mask_x, mask_y, mask_z = bc_masks
    
    # Any BC mask is protected
    bc_protected = (mask_x + mask_y + mask_z) > 0
    
    # Force application points are protected
    force_protected = force_magnitude > 0
    
    return (bc_protected | force_protected).astype(np.float32)
