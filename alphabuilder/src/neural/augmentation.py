"""
Data Augmentation for AlphaBuilder v3.1.

⚡ Performance Target: < 5ms per sample (200+ samples/second)
- All operations use NumPy vectorized operations
- Pre-computed erosion kernels
- No I/O during augmentation
"""
import numpy as np
from scipy import ndimage
from typing import Tuple

# Pre-computed 3D erosion/dilation kernel (3x3x3 cross)
EROSION_KERNEL = ndimage.generate_binary_structure(3, 1)


def rotate_90_z(
    state: np.ndarray, 
    policy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate state and policy 90° around Z axis.
    
    Correctly inverts force vectors:
    - Fx -> Fy
    - Fy -> -Fx
    
    State channels: [density, mask_x, mask_y, mask_z, fx, fy, fz]
    Policy channels: [add, remove]
    
    Args:
        state: (7, D, H, W) tensor
        policy: (2, D, H, W) tensor
        
    Returns:
        Rotated (state, policy) tuple
    """
    # Rotate spatial dimensions (axes 1,2 = D,H -> rotate in D-H plane around W)
    # np.rot90 rotates in plane of (axes[0], axes[1])
    state_rot = np.rot90(state, k=1, axes=(1, 2)).copy()
    policy_rot = np.rot90(policy, k=1, axes=(1, 2)).copy()
    
    # Rotate force vectors (channels 4,5 = fx, fy)
    # Rotation 90° CCW: (fx, fy) -> (fy, -fx)
    old_fx = state_rot[4].copy()
    old_fy = state_rot[5].copy()
    state_rot[4] = old_fy      # New Fx = old Fy
    state_rot[5] = -old_fx     # New Fy = -old Fx
    
    # Rotate mask vectors (channels 1,2 = mask_x, mask_y)
    old_mx = state_rot[1].copy()
    old_my = state_rot[2].copy()
    state_rot[1] = old_my      # New Mask_X = old Mask_Y
    state_rot[2] = old_mx      # New Mask_Y = old Mask_X (masks don't negate)
    
    return state_rot, policy_rot


def flip_y(
    state: np.ndarray, 
    policy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flip state and policy along Y axis (height).
    
    Correctly inverts Fy force component.
    
    Args:
        state: (7, D, H, W) tensor
        policy: (2, D, H, W) tensor
        
    Returns:
        Flipped (state, policy) tuple
    """
    # Flip along axis 2 (H dimension)
    state_flip = np.flip(state, axis=2).copy()
    policy_flip = np.flip(policy, axis=2).copy()
    
    # Invert Fy (channel 5)
    state_flip[5] = -state_flip[5]
    
    return state_flip, policy_flip


def random_pad_to_target(
    state: np.ndarray,
    policy: np.ndarray,
    divisor: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad tensor to target size (divisible by divisor) with random positioning.
    
    Instead of always padding on the right (deterministic), this places
    the original structure at a random position within the padded volume.
    
    This teaches translational invariance - the network cannot rely
    on the structure being in a fixed corner.
    
    Args:
        state: (C, D, H, W) tensor
        policy: (2, D, H, W) tensor
        divisor: Target dimensions will be ceil to nearest multiple of this
        
    Returns:
        Padded (state, policy) tuple with random positioning
    """
    C, D, H, W = state.shape
    
    # Calculate target dimensions (next multiple of divisor)
    def next_multiple(x, div):
        return ((x + div - 1) // div) * div
    
    target_d = next_multiple(D, divisor)
    target_h = next_multiple(H, divisor)
    target_w = next_multiple(W, divisor)
    
    # Calculate total padding needed
    pad_d = target_d - D
    pad_h = target_h - H
    pad_w = target_w - W
    
    # If no padding needed, return as-is
    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return state, policy
    
    # Randomly distribute padding between left and right for each dimension
    pad_d_left = np.random.randint(0, pad_d + 1) if pad_d > 0 else 0
    pad_d_right = pad_d - pad_d_left
    
    pad_h_left = np.random.randint(0, pad_h + 1) if pad_h > 0 else 0
    pad_h_right = pad_h - pad_h_left
    
    pad_w_left = np.random.randint(0, pad_w + 1) if pad_w > 0 else 0
    pad_w_right = pad_w - pad_w_left
    
    # Create padded arrays
    state_padded = np.zeros((C, target_d, target_h, target_w), dtype=state.dtype)
    policy_padded = np.zeros((2, target_d, target_h, target_w), dtype=policy.dtype)
    
    # Place original data at random position
    state_padded[
        :,
        pad_d_left:pad_d_left + D,
        pad_h_left:pad_h_left + H,
        pad_w_left:pad_w_left + W
    ] = state
    
    policy_padded[
        :,
        pad_d_left:pad_d_left + D,
        pad_h_left:pad_h_left + H,
        pad_w_left:pad_w_left + W
    ] = policy
    
    return state_padded, policy_padded


def erosion_attack(
    state: np.ndarray,
    policy: np.ndarray,
    value: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Apply erosion to density channel (Negative Sampling).
    
    Used for final states to teach the network to detect
    structures that are too thin.
    
    Args:
        state: (7, D, H, W) tensor
        policy: (2, D, H, W) tensor
        value: Original fitness value
        
    Returns:
        (eroded_state, repair_policy, -1.0) tuple
    """
    state_eroded = state.copy()
    
    # Get density channel (binary for erosion)
    density = state[0] > 0.5
    
    # Apply binary erosion
    density_eroded = ndimage.binary_erosion(density, structure=EROSION_KERNEL)
    state_eroded[0] = density_eroded.astype(np.float32)
    
    # Policy: Add where eroded (to repair)
    diff = density.astype(np.float32) - density_eroded.astype(np.float32)
    policy_repair = np.zeros_like(policy)
    policy_repair[0] = np.maximum(0, diff)  # Add channel
    policy_repair[1] = 0  # Remove channel (nothing to remove)
    
    return state_eroded, policy_repair, -1.0


def load_multiplier(
    state: np.ndarray,
    policy: np.ndarray,
    value: float,
    k: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Multiply force vectors by factor k (stress test).
    
    Simulates overloading the structure.
    
    Args:
        state: (7, D, H, W) tensor
        policy: (2, D, H, W) tensor
        value: Original fitness value
        k: Force multiplication factor
        
    Returns:
        (stressed_state, add_policy, -0.8) tuple
    """
    state_stressed = state.copy()
    
    # Multiply force channels (4, 5, 6) by k
    state_stressed[4] *= k  # Fx
    state_stressed[5] *= k  # Fy
    state_stressed[6] *= k  # Fz
    
    # Policy: Reinforce near load points
    # Find where forces are applied
    force_mask = (np.abs(state[4]) + np.abs(state[5]) + np.abs(state[6])) > 0.1
    
    # Dilate force region to suggest reinforcement area
    reinforce_region = ndimage.binary_dilation(force_mask, structure=EROSION_KERNEL, iterations=2)
    
    policy_reinforce = np.zeros_like(policy)
    policy_reinforce[0] = reinforce_region.astype(np.float32)  # Add around load
    
    return state_stressed, policy_reinforce, -0.8


def sabotage(
    state: np.ndarray,
    policy: np.ndarray,
    value: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Remove voxels at critical connection points.
    
    Targets nodes near the support (X=0) to create disconnection.
    
    Args:
        state: (7, D, H, W) tensor
        policy: (2, D, H, W) tensor
        value: Original fitness value
        
    Returns:
        (sabotaged_state, repair_policy, -1.0) tuple
    """
    state_sab = state.copy()
    density = state[0].copy()
    
    # Find material near support (X < 5)
    D, H, W = density.shape
    support_region = np.zeros_like(density, dtype=bool)
    support_region[:min(5, D), :, :] = True
    
    # Remove material in support region
    material_near_support = (density > 0.5) & support_region
    
    # Remove a portion (50%)
    remove_mask = material_near_support & (np.random.random(density.shape) < 0.5)
    
    state_sab[0] = np.where(remove_mask, 0.0, density)
    
    # Policy: Repair removed voxels
    policy_repair = np.zeros_like(policy)
    policy_repair[0] = remove_mask.astype(np.float32)  # Add what was removed
    
    return state_sab, policy_repair, -1.0


def saboteur(
    state: np.ndarray,
    policy: np.ndarray,
    value: float,
    cube_size: int = 3
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Remove a random cube of voxels to disconnect load path.
    
    Args:
        state: (7, D, H, W) tensor
        policy: (2, D, H, W) tensor
        value: Original fitness value
        cube_size: Size of cube to remove
        
    Returns:
        (sabotaged_state, repair_policy, -1.0) tuple
    """
    state_sab = state.copy()
    density = state[0].copy()
    D, H, W = density.shape
    
    # Find voxels with material
    material_coords = np.argwhere(density > 0.5)
    
    if len(material_coords) == 0:
        return state, policy, value
    
    # Pick random material voxel as center
    idx = np.random.randint(0, len(material_coords))
    cx, cy, cz = material_coords[idx]
    
    # Define cube bounds
    half = cube_size // 2
    x_min, x_max = max(0, cx - half), min(D, cx + half + 1)
    y_min, y_max = max(0, cy - half), min(H, cy + half + 1)
    z_min, z_max = max(0, cz - half), min(W, cz + half + 1)
    
    # Create removal mask
    remove_mask = np.zeros_like(density, dtype=bool)
    remove_mask[x_min:x_max, y_min:y_max, z_min:z_max] = True
    remove_mask = remove_mask & (density > 0.5)  # Only remove material
    
    # Apply removal
    state_sab[0] = np.where(remove_mask, 0.0, density)
    
    # Policy: Repair removed voxels
    policy_repair = np.zeros_like(policy)
    policy_repair[0] = remove_mask.astype(np.float32)
    
    return state_sab, policy_repair, -1.0

