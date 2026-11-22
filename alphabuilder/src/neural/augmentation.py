"""
Data Augmentation for AlphaBuilder v3.3.

Implements full 3D rigid symmetry transformations (D4 x Z2 = 16x equivariance)
and physics-consistent negative sampling perturbations:
- Rigid symmetries: flip_x, flip_y, flip_z (8x per-sample) + rotate_90_z (2x per-batch)
- Negative sampling: erosion_attack, load_multiplier, sabotage, saboteur
"""

from typing import Optional, Tuple, Union
import numpy as np
from scipy import ndimage

# Pre-computed 3D 6-connectivity cross kernel
EROSION_KERNEL = ndimage.generate_binary_structure(3, 1)


def _flip_spatial(arr: np.ndarray, spatial_axis_4d: int) -> np.ndarray:
    """Flip a 4D (C, D, H, W) or 3D (D, H, W) array along given 4D spatial axis."""
    axis = spatial_axis_4d if arr.ndim == 4 else (spatial_axis_4d - 1)
    return np.flip(arr, axis=axis).copy()


def rotate_90_z(
    state: np.ndarray,
    policy: np.ndarray,
    blueprint: Optional[np.ndarray] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Rotate state, policy (and optional blueprint) 90 deg around Z axis (axes=(1, 2)).

    Under `np.rot90(..., k=1, axes=(1, 2))`, coordinates map (x, y) -> (H - 1 - y, x),
    so spatial direction vectors transform cleanly as:
    - (Fx, Fy) -> (-Fy, +Fx)
    - (Mask_X, Mask_Y) -> (Mask_Y, Mask_X)
    """
    state_rot = np.rot90(state, k=1, axes=(1, 2)).copy()
    policy_rot = np.rot90(policy, k=1, axes=(1, 2)).copy()

    # Rotate force vectors (channels 4, 5 = Fx, Fy)
    old_fx = state_rot[4].copy()
    old_fy = state_rot[5].copy()
    state_rot[4] = -old_fy
    state_rot[5] = old_fx

    # Swap BC mask axes without sign change (channels 1, 2 = Mask_X, Mask_Y)
    old_mx = state_rot[1].copy()
    old_my = state_rot[2].copy()
    state_rot[1] = old_my
    state_rot[2] = old_mx

    if blueprint is not None:
        axes = (1, 2) if blueprint.ndim == 4 else (0, 1)
        bp_rot = np.rot90(blueprint, k=1, axes=axes).copy()
        return state_rot, policy_rot, bp_rot

    return state_rot, policy_rot


def flip_x(
    state: np.ndarray,
    policy: np.ndarray,
    blueprint: Optional[np.ndarray] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Mirror along X (depth) axis and negate Fx (channel 4)."""
    state_flip = _flip_spatial(state, 1)
    policy_flip = _flip_spatial(policy, 1)
    state_flip[4] = -state_flip[4]

    if blueprint is not None:
        return state_flip, policy_flip, _flip_spatial(blueprint, 1)
    return state_flip, policy_flip


def flip_y(
    state: np.ndarray,
    policy: np.ndarray,
    blueprint: Optional[np.ndarray] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Mirror along Y (height) axis and negate Fy (channel 5)."""
    state_flip = _flip_spatial(state, 2)
    policy_flip = _flip_spatial(policy, 2)
    state_flip[5] = -state_flip[5]

    if blueprint is not None:
        return state_flip, policy_flip, _flip_spatial(blueprint, 2)
    return state_flip, policy_flip


def flip_z(
    state: np.ndarray,
    policy: np.ndarray,
    blueprint: Optional[np.ndarray] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Mirror along Z (width) axis and negate Fz (channel 6)."""
    state_flip = _flip_spatial(state, 3)
    policy_flip = _flip_spatial(policy, 3)
    state_flip[6] = -state_flip[6]

    if blueprint is not None:
        return state_flip, policy_flip, _flip_spatial(blueprint, 3)
    return state_flip, policy_flip


def random_pad_to_target(
    state: np.ndarray,
    policy: np.ndarray,
    divisor: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy helper kept for compatibility; v3.3 passes unpadded (64, 32, 8) states
    and performs centered padding inside AlphaBuilderV31.forward().
    """
    C, D, H, W = state.shape

    def next_multiple(x: int, div: int) -> int:
        return ((x + div - 1) // div) * div

    target_d = next_multiple(D, divisor)
    target_h = next_multiple(H, divisor)
    target_w = next_multiple(W, divisor)

    pad_d = target_d - D
    pad_h = target_h - H
    pad_w = target_w - W

    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return state, policy

    pad_d_left = np.random.randint(0, pad_d + 1) if pad_d > 0 else 0
    pad_h_left = np.random.randint(0, pad_h + 1) if pad_h > 0 else 0
    pad_w_left = np.random.randint(0, pad_w + 1) if pad_w > 0 else 0

    state_padded = np.zeros((C, target_d, target_h, target_w), dtype=state.dtype)
    policy_padded = np.zeros((2, target_d, target_h, target_w), dtype=policy.dtype)

    state_padded[
        :,
        pad_d_left : pad_d_left + D,
        pad_h_left : pad_h_left + H,
        pad_w_left : pad_w_left + W,
    ] = state

    policy_padded[
        :,
        pad_d_left : pad_d_left + D,
        pad_h_left : pad_h_left + H,
        pad_w_left : pad_w_left + W,
    ] = policy

    return state_padded, policy_padded


def erosion_attack(
    state: np.ndarray,
    policy: np.ndarray,
    value: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply 3D morphological erosion to teach the network repair of thinned members."""
    density = state[0] > 0.5
    density_eroded = ndimage.binary_erosion(density, structure=EROSION_KERNEL)
    diff = density.astype(np.float32) - density_eroded.astype(np.float32)

    if not np.any(diff > 0):
        return state, policy, value

    state_eroded = state.copy()
    state_eroded[0] = density_eroded.astype(np.float32)

    policy_repair = np.zeros_like(policy)
    policy_repair[0] = np.maximum(0.0, diff)
    policy_repair[1] = 0.0
    return state_eroded, policy_repair, -1.0


def load_multiplier(
    state: np.ndarray,
    policy: np.ndarray,
    value: float,
    k: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Multiply force vectors by factor k and suggest legal reinforcement around loaded boundary.
    Never puts ADD policy targets on already-solid voxels.
    """
    force_mag = np.abs(state[4]) + np.abs(state[5]) + np.abs(state[6])
    force_mask = force_mag > 0.1
    if not np.any(force_mask):
        return state, policy, value

    solid = state[0] > 0.5
    valid_add = ndimage.binary_dilation(solid, structure=EROSION_KERNEL) & (~solid)
    reinforce_region = ndimage.binary_dilation(force_mask, structure=EROSION_KERNEL, iterations=2)

    target_add = reinforce_region & valid_add
    if not np.any(target_add):
        target_add = reinforce_region & (~solid)
    if not np.any(target_add):
        return state, policy, value

    state_stressed = state.copy()
    state_stressed[4] *= k
    state_stressed[5] *= k
    state_stressed[6] *= k

    policy_reinforce = np.zeros_like(policy)
    policy_reinforce[0] = target_add.astype(np.float32)
    return state_stressed, policy_reinforce, -0.8


def sabotage(
    state: np.ndarray,
    policy: np.ndarray,
    value: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Remove material near supports (BC masks or fallback X < 5) to teach disconnection repair.
    Returns unchanged inputs if no near-support material can be removed.
    """
    density = state[0]
    D = density.shape[0]
    bc_support = (state[1] > 0.5) | (state[2] > 0.5) | (state[3] > 0.5)
    if np.any(bc_support):
        support_region = ndimage.binary_dilation(bc_support, structure=EROSION_KERNEL, iterations=4)
    else:
        support_region = np.zeros_like(density, dtype=bool)
        support_region[: min(5, D), :, :] = True

    material_near_support = (density > 0.5) & support_region
    if not np.any(material_near_support):
        return state, policy, value

    remove_mask = material_near_support & (np.random.random(density.shape) < 0.5)
    if not np.any(remove_mask):
        # Guarantee non-empty removal when eligible material exists
        coords = np.argwhere(material_near_support)
        pick = coords[np.random.randint(0, len(coords))]
        remove_mask = np.zeros_like(density, dtype=bool)
        remove_mask[tuple(pick)] = True

    state_sab = state.copy()
    state_sab[0] = np.where(remove_mask, 0.0, density)

    policy_repair = np.zeros_like(policy)
    policy_repair[0] = remove_mask.astype(np.float32)
    return state_sab, policy_repair, -1.0


def saboteur(
    state: np.ndarray,
    policy: np.ndarray,
    value: float,
    cube_size: int = 3,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Remove a random cube of material to simulate localized fracture and teach repair.
    Returns unchanged inputs if no material exists.
    """
    density = state[0]
    D, H, W = density.shape
    material_coords = np.argwhere(density > 0.5)
    if len(material_coords) == 0:
        return state, policy, value

    idx = np.random.randint(0, len(material_coords))
    cx, cy, cz = material_coords[idx]

    half = cube_size // 2
    x_min, x_max = max(0, cx - half), min(D, cx + half + 1)
    y_min, y_max = max(0, cy - half), min(H, cy + half + 1)
    z_min, z_max = max(0, cz - half), min(W, cz + half + 1)

    remove_mask = np.zeros_like(density, dtype=bool)
    remove_mask[x_min:x_max, y_min:y_max, z_min:z_max] = True
    remove_mask = remove_mask & (density > 0.5)
    if not np.any(remove_mask):
        return state, policy, value

    state_sab = state.copy()
    state_sab[0] = np.where(remove_mask, 0.0, density)

    policy_repair = np.zeros_like(policy)
    policy_repair[0] = remove_mask.astype(np.float32)
    return state_sab, policy_repair, -1.0
