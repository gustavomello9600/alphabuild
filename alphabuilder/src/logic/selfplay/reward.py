"""
Reward Function Module for AlphaBuilder Self-Play MCTS.

Implements an additive reward formula:
  R = compliance_score(C) + volume_bonus(V) - island_penalty

Where:
- Compliance score uses calibrated v3.3 log10 mapping: C=100→0.80, C=1000→0.45, C>=1e6→-0.60
- Volume is ALWAYS inversely proportional: lower V = higher reward
- Island penalty is applied as before

Reference: Calibrated on v3.3 100-episode Pilot (Nov 2026)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from scipy import ndimage

from ..harvest.generators import get_load_region_bounds

# =============================================================================
# New Formula Constants (Calibrated v3.3)
# =============================================================================

# Compliance scoring (log10 mapping, calibrated on Pilot Phase-2 p10..p90=[2.30, 3.84])
COMPLIANCE_BASE = 1.15       # Calibrated v3.3: C=100 (log10=2) -> 0.80, C=1000 (log10=3) -> 0.45
COMPLIANCE_SLOPE = 0.35      # Score decrease per log10 unit
COMPLIANCE_MIN = -0.60       # Minimum compliance score (C >= 1e6)
COMPLIANCE_MAX = 0.90        # Maximum compliance score (very stiff structures)

# Volume scoring (always inversely proportional)
VOLUME_REFERENCE = 0.10      # Reference volume (V=0.10 gives no bonus/penalty)
VOLUME_SENSITIVITY = 2.5     # Bonus/penalty per 0.1 volume change (V=0.15 -> -0.125)
VOLUME_BONUS_MAX = 0.30      # Max bonus for very low V
VOLUME_PENALTY_MAX = 0.60    # Max penalty for very high V

# Legacy constants (kept for backward compatibility in estimate_reward_components)
MU_SCORE = -6.65
SIGMA_SCORE = 2.0
ALPHA_VOL = 12.0
EPSILON = 1e-9

# Displacement limits for collapse detection
DISPLACEMENT_LIMIT_RAW = 128000.0  # (L/250) * 500,000 = 0.256 * 500k = 128k. Allows weak but connected structures.

# Physics Scale Factor (User defined Realism)
INITIAL_CANTILEVER_PROBLEM_REAL_SCALE_FACTOR = 500000.0


# =============================================================================
# Core Reward Functions
# =============================================================================

def calculate_compliance_score(compliance: float) -> float:
    """
    Calculate the compliance component of the reward using calibrated log10 mapping.

    Mapping (v3.3):
    - C=100 (log10=2) → 0.80
    - C=1,000 (log10=3) → 0.45
    - C=1,000,000 (log10=6) → -0.60

    Args:
        compliance: Energy of deformation (Joules). Lower is better.

    Returns:
        Compliance score in [COMPLIANCE_MIN, COMPLIANCE_MAX]
    """
    # log10 mapping
    if compliance <= 0: 
        return float(COMPLIANCE_MAX)
        
    log_c = np.log10(max(compliance, 1.0))
    score = COMPLIANCE_BASE - COMPLIANCE_SLOPE * (log_c - 1.0)
    
    return float(np.clip(score, COMPLIANCE_MIN, COMPLIANCE_MAX))


def calculate_volume_bonus(vol_frac: float) -> float:
    """
    Calculate volume bonus (inversely proportional).
    
    Args:
        vol_frac: Volume fraction (0.0 to 1.0)
        
    Returns:
        Bonus/Penalty value
    """
    adjustment = (VOLUME_REFERENCE - vol_frac) * VOLUME_SENSITIVITY
    return float(np.clip(adjustment, -VOLUME_PENALTY_MAX, VOLUME_BONUS_MAX))


def calculate_raw_score(
    compliance: float,
    vol_frac: float
) -> float:
    """
    Calculate raw score before clipping (Compliance + Volume).
    
    Args:
        compliance: Energy of deformation
        vol_frac: Volume fraction
        
    Returns:
        Unclipped raw score
    """
    c_score = calculate_compliance_score(compliance)
    v_bonus = calculate_volume_bonus(vol_frac)
    return c_score + v_bonus


def reverse_engineer_compliance(fitness_score: float, vol_frac: float) -> float:
    """
    Back-calculate compliance C from legacy fitness score.
    Used for recalculating targets from old data or mixed phase targets.
    
    Args:
        fitness_score: Legacy fitness score [-1, 1]
        vol_frac: Volume fraction
        
    Returns:
        Estimated Compliance (Energy)
    """
    fitness = np.clip(fitness_score, -0.999999, 0.999999)
    # S_fem = tanh((S_raw - mu) / sigma) -> raw = artanh(S)*sigma + mu
    z = np.arctanh(fitness)
    s_raw = z * SIGMA_SCORE + MU_SCORE
    # S_raw = -ln(C) - alpha * V -> C = exp(-raw - alpha*V)
    log_c_natural = -s_raw - (ALPHA_VOL * vol_frac)
    return float(np.exp(log_c_natural))


def calculate_fem_score(
    compliance: float,
    vol_frac: float
) -> float:
    """
    Calculate the PURE physics score (Compliance + Volume).
    
    Target for Value Head Training.
    Ignores game penalties (collapse, connectivity).
    
    Args:
        compliance: Energy of deformation
        vol_frac: Volume fraction
        
    Returns:
        Score in [-1.0, 1.0] representing physical quality.
    """
    # 1. Calculate compliance score (log10 mapping)
    compliance_score = calculate_compliance_score(compliance)
    
    # 2. Calculate volume bonus
    volume_bonus = calculate_volume_bonus(vol_frac)
    
    # 3. Combine
    score = compliance_score + volume_bonus
    
    return float(np.clip(score, -1.0, 1.0))


def calculate_displacement_penalty(
    max_displacement: Optional[float],
    displacement_limit: float = DISPLACEMENT_LIMIT_RAW,
) -> float:
    """
    Continuous displacement warning penalty starting at 70% of collapse threshold
    (Hayashi & Ohsaki, 2020). Zero below 0.70 * limit, ramping smoothly to 0.50 at 1.00 * limit.
    """
    if max_displacement is None or displacement_limit <= 0.0:
        return 0.0
    ratio = float(max_displacement) / float(displacement_limit)
    if ratio <= 0.70:
        return 0.0
    return float(0.50 * np.clip((ratio - 0.70) / 0.30, 0.0, 1.0))


def calculate_reward(
    compliance: float,
    vol_frac: float,
    is_valid: bool,
    max_displacement: Optional[float] = None,
    displacement_limit: float = DISPLACEMENT_LIMIT_RAW
) -> float:
    """
    Calculate the GAME REWARD for MCTS.
    
    Includes Game Rules (Penalties):
    - Invalid/Disconnected -> -1.0
    - Collapse (Displacement > Limit) -> -1.0
    - Smooth pre-collapse warning penalty when U_max > 0.70 * U_lim -> [-0.95, +1.0]
    """
    # 1. Hard constraint: structural validity
    if not is_valid:
        return -1.0
    
    # 2. Hard constraint: collapse detection
    if max_displacement is not None and max_displacement > displacement_limit:
        return -1.0
    
    # 3. Pure physics score minus smooth displacement margin warning
    score = calculate_fem_score(compliance, vol_frac)
    p_disp = calculate_displacement_penalty(max_displacement, displacement_limit)
    if p_disp > 0.0:
        return float(np.clip(score - p_disp, -0.95, 1.0))
    return score


def calculate_grace_displacement_reward(
    max_displacement: Optional[float],
    displacement_limit: float = DISPLACEMENT_LIMIT_RAW,
) -> float:
    """
    Smooth penalty returned during post-connection Phase-2 grace period when a thin
    bridge connects support to load (`u_max > u_lim`) so MCTS gets gradient signal to thicken it.
    """
    if max_displacement is None or displacement_limit <= 0.0:
        return -0.90
    ratio = max(1.0, float(max_displacement) / float(displacement_limit))
    overshoot = min(1.0, float(np.log10(ratio)) / 2.0)
    return float(np.clip(-0.85 - 0.13 * overshoot, -0.98, -0.85))


# =============================================================================
# Phase-Specific Reward Functions
# =============================================================================

def get_phase1_reward(
    connected_to_support: bool,
    reached_load: bool,
    step: int,
    max_steps: int,
    living_penalty: float = -0.01
) -> Tuple[float, str]:
    """
    Calculate reward for Phase 1 (Growth).
    
    In Phase 1, physics is invalid (structure disconnected), so we cannot
    calculate compliance. The agent is guided purely by V_net predictions.
    
    Args:
        connected_to_support: Whether structure connects to boundary (X=0)
        reached_load: Whether structure reaches the load application region
        step: Current step number
        max_steps: Maximum allowed steps in Phase 1
        living_penalty: Small negative reward per step to encourage efficiency
        
    Returns:
        Tuple of (reward, reason_string)
    """
    # Success: Connected support to load
    if connected_to_support and reached_load:
        return 0.5, "CONNECTION_SUCCESS"
    
    # Failure: Max steps reached without connection
    if step >= max_steps:
        return -1.0, "MAX_STEPS_REACHED"
    
    # Normal step: Small living penalty, rely on V_net
    return living_penalty, "CONTINUE"


def check_structure_connectivity(
    density: np.ndarray,
    load_config: Dict[str, Any],
    threshold: float = 0.5,
    bc_masks: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    forces: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[bool, bool]:
    """
    Check if structure connects support (`bc_masks` or X=0) to the load region/patches.
    Supports both legacy single-load cantilevers and arbitrary 3D multi-support/multi-load domains.
    """
    binary = density > threshold
    labeled, n_components = ndimage.label(binary)
    if n_components == 0:
        return False, False

    if bc_masks is not None:
        support_mask = (bc_masks[0] > 0.5) | (bc_masks[1] > 0.5) | (bc_masks[2] > 0.5)
        support_labels = (
            np.unique(labeled[support_mask]) if np.any(support_mask) else np.unique(labeled[0, :, :])
        )
    else:
        support_labels = np.unique(labeled[0, :, :])
    support_labels = support_labels[support_labels > 0]

    if len(support_labels) == 0:
        return False, False

    connected_to_support = True

    if forces is not None:
        force_mag = np.abs(forces[0]) + np.abs(forces[1]) + np.abs(forces[2])
        load_mask = force_mag > 1e-6
        if not np.any(load_mask):
            return connected_to_support, False
        if not np.all(binary[load_mask]):
            return connected_to_support, False
        present_labels = np.unique(labeled[load_mask])
        present_labels = present_labels[present_labels > 0]
        if len(present_labels) == 0:
            return connected_to_support, False
        reached_load = bool(np.all(np.isin(present_labels, support_labels)))
        return connected_to_support, reached_load

    x_idx, y_min, y_max, z_min, z_max = get_load_region_bounds(density.shape, load_config)
    load_slice_labels = labeled[x_idx, y_min:y_max, z_min:z_max]
    present_labels = np.unique(load_slice_labels)
    present_labels = present_labels[present_labels > 0]

    if len(present_labels) == 0:
        return connected_to_support, False

    valid_support_labels = np.intersect1d(support_labels, present_labels)
    if len(valid_support_labels) == 0:
        return connected_to_support, False

    if len(present_labels) > 1:
        return connected_to_support, False

    label = present_labels[0]
    reached_load = label in support_labels
    expected_vol = (y_max - y_min) * (z_max - z_min)
    filled_vol = np.sum(load_slice_labels > 0)
    if filled_vol < expected_vol:
        return connected_to_support, False

    return connected_to_support, reached_load


def calculate_connectivity_reward(
    density: np.ndarray,
    bc_masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
    forces: Tuple[np.ndarray, np.ndarray, np.ndarray],
    threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Calculate reward based on connectivity to support and load.
    
    Rewards:
    - Not connected to support: 0.0
    - Connected to support: 0.1 (Base foundation reward)
    - Connected to load: Scales from 0.1 to 1.0 based on fraction of load voxels connected.
    
    Args:
        density: Density grid (D, H, W)
        bc_masks: (mask_x, mask_y, mask_z) boundary condition masks. 1.0 = fixed.
        forces: (fx, fy, fz) force vectors. Non-zero = load applied.
        threshold: Density threshold for binarization.
        
    Returns:
        Tuple of (reward_bonus, connected_load_fraction)
        - reward_bonus: Value in [0.0, 1.0]
        - connected_load_fraction: Fraction of load voxels connected to support [0.0, 1.0]
    """
    nx, ny, nz = density.shape
    binary = density > threshold
    
    # Label connected components
    labeled, n_components = ndimage.label(binary)
    
    if n_components == 0:
        return 0.0, 0.0
    
    # Identify support voxels (where BCs are applied)
    # BC masks are 1.0 where fixed.
    # We check if any component touches a fixed voxel.
    # Combine masks to find any fixed degree of freedom
    # (Usually fixed ground is enough to be "support")
    support_mask = (bc_masks[0] > 0.5) | (bc_masks[1] > 0.5) | (bc_masks[2] > 0.5)
    
    # Check if we have any support mask points at all
    if not np.any(support_mask):
         return 0.0, 0.0

    # Find labels that touch support
    support_labels_found = np.unique(labeled[support_mask])
    # Remove 0 (background)
    support_labels = support_labels_found[support_labels_found > 0]
    
    if len(support_labels) == 0:
        return 0.0, 0.0
        
    # We have a connection to support! Base reward.
    base_reward = 0.1
    
    # Identify load voxels
    # Force magnitude > 0 implies load
    force_mag = np.abs(forces[0]) + np.abs(forces[1]) + np.abs(forces[2])
    load_mask = force_mag > 1e-6
    
    # Identificar voxels de suporte conectados
    connected_mask = np.isin(labeled, support_labels)
    
    # Check which load voxels are connected to support
    # Get labels at load positions where there is material
    load_material_mask = load_mask & binary
    
    connected_to_load = False
    connected_load_voxels = 0
    total_load_voxels = np.sum(load_mask)
    
    if np.any(load_material_mask):
        labels_at_load = labeled[load_material_mask]
        is_connected = np.isin(labels_at_load, support_labels)
        connected_load_voxels = np.sum(is_connected)
        if connected_load_voxels > 0:
            connected_to_load = True

    # If not connected to load, fraction is 0, bonus is 0.
    # The previous distance heuristic is removed as per new spec.

    
    # If connected to load:
    fraction_connected = connected_load_voxels / total_load_voxels if total_load_voxels > 0 else 0
    
    # Bonus Formula:
    # 0.5 * fraction + 0.5 (if 100% connected)
    bonus = 0.5 * fraction_connected
    if fraction_connected >= 0.999: # Float tolerance
        bonus += 0.5
        
    return float(bonus), float(fraction_connected)


def get_phase2_terminal_reward(
    density: np.ndarray,
    load_config: Dict[str, Any],
    threshold: float = 0.5,
    bc_masks: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    forces: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    max_vol_frac: float = 0.65,
) -> Optional[float]:
    """
    Quick terminal check for Phase 2 (no FEM involved).

    Returns a terminal reward only for true structural failure (disconnection,
    vanishing material < 1%, or extreme solid fill > 65%), allowing normal
    continuous volume carving from >40% downward.
    """
    connected_to_support, reached_load = check_structure_connectivity(
        density, load_config, threshold, bc_masks=bc_masks, forces=forces
    )

    if not connected_to_support or not reached_load:
        return -1.0

    vol_frac = np.mean(density > threshold)
    if vol_frac < 0.01:
        return -1.0
    if vol_frac > max_vol_frac:
        return -0.5

    return None


# =============================================================================
# Reward Estimation (for Frontend Display)
# =============================================================================

def estimate_reward_components(
    value: float,
    vol_frac: float
) -> Dict[str, float]:
    """
    Estimate reward components from Value Head output and volume fraction.
    
    Used for frontend display when actual compliance is unknown.
    
    Args:
        value: Value Head output [-1, 1]
        vol_frac: Volume fraction
        
    Returns:
        Dictionary with estimated components
    """
    # Inverse the normalization to estimate raw_score
    # value = tanh((raw_score - mu) / sigma)
    # atanh(value) = (raw_score - mu) / sigma
    # raw_score = atanh(value) * sigma + mu
    
    # Clamp value to avoid atanh singularity
    clamped_value = np.clip(value, -0.999, 0.999)
    estimated_raw_score = np.arctanh(clamped_value) * SIGMA_SCORE + MU_SCORE
    
    # From raw_score = -ln(C + ε) - α*V, solve for estimated compliance
    # -ln(C + ε) = raw_score + α*V
    # C ≈ exp(-(raw_score + α*V))
    estimated_compliance = np.exp(-(estimated_raw_score + ALPHA_VOL * vol_frac))
    
    return {
        'value': float(value),
        'vol_frac': float(vol_frac),
        'estimated_raw_score': float(estimated_raw_score),
        'estimated_compliance': float(estimated_compliance),
        'alpha_vol': ALPHA_VOL,
        'mu_score': MU_SCORE,
        'sigma_score': SIGMA_SCORE,
    }


# =============================================================================
# Island Analysis (for Connectivity Penalty)
# =============================================================================

def analyze_structure_islands(
    density: np.ndarray,
    load_config: Dict[str, Any],
    threshold: float = 0.5,
    bc_masks: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    forces: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    bidirectional: bool = False,
) -> Dict[str, Any]:
    """
    Analyze connected components (islands) in a structure.

    In Phase 1 (before reaching load), voxels anchored to support do not count as
    `loose_voxels`; when `bidirectional=True`, voxels anchored to the load front also
    count as valid construction fronts rather than loose debris.
    """
    binary = density > threshold

    labeled, n_islands = ndimage.label(binary)
    if n_islands == 0:
        return {
            'n_islands': 0,
            'main_island_label': 0,
            'main_island_mask': np.zeros_like(density, dtype=bool),
            'main_island_voxels': 0,
            'loose_voxels': 0,
            'is_connected': False,
        }

    total_voxels = int(binary.sum())

    if bc_masks is not None:
        support_mask = (bc_masks[0] > 0.5) | (bc_masks[1] > 0.5) | (bc_masks[2] > 0.5)
        support_labels = set(np.unique(labeled[support_mask])) - {0} if np.any(support_mask) else (set(np.unique(labeled[0, :, :])) - {0})
    else:
        support_labels = set(np.unique(labeled[0, :, :])) - {0}

    load_labels_pre: set = set()
    if forces is not None:
        fmag_pre = np.abs(forces[0]) + np.abs(forces[1]) + np.abs(forces[2])
        lmask_pre = fmag_pre > 1e-6
        if np.any(lmask_pre):
            load_labels_pre = set(np.unique(labeled[lmask_pre])) - {0}

    if not support_labels:
        if bidirectional and load_labels_pre:
            front_mask = np.isin(labeled, list(load_labels_pre))
            front_vox = int(front_mask.sum())
            return {
                'n_islands': n_islands,
                'main_island_label': 0,
                'main_island_mask': front_mask,
                'main_island_voxels': front_vox,
                'loose_voxels': total_voxels - front_vox,
                'is_connected': False,
            }
        return {
            'n_islands': n_islands,
            'main_island_label': 0,
            'main_island_mask': np.zeros_like(density, dtype=bool),
            'main_island_voxels': 0,
            'loose_voxels': total_voxels,
            'is_connected': False,
        }

    valid_front_labels = support_labels | load_labels_pre if bidirectional else support_labels
    anchored_mask = np.isin(labeled, list(valid_front_labels))
    anchored_voxels = int(anchored_mask.sum())
    unanchored_voxels = total_voxels - anchored_voxels

    if forces is not None:
        force_mag = np.abs(forces[0]) + np.abs(forces[1]) + np.abs(forces[2])
        load_mask = force_mag > 1e-6
        if not np.any(load_mask):
            return {
                'n_islands': n_islands,
                'main_island_label': 0,
                'main_island_mask': anchored_mask,
                'main_island_voxels': anchored_voxels,
                'loose_voxels': unanchored_voxels,
                'is_connected': False,
            }
        filled_all_load = bool(np.all(binary[load_mask]))
        labels_at_load = set(np.unique(labeled[load_mask])) - {0}
        valid_islands = labels_at_load & support_labels
        if not valid_islands:
            return {
                'n_islands': n_islands,
                'main_island_label': 0,
                'main_island_mask': anchored_mask,
                'main_island_voxels': anchored_voxels,
                'loose_voxels': unanchored_voxels,
                'is_connected': False,
            }
        main_label = max(valid_islands, key=lambda lbl: int((labeled == lbl).sum()))
        main_mask = np.isin(labeled, list(valid_islands))
        main_voxels = int(main_mask.sum())
        loose_voxels = total_voxels - main_voxels
        is_conn = bool(filled_all_load and labels_at_load.issubset(support_labels))
        if not is_conn and bidirectional:
            loose_voxels = unanchored_voxels
        return {
            'n_islands': n_islands,
            'main_island_label': int(main_label),
            'main_island_mask': main_mask,
            'main_island_voxels': main_voxels,
            'loose_voxels': loose_voxels,
            'is_connected': is_conn,
        }

    x_idx, y_min, y_max, z_min, z_max = get_load_region_bounds(density.shape, load_config)
    load_slice_labels = labeled[x_idx, y_min:y_max, z_min:z_max]
    load_slice_binary = binary[x_idx, y_min:y_max, z_min:z_max]

    expected_vol = (y_max - y_min) * (z_max - z_min)
    filled_vol = np.sum(load_slice_binary)
    labels_at_load = set(np.unique(load_slice_labels)) - {0}

    if not labels_at_load:
        return {
            'n_islands': n_islands,
            'main_island_label': 0,
            'main_island_mask': anchored_mask,
            'main_island_voxels': anchored_voxels,
            'loose_voxels': unanchored_voxels,
            'is_connected': False,
        }

    valid_islands = labels_at_load & support_labels
    if not valid_islands:
        return {
            'n_islands': n_islands,
            'main_island_label': 0,
            'main_island_mask': anchored_mask,
            'main_island_voxels': anchored_voxels,
            'loose_voxels': unanchored_voxels,
            'is_connected': False,
        }

    main_label = max(valid_islands, key=lambda lbl: (labeled == lbl).sum())
    main_mask = labeled == main_label
    main_voxels = int(main_mask.sum())
    loose_voxels = total_voxels - main_voxels

    if len(labels_at_load) > 1 or filled_vol < expected_vol:
        return {
            'n_islands': n_islands,
            'main_island_label': main_label,
            'main_island_mask': main_mask,
            'main_island_voxels': main_voxels,
            'loose_voxels': loose_voxels,
            'is_connected': False,
        }

    return {
        'n_islands': n_islands,
        'main_island_label': main_label,
        'main_island_mask': main_mask,
        'main_island_voxels': main_voxels,
        'loose_voxels': loose_voxels,
        'is_connected': True,
    }


def calculate_island_penalty(
    n_islands: int,
    loose_voxels: int,
    total_voxels: int,
    penalty_per_island: float = 0.02,
    penalty_per_loose_voxel_frac: float = 0.1,
    max_penalty: float = 0.3
) -> float:
    """
    Calculate penalty for having multiple disconnected islands.
    
    The penalty is designed to:
    - Discourage loose/floating voxels
    - Not heavily penalize multiple anchor points on support (which are good)
    - Be mild enough to not override good structural decisions
    
    Args:
        n_islands: Number of connected components
        loose_voxels: Voxels not part of the main connected island
        total_voxels: Total voxels in structure
        penalty_per_island: Penalty per extra island beyond 1
        penalty_per_loose_voxel_frac: Penalty scaling with loose voxel fraction
        max_penalty: Maximum penalty cap
        
    Returns:
        Penalty value in [0, max_penalty] (to be subtracted from reward)
    """
    if n_islands <= 1 or total_voxels == 0:
        return 0.0
    
    # Penalty for extra islands (mild)
    extra_islands = n_islands - 1
    island_penalty = penalty_per_island * extra_islands
    
    # Penalty based on fraction of loose voxels (stronger signal)
    loose_fraction = loose_voxels / total_voxels
    loose_penalty = penalty_per_loose_voxel_frac * loose_fraction
    
    total_penalty = island_penalty + loose_penalty
    
    return min(total_penalty, max_penalty)


def get_reward_with_island_penalty(
    base_reward: float,
    density: np.ndarray,
    load_config: Dict[str, Any],
    threshold: float = 0.5
) -> Tuple[float, Dict[str, Any]]:
    """
    Apply island penalty to a base reward.
    
    Args:
        base_reward: The reward before island penalty (from V_net or calculate_reward)
        density: Current density grid
        load_config: Load configuration
        threshold: Binarization threshold
        
    Returns:
        Tuple of (adjusted_reward, island_analysis_dict)
    """
    analysis = analyze_structure_islands(density, load_config, threshold)
    
    total_voxels = int((density > threshold).sum())
    penalty = calculate_island_penalty(
        n_islands=analysis['n_islands'],
        loose_voxels=analysis['loose_voxels'],
        total_voxels=total_voxels
    )
    
    adjusted_reward = base_reward - penalty
    
    # Clamp to valid range
    adjusted_reward = max(-1.0, min(1.0, adjusted_reward))
    
    analysis['island_penalty'] = penalty
    analysis['base_reward'] = base_reward
    analysis['adjusted_reward'] = adjusted_reward
    
    return adjusted_reward, analysis


# =============================================================================
# Decaying Potential-Based Reward Shaping (PBRS) Helpers
# =============================================================================


def get_guidance_alpha(
    generation: int = 0,
    decay_generations: int = 5,
    initial_alpha: float = 0.30,
) -> float:
    """
    Linearly decay heuristic MCTS shaping weight across self-play generations
    (Wang et al., 2020).
    """
    if decay_generations <= 0 or generation >= decay_generations:
        return 0.0
    if generation <= 0:
        return float(initial_alpha)
    return float(initial_alpha * (1.0 - float(generation) / float(decay_generations)))


def calculate_pbrs_potential(
    density: np.ndarray,
    bc_masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
    forces: Tuple[np.ndarray, np.ndarray, np.ndarray],
    load_config: Dict[str, Any],
    w_conn: float = 0.50,
    w_dist: float = 0.25,
    threshold: float = 0.5,
    bidirectional: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute state potential Phi(s) = w_conn * conn_frac + dist_guidance - island_penalty
    for Potential-Based Reward Shaping (Muller & Kudenko, 2025).
    `dist_guidance` smoothly rewards shortening the Euclidean gap from anchored support
    material to the load front (load patch or load-anchored frontier when `bidirectional=True`).
    """
    _, conn_frac = calculate_connectivity_reward(density, bc_masks, forces, threshold=threshold)
    analysis = analyze_structure_islands(
        density, load_config, threshold=threshold, bc_masks=bc_masks, forces=forces, bidirectional=bidirectional
    )
    total_voxels = int((density > threshold).sum())
    eff_islands = (
        1
        if (bidirectional and not analysis['is_connected'] and analysis['loose_voxels'] == 0)
        else analysis['n_islands']
    )
    island_pen = calculate_island_penalty(
        eff_islands,
        analysis['loose_voxels'],
        total_voxels,
    )

    dist_bonus = 0.0
    if w_dist > 0.0 and conn_frac < 0.999 and total_voxels > 0:
        binary = density > threshold
        labeled, n_comp = ndimage.label(binary)
        if n_comp > 0:
            support_mask = (bc_masks[0] > 0.5) | (bc_masks[1] > 0.5) | (bc_masks[2] > 0.5)
            sup_lbls = set(np.unique(labeled[support_mask])) - {0} if np.any(support_mask) else set()
            force_mag = np.abs(forces[0]) + np.abs(forces[1]) + np.abs(forces[2])
            load_mask = force_mag > 1e-6
            if sup_lbls and np.any(load_mask):
                anchored_sup = np.isin(labeled, list(sup_lbls))
                unconn_load = load_mask & (~anchored_sup)
                target_front = unconn_load
                if bidirectional:
                    load_lbls = set(np.unique(labeled[load_mask])) - {0} - sup_lbls
                    if load_lbls:
                        anchored_load = np.isin(labeled, list(load_lbls))
                        target_front = anchored_load ^ ndimage.binary_erosion(anchored_load)
                        if not np.any(target_front):
                            target_front = anchored_load
                if np.any(target_front):
                    surface = anchored_sup ^ ndimage.binary_erosion(anchored_sup)
                    anc_pts = np.argwhere(surface if np.any(surface) else anchored_sup)
                    if len(anc_pts) > 512:
                        step = max(1, len(anc_pts) // 512)
                        anc_pts = anc_pts[::step]
                    tgt_pts = np.argwhere(target_front)
                    if len(tgt_pts) > 512:
                        step_t = max(1, len(tgt_pts) // 512)
                        tgt_pts = tgt_pts[::step_t]
                    diff = anc_pts[:, None, :].astype(np.float32) - tgt_pts[None, :, :].astype(np.float32)
                    min_dist = float(np.sqrt(np.min(np.sum(diff * diff, axis=-1))))
                    diag = float(np.sqrt(sum(s * s for s in density.shape)))
                    dist_bonus = float(w_dist * max(0.0, 1.0 - min_dist / max(diag, 1.0)))

    phi = float(w_conn * conn_frac + dist_bonus - island_pen)
    analysis['island_penalty'] = island_pen
    analysis['connected_load_fraction'] = float(conn_frac)
    analysis['distance_bonus'] = dist_bonus
    analysis['potential'] = phi
    return phi, analysis
