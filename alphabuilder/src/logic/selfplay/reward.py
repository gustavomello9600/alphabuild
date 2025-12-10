"""
Reward Function Module for AlphaBuilder Self-Play MCTS.

Implements the reward function specification that mirrors the training normalization.
This ensures mathematical consistency between the Value Head predictions and MCTS rewards.

Reference: MCTS Spec Section 7 (Reward Function Specification)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from scipy import ndimage

# =============================================================================
# Normalization Constants (Mirrored from Training)
# =============================================================================
# These MUST match the constants in alphabuilder/src/logic/harvest/config.py

MU_SCORE = -6.65        # Mean of raw score distribution
SIGMA_SCORE = 2.0       # Std dev for Tanh smoothing
ALPHA_VOL = 12.0        # Volume penalty weight (20% more than compliance)
EPSILON = 1e-9          # Numerical stability

# Displacement limits for collapse detection
DEFAULT_DISPLACEMENT_LIMIT = 10.0  # meters (very generous for small structures)


# =============================================================================
# Core Reward Functions
# =============================================================================

def calculate_raw_score(compliance: float, vol_frac: float) -> float:
    """
    Calculate the raw score before normalization.
    
    S_raw = -ln(C + ε) - α·V_frac
    
    Args:
        compliance: Energy of deformation (Joules). Lower is better.
        vol_frac: Volume fraction (0.0 to 1.0). Lower is better.
        
    Returns:
        Raw score (higher is better after log inversion)
    """
    return -np.log(compliance + EPSILON) - ALPHA_VOL * vol_frac


def calculate_reward(
    compliance: float,
    vol_frac: float,
    is_valid: bool,
    max_displacement: Optional[float] = None,
    displacement_limit: float = DEFAULT_DISPLACEMENT_LIMIT
) -> float:
    """
    Calculate the normalized reward for MCTS.
    
    R = tanh((S_raw - μ) / σ)
    
    This function mirrors the normalization used during training to ensure
    the MCTS speaks the same language as the Value Head.
    
    Args:
        compliance: Energy of deformation (Joules)
        vol_frac: Volume fraction (0.0 to 1.0)
        is_valid: Whether the physical solution is valid (structure connected, FEM solved)
        max_displacement: Maximum displacement in the structure (optional)
        displacement_limit: Threshold for collapse detection
        
    Returns:
        Normalized reward in range [-1, 1]
        - Returns -1.0 for catastrophic failures (collapse, disconnection)
        - Returns 0.0 for average structures (raw_score ≈ μ)
        - Returns positive for better structures
    """
    # 1. Hard constraint: structural validity
    if not is_valid:
        return -1.0
    
    # 2. Hard constraint: collapse detection
    if max_displacement is not None and max_displacement > displacement_limit:
        return -1.0
    
    # 3. Calculate raw score
    raw_score = calculate_raw_score(compliance, vol_frac)
    
    # 4. Normalize with Tanh
    normalized_reward = np.tanh((raw_score - MU_SCORE) / SIGMA_SCORE)
    
    return float(normalized_reward)


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
    threshold: float = 0.5
) -> Tuple[bool, bool]:
    """
    Check if structure connects support to load region.
    
    This is used for Phase 1 -> Phase 2 transition detection.
    
    Args:
        density: Density grid (D, H, W)
        load_config: Dictionary with 'x', 'y', 'z_start', 'z_end' keys
        threshold: Binarization threshold
        
    Returns:
        Tuple of (connected_to_support, reached_load)
    """
    nx, ny, nz = density.shape
    
    # Binarize
    binary = density > threshold
    
    # Label connected components
    labeled, n_components = ndimage.label(binary)
    
    if n_components == 0:
        return False, False
    
    # Check support connection (X=0 plane)
    support_labels = np.unique(labeled[0, :, :])
    support_labels = support_labels[support_labels > 0]
    
    if len(support_labels) == 0:
        return False, False
    
    connected_to_support = True
    
    # Check load region connection
    lx = min(load_config.get('x', nx-1), nx-1)
    ly = min(load_config.get('y', ny//2), ny-1)
    lz_s = max(0, load_config.get('z_start', 0))
    lz_e = min(nz, load_config.get('z_end', nz))
    
    # Identify voxels with active load
    # In generate_random_load_config/forces, we apply load at:
    # x=lx, y=ly+/-1, z=lz_s to lz_e
    # Need to match exact logic from create_empty_state to know where force IS.
    
    # Reconstruct load region bounds (matching forces logic in runner.py)
    load_half_width = 1.0
    y_min = max(0, int(ly - load_half_width))
    y_max = min(ny, int(ly + load_half_width) + 1)
    z_min = max(0, int((lz_s + lz_e)/2.0 - load_half_width))
    z_max = min(nz, int((lz_s + lz_e)/2.0 + load_half_width) + 1)
    x_idx = lx
    
    # Get labels exactly at load voxels
    load_slice_labels = labeled[x_idx, y_min:y_max, z_min:z_max]
    
    # Get unique labels in load region that are not background (0)
    present_labels = np.unique(load_slice_labels)
    present_labels = present_labels[present_labels > 0]
    
    if len(present_labels) == 0:
        return connected_to_support, False

    # Stricter Check:
    # 1. All loaded voxels must be occupied (no gaps under load) -> IMPLICIT via present_labels?
    # No, effectively we check if 'forces' are applied to material.
    # But here we just check if the "load region" has material and is connected.
    
    # 2. All occupied load voxels must belong to a support-connected component
    # Find labels that touch support
    valid_support_labels = np.intersect1d(support_labels, present_labels)
    
    if len(valid_support_labels) == 0:
        return connected_to_support, False
    
    # Check if ALL present load voxels belong to valid support labels
    # i.e., are there any islands in load region NOT connected to support?
    # Ideally should be a SINGLE component.
    
    # Requirement: "all points of application are connected to supports by the SAME island"
    # So we should find ONE label that covers ALL occupied load points? 
    # Or just that all occupied load points are connected labels?
    # "By A SAME island" implies 1 component.
    
    if len(present_labels) > 1:
        # Multiple disjoint islands in load region - fail
        return connected_to_support, False
        
    # Only 1 label in load region. Is it connected to support?
    label = present_labels[0]
    reached_load = label in support_labels
    
    # Also verify coverage? 
    # If load implies 9 voxels, but we only have 1 voxel filled, is that "reached load"?
    # The prompt says "all points of application ... connected".
    # This implies all points where force is applied MUST have material AND be connected.
    # So we check if load region is fully filled?
    
    # Count expected voxels volume
    expected_vol = (y_max - y_min) * (z_max - z_min)
    filled_vol = np.sum(load_slice_labels > 0)
    
    if filled_vol < expected_vol:
        # Not fully filled under load
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
    threshold: float = 0.5
) -> Optional[float]:
    """
    Quick terminal check for Phase 2 (no FEM involved).
    
    Returns a terminal reward if the state is obviously bad (disconnected),
    otherwise returns None to indicate continuation.
    
    This is used INSIDE MCTS simulations where running FEM is too expensive.
    
    Args:
        density: Density grid (D, H, W)
        load_config: Load configuration
        threshold: Binarization threshold
        
    Returns:
        -1.0 if disconnected (terminal), None otherwise (continue)
    """
    connected_to_support, reached_load = check_structure_connectivity(
        density, load_config, threshold
    )
    
    # If removed a critical connection, game over
    if not connected_to_support or not reached_load:
        return -1.0
    
    # Volume fraction check
    vol_frac = np.mean(density > threshold)
    if vol_frac < 0.01:  # Structure too small
        return -1.0
    if vol_frac > 0.4:   # Structure too large
        return -0.5      # Penalty but not terminal
    
    return None  # Continue, not terminal


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
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Analyze the connected components (islands) in a structure.
    
    Connectivity criteria (STRICT - matches check_structure_connectivity):
    - The structure is "connected" if and only if:
      1. There exists at least one island touching both support (X=0) AND load region
      2. ALL load region voxels are filled (100% coverage)
      3. All filled load voxels belong to a SINGLE island connected to support
    
    Args:
        density: Density grid (D, H, W)
        load_config: Dictionary with 'x', 'y', 'z_start', 'z_end' keys
        threshold: Binarization threshold
        
    Returns:
        Dictionary with island analysis:
        - n_islands: Total number of connected components
        - main_island_label: Label of the main (support-to-load) island (0 if none)
        - main_island_mask: Binary mask of the main island
        - main_island_voxels: Number of voxels in main island
        - loose_voxels: Number of voxels NOT in main island
        - is_connected: Whether a valid main island exists (strict criteria)
    """
    nx, ny, nz = density.shape
    binary = density > threshold
    
    # Label all connected components
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
    
    # Find labels touching support (X=0)
    support_labels = set(np.unique(labeled[0, :, :])) - {0}
    
    if not support_labels:
        total_voxels = int(binary.sum())
        return {
            'n_islands': n_islands,
            'main_island_label': 0,
            'main_island_mask': np.zeros_like(density, dtype=bool),
            'main_island_voxels': 0,
            'loose_voxels': total_voxels,
            'is_connected': False,
        }
    
    # === Load region calculation (MUST MATCH check_structure_connectivity) ===
    lx = min(load_config.get('x', nx-1), nx-1)
    ly = min(load_config.get('y', ny//2), ny-1)
    lz_s = max(0, load_config.get('z_start', 0))
    lz_e = min(nz, load_config.get('z_end', nz))
    
    load_half_width = 1.0
    y_min = max(0, int(ly - load_half_width))
    y_max = min(ny, int(ly + load_half_width) + 1)
    z_min = max(0, int((lz_s + lz_e)/2.0 - load_half_width))
    z_max = min(nz, int((lz_s + lz_e)/2.0 + load_half_width) + 1)
    x_idx = lx
    
    # Get labels exactly at load voxels (same slice as check_structure_connectivity)
    load_slice_labels = labeled[x_idx, y_min:y_max, z_min:z_max]
    load_slice_binary = binary[x_idx, y_min:y_max, z_min:z_max]
    
    # Expected voxels in load region
    expected_vol = (y_max - y_min) * (z_max - z_min)
    filled_vol = np.sum(load_slice_binary)
    
    # Get labels at load positions where there IS material
    labels_at_load = set(np.unique(load_slice_labels)) - {0}
    total_voxels = int(binary.sum())
    
    # Check 1: Any material at load?
    if not labels_at_load:
        return {
            'n_islands': n_islands,
            'main_island_label': 0,
            'main_island_mask': np.zeros_like(density, dtype=bool),
            'main_island_voxels': 0,
            'loose_voxels': total_voxels,
            'is_connected': False,
        }
    
    # Check 2: Which labels touch both support AND load?
    valid_islands = labels_at_load & support_labels
    
    if not valid_islands:
        return {
            'n_islands': n_islands,
            'main_island_label': 0,
            'main_island_mask': np.zeros_like(density, dtype=bool),
            'main_island_voxels': 0,
            'loose_voxels': total_voxels,
            'is_connected': False,
        }
    
    # Pick main island (largest valid one)
    main_label = max(valid_islands, key=lambda lbl: (labeled == lbl).sum())
    main_mask = labeled == main_label
    main_voxels = int(main_mask.sum())
    loose_voxels = total_voxels - main_voxels
    
    # Check 3: Multiple disjoint labels at load region?
    if len(labels_at_load) > 1:
        return {
            'n_islands': n_islands,
            'main_island_label': main_label,
            'main_island_mask': main_mask,
            'main_island_voxels': main_voxels,
            'loose_voxels': loose_voxels,
            'is_connected': False,  # Fragmented at load
        }
    
    # Check 4: Load region fully filled? (STRICT criterion)
    if filled_vol < expected_vol:
        return {
            'n_islands': n_islands,
            'main_island_label': main_label,
            'main_island_mask': main_mask,
            'main_island_voxels': main_voxels,
            'loose_voxels': loose_voxels,
            'is_connected': False,  # Not fully filled under load
        }
    
    # All checks passed - structure is connected
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
