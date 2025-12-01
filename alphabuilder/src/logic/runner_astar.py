"""
Episode runner using A* for Phase 1 connectivity.

NOTE: MCTS is NOT used during data harvest. It will only be used
during self-play after the model is trained.
"""

import numpy as np
from typing import Optional
from .game_rules import GameState
from alphabuilder.src.core.physics_model import FEMContext, PhysicalProperties
from alphabuilder.src.core.solver import solve_topology_3d


def run_episode_astar(
    ctx: FEMContext,
    props: PhysicalProperties,
    resolution: tuple = (64, 32, 8),
    model=None
) -> GameState:
    """
    Run a single episode using A* for Phase 1 connectivity.
    
    This creates a connected backbone structure from loads to supports,
    then returns the state for optional Phase 2 refinement (SIMP).
    
    Args:
        ctx: FEM context with mesh and solver
        props: Physical properties
        resolution: Grid resolution (D, H, W)
        model: Neural network model (NOT USED in data harvest, only for self-play)
        
    Returns:
        Final game state with connected backbone
    """
    D, H, W = resolution
    
    # Initialize State Tensor (5, D, H, W)
    tensor = np.zeros((5, D, H, W), dtype=np.float32)
    
    # Set BCs (Left Wall Support - entire left face)
    # Fix x=0 (d=0)
    tensor[1, 0, :, :] = 1.0
    
    # Set Load (Tip - center of right edge)
    # Load at x=L (d=D-1), y=mid, z=mid
    tensor[3, D-1, H//2, W//2] = -1.0
    
    # PHASE 1: Build Connectivity Backbone with A*
    print("Phase 1: Building connectivity with A*...")
    from .astar_pathfinder import (
        build_connectivity_backbone,
        extract_load_points,
        extract_support_points
    )
    
    # Extract coordinates
    load_coords = extract_load_points(tensor)
    support_coords = extract_support_points(tensor)
    
    print(f"  Found {len(load_coords)} load points, {len(support_coords)} support points")
    print(f"  DEBUG: Load Points: {load_coords}")
    print(f"  DEBUG: Support Points (first 5): {support_coords[:5]}")
    
    # Build backbone
    backbone_coords = build_connectivity_backbone(
        load_coords, support_coords, (D, H, W)
    )
    
    # Thicken the backbone for structural robustness
    from .astar_pathfinder import thicken_backbone
    thickened_coords = thicken_backbone(backbone_coords, (D, H, W), thickness=4)
    
    # Apply to tensor
    for d, h, w in thickened_coords:
        tensor[0, d, h, w] = 1.0
    
    volume_fraction = len(thickened_coords) / (D * H * W)
    print(f"  Built backbone with {len(backbone_coords)} voxels (thin)")
    print(f"  Thickened to {len(thickened_coords)} voxels ({volume_fraction:.2%} volume)")
    
    # DEBUG: Check backbone Z-alignment
    zs = [c[2] for c in thickened_coords]
    print(f"  DEBUG: Backbone Z mean: {np.mean(zs):.2f}, min: {np.min(zs)}, max: {np.max(zs)}")
    if len(zs) > 0:
        from collections import Counter
        print(f"  DEBUG: Backbone Z distribution: {dict(Counter(zs))}")
    
    # Create final state (Phase 2 ready)
    state = GameState(tensor=tensor, phase='REFINEMENT', step_count=0)
    
    # Solve physics once to get compliance
    sim_result = solve_topology_3d(tensor, ctx, props)
    print(f"  Backbone compliance: {sim_result.compliance:.2f}, max_disp: {sim_result.max_displacement:.2f}")
    
    return state
