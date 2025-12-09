
import time
import random
import gc
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass

from ..storage import Phase
from ..mcts import AlphaBuilderMCTS, get_legal_moves
from ..mcts.engine import (
    MCTSConfig, 
    PHASE1_CONFIG, 
    PHASE2_CONFIG,
    run_mcts_search,
    build_state_tensor,
    SearchResult,
)
from ...neural.inference import AlphaBuilderInference
from ..harvest.generators import (
    generate_random_load_config,
    generate_bc_masks,
)
from .evaluator import evaluate_main_island
from .reward import calculate_reward
from . import (
    save_game,
    save_game_step,
    update_game_final,
    generate_game_id,
    GameInfo,
    GameStep as StorageGameStep,
    SelectedAction,
    MCTSStats,
    check_structure_connectivity,
    get_phase2_terminal_reward,
    analyze_structure_islands,
    calculate_island_penalty,
    calculate_connectivity_reward,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EpisodeConfig:
    """Configuration for a single self-play episode."""
    max_steps: int = 600
    max_phase1_steps: int = 200  # Max steps before forcing phase transition
    target_volume_fraction: float = 0.15
    resolution: Tuple[int, int, int] = (64, 32, 8)
    bc_type: str = "FULL_CLAMP"
    

# =============================================================================
# Episode State
# =============================================================================

@dataclass
class EpisodeState:
    """Mutable state for the current episode."""
    density: np.ndarray
    bc_masks: np.ndarray  # (3, D, H, W)
    forces: np.ndarray    # (3, D, H, W)
    load_config: Dict[str, Any]
    current_step: int = 0
    phase: Phase = Phase.GROWTH
    is_connected: bool = False
    game_id: str = ""
    

# =============================================================================
# Initialization
# =============================================================================

def create_empty_state(
    resolution: Tuple[int, int, int],
    load_config: Dict[str, Any],
    bc_type: str = "FULL_CLAMP"
) -> EpisodeState:
    """
    Create episode state with EMPTY grid and boundary conditions.
    """
    nx, ny, nz = resolution
    
    # Completely empty density grid
    density = np.zeros(resolution, dtype=np.float32)
    
    # Generate BC masks
    mask_x, mask_y, mask_z = generate_bc_masks(resolution, bc_type)
    bc_masks = np.stack([mask_x, mask_y, mask_z], axis=0).astype(np.float32)
    
    # Generate force field
    fx = np.zeros(resolution, dtype=np.float32)
    fy = np.zeros(resolution, dtype=np.float32)
    fz = np.zeros(resolution, dtype=np.float32)
    
    # Apply force at load region
    lx = load_config.get('x', nx - 1)
    ly = load_config.get('y', ny // 2)
    lz_center = (load_config.get('z_start', 0) + load_config.get('z_end', nz)) / 2.0
    load_half_width = 1.0
    
    y_min = max(0, int(ly - load_half_width))
    y_max = min(ny, int(ly + load_half_width) + 1)
    z_min = max(0, int(lz_center - load_half_width))
    z_max = min(nz, int(lz_center + load_half_width) + 1)
    x_idx = min(lx, nx - 1)
    
    fy[x_idx, y_min:y_max, z_min:z_max] = -1.0
    forces = np.stack([fx, fy, fz], axis=0).astype(np.float32)
    
    return EpisodeState(
        density=density,
        bc_masks=bc_masks,
        forces=forces,
        load_config=load_config,
        current_step=0,
        phase=Phase.GROWTH,
        is_connected=False,
        game_id=generate_game_id()
    )


# =============================================================================
# Phase Transition
# =============================================================================

def check_phase_transition(state: EpisodeState) -> bool:
    """
    Check if we should transition from Phase 1 to Phase 2.
    """
    connected_to_support, reached_load = check_structure_connectivity(
        state.density, state.load_config, threshold=0.5
    )
    return connected_to_support and reached_load


# =============================================================================
# MCTS Step Execution
# =============================================================================

def execute_mcts_step(
    state: EpisodeState,
    mcts: AlphaBuilderMCTS
) -> Tuple[SearchResult, float, np.ndarray, np.ndarray]:
    """
    Execute one MCTS search and return results.
    """
    # Build state tensor (7 channels) for recording raw network prediction
    # Note: MCTS wrapper handles this internally for search, but we want it for logging
    state_tensor = build_state_tensor(
        state.density,
        (state.bc_masks[0], state.bc_masks[1], state.bc_masks[2]),
        (state.forces[0], state.forces[1], state.forces[2])
    )
    
    # Get neural prediction for recording (using wrapper's predict_fn)
    value, policy_add, policy_remove = mcts.predict_fn(state_tensor)
    
    # Run MCTS search using stateful wrapper
    result = mcts.search(
        density=state.density,
        bc_masks=(state.bc_masks[0], state.bc_masks[1], state.bc_masks[2]),
        forces=(state.forces[0], state.forces[1], state.forces[2])
    )
    
    return result, value, policy_add, policy_remove


def apply_micro_batch(
    state: EpisodeState,
    actions: List[Tuple[int, int, int, int]]
) -> None:
    """
    Apply micro-batch of actions to state (in-place).
    """
    for channel, x, y, z in actions:
        if channel == 0:  # Add
            state.density[x, y, z] = 1.0
        else:  # Remove
            state.density[x, y, z] = 0.0


# =============================================================================
# Step Recording
# =============================================================================

def record_step(
    db_path: Path,
    state: EpisodeState,
    result: SearchResult,
    value: float,
    policy_add: np.ndarray,
    policy_remove: np.ndarray,
    island_analysis: dict = None,
    compliance_fem: float = None,
    max_displacement: float = None,
    island_penalty: float = 0.0,
    volume_fraction: float = 0.0
) -> None:
    """
    Record a single step to the database.
    """
    # Build visit count arrays
    resolution = state.density.shape
    mcts_visit_add = np.zeros(resolution, dtype=np.float32)
    mcts_visit_remove = np.zeros(resolution, dtype=np.float32)
    mcts_q_add = np.zeros(resolution, dtype=np.float32)
    mcts_q_remove = np.zeros(resolution, dtype=np.float32)
    
    # Fill from visit distribution
    for action, visits in result.visit_distribution.items():
        channel, x, y, z = action
        if channel == 0:
            mcts_visit_add[x, y, z] = visits
        else:
            mcts_visit_remove[x, y, z] = visits
    
    # Build selected actions list
    selected_actions = []
    for action in result.actions:
        channel, x, y, z = action
        visits = result.visit_distribution.get(action, 0)
        selected_actions.append(SelectedAction(
            channel=channel,
            x=x, y=y, z=z,
            visits=visits,
            q_value=0.0  # TODO: Get from node
        ))
    
    # MCTS stats
    mcts_stats = MCTSStats(
        num_simulations=result.num_simulations,
        nodes_expanded=len(result.visit_distribution),
        max_depth=0,  # TODO: Track in search
        cache_hits=0,  # TODO: Track in search
        top8_concentration=sum(sorted(result.visit_distribution.values(), reverse=True)[:8]) / max(1, sum(result.visit_distribution.values())),
        refutation=False  # TODO: Calculate
    )
    
    # Extract island analysis data
    n_islands = 1
    loose_voxels = 0
    is_connected = False
    if island_analysis is not None:
        n_islands = island_analysis.get('n_islands', 1)
        loose_voxels = island_analysis.get('loose_voxels', 0)
        is_connected = island_analysis.get('is_connected', False)
    
    # Create step record with all fields
    step_record = StorageGameStep(
        game_id=state.game_id,
        step=state.current_step,
        phase=state.phase,
        density=state.density.copy(),
        policy_add=policy_add,
        policy_remove=policy_remove,
        mcts_visit_add=mcts_visit_add,
        mcts_visit_remove=mcts_visit_remove,
        mcts_q_add=mcts_q_add,
        mcts_q_remove=mcts_q_remove,
        selected_actions=selected_actions,
        value=value,
        mcts_stats=mcts_stats,
        n_islands=n_islands,
        loose_voxels=loose_voxels,
        is_connected=is_connected,
        compliance_fem=compliance_fem,
        max_displacement=max_displacement,
        island_penalty=island_penalty,
        volume_fraction=volume_fraction
    )
    
    save_game_step(db_path, step_record)


# =============================================================================
# Main Episode Loop
# =============================================================================

def run_episode(
    model: AlphaBuilderInference,
    db_path: Path,
    config: EpisodeConfig
) -> Tuple[str, float, int]:
    """
    Run a complete self-play episode.
    
    Returns:
        Tuple of (game_id, final_score, total_steps)
    """
    print(f"\n{'='*60}")
    print("Starting Self-Play Episode")
    print(f"{'='*60}")
    
    # Generate random load configuration
    load_config = generate_random_load_config(config.resolution)
    print(f"Load Config: x={load_config['x']}, y={load_config['y']}, z={load_config['z_start']}-{load_config['z_end']}")
    
    # Create initial state
    state = create_empty_state(
        resolution=config.resolution,
        load_config=load_config,
        bc_type=config.bc_type
    )
    print(f"Game ID: {state.game_id}")
    print(f"Initial density: {np.sum(state.density > 0.5)} voxels")
    
    # Save game metadata
    game_info = GameInfo(
        game_id=state.game_id,
        neural_engine="simple",
        checkpoint_version="warmup",
        bc_masks=state.bc_masks,
        forces=state.forces,
        load_config=load_config,
        bc_type=config.bc_type,
        resolution=config.resolution
    )
    save_game(db_path, game_info)
    
    # Episode loop
    final_score = 0.0
    start_time = time.time()
    
    # Initialize Stateful MCTS
    # We use a wrapper predictor to inject analytical rewards in Phase 1
    def hybrid_predictor(state_tensor: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        # Get raw network prediction
        val, p_add, p_remove = model.predict(state_tensor)
        
        # In Phase 1 (Growth), inject analytical connectivity reward
        # State tensor layout: [density, mask_x, mask_y, mask_z, fx, fy, fz]
        if state.phase == Phase.GROWTH:
            density_grid = state_tensor[0]
            masks = (state_tensor[1], state_tensor[2], state_tensor[3])
            force_vecs = (state_tensor[4], state_tensor[5], state_tensor[6])
            
            # 1. Connectivity Bonus (+0.0 to +0.5)
            connectivity_bonus = calculate_connectivity_reward(
                density_grid, masks, force_vecs
            )
            
            # 2. Island Penalty
            # Need to re-analyze islands for the *current* state tensor
            # Ideally this info is passed in, but we have to recompute from tensor here
            # to keep predictor stateless/consistent with MCTS node expansion.
            # Convert channel 0 density back to standard format
            island_analysis = analyze_structure_islands(
                density_grid, 
                state.load_config,
                threshold=0.5
            )
            island_penalty = calculate_island_penalty(
                island_analysis['n_islands'], 
                island_analysis['loose_voxels'], 
                total_voxels=int((density_grid > 0.5).sum())
            )
            
            # 3. Combine Components
            # Base: Network Value [-1, 1]
            # Bonus: Connectivity [+0, +0.5]
            # Penalty: Islands [-0, -0.3]
            # Penalty: Living [-0.01 per step]
            living_penalty = 0.01
            
            val = val + connectivity_bonus - island_penalty - living_penalty
            
            # Clamp to Valid Value Head Range [-1, 1]
            val = max(-1.0, min(1.0, val))
            
        return val, p_add, p_remove

    mcts_wrapper = AlphaBuilderMCTS(
        predict_fn=hybrid_predictor,
        num_simulations=PHASE1_CONFIG.num_simulations,
        batch_size=PHASE1_CONFIG.batch_size,
        c_puct=PHASE1_CONFIG.c_puct
    )
    
    while state.current_step < config.max_steps:
        step_start = time.time()
        
        # Get phase-specific config
        current_config = PHASE1_CONFIG if state.phase == Phase.GROWTH else PHASE2_CONFIG
        
        # Update MCTS wrapper config on phase change or every step
        mcts_wrapper.config = mcts_wrapper.config._replace(
            num_simulations=current_config.num_simulations,
            batch_size=current_config.batch_size,
            c_puct=current_config.c_puct,
            phase=current_config.phase
        )
        
        # Execute MCTS step
        result, value, policy_add, policy_remove = execute_mcts_step(
            state, mcts_wrapper
        )
        
        # Analyze islands BEFORE recording (for current state before actions)
        vol_frac = np.mean(state.density > 0.5)
        island_analysis = analyze_structure_islands(state.density, state.load_config)
        n_islands = island_analysis['n_islands']
        loose_voxels = island_analysis['loose_voxels']
        total_voxels = int((state.density > 0.5).sum())
        island_penalty = calculate_island_penalty(n_islands, loose_voxels, total_voxels)
        
        # In Phase 2, run FEM on main island to get real compliance
        compliance = None
        max_disp = None
        fem_reward = None
        if state.phase == Phase.REFINEMENT and island_analysis['is_connected']:
            # Evaluate only the main island (connected component)
            main_mask = island_analysis['main_island_mask']
            fem_result = evaluate_main_island(
                state.density, 
                state.load_config, 
                main_mask
            )
            if fem_result.valid:
                compliance = fem_result.compliance
                max_disp = fem_result.max_displacement
                fem_reward = calculate_reward(
                    compliance=compliance,
                    vol_frac=vol_frac,
                    is_valid=True,
                    max_displacement=max_disp
                )
        
        # Record step with island and FEM data
        record_step(
            db_path, state, result, value, policy_add, policy_remove,
            island_analysis=island_analysis,
            compliance_fem=compliance,
            max_displacement=max_disp,
            island_penalty=island_penalty,
            volume_fraction=vol_frac
        )
        
        # Apply micro-batch
        if result.actions:
            # Advance tree state along the FIRST action (primary choice)
            # This enables Tree Reuse for next step
            primary_action = result.actions[0]
            mcts_wrapper.step(primary_action)
            
            # Apply actions (including Micro-Batch if multiple)
            # Note: Tree reuse only follows the primary path. Secondary micro-batch actions
            # effectively cause a "drift" from the tree's expected state if they are significant.
            # But usually micro-batch actions are compatible or distant.
            # The tree root for next step will assume we took primary_action.
            # If we apply more, the next root_density will differ from what the tree expects.
            # This invalidates the tree reuse slightly.
            # For strict correctness with Tree Reuse, we should ideally only take 1 action per step.
            # OR we accept that the tree state is an approximation.
            # Given we clear the root if state diverges?
            # AlphaBuilderMCTS.step() just moves the pointer.
            # The next search() calcs root state.
            # If actual state != expected state from tree walk, tree reuse might be harmful?
            # Actually, MCTSNode doesn't store state. It just stores stats.
            # But the children are conditioned on the parent's action.
            # If we apply Action A AND Action B, but only traverse A...
            # The new root corresponds to State_after_A.
            # But actual simulation starts from State_after_A_and_B.
            # This is a discrepancy!
            
            # For now, to support Tree Reuse safely, we should disable micro-batches 
            # OR clear tree if batch_size > 1.
            
            if len(result.actions) > 1:
                # If we take multiple actions, we can't reuse the tree easily 
                # because the tree branch assumes only ONE action was taken.
                # So we reset the tree.
                mcts_wrapper.reset()
            
            apply_micro_batch(state, result.actions)
        
        # Check phase transition
        if state.phase == Phase.GROWTH:
            if check_phase_transition(state):
                print(f"\n>>> Phase Transition at step {state.current_step}! <<<")
                state.phase = Phase.REFINEMENT
                state.is_connected = True
            elif state.current_step >= config.max_phase1_steps:
                print(f"\n>>> Max Phase 1 steps reached, forcing transition <<<")
                state.phase = Phase.REFINEMENT
        
        # Check terminal conditions for Phase 2
        if state.phase == Phase.REFINEMENT:
            terminal_reward = get_phase2_terminal_reward(state.density, state.load_config)
            if terminal_reward is not None:
                print(f"\n>>> Terminal state detected: reward={terminal_reward} <<<")
                final_score = terminal_reward
                break
        
        # Calculate adjusted value with island penalty
        if fem_reward is not None:
            adjusted_value = max(-1.0, min(1.0, fem_reward - island_penalty))
        else:
            adjusted_value = max(-1.0, min(1.0, value - island_penalty))
        
        # Progress logging
        step_time = time.time() - step_start
        if state.current_step % 10 == 0:
            comp_str = f"C: {compliance:.1f}" if compliance is not None else "C: ---"
            print(
                f"Step {state.current_step:4d} | "
                f"Phase: {state.phase.value:10s} | "
                f"V: {value:+.3f} | "
                f"{comp_str} | "
                f"Vol: {vol_frac:.3f} | "
                f"Isl: {n_islands} | "
                f"T: {step_time:.1f}s"
            )
        
        state.current_step += 1
        final_score = adjusted_value  # Use adjusted value with island penalty
        
        # Update total_steps incrementally so frontend can track progress
        update_game_final(
            db_path=db_path,
            game_id=state.game_id,
            final_score=adjusted_value,
            final_volume=vol_frac,
            total_steps=state.current_step
        )
    
    # Calculate final stats
    total_time = time.time() - start_time
    vol_frac = np.mean(state.density > 0.5)
    
    print(f"\n{'='*60}")
    print(f"Episode Complete!")
    print(f"  Total steps: {state.current_step}")
    print(f"  Final phase: {state.phase.value}")
    print(f"  Final value: {final_score:.4f}")
    print(f"  Final volume: {vol_frac:.3f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg time/step: {total_time/max(1, state.current_step):.2f}s")
    print(f"{'='*60}")
    
    # Update game with final results
    update_game_final(
        db_path=db_path,
        game_id=state.game_id,
        final_score=final_score,
        final_volume=vol_frac,
        total_steps=state.current_step
    )
    
    return state.game_id, final_score, state.current_step
