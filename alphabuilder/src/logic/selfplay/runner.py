
import time
import random
import gc
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, field

from ..storage import Phase
from ..mcts import AlphaBuilderMCTS, get_legal_moves
from ..mcts.engine import (
    MCTSConfig, 
    PHASE1_CONFIG, 
    PHASE2_CONFIG,
    run_mcts_search,
    build_state_tensor,
    SearchResult,
    select_pv_sequences,
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
    save_game,
    record_step,
    update_game_final,
    generate_game_id,
    load_game,
    load_game_step,
    get_last_step,
    GameInfo,
    GameStep as StorageGameStep,
    SelectedAction,
    MCTSStats,
    check_structure_connectivity,
    get_phase2_terminal_reward,
    analyze_structure_islands,
    calculate_island_penalty,
    calculate_connectivity_reward,
    backpropagate_value_targets,
)



MCTS_CONFIG = MCTSConfig(
    num_simulations=80,
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
    use_guided_value: bool = True
    

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
    compliance_history: List[float] = field(default_factory=list)
    

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
        resolution=config.resolution,
        final_compliance=None,
        final_volume=None,
        total_steps=0,
        initial_cantilever_problem_real_scale_factor=500000.0,  # 16.0 units -> 0.032mm (High-Stiffness Steel)
        stiffness_E=200e9         # Steel (Reference)
    )
    save_game(db_path, game_info)
    
    # Episode loop
    final_score = 0.0
    start_time = time.time()
    
    # Initialize Stateful MCTS
    # We use a wrapper predictor to inject analytical rewards in Phase 1
    def hybrid_predictor(state_tensor: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        # Get raw network prediction
        net_val, p_add, p_remove = model.predict(state_tensor)
        
        # Calculate Aux components for Guidance
        density_grid = state_tensor[0]
        masks = (state_tensor[1], state_tensor[2], state_tensor[3])
        force_vecs = (state_tensor[4], state_tensor[5], state_tensor[6])
        
        # 1. Connectivity Bonus (Phase 1 focus, but valid metric always)
        connectivity_bonus, _ = calculate_connectivity_reward(
            density_grid, masks, force_vecs
        )
        # Note: calculate_connectivity_reward now returns specific [0, 1] bonus
        
        # 2. Island Penalty
        # Need full config for island analysis
        # We need to reconstruction load_config from force_vecs? 
        # Or just pass current state.load_config via closure
        # Issue: The tensor might represent a LEAF state, different from current 'state'.
        # But load_config is constant for the episode (defines where load IS).
        # So we can use 'load_config' from outer scope.
        
        # However, density_grid is from TENSOR (leaf).
        island_analysis = analyze_structure_islands(density_grid, load_config)
        n_islands = island_analysis['n_islands']
        loose_voxels = island_analysis['loose_voxels']
        total_voxels = int((density_grid > 0.5).sum())
        
        island_penalty = calculate_island_penalty(n_islands, loose_voxels, total_voxels)
        
        if state.phase == Phase.REFINEMENT:
             # Catastrophic Failure Check:
             # If structure loses connectivity (no valid main island connecting support to load),
             # immediate terminal penalty. This overrides everything.
             if not island_analysis['is_connected']:
                 return -1.0, p_add, p_remove

        if config.use_guided_value:
            # Guided Value = Net + Bonus - Penalty
            # Spec: Connectivity Bonus applies only in Phase 1. In Phase 2 it is 0.
            bonus_to_apply = connectivity_bonus if state.phase == Phase.GROWTH else 0.0
            
            guided_val = net_val + bonus_to_apply - island_penalty
            
            # Clamp to valid range [-1, 1]
            val = max(-1.0, min(1.0, guided_val))
        else:
            val = net_val

        return val, p_add, p_remove

    def hybrid_batch_predictor(state_tensors: List[np.ndarray]) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
        # 1. Run batch inference (efficient NPU call)
        net_vals, ps_add, ps_remove = model.predict_batch(state_tensors)
        
        final_vals = []
        
        # 2. Iterate and apply hybrid logic
        for i, (net_val, state_tensor) in enumerate(zip(net_vals, state_tensors)):
            density_grid = state_tensor[0]
            masks = (state_tensor[1], state_tensor[2], state_tensor[3])
            force_vecs = (state_tensor[4], state_tensor[5], state_tensor[6])
            
            # Connectivity Bonus
            connectivity_bonus, _ = calculate_connectivity_reward(
                density_grid, masks, force_vecs
            )
            
            # Island Penalty
            island_analysis = analyze_structure_islands(density_grid, load_config)
            n_islands = island_analysis['n_islands']
            loose_voxels = island_analysis['loose_voxels']
            total_voxels = int((density_grid > 0.5).sum())
            
            island_penalty = calculate_island_penalty(n_islands, loose_voxels, total_voxels)
            
            # Catastrophic failure check in Phase 2
            if state.phase == Phase.REFINEMENT and not island_analysis['is_connected']:
                final_vals.append(-1.0)
                continue

            if config.use_guided_value:
                bonus_to_apply = connectivity_bonus if state.phase == Phase.GROWTH else 0.0
                guided_val = net_val + bonus_to_apply - island_penalty
                val = max(-1.0, min(1.0, guided_val))
            else:
                val = net_val
            
            final_vals.append(val)
            
        return final_vals, ps_add, ps_remove

    mcts_wrapper = AlphaBuilderMCTS(
        predict_fn=hybrid_predictor,
        num_simulations=PHASE1_CONFIG.num_simulations,
        batch_size=PHASE1_CONFIG.batch_size,
        c_puct=PHASE1_CONFIG.c_puct,
        top_k_expansion=PHASE1_CONFIG.top_k_expansion
    )
    
    while state.current_step < config.max_steps:
        step_start = time.time()
        
        # Get phase-specific config
        current_config = PHASE1_CONFIG if state.phase == Phase.GROWTH else PHASE2_CONFIG
        
        # Update MCTS wrapper config on phase change or every step
        mcts_wrapper.config = mcts_wrapper.config._replace(
            num_simulations=current_config.num_simulations,
            batch_size=current_config.batch_size,
            action_batch_size=current_config.action_batch_size,
            c_puct=current_config.c_puct,
            top_k_expansion=current_config.top_k_expansion,
            min_volume_fraction=current_config.min_volume_fraction,
            max_volume_fraction=current_config.max_volume_fraction,
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
        displacement_map = None
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
                displacement_map = fem_result.displacement_map
                fem_reward = calculate_reward(
                    compliance=compliance,
                    vol_frac=vol_frac,
                    is_valid=True,
                    max_displacement=max_disp
                )
                # Track compliance
                state.compliance_history.append(compliance)
        
        # Calculate connectivity bonus (for Phase 1 reward components)
        connectivity_bonus = 0.0
        connected_load_fraction = 0.0
        if state.phase == Phase.GROWTH:
            connectivity_bonus, connected_load_fraction = calculate_connectivity_reward(
                state.density,
                (state.bc_masks[0], state.bc_masks[1], state.bc_masks[2]),
                (state.forces[0], state.forces[1], state.forces[2])
            )

        # Build reward components for recording
        reward_components = {
            "base_reward": 0.0,
            "connectivity_bonus": connectivity_bonus,
            "connected_load_fraction": connected_load_fraction,
            "fem_reward": fem_reward if fem_reward is not None else 0.0,
            "island_penalty": island_penalty,
            "loose_penalty": calculate_island_penalty(1, loose_voxels, total_voxels, penalty_per_island=0.0) if loose_voxels > 0 else 0.0,
            "volume_penalty": 0.0,
            "validity_penalty": -1.0 if (fem_reward == -1.0) else 0.0,
            "n_islands": island_analysis['n_islands'] if island_analysis else 1,
            "loose_voxels": loose_voxels,
            "disconnected_volume_fraction": loose_voxels / total_voxels if total_voxels > 0 else 0.0,
            "total": (fem_reward if fem_reward is not None else connectivity_bonus) - island_penalty
        }

        # Extract PV sequences for micro-batch (32 actions, max depth 4)
        pv_actions = select_pv_sequences(
            mcts_wrapper.root,
            max_actions=current_config.action_batch_size,
            max_depth=4
        )
        
        # Determine executed actions: prefer PV sequences, fallback to result.actions
        executed_actions = None
        if pv_actions:
            executed_actions = pv_actions
        elif result.actions:
            executed_actions = result.actions

        # Determine value target for training (FEM reward in Phase 2)
        value_target = None
        if state.phase == Phase.REFINEMENT and fem_reward is not None:
            value_target = fem_reward

        # Record step with all data
        record_step(
            db_path, state, result, value, policy_add, policy_remove,
            island_analysis=island_analysis,
            compliance_fem=compliance,
            max_displacement=max_disp,
            island_penalty=island_penalty,
            volume_fraction=vol_frac,
            reward_components=reward_components,
            executed_actions=executed_actions,
            connected_load_fraction=connected_load_fraction,
            value_target=value_target, displacement_map=displacement_map
        )
        
        # Apply micro-batch actions to density
        if pv_actions:
            apply_micro_batch(state, pv_actions)
            # Step through tree for each PV action
            for action in pv_actions:
                mcts_wrapper.step(action)
        elif result.actions:
            primary_action = result.actions[0]
            mcts_wrapper.step(primary_action)
            apply_micro_batch(state, result.actions)
            
            if len(result.actions) > 1:
                # Multiple actions invalidate tree reuse
                mcts_wrapper.reset()
        
        # Check phase transition
        if state.phase == Phase.GROWTH:
            if check_phase_transition(state):
                print(f"\n>>> Phase Transition at step {state.current_step}! <<<")
                state.phase = Phase.REFINEMENT
                state.is_connected = True
                state.compliance_history = [] # Reset history
            elif state.current_step >= config.max_phase1_steps:
                print(f"\n>>> Max Phase 1 steps reached, forcing transition <<<")
                state.phase = Phase.REFINEMENT
                state.compliance_history = [] # Reset history
        
        # Check terminal conditions for Phase 2
        if state.phase == Phase.REFINEMENT:
            terminal_reward = get_phase2_terminal_reward(state.density, state.load_config)
            if terminal_reward is not None:
                print(f"\n>>> Terminal state detected: reward={terminal_reward} <<<")
                final_score = terminal_reward
                break
        
        # Calculate Adjusted Value for Final Evaluation / Recording
        # Formula: 0.5 * Eval_Score + 0.5 * Value_Head + Bonus - Penalty
        
        # 1. Get current state components
        state_tensor = build_state_tensor(
             state.density,
             (state.bc_masks[0], state.bc_masks[1], state.bc_masks[2]),
             (state.forces[0], state.forces[1], state.forces[2])
        )
        net_val, _, _ = model.predict(state_tensor)
        
        connectivity_bonus, _ = calculate_connectivity_reward(
            state.density, state.bc_masks, state.forces
        )
        
        # 2. Determine Eval Score (FEM or Surrogate)
        if fem_reward is not None:
             eval_score = fem_reward
        else:
             eval_score = net_val # Fallback
             
        # 3. Apply Formula
        bonus_to_apply = connectivity_bonus if state.phase == Phase.GROWTH else 0.0
        
        if state.phase == Phase.REFINEMENT and fem_reward is not None:
             adjusted_value = 0.5 * eval_score + 0.5 * net_val + bonus_to_apply - island_penalty
        else:
             adjusted_value = net_val + bonus_to_apply - island_penalty
             
        # Clamp
        adjusted_value = max(-1.0, min(1.0, adjusted_value))
        
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
                f"B: {connectivity_bonus:.2f} | "
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
        
        # Convergence Termination Check (Success)
        if state.phase == Phase.REFINEMENT and len(state.compliance_history) >= 20:
             window = state.compliance_history[-20:]
             min_c = min(window)
             max_c = max(window)
             if max_c > 1e-9:
                rel_range = (max_c - min_c) / max_c
                if rel_range < 1e-3:
                     print(f"\n>>> Convergence Detected! Compliance stable over 20 steps (Range: {rel_range:.2%}) <<<")
                     break


    
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
    
    # Backpropagate Value Targets (Phase 1 = Final, Phase 2 = Mixed)
    print("Backpropagating Value Targets...")
    backpropagate_value_targets(db_path, state.game_id)

    return state.game_id, final_score, state.current_step


# =============================================================================
# Resume Episode
# =============================================================================

def resume_episode(
    model: AlphaBuilderInference,
    db_path: Path,
    game_id: str,
    config: EpisodeConfig
) -> Tuple[str, float, int]:
    """
    Resume an interrupted self-play episode from the last checkpoint.
    
    Args:
        model: Neural network inference wrapper
        db_path: Path to the SQLite database
        game_id: ID of the game to resume
        config: Episode configuration
        
    Returns:
        Tuple of (game_id, final_score, total_steps)
        
    Raises:
        ValueError: If game doesn't exist or has no steps to resume from
    """
    print(f"\n{'='*60}")
    print(f"Resuming Self-Play Episode: {game_id}")
    print(f"{'='*60}")
    
    # Load game metadata
    game_info = load_game(db_path, game_id)
    if game_info is None:
        raise ValueError(f"Game {game_id} not found in database")
    
    # Get the last step
    last_step = get_last_step(db_path, game_id)
    if last_step is None:
        raise ValueError(f"Game {game_id} has no recorded steps to resume from")
    
    print(f"Found game with {game_info.total_steps} total steps")
    print(f"Last recorded step: {last_step}")
    
    # Load the last step to get density state
    last_game_step = load_game_step(db_path, game_id, last_step, game_info.resolution)
    if last_game_step is None:
        raise ValueError(f"Could not load step {last_step} for game {game_id}")
    
    # Reconstruct EpisodeState from saved data
    # NOTE: The density saved in DB is the state BEFORE actions were applied (for training correctness).
    # We need to apply the last step's actions to get the actual post-action state.
    state = EpisodeState(
        density=last_game_step.density.copy(),
        bc_masks=game_info.bc_masks.copy(),
        forces=game_info.forces.copy(),
        load_config=game_info.load_config.copy(),
        current_step=last_step + 1,  # Start from next step
        phase=last_game_step.phase,
        is_connected=last_game_step.is_connected,
        game_id=game_id,
        compliance_history=[] # Start fresh history on resume
    )
    
    # ... (skipping to loop) ...

    # Same loop structure as run_episode, so we need to insert the logic
    # Wait, the tool view output cut off the loop content in resume_episode.
    # I need to see the loop in resume_episode first to know where to insert.
    # It seems resume_episode duplicates logic. I should ideally refactor but I will duplicate for now as requested.
    
    # Actually, let's look at the view output again. It shows resume_episode start but cuts off at line 800.
    # The loop starts around 800+?
    # No, run_episode was 207-630. resume_episode starts 636.
    # I need to see the loop inside resume_episode.

    
    # Apply the last step's actions to reconstruct S_{n+1} from S_n
    # The actions were recorded but not yet applied to the saved density
    if last_game_step.selected_actions:
        print(f"  Replaying {len(last_game_step.selected_actions)} actions from step {last_step}...")
        for action in last_game_step.selected_actions:
            # SelectedAction is a dataclass with channel, x, y, z attributes
            if action.channel == 0:  # Add
                state.density[action.x, action.y, action.z] = 1.0
            else:  # Remove
                state.density[action.x, action.y, action.z] = 0.0
    
    print(f"Restored state at step {last_step}:")
    print(f"  Phase: {state.phase.value}")
    print(f"  Density voxels: {np.sum(state.density > 0.5)}")
    print(f"  Volume fraction: {np.mean(state.density > 0.5):.4f}")
    
    # Phase correction: Ensure phase matches step count
    # Phase 1 (GROWTH) should end by max_phase1_steps
    print(f"  [DEBUG] Phase={state.phase}, current_step={state.current_step}, max_phase1_steps={config.max_phase1_steps}")
    if state.phase == Phase.GROWTH and state.current_step >= config.max_phase1_steps:
        print(f"  Correcting phase: Step {state.current_step} >= max_phase1_steps ({config.max_phase1_steps})")
        state.phase = Phase.REFINEMENT
        print(f"  Phase corrected to: {state.phase.value}")
    
    print(f"  Continuing from step {state.current_step}...")
    
    # Load config from saved game
    load_config = game_info.load_config
    
    # Episode loop - same as run_episode from here on
    final_score = 0.0
    start_time = time.time()
    
    # Initialize Stateful MCTS with hybrid predictor
    def hybrid_predictor(state_tensor: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        net_val, p_add, p_remove = model.predict(state_tensor)
        
        density_grid = state_tensor[0]
        masks = (state_tensor[1], state_tensor[2], state_tensor[3])
        force_vecs = (state_tensor[4], state_tensor[5], state_tensor[6])
        
        connectivity_bonus = calculate_connectivity_reward(
            density_grid, masks, force_vecs
        )
        
        island_analysis = analyze_structure_islands(density_grid, load_config)
        n_islands = island_analysis['n_islands']
        loose_voxels = island_analysis['loose_voxels']
        total_voxels = int((density_grid > 0.5).sum())
        
        island_penalty = calculate_island_penalty(n_islands, loose_voxels, total_voxels)
        
        if state.phase == Phase.REFINEMENT:
            if not island_analysis['is_connected']:
                return -1.0, p_add, p_remove

        if config.use_guided_value:
            bonus_to_apply = connectivity_bonus if state.phase == Phase.GROWTH else 0.0
            guided_val = net_val + bonus_to_apply - island_penalty
            val = max(-1.0, min(1.0, guided_val))
        else:
            val = net_val

        return val, p_add, p_remove

    def hybrid_batch_predictor(state_tensors: List[np.ndarray]) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
        """Batch version of hybrid_predictor for efficient NPU inference."""
        # 1. Run batch inference (efficient NPU call)
        net_vals, ps_add, ps_remove = model.predict_batch(state_tensors)
        
        final_vals = []
        
        # 2. Iterate and apply hybrid logic
        for i, (net_val, state_tensor) in enumerate(zip(net_vals, state_tensors)):
            density_grid = state_tensor[0]
            masks = (state_tensor[1], state_tensor[2], state_tensor[3])
            force_vecs = (state_tensor[4], state_tensor[5], state_tensor[6])
            
            # Connectivity Bonus
            connectivity_bonus = calculate_connectivity_reward(
                density_grid, masks, force_vecs
            )
            
            # Island Penalty
            island_analysis = analyze_structure_islands(density_grid, load_config)
            n_islands = island_analysis['n_islands']
            loose_voxels = island_analysis['loose_voxels']
            total_voxels = int((density_grid > 0.5).sum())
            
            island_penalty = calculate_island_penalty(n_islands, loose_voxels, total_voxels)
            
            # Catastrophic failure check in Phase 2
            if state.phase == Phase.REFINEMENT and not island_analysis['is_connected']:
                final_vals.append(-1.0)
                continue

            if config.use_guided_value:
                bonus_to_apply = connectivity_bonus if state.phase == Phase.GROWTH else 0.0
                guided_val = net_val + bonus_to_apply - island_penalty
                val = max(-1.0, min(1.0, guided_val))
            else:
                val = net_val
            
            final_vals.append(val)
            
        return final_vals, ps_add, ps_remove

    mcts_wrapper = AlphaBuilderMCTS(
        predict_fn=hybrid_predictor,
        num_simulations=PHASE1_CONFIG.num_simulations,
        batch_size=PHASE1_CONFIG.batch_size,
        action_batch_size=PHASE1_CONFIG.action_batch_size,
        c_puct=PHASE1_CONFIG.c_puct,
        top_k_expansion=PHASE1_CONFIG.top_k_expansion,
        predict_batch_fn=hybrid_batch_predictor
    )
    
    while state.current_step < config.max_steps:
        step_start = time.time()
        
        current_config = PHASE1_CONFIG if state.phase == Phase.GROWTH else PHASE2_CONFIG
        
        mcts_wrapper.config = mcts_wrapper.config._replace(
            num_simulations=current_config.num_simulations,
            batch_size=current_config.batch_size,
            action_batch_size=current_config.action_batch_size,
            c_puct=current_config.c_puct,
            top_k_expansion=current_config.top_k_expansion,
            min_volume_fraction=current_config.min_volume_fraction,
            max_volume_fraction=current_config.max_volume_fraction,
            phase=current_config.phase
        )
        
        result, value, policy_add, policy_remove = execute_mcts_step(
            state, mcts_wrapper
        )
        
        vol_frac = np.mean(state.density > 0.5)
        island_analysis = analyze_structure_islands(state.density, state.load_config)
        n_islands = island_analysis['n_islands']
        loose_voxels = island_analysis['loose_voxels']
        total_voxels = int((state.density > 0.5).sum())
        island_penalty = calculate_island_penalty(n_islands, loose_voxels, total_voxels)
        
        compliance = None
        max_disp = None
        fem_reward = None
        if state.phase == Phase.REFINEMENT and island_analysis['is_connected']:
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
                # Track compliance
                state.compliance_history.append(compliance)
        
        connectivity_bonus = 0.0
        connected_load_fraction = 0.0
        if state.phase == Phase.GROWTH:
            connectivity_bonus, connected_load_fraction = calculate_connectivity_reward(
                state.density,
                (state.bc_masks[0], state.bc_masks[1], state.bc_masks[2]),
                (state.forces[0], state.forces[1], state.forces[2])
            )

        reward_components = {
            "base_reward": 0.0,
            "connectivity_bonus": connectivity_bonus,
            "connected_load_fraction": connected_load_fraction,
            "fem_reward": fem_reward if fem_reward is not None else 0.0,
            "island_penalty": island_penalty,
            "loose_penalty": calculate_island_penalty(1, loose_voxels, total_voxels, penalty_per_island=0.0) if loose_voxels > 0 else 0.0,
            "volume_penalty": 0.0,
            "validity_penalty": -1.0 if (fem_reward == -1.0) else 0.0,
            "n_islands": island_analysis['n_islands'] if island_analysis else 1,
            "loose_voxels": loose_voxels,
            "disconnected_volume_fraction": loose_voxels / total_voxels if total_voxels > 0 else 0.0,
            "total": (fem_reward if fem_reward is not None else connectivity_bonus) - island_penalty
        }

        pv_actions = select_pv_sequences(
            mcts_wrapper.root,
            max_actions=current_config.action_batch_size,
            max_depth=4
        )
        
        executed_actions = None
        if pv_actions:
            executed_actions = pv_actions
        elif result.actions:
            executed_actions = result.actions

        value_target = None
        if state.phase == Phase.REFINEMENT and fem_reward is not None:
            value_target = fem_reward

        record_step(
            db_path, state, result, value, policy_add, policy_remove,
            island_analysis=island_analysis,
            compliance_fem=compliance,
            max_displacement=max_disp,
            island_penalty=island_penalty,
            volume_fraction=vol_frac,
            reward_components=reward_components,
            executed_actions=executed_actions,
            connected_load_fraction=connected_load_fraction,
            value_target=value_target
        )
        
        if pv_actions:
            apply_micro_batch(state, pv_actions)
            for action in pv_actions:
                mcts_wrapper.step(action)
        elif result.actions:
            primary_action = result.actions[0]
            mcts_wrapper.step(primary_action)
            apply_micro_batch(state, result.actions)
            
            if len(result.actions) > 1:
                mcts_wrapper.reset()
            
            apply_micro_batch(state, result.actions)
        
        if state.phase == Phase.GROWTH:
            if check_phase_transition(state):
                print(f"\n>>> Phase Transition at step {state.current_step}! <<<")
                state.phase = Phase.REFINEMENT
                state.is_connected = True
                state.compliance_history = [] # Reset history
            elif state.current_step >= config.max_phase1_steps:
                print(f"\n>>> Max Phase 1 steps reached, forcing transition <<<")
                state.phase = Phase.REFINEMENT
                state.compliance_history = [] # Reset history
        
        if state.phase == Phase.REFINEMENT:
            terminal_reward = get_phase2_terminal_reward(state.density, state.load_config)
            if terminal_reward is not None:
                print(f"\n>>> Terminal state detected: reward={terminal_reward} <<<")
                final_score = terminal_reward
                break
        
        state_tensor = build_state_tensor(
             state.density,
             (state.bc_masks[0], state.bc_masks[1], state.bc_masks[2]),
             (state.forces[0], state.forces[1], state.forces[2])
        )
        net_val, _, _ = model.predict(state_tensor)
        
        connectivity_bonus, _ = calculate_connectivity_reward(
            state.density, state.bc_masks, state.forces
        )
        
        if fem_reward is not None:
             eval_score = fem_reward
        else:
             eval_score = net_val
             
        bonus_to_apply = connectivity_bonus if state.phase == Phase.GROWTH else 0.0
        
        if state.phase == Phase.REFINEMENT and fem_reward is not None:
             adjusted_value = 0.5 * eval_score + 0.5 * net_val + bonus_to_apply - island_penalty
        else:
             adjusted_value = net_val + bonus_to_apply - island_penalty
             
        adjusted_value = max(-1.0, min(1.0, adjusted_value))
        
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
                f"B: {connectivity_bonus:.2f} | "
                f"T: {step_time:.1f}s"
            )
        
        state.current_step += 1
        final_score = adjusted_value
        
        update_game_final(
            db_path=db_path,
            game_id=state.game_id,
            final_score=adjusted_value,
            final_volume=vol_frac,
            total_steps=state.current_step
        )

        # Convergence Termination Check (Success)
        if state.phase == Phase.REFINEMENT and len(state.compliance_history) >= 20:
             window = state.compliance_history[-20:]
             min_c = min(window)
             max_c = max(window)
             if max_c > 1e-9:
                rel_range = (max_c - min_c) / max_c
                if rel_range < 1e-3:
                     print(f"\n>>> Convergence Detected! Compliance stable over 20 steps (Range: {rel_range:.2%}) <<<")
                     break
    
    total_time = time.time() - start_time
    vol_frac = np.mean(state.density > 0.5)
    
    print(f"\n{'='*60}")
    print(f"Resumed Episode Complete!")
    print(f"  Total steps: {state.current_step}")
    print(f"  Final phase: {state.phase.value}")
    print(f"  Final value: {final_score:.4f}")
    print(f"  Final volume: {vol_frac:.3f}")
    print(f"  Resume time: {total_time:.1f}s")
    print(f"{'='*60}")
    
    update_game_final(
        db_path=db_path,
        game_id=state.game_id,
        final_score=final_score,
        final_volume=vol_frac,
        total_steps=state.current_step
    )
    
    # Backpropagate Value Targets (Phase 1 = Final, Phase 2 = Mixed)
    print("Backpropagating Value Targets...")
    backpropagate_value_targets(db_path, state.game_id)

    return state.game_id, final_score, state.current_step

