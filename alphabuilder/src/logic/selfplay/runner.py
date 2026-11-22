"""
Self-Play Episode Runner for AlphaBuilder v3.3.

Implements complete MCTS self-play loop with:
- Batch MCTS inference (`predict_batch_fn`) with Virtual Loss
- Decaying Potential-Based Reward Shaping (PBRS) in Phase 1 (growth)
- Pure physics FEM evaluation (`evaluate_main_island` matching FEniTop physics) in Phase 2
- Faithful PV branch storage (`executed_action_stats`), tree/cache reset on phase transition
- Physical collapse termination and reward-aligned backpropagation (`backpropagate_value_targets`)
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ..storage import Phase
from ..mcts.engine import (
    AlphaBuilderMCTS,
    MCTSConfig,
    PHASE1_CONFIG,
    PHASE2_CONFIG,
    SearchResult,
    build_state_tensor,
    select_pv_branches,
)
from ...neural.inference import AlphaBuilderInference
from ..harvest.generators import (
    generate_bc_masks,
    generate_random_load_config,
    get_load_region_bounds,
)
from .evaluator import evaluate_main_island
from .reward import (
    analyze_structure_islands,
    calculate_connectivity_reward,
    calculate_island_penalty,
    calculate_pbrs_potential,
    calculate_reward,
    check_structure_connectivity,
    get_guidance_alpha,
    get_phase2_terminal_reward,
)
from . import (
    GameInfo,
    backpropagate_value_targets,
    generate_game_id,
    get_last_step,
    load_game,
    load_game_step,
    record_step,
    save_game,
    update_game_final,
)


MCTS_CONFIG = MCTSConfig(num_simulations=80)


@dataclass
class EpisodeConfig:
    """Configuration for a single self-play episode."""

    max_steps: int = 600
    max_phase1_steps: int = 200
    target_volume_fraction: float = 0.15
    resolution: Tuple[int, int, int] = (64, 32, 8)
    bc_type: str = "FULL_CLAMP"
    use_guided_value: bool = True
    generation: int = 0
    guidance_decay_generations: int = 5


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


def create_empty_state(
    resolution: Tuple[int, int, int],
    load_config: Dict[str, Any],
    bc_type: str = "FULL_CLAMP",
) -> EpisodeState:
    """Create initial empty episode state with aligned BC masks and force patch."""
    density = np.zeros(resolution, dtype=np.float32)

    mask_x, mask_y, mask_z = generate_bc_masks(resolution, bc_type)
    bc_masks = np.stack([mask_x, mask_y, mask_z], axis=0).astype(np.float32)

    fx = np.zeros(resolution, dtype=np.float32)
    fy = np.zeros(resolution, dtype=np.float32)
    fz = np.zeros(resolution, dtype=np.float32)

    lx, y_min, y_max, z_min, z_max = get_load_region_bounds(resolution, load_config)
    fy[lx, y_min:y_max, z_min:z_max] = -1.0
    forces = np.stack([fx, fy, fz], axis=0).astype(np.float32)

    cfg_copy = dict(load_config)
    cfg_copy["bc_type"] = bc_type

    return EpisodeState(
        density=density,
        bc_masks=bc_masks,
        forces=forces,
        load_config=cfg_copy,
        current_step=0,
        phase=Phase.GROWTH,
        is_connected=False,
        game_id=generate_game_id(),
    )


def check_phase_transition(state: EpisodeState) -> bool:
    """Check if structure connects support to the full load patch."""
    connected_to_support, reached_load = check_structure_connectivity(
        state.density, state.load_config, threshold=0.5
    )
    return connected_to_support and reached_load


def execute_mcts_step(
    state: EpisodeState,
    mcts: AlphaBuilderMCTS,
    raw_predict_fn: Optional[Callable[[np.ndarray], Tuple[float, np.ndarray, np.ndarray]]] = None,
) -> Tuple[SearchResult, float, np.ndarray, np.ndarray]:
    """Run MCTS search and also capture raw unshaped network prediction at root."""
    bc_tuple = (state.bc_masks[0], state.bc_masks[1], state.bc_masks[2])
    force_tuple = (state.forces[0], state.forces[1], state.forces[2])
    state_tensor = build_state_tensor(state.density, bc_tuple, force_tuple)

    raw_fn = raw_predict_fn if raw_predict_fn is not None else mcts.predict_fn
    value, policy_add, policy_remove = raw_fn(state_tensor)

    result = mcts.search(
        density=state.density,
        bc_masks=bc_tuple,
        forces=force_tuple,
    )
    return result, float(value), policy_add, policy_remove


def apply_micro_batch(
    state: EpisodeState,
    actions: List[Tuple[int, int, int, int]],
) -> None:
    """Apply micro-batch of unitary voxel edits in place."""
    for channel, x, y, z in actions:
        state.density[x, y, z] = 1.0 if channel == 0 else 0.0


def _run_episode_loop(
    state: EpisodeState,
    model: AlphaBuilderInference,
    db_path: Path,
    config: EpisodeConfig,
) -> Tuple[str, float, int]:
    """Unified execution loop shared cleanly by run_episode and resume_episode."""
    load_config = state.load_config
    bc_type = str(load_config.get("bc_type", config.bc_type))
    alpha_guide = (
        get_guidance_alpha(
            generation=config.generation,
            decay_generations=config.guidance_decay_generations,
        )
        if config.use_guided_value
        else 0.0
    )

    current_phi_root = [0.0]

    def _shape_value(net_val: float, density_grid: np.ndarray, masks, force_vecs) -> float:
        isl = analyze_structure_islands(density_grid, load_config, bc_masks=masks)
        total_voxels = int((density_grid > 0.5).sum())
        island_pen = calculate_island_penalty(isl["n_islands"], isl["loose_voxels"], total_voxels)

        if state.phase == Phase.REFINEMENT:
            if not isl["is_connected"]:
                return -1.0
            val = float(net_val) - island_pen
            return max(-1.0, min(1.0, val))

        if alpha_guide > 0.0:
            phi_leaf, _ = calculate_pbrs_potential(density_grid, masks, force_vecs, load_config)
            shaping = alpha_guide * (phi_leaf - current_phi_root[0])
            return max(-1.0, min(1.0, float(net_val) + shaping))
        return max(-1.0, min(1.0, float(net_val)))

    def hybrid_predictor(state_tensor: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        net_val, p_add, p_remove = model.predict(state_tensor)
        masks = (state_tensor[1], state_tensor[2], state_tensor[3])
        force_vecs = (state_tensor[4], state_tensor[5], state_tensor[6])
        return _shape_value(net_val, state_tensor[0], masks, force_vecs), p_add, p_remove

    def hybrid_batch_predictor(
        state_tensors: List[np.ndarray],
    ) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
        net_vals, ps_add, ps_remove = model.predict_batch(state_tensors)
        final_vals = [
            _shape_value(
                nv,
                st[0],
                (st[1], st[2], st[3]),
                (st[4], st[5], st[6]),
            )
            for nv, st in zip(net_vals, state_tensors)
        ]
        return final_vals, ps_add, ps_remove

    init_cfg = PHASE1_CONFIG if state.phase == Phase.GROWTH else PHASE2_CONFIG
    mcts_wrapper = AlphaBuilderMCTS(
        predict_fn=hybrid_predictor,
        num_simulations=init_cfg.num_simulations,
        batch_size=init_cfg.batch_size,
        action_batch_size=init_cfg.action_batch_size,
        c_puct=init_cfg.c_puct,
        top_k_expansion=init_cfg.top_k_expansion,
        predict_batch_fn=hybrid_batch_predictor,
    )

    final_score = 0.0
    last_valid_compliance: Optional[float] = None
    start_time = time.time()

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
            phase=current_config.phase,
            policy_temperature=current_config.policy_temperature,
            mixmax_beta=current_config.mixmax_beta,
        )

        bc_tuple = (state.bc_masks[0], state.bc_masks[1], state.bc_masks[2])
        force_tuple = (state.forces[0], state.forces[1], state.forces[2])
        current_phi_root[0], _ = calculate_pbrs_potential(state.density, bc_tuple, force_tuple, state.load_config)

        result, raw_net_val, policy_add, policy_remove = execute_mcts_step(
            state, mcts_wrapper, raw_predict_fn=model.predict
        )

        vol_frac = float(np.mean(state.density > 0.5))
        island_analysis = analyze_structure_islands(state.density, state.load_config, bc_masks=bc_tuple)
        n_islands = island_analysis["n_islands"]
        loose_voxels = island_analysis["loose_voxels"]
        total_voxels = int((state.density > 0.5).sum())
        island_penalty = calculate_island_penalty(n_islands, loose_voxels, total_voxels)

        compliance: Optional[float] = None
        max_disp: Optional[float] = None
        fem_reward: Optional[float] = None
        displacement_map: Optional[np.ndarray] = None
        fem_collapsed = False

        if state.phase == Phase.REFINEMENT:
            if not island_analysis["is_connected"]:
                fem_reward = -1.0
                fem_collapsed = True
            else:
                fem_result = evaluate_main_island(
                    state.density,
                    state.load_config,
                    island_analysis["main_island_mask"],
                    bc_type=bc_type,
                )
                if fem_result.valid:
                    compliance = fem_result.compliance
                    max_disp = fem_result.max_displacement
                    displacement_map = fem_result.displacement_map
                    fem_reward = calculate_reward(
                        compliance=compliance,
                        vol_frac=vol_frac,
                        is_valid=True,
                        max_displacement=max_disp,
                    )
                    last_valid_compliance = compliance
                    if fem_reward == -1.0:
                        fem_collapsed = True
                    else:
                        state.compliance_history.append(compliance)
                else:
                    fem_reward = -1.0
                    fem_collapsed = True

        connectivity_bonus = 0.0
        connected_load_fraction = 0.0
        if state.phase == Phase.GROWTH:
            connectivity_bonus, connected_load_fraction = calculate_connectivity_reward(
                state.density, bc_tuple, force_tuple
            )

        reward_components = {
            "base_reward": 0.0,
            "connectivity_bonus": connectivity_bonus,
            "connected_load_fraction": connected_load_fraction,
            "fem_reward": fem_reward if fem_reward is not None else 0.0,
            "island_penalty": island_penalty,
            "loose_penalty": calculate_island_penalty(1, loose_voxels, total_voxels, penalty_per_island=0.0)
            if loose_voxels > 0
            else 0.0,
            "volume_penalty": 0.0,
            "validity_penalty": -1.0 if fem_collapsed else 0.0,
            "n_islands": n_islands,
            "loose_voxels": loose_voxels,
            "disconnected_volume_fraction": loose_voxels / total_voxels if total_voxels > 0 else 0.0,
            "total": (fem_reward if fem_reward is not None else connectivity_bonus) - island_penalty,
        }

        pv_branches = select_pv_branches(
            mcts_wrapper.root,
            max_actions=current_config.action_batch_size,
            max_depth=4,
        )
        if pv_branches:
            executed_actions = [act for branch in pv_branches for act, _, _ in branch]
            executed_action_stats = [entry for branch in pv_branches for entry in branch]
        elif result.actions:
            executed_actions = list(result.actions)
            executed_action_stats = None
        else:
            executed_actions = []
            executed_action_stats = None

        value_target = fem_reward if (state.phase == Phase.REFINEMENT and fem_reward is not None) else None

        record_step(
            db_path,
            state,
            result,
            raw_net_val,
            policy_add,
            policy_remove,
            island_analysis=island_analysis,
            compliance_fem=compliance,
            max_displacement=max_disp,
            island_penalty=island_penalty,
            volume_fraction=vol_frac,
            reward_components=reward_components,
            executed_actions=executed_actions,
            connected_load_fraction=connected_load_fraction,
            value_target=value_target,
            fem_reward=fem_reward,
            displacement_map=displacement_map,
            executed_action_stats=executed_action_stats,
        )

        # Terminate immediately upon physical collapse or structural disconnection in Phase 2
        if fem_collapsed:
            print(f"\n>>> Physical Collapse / Disconnection in Phase 2 at step {state.current_step}! <<<")
            final_score = -1.0
            state.current_step += 1
            break

        # Apply executed edits once and update MCTS tree properly for single vs multi-branch
        if executed_actions:
            apply_micro_batch(state, executed_actions)
            if len(pv_branches) == 1:
                for act, _, _ in pv_branches[0]:
                    mcts_wrapper.step(act)
            elif not pv_branches and len(executed_actions) == 1:
                mcts_wrapper.step(executed_actions[0])
            else:
                mcts_wrapper.clear_tree()

        # Check Phase 1 -> Phase 2 transition; ALWAYS reset tree & cache across phases
        if state.phase == Phase.GROWTH:
            if check_phase_transition(state):
                print(f"\n>>> Phase Transition at step {state.current_step}! <<<")
                state.phase = Phase.REFINEMENT
                state.is_connected = True
                state.compliance_history = []
                mcts_wrapper.reset()
            elif state.current_step >= config.max_phase1_steps:
                print("\n>>> Max Phase 1 steps reached, forcing transition <<<")
                state.phase = Phase.REFINEMENT
                state.compliance_history = []
                mcts_wrapper.reset()

        # Check cheap structural terminal conditions after Phase 2 action
        if state.phase == Phase.REFINEMENT:
            terminal_reward = get_phase2_terminal_reward(state.density, state.load_config)
            if terminal_reward is not None:
                print(f"\n>>> Terminal state detected: reward={terminal_reward} <<<")
                final_score = float(terminal_reward)
                state.current_step += 1
                break

        if state.phase == Phase.REFINEMENT and fem_reward is not None:
            final_score = float(max(-1.0, min(1.0, 0.5 * fem_reward + 0.5 * raw_net_val - island_penalty)))
        else:
            final_score = float(max(-1.0, min(1.0, raw_net_val)))

        step_time = time.time() - step_start
        if state.current_step % 10 == 0:
            comp_str = f"C: {compliance:.1f}" if compliance is not None else "C: ---"
            print(
                f"Step {state.current_step:4d} | Phase: {state.phase.value:10s} | "
                f"V: {raw_net_val:+.3f} | {comp_str} | Vol: {vol_frac:.3f} | "
                f"Isl: {n_islands} | B: {connectivity_bonus:.2f} | T: {step_time:.1f}s"
            )

        state.current_step += 1
        update_game_final(
            db_path=db_path,
            game_id=state.game_id,
            final_score=final_score,
            final_compliance=last_valid_compliance,
            final_volume=float(np.mean(state.density > 0.5)),
            total_steps=state.current_step,
        )

        if state.phase == Phase.REFINEMENT and len(state.compliance_history) >= 20:
            window = state.compliance_history[-20:]
            min_c, max_c = min(window), max(window)
            if max_c > 1e-9 and (max_c - min_c) / max_c < 1e-3:
                print("\n>>> Convergence Detected! Compliance stable over 20 steps <<<")
                break

    total_time = time.time() - start_time
    vol_frac = float(np.mean(state.density > 0.5))
    print(f"\n{'='*60}")
    print(
        f"Episode Complete! Steps={state.current_step}, Phase={state.phase.value}, "
        f"Score={final_score:.4f}, Vol={vol_frac:.3f}, Time={total_time:.1f}s"
    )
    print(f"{'='*60}")

    update_game_final(
        db_path=db_path,
        game_id=state.game_id,
        final_score=final_score,
        final_compliance=last_valid_compliance,
        final_volume=vol_frac,
        total_steps=state.current_step,
    )
    backpropagate_value_targets(db_path, state.game_id)
    return state.game_id, final_score, state.current_step


def run_episode(
    model: AlphaBuilderInference,
    db_path: Path,
    config: EpisodeConfig,
) -> Tuple[str, float, int]:
    """Run a fresh self-play episode."""
    load_config = generate_random_load_config(config.resolution)
    state = create_empty_state(
        resolution=config.resolution,
        load_config=load_config,
        bc_type=config.bc_type,
    )
    engine_name = "swin-unetr" if getattr(model, "use_swin", False) else "simple"
    game_info = GameInfo(
        game_id=state.game_id,
        neural_engine=engine_name,
        checkpoint_version=f"selfplay-gen{config.generation}",
        bc_masks=state.bc_masks,
        forces=state.forces,
        load_config=state.load_config,
        bc_type=config.bc_type,
        resolution=config.resolution,
        final_compliance=None,
        final_volume=None,
        total_steps=0,
        initial_cantilever_problem_real_scale_factor=500000.0,
        stiffness_E=200e9,
    )
    save_game(db_path, game_info)
    return _run_episode_loop(state, model, db_path, config)


def resume_episode(
    model: AlphaBuilderInference,
    db_path: Path,
    game_id: str,
    config: EpisodeConfig,
) -> Tuple[str, float, int]:
    """Resume an interrupted self-play episode cleanly from the last recorded step."""
    game_info = load_game(db_path, game_id)
    if game_info is None:
        raise ValueError(f"Game {game_id} not found in database")

    last_step = get_last_step(db_path, game_id)
    if last_step is None:
        raise ValueError(f"Game {game_id} has no recorded steps to resume from")

    last_game_step = load_game_step(db_path, game_id, last_step, game_info.resolution)
    if last_game_step is None:
        raise ValueError(f"Could not load step {last_step} for game {game_id}")

    load_cfg = dict(game_info.load_config)
    load_cfg.setdefault("bc_type", game_info.bc_type)

    state = EpisodeState(
        density=last_game_step.density.copy(),
        bc_masks=game_info.bc_masks.copy(),
        forces=game_info.forces.copy(),
        load_config=load_cfg,
        current_step=last_step + 1,
        phase=last_game_step.phase,
        is_connected=last_game_step.is_connected,
        game_id=game_id,
        compliance_history=[],
    )

    if last_game_step.selected_actions:
        actions = [
            (a.channel, a.x, a.y, a.z) for a in last_game_step.selected_actions
        ]
        apply_micro_batch(state, actions)

    if state.phase == Phase.GROWTH and state.current_step >= config.max_phase1_steps:
        state.phase = Phase.REFINEMENT

    return _run_episode_loop(state, model, db_path, config)
