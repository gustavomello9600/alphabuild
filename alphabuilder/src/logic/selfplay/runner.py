"""
Self-Play Episode Runner for AlphaBuilder v4.0 / v4.1 (Stage 6).

Implements the complete MCTS-DAG self-play loop with:
- 4-Tier progressive domain/BC/load/keep-out/multi-res scenarios (`scenarios_stage6.py`)
- v4.1 MCTS-DAG upgrades: relative FPU, surface-only removals + BFS articulation safety,
  bidirectional Phase-1 growth + PBRS, composite macro-actions (ADD/REMOVE/SWAP/STOP),
  and real subtree/DAG reuse (`mcts.step(macro_act)`)
- Protected lifecycle: post-connection Grace Period, adaptive trust-region batching
  (16 -> 8 -> 4 -> 2), cycle detection, R_total patience, and best-of-episode targets
- Fast PyAMG FEM (`fast_amg`) + SED_e supervision + counterfactual SED probes (`fem_probes`)
"""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import random
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy import ndimage

from ..storage import Phase
from ..mcts.engine import (
    AlphaBuilderMCTS,
    MCTSConfig,
    PHASE1_CONFIG,
    PHASE1_V41_CONFIG,
    PHASE2_CONFIG,
    PHASE2_V41_CONFIG,
    SearchResult,
    build_state_tensor,
    select_pv_branches,
)
from ..mcts.legal_moves import (
    filter_disconnect_safe_batch,
    get_legal_add_moves,
    get_legal_remove_moves,
)
from ..mcts.node import MCTSNode, STOP_ACTION, flatten_tree_action, is_stop_action
from ...neural.inference import AlphaBuilderInference
from ..harvest.generators import (
    generate_bc_masks,
    get_load_region_bounds,
)
from .evaluator import evaluate_main_island
from .reward import (
    DISPLACEMENT_LIMIT_RAW,
    analyze_structure_islands,
    calculate_connectivity_reward,
    calculate_grace_displacement_reward,
    calculate_island_penalty,
    calculate_pbrs_potential,
    calculate_reward,
    check_structure_connectivity,
    compute_geodesic_load_distance,
    get_guidance_alpha,
    get_phase2_terminal_reward,
)
from .scenarios_stage6 import (
    HELD_OUT_CANONICAL_SCENARIOS,
    SelfPlayScenario,
    is_near_canonical_scenario,
    sample_curriculum_scenario,
)
from . import (
    GameInfo,
    backpropagate_value_targets,
    generate_game_id,
    get_last_step,
    load_game,
    load_game_step,
    record_fem_probe,
    record_step,
    save_game,
    update_game_final,
)


MCTS_CONFIG = MCTSConfig(num_simulations=80)


def sample_selfplay_scenario(
    resolution: Tuple[int, int, int] = (64, 32, 8),
    rng: Optional[random.Random] = None,
    rail_prob: float = 0.45,
    eccentric_z_prob: float = 0.45,
) -> Dict[str, Any]:
    """Sample a Tier-0 cantilever load configuration disjoint from held-out S1-S6."""
    r = rng if rng is not None else random.Random()
    nx, ny, nz = resolution
    x_lo = min(nx - 1, max(nx // 2, 24))
    x_hi = nx - 1
    y_lo = min(ny - 4, 4)
    y_hi = max(y_lo, ny - 5)

    for _ in range(128):
        lx = int(r.randint(x_lo, x_hi))
        ly = int(r.randint(y_lo, y_hi))
        if nz >= 6 and r.random() < eccentric_z_prob:
            z_start = int(r.randint(0, nz - 2))
            z_end = min(nz, z_start + 2)
        else:
            zc = nz // 2
            z_start = max(0, zc - 1)
            z_end = min(nz, zc + 1)
        bc_type = "RAIL_XY" if r.random() < rail_prob else "FULL_CLAMP"
        cfg = {"x": lx, "y": ly, "z_start": z_start, "z_end": z_end, "bc_type": bc_type}
        if resolution != (64, 32, 8) or not is_near_canonical_scenario(cfg, bc_type, tol_xy=1):
            return cfg

    return {"x": min(nx - 1, 48), "y": ny // 2, "z_start": 3, "z_end": 5, "bc_type": "FULL_CLAMP"}


@dataclass
class EpisodeConfig:
    """Configuration for a single self-play episode."""

    max_steps: int = 600
    max_phase1_steps: int = 120
    max_phase2_steps: int = 64
    target_volume_fraction: float = 0.15
    resolution: Tuple[int, int, int] = (64, 32, 8)
    bc_type: str = "FULL_CLAMP"
    use_guided_value: bool = True
    generation: int = 1
    guidance_decay_generations: int = 5
    num_simulations_p1: int = 160
    num_simulations_p2: int = 160
    mcts_batch_size: int = 8
    add_root_noise: bool = True
    p1_explore_steps: int = 12
    p1_early_temperature: float = 1.35
    p1_early_dirichlet_frac: float = 0.35
    p1_stagnation_patience: int = 20
    p2_c_puct: float = 1.0
    p2_action_batch_size: int = 16
    p2_action_batch_size_late: int = 8
    p2_batch_schedule: Tuple[int, ...] = (16, 8, 4, 2)
    p2_min_remove_candidates: int = 16
    p2_remove_prior_floor: float = 0.25
    p2_remove_probe_prob: float = 0.50
    p2_remove_probe_min_vol: float = 0.055
    p2_grace_steps: int = 5
    p2_patience: int = 12
    p2_min_stop_steps: int = 6
    p2_max_vol_terminal: float = 0.65
    use_v41_mcts: bool = True
    use_best_of_episode_targets: bool = True
    fem_backend: Optional[str] = None
    record_displacement_map: bool = False
    record_sed_map: bool = True
    tier_weights: Optional[Tuple[float, float, float, float]] = None
    seed: Optional[int] = None
    verbose: bool = True


@dataclass
class EpisodeState:
    """Mutable state for the current episode."""

    density: np.ndarray
    bc_masks: np.ndarray                 # (3, D, H, W)
    forces: np.ndarray                   # (3, D, H, W)
    load_config: Dict[str, Any]
    current_step: int = 0
    phase: Phase = Phase.GROWTH
    is_connected: bool = False
    game_id: str = ""
    compliance_history: List[float] = field(default_factory=list)
    domain_mask: Optional[np.ndarray] = None  # (D, H, W) optional keep-out mask


def create_empty_state(
    resolution: Tuple[int, int, int],
    load_config: Dict[str, Any],
    bc_type: str = "FULL_CLAMP",
    bc_masks: Optional[np.ndarray] = None,
    forces: Optional[np.ndarray] = None,
    domain_mask: Optional[np.ndarray] = None,
) -> EpisodeState:
    """Create initial empty episode state supporting arbitrary BCs, 3D forces, and keep-out masks."""
    density = np.zeros(resolution, dtype=np.float32)
    resolved_bc = str(load_config.get("bc_type", bc_type)).upper()

    if bc_masks is None:
        mask_x, mask_y, mask_z = generate_bc_masks(resolution, resolved_bc)
        bc_arr = np.stack([mask_x, mask_y, mask_z], axis=0).astype(np.float32)
    else:
        bc_arr = np.asarray(bc_masks, dtype=np.float32)

    if forces is None:
        fx = np.zeros(resolution, dtype=np.float32)
        fy = np.zeros(resolution, dtype=np.float32)
        fz = np.zeros(resolution, dtype=np.float32)
        lx, y_min, y_max, z_min, z_max = get_load_region_bounds(resolution, load_config)
        fy[lx, y_min:y_max, z_min:z_max] = -1.0
        forces_arr = np.stack([fx, fy, fz], axis=0).astype(np.float32)
    else:
        forces_arr = np.asarray(forces, dtype=np.float32)

    cfg_copy = dict(load_config)
    cfg_copy["bc_type"] = resolved_bc

    return EpisodeState(
        density=density,
        bc_masks=bc_arr,
        forces=forces_arr,
        load_config=cfg_copy,
        current_step=0,
        phase=Phase.GROWTH,
        is_connected=False,
        game_id=generate_game_id(),
        domain_mask=np.asarray(domain_mask, dtype=np.float32) if domain_mask is not None else None,
    )


def check_phase_transition(state: EpisodeState) -> bool:
    """Check if structure connects support to all load patches."""
    connected_to_support, reached_load = check_structure_connectivity(
        state.density,
        state.load_config,
        threshold=0.5,
        bc_masks=(state.bc_masks[0], state.bc_masks[1], state.bc_masks[2]),
        forces=(state.forces[0], state.forces[1], state.forces[2]),
    )
    return connected_to_support and reached_load


def _raw_predict_with_optional_op(
    model_or_fn: Any,
    state_tensor: np.ndarray,
    domain_mask: Optional[np.ndarray],
) -> Tuple[float, np.ndarray, np.ndarray, Optional[Tuple[float, float, float]]]:
    """Query raw network prediction along with optional `op_probs` when `predict_with_aux` exists."""
    op_probs: Optional[Tuple[float, float, float]] = None
    if hasattr(model_or_fn, "predict_with_aux"):
        try:
            v, pa, pr, aux = model_or_fn.predict_with_aux(state_tensor, compute_sed=False, domain_mask=domain_mask)
            raw_op = aux.get("op_probs") if isinstance(aux, dict) else None
            if raw_op is not None and len(raw_op) == 3:
                op_probs = (float(raw_op[0]), float(raw_op[1]), float(raw_op[2]))
            return float(v), pa, pr, op_probs
        except TypeError:
            pass

    pred_fn = model_or_fn.predict if hasattr(model_or_fn, "predict") else model_or_fn
    try:
        v, pa, pr = pred_fn(state_tensor, domain_mask=domain_mask)
    except TypeError:
        v, pa, pr = pred_fn(state_tensor)
    return float(v), pa, pr, None


def execute_mcts_step(
    state: EpisodeState,
    mcts: AlphaBuilderMCTS,
    raw_predict_fn: Optional[Callable[..., Tuple[float, np.ndarray, np.ndarray]]] = None,
    stop_value: Optional[float] = None,
    raw_model: Optional[Any] = None,
) -> Tuple[SearchResult, float, np.ndarray, np.ndarray]:
    """Run MCTS search and also capture raw unshaped network prediction at root."""
    bc_tuple = (state.bc_masks[0], state.bc_masks[1], state.bc_masks[2])
    force_tuple = (state.forces[0], state.forces[1], state.forces[2])
    state_tensor = build_state_tensor(state.density, bc_tuple, force_tuple)

    if raw_model is not None:
        value, policy_add, policy_remove, op_probs = _raw_predict_with_optional_op(
            raw_model, state_tensor, state.domain_mask
        )
    elif raw_predict_fn is not None:
        value, policy_add, policy_remove, op_probs = _raw_predict_with_optional_op(
            raw_predict_fn, state_tensor, state.domain_mask
        )
    else:
        value, policy_add, policy_remove = mcts.predict_fn(state_tensor)
        op_probs = None

    result = mcts.search(
        density=state.density,
        bc_masks=bc_tuple,
        forces=force_tuple,
        domain_mask=state.domain_mask,
        stop_value=stop_value,
        root_op_probs=op_probs,
    )
    return result, float(value), policy_add, policy_remove


def apply_micro_batch(
    state: EpisodeState,
    actions: List[Tuple[int, int, int, int]],
) -> None:
    """Apply micro-batch of unitary voxel edits in place."""
    for channel, x, y, z in actions:
        state.density[x, y, z] = 1.0 if channel == 0 else 0.0


def _build_p2_batch_schedule(early_k: int, late_k: int, custom_sched: Tuple[int, ...]) -> List[int]:
    """Build monotonic coarse-to-fine batch schedule compatible with both full grids and unit tests."""
    if early_k == 16 and late_k == 8 and custom_sched:
        return [int(v) for v in custom_sched if v >= 1]
    cands = [int(early_k), max(2, int(early_k) // 2), int(late_k), max(2, int(late_k) // 2)]
    out: List[int] = []
    for c in cands:
        if c >= 1 and (not out or c < out[-1]):
            out.append(c)
    return out or [max(1, int(late_k))]


def _build_sed_or_policy_probe_batch(
    density: np.ndarray,
    bc_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray],
    force_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray],
    load_config: Dict[str, Any],
    policy_remove: np.ndarray,
    sed_map: Optional[np.ndarray] = None,
    domain_mask: Optional[np.ndarray] = None,
    max_k: int = 12,
    min_vol_frac: float = 0.015,
) -> List[Tuple[int, int, int, int]]:
    """
    Build a counterfactual Phase-2 batch that either:
    - With `sed_map`: removes lowest-SED surface voxels (safely keeping connectivity)
      AND adds highest-neighbor-SED legal boundary voxels (shifting mass from neutral core to stressed chords).
    - Without `sed_map`: removes top policy_remove legal surface voxels safely keeping connectivity.
    """
    base_isl = analyze_structure_islands(
        density, load_config, bc_masks=bc_tuple, forces=force_tuple
    )
    max_allowed_islands = max(1, int(base_isl["n_islands"]))
    cand_density = density.copy()
    chosen: List[Tuple[int, int, int, int]] = []

    valid_rem = get_legal_remove_moves(
        density, threshold=0.5, bc_masks=bc_tuple, forces=force_tuple, surface_only=True
    )
    rem_coords = np.argwhere(valid_rem > 0)

    n_rem_target = max(1, max_k // 2) if sed_map is not None else max_k
    if rem_coords.size > 0:
        if sed_map is not None:
            rem_scores = sed_map[rem_coords[:, 0], rem_coords[:, 1], rem_coords[:, 2]]
            order = np.argsort(rem_scores)
        else:
            rem_scores = policy_remove[rem_coords[:, 0], rem_coords[:, 1], rem_coords[:, 2]]
            order = np.argsort(rem_scores)[::-1]

        limit = min(len(order), max(n_rem_target * 5, 40))
        for idx in order[:limit]:
            if len(chosen) >= n_rem_target:
                break
            x, y, z = int(rem_coords[idx, 0]), int(rem_coords[idx, 1]), int(rem_coords[idx, 2])
            cand_density[x, y, z] = 0.0
            if float(np.mean(cand_density > 0.5)) < min_vol_frac:
                cand_density[x, y, z] = 1.0
                continue
            post_isl = analyze_structure_islands(
                cand_density, load_config, bc_masks=bc_tuple, forces=force_tuple
            )
            if post_isl["is_connected"] and int(post_isl["n_islands"]) <= max_allowed_islands:
                chosen.append((1, x, y, z))
            else:
                cand_density[x, y, z] = 1.0

    if sed_map is not None and len(chosen) < max_k:
        n_add_needed = max_k - len(chosen)
        valid_add = get_legal_add_moves(
            density, threshold=0.5, bc_masks=bc_tuple, domain_mask=domain_mask
        )
        add_coords = np.argwhere(valid_add > 0)
        if add_coords.size > 0:
            sed_nbr = ndimage.maximum_filter(sed_map, size=3)
            add_scores = sed_nbr[add_coords[:, 0], add_coords[:, 1], add_coords[:, 2]]
            add_order = np.argsort(add_scores)[::-1]
            for idx in add_order[:n_add_needed]:
                x, y, z = int(add_coords[idx, 0]), int(add_coords[idx, 1]), int(add_coords[idx, 2])
                chosen.append((0, x, y, z))

    return chosen


def _inject_winning_probe_visits(
    result: SearchResult,
    probe_actions: List[Tuple[int, int, int, int]],
    probe_reward: float,
) -> List[Tuple[Tuple[int, int, int, int], int, float]]:
    """Record counterfactual winning actions into root visit counts for policy distillation."""
    root = getattr(result, "root", None)
    existing_total = sum(int(c.visit_count) for c in root.children.values()) if (root and root.children) else 0
    per_action_base = max(12, existing_total // max(len(probe_actions), 1))
    action_stats: List[Tuple[Tuple[int, int, int, int], int, float]] = []

    if root is not None:
        macro_key = tuple(sorted(probe_actions))
        child_macro = root.children.get(macro_key)
        if child_macro is not None:
            child_macro.visit_count += per_action_base * max(1, len(probe_actions))
            child_macro.value_sum += float(per_action_base * max(1, len(probe_actions))) * float(probe_reward)
            child_macro.max_value = max(child_macro.max_value, float(probe_reward))

    for rank_idx, act in enumerate(probe_actions):
        boost = per_action_base + (len(probe_actions) - rank_idx)
        if result.visit_distribution is not None:
            result.visit_distribution[act] = int(result.visit_distribution.get(act, 0)) + boost
        if root is not None and len(root.children) > 0:
            sample_key = next(iter(root.children.keys()))
            is_single_tree = isinstance(sample_key, tuple) and len(sample_key) == 4 and isinstance(sample_key[0], int)
            if is_single_tree:
                child = root.children.get(act)
                if child is None:
                    child = MCTSNode(prior=1.0 / max(len(probe_actions), 1), action_to_parent=act, parent=root)
                    root.children[act] = child
                child.visit_count += boost
                child.value_sum += float(boost) * float(probe_reward)
                child.max_value = max(child.max_value, float(probe_reward))
                v_count = int(child.visit_count)
            else:
                v_count = int(result.visit_distribution[act]) if result.visit_distribution else boost
        else:
            v_count = int(result.visit_distribution[act]) if result.visit_distribution else boost
        action_stats.append((act, v_count, float(probe_reward)))

    return action_stats


def _density_hash(density: np.ndarray) -> str:
    return hashlib.md5((density > 0.5).astype(np.uint8).tobytes()).hexdigest()


def _run_episode_loop(
    state: EpisodeState,
    model: AlphaBuilderInference,
    db_path: Path,
    config: EpisodeConfig,
    tier: int = 0,
    family: str = "cantilever",
) -> Tuple[str, float, int, Dict[str, Any]]:
    """Unified execution loop shared cleanly by run_episode and resume_episode."""
    load_config = state.load_config
    bc_type = str(load_config.get("bc_type", config.bc_type)).upper()
    rng = random.Random(config.seed) if config.seed is not None else random.Random()
    bc_masks_arg = state.bc_masks
    forces_arg = state.forces

    alpha_guide = (
        get_guidance_alpha(
            generation=config.generation,
            decay_generations=config.guidance_decay_generations,
        )
        if config.use_guided_value
        else 0.0
    )

    p1_template = PHASE1_V41_CONFIG if config.use_v41_mcts else PHASE1_CONFIG
    p2_template = PHASE2_V41_CONFIG if config.use_v41_mcts else PHASE2_CONFIG
    bidirectional_p1 = bool(p1_template.bidirectional_p1)
    geodesic_dist = compute_geodesic_load_distance(
        state.density.shape,
        (state.forces[0], state.forces[1], state.forces[2]),
        domain_mask=state.domain_mask,
    )
    current_phi_root = [0.0]

    def _shape_value(net_val: float, density_grid: np.ndarray, masks, force_vecs) -> float:
        is_p1 = state.phase == Phase.GROWTH
        isl = analyze_structure_islands(
            density_grid,
            load_config,
            bc_masks=masks,
            forces=force_vecs,
            bidirectional=(bidirectional_p1 and is_p1),
        )
        total_voxels = int((density_grid > 0.5).sum())
        eff_isl = (
            1
            if (bidirectional_p1 and is_p1 and not isl["is_connected"] and isl["loose_voxels"] == 0)
            else isl["n_islands"]
        )
        island_pen = calculate_island_penalty(eff_isl, isl["loose_voxels"], total_voxels)

        if state.phase == Phase.REFINEMENT:
            if not isl["is_connected"]:
                return -1.0
            val = float(net_val) - island_pen
            return max(-1.0, min(1.0, val))

        if alpha_guide > 0.0:
            phi_leaf, _ = calculate_pbrs_potential(
                density_grid,
                masks,
                force_vecs,
                load_config,
                bidirectional=bidirectional_p1,
                dist_field=geodesic_dist,
            )
            shaping = alpha_guide * (phi_leaf - current_phi_root[0])
            return max(-1.0, min(1.0, float(net_val) + shaping))
        return max(-1.0, min(1.0, float(net_val)))

    def hybrid_predictor(state_tensor: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        try:
            net_val, p_add, p_remove = model.predict(state_tensor, domain_mask=state.domain_mask)
        except TypeError:
            net_val, p_add, p_remove = model.predict(state_tensor)
        masks = (state_tensor[1], state_tensor[2], state_tensor[3])
        force_vecs = (state_tensor[4], state_tensor[5], state_tensor[6])
        return _shape_value(net_val, state_tensor[0], masks, force_vecs), p_add, p_remove

    def hybrid_batch_predictor(
        state_tensors: List[np.ndarray],
    ) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
        try:
            net_vals, ps_add, ps_remove = model.predict_batch(state_tensors, domain_mask=state.domain_mask)
        except TypeError:
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

    # Resolution-scaled P1 batch & budget (canonical 64x32x8 keeps exact baseline values)
    domain_scale = (float(state.density.size) / 16384.0) ** (1.0 / 3.0)
    p1_action_batch = max(4, min(p1_template.action_batch_size, int(round(p1_template.action_batch_size * domain_scale))))
    eff_max_p1_steps = config.max_phase1_steps

    p2_schedule = _build_p2_batch_schedule(
        config.p2_action_batch_size, config.p2_action_batch_size_late, config.p2_batch_schedule
    )

    base_p1_cfg = p1_template._replace(
        num_simulations=config.num_simulations_p1,
        batch_size=config.mcts_batch_size,
        action_batch_size=p1_action_batch,
        add_root_noise=config.add_root_noise,
        min_remove_candidates=0,
        remove_prior_floor=0.0,
    )
    base_p2_cfg = p2_template._replace(
        num_simulations=config.num_simulations_p2,
        batch_size=config.mcts_batch_size,
        action_batch_size=p2_schedule[0],
        c_puct=config.p2_c_puct,
        add_root_noise=config.add_root_noise,
        min_remove_candidates=config.p2_min_remove_candidates,
        remove_prior_floor=config.p2_remove_prior_floor,
    )

    init_cfg = base_p1_cfg if state.phase == Phase.GROWTH else base_p2_cfg
    mcts_wrapper = AlphaBuilderMCTS(
        predict_fn=hybrid_predictor,
        num_simulations=init_cfg.num_simulations,
        batch_size=init_cfg.batch_size,
        action_batch_size=init_cfg.action_batch_size,
        c_puct=init_cfg.c_puct,
        top_k_expansion=init_cfg.top_k_expansion,
        predict_batch_fn=hybrid_batch_predictor,
    )
    mcts_wrapper.config = init_cfg

    final_score = -1.0
    last_valid_compliance: Optional[float] = None
    last_recorded_step = -1
    phase1_steps_to_connect: Optional[int] = None
    phase2_steps_run = 0
    p2_add_actions = 0
    p2_rem_actions = 0
    probes_run = 0
    probes_adopted = 0
    grace_saves = 0
    tree_reuse_hits = 0
    end_reason = "MAX_STEPS"
    start_time = time.time()

    best_p1_phi = -float("inf")
    p1_no_improve = 0
    best_p2_score = -float("inf")
    best_p2_compliance: Optional[float] = None
    best_p2_volume: Optional[float] = None
    best_p2_step: Optional[int] = None
    p2_no_improve_streak = 0
    prev_p2_score: Optional[float] = None
    p2_batch_idx = 0
    seen_p2_hashes: Set[str] = set()

    bc_tuple = (state.bc_masks[0], state.bc_masks[1], state.bc_masks[2])
    force_tuple = (state.forces[0], state.forces[1], state.forces[2])
    has_forces = float(np.abs(state.forces).sum()) > 1e-6

    while state.current_step < config.max_steps:
        step_start = time.time()
        cur_phase = state.phase
        if cur_phase == Phase.GROWTH:
            if config.add_root_noise and state.current_step < config.p1_explore_steps:
                current_config = base_p1_cfg._replace(
                    policy_temperature=config.p1_early_temperature,
                    root_exploration_fraction=config.p1_early_dirichlet_frac,
                )
            else:
                current_config = base_p1_cfg
        else:
            p2_batch = p2_schedule[min(p2_batch_idx, len(p2_schedule) - 1)]
            allow_stop = bool(config.use_v41_mcts and phase2_steps_run >= config.p2_min_stop_steps)
            current_config = base_p2_cfg._replace(
                action_batch_size=p2_batch,
                allow_stop_action=allow_stop,
            )

        mcts_wrapper.config = current_config

        vol_frac = float(np.mean(state.density > 0.5))
        island_analysis = analyze_structure_islands(
            state.density,
            state.load_config,
            bc_masks=bc_tuple,
            forces=force_tuple if has_forces else None,
            bidirectional=(bidirectional_p1 and cur_phase == Phase.GROWTH),
        )
        n_islands = island_analysis["n_islands"]
        loose_voxels = island_analysis["loose_voxels"]
        total_voxels = int((state.density > 0.5).sum())
        eff_isl = (
            1
            if (bidirectional_p1 and cur_phase == Phase.GROWTH and not island_analysis["is_connected"] and loose_voxels == 0)
            else n_islands
        )
        island_penalty = calculate_island_penalty(eff_isl, loose_voxels, total_voxels)

        compliance: Optional[float] = None
        max_disp: Optional[float] = None
        fem_reward: Optional[float] = None
        displacement_map: Optional[np.ndarray] = None
        sed_map: Optional[np.ndarray] = None
        fem_collapsed = False
        root_stop_val: Optional[float] = None

        # Evaluate physical FEM state first so MCTS STOP has exact ground-truth terminal value
        if cur_phase == Phase.REFINEMENT:
            seen_p2_hashes.add(_density_hash(state.density))
            if not island_analysis["is_connected"]:
                fem_reward = -1.0
                fem_collapsed = True
                root_stop_val = -1.0
            else:
                fem_result = evaluate_main_island(
                    state.density,
                    state.load_config,
                    island_analysis["main_island_mask"],
                    bc_type=bc_type,
                    backend=config.fem_backend,
                    bc_masks=bc_masks_arg,
                    forces=forces_arg,
                    compute_sed=config.record_sed_map,
                )
                if fem_result.valid:
                    compliance = float(fem_result.compliance)
                    max_disp = float(fem_result.max_displacement)
                    displacement_map = fem_result.displacement_map if config.record_displacement_map else None
                    sed_map = fem_result.sed_map if config.record_sed_map else None
                    fem_reward = float(
                        calculate_reward(
                            compliance=compliance,
                            vol_frac=vol_frac,
                            is_valid=True,
                            max_displacement=max_disp,
                        )
                    )
                    last_valid_compliance = compliance
                    if fem_reward == -1.0:
                        if max_disp is not None and max_disp > DISPLACEMENT_LIMIT_RAW and phase2_steps_run < config.p2_grace_steps:
                            fem_reward = calculate_grace_displacement_reward(max_disp)
                            fem_collapsed = False
                            grace_saves += 1
                        else:
                            fem_collapsed = True
                    else:
                        state.compliance_history.append(compliance)

                    if not fem_collapsed:
                        cur_total = float(max(-1.0, min(1.0, fem_reward - island_penalty)))
                        root_stop_val = cur_total
                        if cur_total > best_p2_score + 1e-3:
                            best_p2_score = cur_total
                            best_p2_compliance = compliance
                            best_p2_volume = vol_frac
                            best_p2_step = state.current_step
                            p2_no_improve_streak = 0
                        else:
                            p2_no_improve_streak += 1
                            if cur_total > best_p2_score:
                                best_p2_score = cur_total
                                best_p2_compliance = compliance
                                best_p2_volume = vol_frac
                                best_p2_step = state.current_step
                        if (p2_no_improve_streak > 0 and p2_no_improve_streak % 3 == 0) or (
                            prev_p2_score is not None and cur_total < prev_p2_score - 0.005
                        ):
                            if p2_batch_idx < len(p2_schedule) - 1:
                                p2_batch_idx += 1
                        prev_p2_score = cur_total
                else:
                    fem_reward = -1.0
                    fem_collapsed = True
                    root_stop_val = -1.0

        current_phi_root[0], _ = calculate_pbrs_potential(
            state.density,
            bc_tuple,
            force_tuple,
            state.load_config,
            bidirectional=bidirectional_p1,
            dist_field=geodesic_dist,
        )
        if cur_phase == Phase.GROWTH:
            if current_phi_root[0] > best_p1_phi + 1e-3:
                best_p1_phi = current_phi_root[0]
                p1_no_improve = 0
            else:
                p1_no_improve += 1

        result, raw_net_val, policy_add, policy_remove = execute_mcts_step(
            state,
            mcts_wrapper,
            raw_predict_fn=model.predict,
            stop_value=root_stop_val,
            raw_model=model,
        )

        connectivity_bonus = 0.0
        connected_load_fraction = 0.0
        if cur_phase == Phase.GROWTH:
            connectivity_bonus, connected_load_fraction = calculate_connectivity_reward(
                state.density, bc_tuple, force_tuple
            )
        else:
            _, connected_load_fraction = calculate_connectivity_reward(
                state.density, bc_tuple, force_tuple
            )

        best_child_pair = mcts_wrapper.root.get_best_child() if mcts_wrapper.root is not None else None
        chosen_tree_action = best_child_pair[0] if best_child_pair is not None else None
        stopped_by_mcts = bool(
            cur_phase == Phase.REFINEMENT
            and not fem_collapsed
            and chosen_tree_action is not None
            and is_stop_action(chosen_tree_action)
            and phase2_steps_run >= config.p2_min_stop_steps
        )

        executed_actions: List[Tuple[int, int, int, int]] = []
        executed_action_stats = None
        pv_branches: List[List[Tuple[Tuple[int, int, int, int], int, float]]] = []

        if stopped_by_mcts:
            executed_actions = []
            executed_action_stats = None
        else:
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

        adopted_probe = False
        if (
            cur_phase == Phase.REFINEMENT
            and not fem_collapsed
            and not stopped_by_mcts
            and fem_reward is not None
            and config.p2_remove_probe_prob > 0.0
            and vol_frac >= config.p2_remove_probe_min_vol
            and rng.random() < config.p2_remove_probe_prob
        ):
            probe_actions = _build_sed_or_policy_probe_batch(
                state.density,
                bc_tuple,
                force_tuple,
                state.load_config,
                policy_remove,
                sed_map=sed_map,
                domain_mask=state.domain_mask,
                max_k=current_config.action_batch_size,
            )
            if probe_actions:
                probes_run += 1
                probe_density = state.density.copy()
                for ch, ax, ay, az in probe_actions:
                    probe_density[ax, ay, az] = 1.0 if ch == 0 else 0.0
                probe_vol = float(np.mean(probe_density > 0.5))
                probe_isl = analyze_structure_islands(
                    probe_density, state.load_config, bc_masks=bc_tuple, forces=force_tuple
                )
                probe_sed: Optional[np.ndarray] = None
                if probe_isl["is_connected"]:
                    probe_fem = evaluate_main_island(
                        probe_density,
                        state.load_config,
                        probe_isl["main_island_mask"],
                        bc_type=bc_type,
                        backend=config.fem_backend,
                        bc_masks=bc_masks_arg,
                        forces=forces_arg,
                        compute_sed=config.record_sed_map,
                    )
                    probe_comp = float(probe_fem.compliance) if probe_fem.valid else None
                    probe_disp = float(probe_fem.max_displacement) if probe_fem.valid else None
                    probe_sed = probe_fem.sed_map if (probe_fem.valid and config.record_sed_map) else None
                    probe_fem_r = (
                        float(calculate_reward(probe_comp, probe_vol, True, probe_disp))
                        if probe_fem.valid and probe_comp is not None
                        else -1.0
                    )
                else:
                    probe_comp = None
                    probe_disp = None
                    probe_fem_r = -1.0

                probe_tag = "sed_hybrid_topk" if sed_map is not None else "remove_topk"
                record_fem_probe(
                    db_path=db_path,
                    game_id=state.game_id,
                    step=state.current_step,
                    probe_type=probe_tag,
                    density=probe_density,
                    is_connected=bool(probe_isl["is_connected"]),
                    compliance_fem=probe_comp,
                    max_displacement=probe_disp,
                    volume_fraction=probe_vol,
                    fem_reward=probe_fem_r,
                    sed_map=probe_sed,
                )

                has_remove_in_mcts = any(ch == 1 for ch, _, _, _ in executed_actions)
                if probe_isl["is_connected"] and probe_fem_r > -1.0 and ((not has_remove_in_mcts) or sed_map is not None):
                    mcts_next_r = float(fem_reward)
                    if executed_actions:
                        mcts_next_dens = state.density.copy()
                        for ch, ax, ay, az in executed_actions:
                            mcts_next_dens[ax, ay, az] = 1.0 if ch == 0 else 0.0
                        mcts_isl = analyze_structure_islands(
                            mcts_next_dens, state.load_config, bc_masks=bc_tuple, forces=force_tuple
                        )
                        if mcts_isl["is_connected"]:
                            mcts_fem = evaluate_main_island(
                                mcts_next_dens,
                                state.load_config,
                                mcts_isl["main_island_mask"],
                                bc_type=bc_type,
                                backend=config.fem_backend,
                                bc_masks=bc_masks_arg,
                                forces=forces_arg,
                            )
                            if mcts_fem.valid:
                                mcts_next_r = float(
                                    calculate_reward(
                                        float(mcts_fem.compliance),
                                        float(np.mean(mcts_next_dens > 0.5)),
                                        True,
                                        float(mcts_fem.max_displacement),
                                    )
                                )
                            else:
                                mcts_next_r = -1.0
                        else:
                            mcts_next_r = -1.0

                    if probe_fem_r > mcts_next_r and probe_fem_r >= fem_reward - 0.002:
                        executed_actions = probe_actions
                        executed_action_stats = _inject_winning_probe_visits(
                            result, probe_actions, probe_fem_r
                        )
                        adopted_probe = True
                        probes_adopted += 1

        batch_filtered_by_bfs = False
        if (
            cur_phase == Phase.REFINEMENT
            and not fem_collapsed
            and not adopted_probe
            and config.use_v41_mcts
            and executed_actions
            and any(ch == 1 for ch, _, _, _ in executed_actions)
        ):
            safe_actions = filter_disconnect_safe_batch(
                state.density,
                executed_actions,
                state.load_config,
                bc_masks=bc_tuple,
                forces=force_tuple if has_forces else None,
            )
            if safe_actions != executed_actions:
                batch_filtered_by_bfs = True
                executed_actions = safe_actions
                if executed_action_stats:
                    allowed_set = set(safe_actions)
                    executed_action_stats = [e for e in executed_action_stats if e[0] in allowed_set]

        reward_components = {
            "base_reward": 0.0,
            "connectivity_bonus": float(connectivity_bonus),
            "connected_load_fraction": float(connected_load_fraction),
            "fem_reward": float(fem_reward) if fem_reward is not None else 0.0,
            "island_penalty": float(island_penalty),
            "loose_penalty": calculate_island_penalty(1, loose_voxels, total_voxels, penalty_per_island=0.0)
            if loose_voxels > 0
            else 0.0,
            "volume_penalty": 0.0,
            "validity_penalty": -1.0 if fem_collapsed else 0.0,
            "n_islands": int(n_islands),
            "loose_voxels": int(loose_voxels),
            "disconnected_volume_fraction": loose_voxels / total_voxels if total_voxels > 0 else 0.0,
            "total": (float(fem_reward) if fem_reward is not None else float(connectivity_bonus)) - float(island_penalty),
        }

        value_target = (
            float(max(-1.0, min(1.0, fem_reward - island_penalty)))
            if (cur_phase == Phase.REFINEMENT and fem_reward is not None)
            else None
        )

        record_step(
            db_path,
            state,
            result,
            float(raw_net_val),
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
            sed_map=sed_map,
        )
        last_recorded_step = state.current_step

        if fem_collapsed:
            if config.verbose:
                print(f"\n>>> Physical Collapse / Disconnection in Phase 2 at step {state.current_step}! <<<")
            final_score = -1.0
            end_reason = "COLLAPSE"
            state.current_step += 1
            break

        if stopped_by_mcts:
            final_score = float(max(-1.0, min(1.0, (fem_reward or -1.0) - island_penalty)))
            end_reason = "MCTS_STOP"
            state.current_step += 1
            break

        if cur_phase == Phase.REFINEMENT:
            phase2_steps_run += 1
            p2_add_actions += sum(1 for c, _, _, _ in executed_actions if c == 0)
            p2_rem_actions += sum(1 for c, _, _, _ in executed_actions if c == 1)

        if executed_actions:
            apply_micro_batch(state, executed_actions)

        if cur_phase == Phase.REFINEMENT:
            terminal_reward = get_phase2_terminal_reward(
                state.density,
                state.load_config,
                bc_masks=bc_tuple,
                forces=force_tuple,
                max_vol_frac=config.p2_max_vol_terminal,
            )
            if terminal_reward is not None:
                if config.verbose:
                    print(f"\n>>> Terminal state detected: reward={terminal_reward} <<<")
                final_score = float(terminal_reward)
                end_reason = "TERMINAL"
                state.current_step += 1
                break

            post_h = _density_hash(state.density)
            if phase2_steps_run >= config.p2_min_stop_steps and post_h in seen_p2_hashes:
                end_reason = "CYCLE_DETECTED"
                state.current_step += 1
                break

            if p2_no_improve_streak >= config.p2_patience:
                end_reason = "REWARD_CONVERGENCE"
                state.current_step += 1
                break

            if phase2_steps_run >= config.max_phase2_steps:
                end_reason = "MAX_PHASE2_STEPS"
                state.current_step += 1
                break

        # Clean tree/DAG advancement across both macro-edges and unitary branches
        if executed_actions and not adopted_probe and not batch_filtered_by_bfs:
            if (
                current_config.macro_action_mode
                and chosen_tree_action is not None
                and not is_stop_action(chosen_tree_action)
                and mcts_wrapper.step(chosen_tree_action)
            ):
                tree_reuse_hits += 1
                if cur_phase == Phase.GROWTH:
                    mcts_wrapper.eval_cache = {}
            elif len(pv_branches) == 1 and pv_branches[0]:
                reused_all = True
                for act, _, _ in pv_branches[0]:
                    if not mcts_wrapper.step(act):
                        reused_all = False
                        break
                if reused_all:
                    tree_reuse_hits += 1
                if cur_phase == Phase.GROWTH:
                    mcts_wrapper.eval_cache = {}
            elif not pv_branches and len(executed_actions) == 1:
                if mcts_wrapper.step(executed_actions[0]):
                    tree_reuse_hits += 1
                if cur_phase == Phase.GROWTH:
                    mcts_wrapper.eval_cache = {}
            else:
                if cur_phase == Phase.GROWTH:
                    mcts_wrapper.reset()
                else:
                    mcts_wrapper.clear_tree()
        else:
            if cur_phase == Phase.GROWTH:
                mcts_wrapper.reset()
            else:
                mcts_wrapper.clear_tree()

        if cur_phase == Phase.GROWTH:
            if check_phase_transition(state):
                phase1_steps_to_connect = state.current_step + 1
                if config.verbose:
                    print(f"\n>>> Phase Transition at step {phase1_steps_to_connect}! <<<")
                state.phase = Phase.REFINEMENT
                state.is_connected = True
                state.compliance_history = []
                mcts_wrapper.reset()
            elif p1_no_improve >= config.p1_stagnation_patience and state.current_step + 1 >= config.p1_stagnation_patience:
                final_score = -1.0
                end_reason = "PHASE1_STAGNATION"
                state.current_step += 1
                break
            elif state.current_step + 1 >= eff_max_p1_steps:
                if config.verbose:
                    print(f"\n>>> Phase 1 timeout at step {state.current_step + 1} <<<")
                final_score = -1.0
                end_reason = "PHASE1_TIMEOUT"
                state.current_step += 1
                break

        if cur_phase == Phase.REFINEMENT and fem_reward is not None:
            final_score = float(max(-1.0, min(1.0, fem_reward - island_penalty)))
        else:
            final_score = float(max(-1.0, min(1.0, raw_net_val)))

        step_time = time.time() - step_start
        if config.verbose and state.current_step % 10 == 0:
            comp_str = f"C: {compliance:.1f}" if compliance is not None else "C: ---"
            print(
                f"Step {state.current_step:4d} | Phase: {cur_phase.value:10s} | "
                f"V: {raw_net_val:+.3f} | {comp_str} | Vol: {vol_frac:.3f} | "
                f"Isl: {n_islands} | B: {connectivity_bonus:.2f} | T: {step_time:.1f}s"
            )

        state.current_step += 1

    # Record final post-action terminal snapshot grounded in real FEM
    isl_final = analyze_structure_islands(
        state.density, state.load_config, bc_masks=bc_tuple, forces=force_tuple
    )
    vol_final = float(np.mean(state.density > 0.5))
    vox_final = int((state.density > 0.5).sum())
    isl_pen_final = calculate_island_penalty(isl_final["n_islands"], isl_final["loose_voxels"], vox_final)
    _, conn_frac_final = calculate_connectivity_reward(state.density, bc_tuple, force_tuple)

    comp_final: Optional[float] = None
    max_disp_final: Optional[float] = None
    fem_reward_final: Optional[float] = None
    disp_map_final: Optional[np.ndarray] = None
    sed_map_final: Optional[np.ndarray] = None
    if isl_final["is_connected"]:
        fem_res_f = evaluate_main_island(
            state.density,
            state.load_config,
            isl_final["main_island_mask"],
            bc_type=bc_type,
            backend=config.fem_backend,
            bc_masks=bc_masks_arg,
            forces=forces_arg,
            compute_sed=config.record_sed_map,
        )
        if fem_res_f.valid:
            comp_final = float(fem_res_f.compliance)
            max_disp_final = float(fem_res_f.max_displacement)
            disp_map_final = fem_res_f.displacement_map if config.record_displacement_map else None
            sed_map_final = fem_res_f.sed_map if config.record_sed_map else None
            fem_reward_final = float(calculate_reward(comp_final, vol_final, True, max_disp_final))
            last_valid_compliance = comp_final
            final_score = float(max(-1.0, min(1.0, fem_reward_final - isl_pen_final)))
            if final_score > best_p2_score:
                best_p2_score = final_score
                best_p2_compliance = comp_final
                best_p2_volume = vol_final
                best_p2_step = state.current_step
        else:
            final_score = -1.0
    else:
        final_score = -1.0

    state.is_connected = bool(isl_final["is_connected"])

    if state.current_step != last_recorded_step:
        st_final = build_state_tensor(state.density, bc_tuple, force_tuple)
        raw_val_final, pa_final, pr_final, _ = _raw_predict_with_optional_op(
            model, st_final, state.domain_mask
        )
        dummy_res = SearchResult(
            actions=[],
            visit_distribution={},
            root_value=float(raw_val_final),
            num_simulations=0,
            root=None,
        )
        rc_final = {
            "base_reward": 0.0,
            "connectivity_bonus": 0.0,
            "connected_load_fraction": float(conn_frac_final),
            "fem_reward": float(fem_reward_final) if fem_reward_final is not None else -1.0,
            "island_penalty": float(isl_pen_final),
            "loose_penalty": 0.0,
            "volume_penalty": 0.0,
            "validity_penalty": 0.0 if isl_final["is_connected"] else -1.0,
            "n_islands": int(isl_final["n_islands"]),
            "loose_voxels": int(isl_final["loose_voxels"]),
            "disconnected_volume_fraction": isl_final["loose_voxels"] / vox_final if vox_final > 0 else 0.0,
            "total": float(final_score),
        }
        record_step(
            db_path,
            state,
            dummy_res,
            float(raw_val_final),
            pa_final,
            pr_final,
            island_analysis=isl_final,
            compliance_fem=comp_final,
            max_displacement=max_disp_final,
            island_penalty=isl_pen_final,
            volume_fraction=vol_final,
            reward_components=rc_final,
            executed_actions=[],
            connected_load_fraction=conn_frac_final,
            value_target=float(final_score) if isl_final["is_connected"] else -1.0,
            fem_reward=fem_reward_final,
            displacement_map=disp_map_final,
            executed_action_stats=None,
            sed_map=sed_map_final,
        )

    total_recorded_steps = state.current_step + 1
    total_time = time.time() - start_time
    if config.verbose:
        print(f"\n{'='*60}")
        print(
            f"Episode Complete ({end_reason}, T{tier}:{family})! Steps={total_recorded_steps}, Phase={state.phase.value}, "
            f"Score={final_score:.4f}, Best={best_p2_score:.4f}, Vol={vol_final:.3f}, C={comp_final}, Time={total_time:.1f}s"
        )
        print(f"{'='*60}")

    update_game_final(
        db_path=db_path,
        game_id=state.game_id,
        final_score=final_score,
        final_compliance=last_valid_compliance,
        final_volume=vol_final,
        total_steps=total_recorded_steps,
    )
    backpropagate_value_targets(
        db_path,
        state.game_id,
        use_best_of_episode=config.use_best_of_episode_targets,
    )

    stats = {
        "game_id": state.game_id,
        "tier": int(tier),
        "family": str(family),
        "resolution": list(state.density.shape),
        "bc_type": bc_type,
        "load_config": dict(state.load_config),
        "end_reason": end_reason,
        "connected_final": bool(isl_final["is_connected"]),
        "reached_phase2": bool(phase1_steps_to_connect is not None or state.phase == Phase.REFINEMENT),
        "phase1_steps": phase1_steps_to_connect,
        "phase2_steps": phase2_steps_run,
        "total_steps": total_recorded_steps,
        "final_score": float(final_score),
        "best_score": float(best_p2_score) if best_p2_score > -float("inf") else float(final_score),
        "best_step": best_p2_step,
        "best_compliance": best_p2_compliance,
        "best_volume": best_p2_volume,
        "final_compliance": comp_final,
        "final_volume": float(vol_final),
        "p2_add_actions": p2_add_actions,
        "p2_rem_actions": p2_rem_actions,
        "probes_run": probes_run,
        "probes_adopted": probes_adopted,
        "grace_saves": grace_saves,
        "tree_reuse_hits": tree_reuse_hits,
        "dag_transpositions": int(mcts_wrapper.total_dag_hits),
        "elapsed_sec": round(total_time, 2),
    }
    return state.game_id, float(final_score), total_recorded_steps, stats


def run_episode_with_stats(
    model: AlphaBuilderInference,
    db_path: Path,
    config: EpisodeConfig,
    load_config: Optional[Dict[str, Any]] = None,
    scenario: Optional[SelfPlayScenario] = None,
) -> Tuple[str, float, int, Dict[str, Any]]:
    """Run a fresh self-play episode with optional `SelfPlayScenario` or `load_config` and return telemetry."""
    rng = random.Random(config.seed) if config.seed is not None else None
    if scenario is not None:
        res = scenario.resolution
        cfg = dict(scenario.load_config)
        resolved_bc = scenario.bc_type
        bc_masks = scenario.bc_masks
        forces = scenario.forces
        domain_mask = scenario.domain_mask
        tier = scenario.tier
        family = scenario.family
    else:
        res = config.resolution
        cfg = dict(load_config) if load_config is not None else None
        if cfg is None:
            scen = sample_curriculum_scenario(
                generation=config.generation,
                tier_weights=config.tier_weights,
                rng=rng,
            )
            res = scen.resolution
            cfg = dict(scen.load_config)
            resolved_bc = scen.bc_type
            bc_masks = scen.bc_masks
            forces = scen.forces
            domain_mask = scen.domain_mask
            tier = scen.tier
            family = scen.family
        else:
            resolved_bc = str(cfg.get("bc_type", config.bc_type)).upper()
            cfg["bc_type"] = resolved_bc
            bc_masks = None
            forces = None
            domain_mask = None
            tier = 0
            family = "custom"

    state = create_empty_state(
        resolution=res,
        load_config=cfg,
        bc_type=resolved_bc,
        bc_masks=bc_masks,
        forces=forces,
        domain_mask=domain_mask,
    )
    engine_name = getattr(
        model,
        "architecture",
        "swin-unetr" if getattr(model, "use_swin", False) else "simple",
    )
    game_info = GameInfo(
        game_id=state.game_id,
        neural_engine=engine_name,
        checkpoint_version=f"selfplay-gen{config.generation}-T{tier}",
        bc_masks=state.bc_masks,
        forces=state.forces,
        load_config=state.load_config,
        bc_type=resolved_bc,
        resolution=res,
        final_compliance=None,
        final_volume=None,
        total_steps=0,
        initial_cantilever_problem_real_scale_factor=500000.0,
        stiffness_E=200e9,
        domain_mask=state.domain_mask,
    )
    save_game(db_path, game_info)
    return _run_episode_loop(state, model, db_path, config, tier=tier, family=family)


def run_episode(
    model: AlphaBuilderInference,
    db_path: Path,
    config: EpisodeConfig,
    load_config: Optional[Dict[str, Any]] = None,
    scenario: Optional[SelfPlayScenario] = None,
) -> Tuple[str, float, int]:
    """Run a fresh self-play episode."""
    game_id, score, steps, _ = run_episode_with_stats(
        model, db_path, config, load_config=load_config, scenario=scenario
    )
    return game_id, score, steps


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
        domain_mask=game_info.domain_mask.copy() if game_info.domain_mask is not None else None,
    )

    if last_game_step.selected_actions:
        actions = [
            (a.channel, a.x, a.y, a.z) for a in last_game_step.selected_actions
        ]
        apply_micro_batch(state, actions)

    if state.phase == Phase.GROWTH and check_phase_transition(state):
        state.phase = Phase.REFINEMENT
        state.is_connected = True

    gid, score, steps, _ = _run_episode_loop(state, model, db_path, config)
    return gid, score, steps
