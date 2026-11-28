"""
MCTS Engine for AlphaBuilder v3.3 / v4.1 (MCTS-DAG + Macro-Actions + SWAP + STOP).

Implements:
- Batch/sequential MCTS with Virtual Loss & relative FPU (`parent_Q - fpu_reduction`)
- Factored policy priors `P(op) * P(voxel | op)` + surface-only removals
- Bidirectional Phase-1 seeding (support + load patches)
- Composite macro-actions (ADD, REMOVE, mass-conserving SWAP, STOP)
- Permutation transposition DAG (`A+B == B+A`) for genuine lookahead depth & subtree reuse
- Root-only voxel policy targets without depth 2..4 illegal leakage
"""

import hashlib
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple
import numpy as np

from .legal_moves import get_legal_moves, is_terminal_state
from .node import (
    Action,
    MCTSNode,
    STOP_ACTION,
    TreeAction,
    flatten_tree_action,
    is_stop_action,
)


class MCTSConfig(NamedTuple):
    """Configuration for MCTS search."""

    num_simulations: int = 80
    batch_size: int = 8
    action_batch_size: int = 1
    c_puct: float = 1.25
    density_threshold: float = 0.5
    top_k_expansion: Optional[int] = None
    min_prior_threshold: float = 1e-6
    max_volume_fraction: float = 0.3
    min_volume_fraction: float = 0.01
    phase: str = "REFINEMENT"
    add_root_noise: bool = False
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    policy_temperature: float = 1.0
    mixmax_beta: float = 1.0
    min_remove_candidates: int = 0
    remove_prior_floor: float = 0.0
    # v4.1 / v4.2 surgical & structural upgrades (safe defaults preserve legacy behavior)
    fpu_reduction: Optional[float] = None
    surface_only_remove: bool = False
    bidirectional_p1: bool = False
    use_transposition_dag: bool = False
    macro_action_mode: bool = False
    factored_policy: bool = False
    root_only_policy_target: bool = False
    allow_stop_action: bool = False
    macro_num_candidates: int = 16
    remove_min_vol_frac: float = 0.0
    remove_floor_vol_frac: float = 0.0


# Legacy v3.3 / Stage 5 configurations (preserved for exact regression & ablation)
PHASE1_CONFIG = MCTSConfig(
    num_simulations=160,
    c_puct=1.2,
    batch_size=8,
    action_batch_size=32,
    top_k_expansion=64,
    min_volume_fraction=0.0,
    max_volume_fraction=0.9,
    phase="GROWTH",
    mixmax_beta=0.75,
)

PHASE2_CONFIG = MCTSConfig(
    num_simulations=160,
    c_puct=0.8,
    batch_size=8,
    action_batch_size=8,
    top_k_expansion=64,
    min_volume_fraction=0.01,
    max_volume_fraction=0.9,
    phase="REFINEMENT",
    mixmax_beta=1.0,
)

# Upgraded Stage 6 v4.1 configurations (100% compatible with v4_stage5 checkpoints)
PHASE1_V41_CONFIG = MCTSConfig(
    num_simulations=160,
    c_puct=1.2,
    batch_size=8,
    action_batch_size=32,
    top_k_expansion=64,
    min_volume_fraction=0.0,
    max_volume_fraction=0.9,
    phase="GROWTH",
    mixmax_beta=0.75,
    fpu_reduction=0.20,
    surface_only_remove=True,
    bidirectional_p1=False,
    use_transposition_dag=True,
    macro_action_mode=False,
    factored_policy=False,
    root_only_policy_target=True,
    allow_stop_action=False,
)

PHASE2_V41_CONFIG = MCTSConfig(
    num_simulations=160,
    c_puct=1.0,
    batch_size=8,
    action_batch_size=8,
    top_k_expansion=64,
    min_volume_fraction=0.01,
    max_volume_fraction=0.9,
    phase="REFINEMENT",
    mixmax_beta=1.0,
    min_remove_candidates=16,
    remove_prior_floor=0.25,
    fpu_reduction=0.25,
    surface_only_remove=True,
    bidirectional_p1=False,
    use_transposition_dag=True,
    macro_action_mode=True,
    factored_policy=True,
    root_only_policy_target=True,
    allow_stop_action=True,
    macro_num_candidates=16,
)

# Upgraded Stage 6 v4.2 configurations (volume-aware expansion + anti-throttle guard)
PHASE1_V42_CONFIG = PHASE1_V41_CONFIG
PHASE2_V42_CONFIG = PHASE2_V41_CONFIG._replace(
    action_batch_size=48,
    remove_prior_floor=0.25,
    remove_min_vol_frac=0.080,
    remove_floor_vol_frac=0.105,
    macro_num_candidates=16,
)


class SearchResult(NamedTuple):
    """Result from MCTS search."""

    actions: List[Action]
    visit_distribution: Dict[Action, int]
    root_value: float
    num_simulations: int
    root: Any


def apply_action(density: np.ndarray, action: Action) -> np.ndarray:
    """Pure function applying a single `(channel, x, y, z)` action."""
    new_density = density.copy()
    channel, x, y, z = action
    new_density[x, y, z] = 1.0 if channel == 0 else 0.0
    return new_density


def apply_actions(density: np.ndarray, actions: List[Action]) -> np.ndarray:
    """Pure function applying a sequence of actions."""
    new_density = density.copy()
    for channel, x, y, z in actions:
        new_density[x, y, z] = 1.0 if channel == 0 else 0.0
    return new_density


def apply_tree_action(density: np.ndarray, tree_action: TreeAction) -> np.ndarray:
    """Pure function applying a unitary, macro, or STOP tree edge."""
    return apply_actions(density, flatten_tree_action(tree_action))


def _compute_context_key(
    bc_masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
    forces: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> str:
    h = hashlib.md5()
    for arr in (*bc_masks, *forces):
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return h.hexdigest()


def compute_state_hash(
    density: np.ndarray,
    phase: str = "",
    context_key: str = "",
) -> str:
    """Compute cache hash incorporating density, active MCTS phase, and BC/force context."""
    binary = (density > 0.5).astype(np.uint8)
    h = hashlib.md5()
    h.update(f"{phase}|{context_key}|".encode("utf-8"))
    h.update(binary.tobytes())
    return h.hexdigest()


def build_state_tensor(
    density: np.ndarray,
    bc_masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
    forces: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Build 7-channel network state tensor `(7, D, H, W)`."""
    return np.stack(
        [
            density,
            bc_masks[0],
            bc_masks[1],
            bc_masks[2],
            forces[0],
            forces[1],
            forces[2],
        ],
        axis=0,
    ).astype(np.float32)


def select_leaf(
    root: MCTSNode,
    c_puct: float,
    mixmax_beta: float = 1.0,
    fpu_reduction: Optional[float] = None,
) -> Tuple[MCTSNode, List[MCTSNode]]:
    """Select an unexpanded leaf by traversing children via PUCT (legacy 2-tuple signature)."""
    leaf, path, _ = select_leaf_with_actions(
        root, c_puct, mixmax_beta=mixmax_beta, fpu_reduction=fpu_reduction
    )
    return leaf, path


def select_leaf_with_actions(
    root: MCTSNode,
    c_puct: float,
    mixmax_beta: float = 1.0,
    fpu_reduction: Optional[float] = None,
) -> Tuple[MCTSNode, List[MCTSNode], List[TreeAction]]:
    """
    DAG-safe leaf selection tracking exact traversed edges `(leaf, path_nodes, path_actions)`.
    """
    node = root
    path_nodes = [node]
    path_actions: List[TreeAction] = []
    seen_ids: Set[int] = {id(node)}
    while node.is_expanded and node.children:
        act, next_node = node.select_child(
            c_puct,
            mixmax_beta=mixmax_beta,
            fpu_reduction=fpu_reduction,
            exclude_ids=seen_ids,
        )
        if id(next_node) in seen_ids:
            break
        seen_ids.add(id(next_node))
        path_actions.append(act)
        path_nodes.append(next_node)
        node = next_node
    return node, path_nodes, path_actions


def select_pv_branches(
    root: MCTSNode,
    max_actions: int = 32,
    max_depth: int = 4,
) -> List[List[Tuple[Action, int, float]]]:
    """
    Extract Principal Variation branches with spatial voxel locking.

    For macro-action trees (`macro_action_mode=True`), selects the highest-visit
    macro child cleanly; for legacy 1-voxel trees, stitches up to `max_actions`
    across top root branches.
    """
    if not getattr(root, "children", None):
        return []

    root_children_sorted = sorted(
        root.children.items(),
        key=lambda item: (item[1].visit_count, item[1].mean_value),
        reverse=True,
    )

    top_tree_act, top_child = root_children_sorted[0]
    if is_stop_action(top_tree_act):
        return []
    flat_top = flatten_tree_action(top_tree_act)
    if len(flat_top) > 1 or not isinstance(top_tree_act[0], (int, np.integer)):
        branch = [
            (a, int(top_child.visit_count), float(top_child.mean_value))
            for a in flat_top[:max_actions]
        ]
        return [branch] if branch else []

    branches: List[List[Tuple[Action, int, float]]] = []
    total_actions = 0
    locked_voxels = set()

    def is_locked(act: Action) -> bool:
        return (act[1], act[2], act[3]) in locked_voxels

    for tree_action, child in root_children_sorted:
        if total_actions >= max_actions:
            break
        if is_stop_action(tree_action):
            continue
        flat = flatten_tree_action(tree_action)
        if not flat:
            continue
        action = flat[0]
        if is_locked(action):
            continue

        seq: List[Tuple[Action, int, float]] = [
            (action, int(child.visit_count), float(child.mean_value))
        ]
        locked_voxels.add((action[1], action[2], action[3]))
        current_node = child

        for _ in range(max_depth - 1):
            if not current_node.children:
                break

            sorted_children = sorted(
                current_node.children.items(),
                key=lambda item: (item[1].visit_count, item[1].mean_value),
                reverse=True,
            )

            best_action = None
            best_child = None
            for child_tree_act, child_node in sorted_children:
                if is_stop_action(child_tree_act):
                    continue
                c_flat = flatten_tree_action(child_tree_act)
                if not c_flat:
                    continue
                child_act = c_flat[0]
                if not is_locked(child_act):
                    best_action = child_act
                    best_child = child_node
                    break

            if best_action is None or best_child is None:
                break

            seq.append((best_action, int(best_child.visit_count), float(best_child.mean_value)))
            locked_voxels.add((best_action[1], best_action[2], best_action[3]))
            current_node = best_child

        remaining_slots = max_actions - total_actions
        if len(seq) > remaining_slots:
            seq = seq[:remaining_slots]

        if seq:
            branches.append(seq)
            total_actions += len(seq)

    return branches


def select_pv_sequences(
    root: MCTSNode,
    max_actions: int = 32,
    max_depth: int = 4,
) -> List[Action]:
    """Flatten `select_pv_branches` into an ordered list of actions."""
    branches = select_pv_branches(root, max_actions=max_actions, max_depth=max_depth)
    return [act for branch in branches for act, _, _ in branch]


def select_non_cycling_root_action(
    root: Optional[MCTSNode],
    density: np.ndarray,
    seen_hashes: Set[str],
    allow_stop: bool = False,
    max_actions: Optional[int] = None,
) -> Tuple[Optional[TreeAction], List[Action], Optional[List[Tuple[Action, int, float]]], bool, bool]:
    """
    Select highest-visit non-STOP root edge whose post-action state has not already been visited.
    Returns `(chosen_tree_action, executed_actions, executed_action_stats, stopped_by_mcts, avoided_cycle)`.
    """
    if root is None or not getattr(root, "children", None):
        return None, [], None, False, False

    sorted_children = sorted(
        root.children.items(),
        key=lambda item: (item[1].visit_count, item[1].mean_value),
        reverse=True,
    )

    top_tree_act, _ = sorted_children[0]
    if is_stop_action(top_tree_act) and allow_stop:
        return STOP_ACTION, [], None, True, False

    non_stop_rank = 0
    for tree_act, child in sorted_children:
        if is_stop_action(tree_act):
            continue
        flat = flatten_tree_action(tree_act)
        if max_actions is not None:
            flat = flat[: max(1, int(max_actions))]
        if not flat:
            continue
        cand = apply_actions(density, flat)
        h = hashlib.md5((cand > 0.5).astype(np.uint8).tobytes()).hexdigest()
        if h not in seen_hashes:
            stats = [(a, int(child.visit_count), float(child.mean_value)) for a in flat]
            return tree_act, flat, stats, False, (non_stop_rank > 0)
        non_stop_rank += 1

    if allow_stop and STOP_ACTION in root.children:
        return STOP_ACTION, [], None, True, True
    return None, [], None, False, True


def expand_node(
    node: MCTSNode,
    policy_add: np.ndarray,
    policy_remove: np.ndarray,
    valid_add: np.ndarray,
    valid_remove: np.ndarray,
    config: MCTSConfig,
    op_probs: Optional[Tuple[float, float, float]] = None,
    current_vol_frac: Optional[float] = None,
) -> None:
    """Expand a leaf node using either composite macro-actions (v4.1/v4.2) or unitary voxel edges."""
    if config.macro_action_mode:
        eff_rem_floor = (
            config.remove_prior_floor
            if (config.remove_min_vol_frac > 0.0 or config.remove_floor_vol_frac > 0.0)
            else max(0.20, config.remove_prior_floor)
        )
        node.expand_macro(
            policy_add=policy_add,
            policy_remove=policy_remove,
            valid_add=valid_add,
            valid_remove=valid_remove,
            action_batch_size=config.action_batch_size,
            phase=config.phase,
            temperature=config.policy_temperature,
            remove_prior_floor=eff_rem_floor,
            op_probs=op_probs,
            allow_stop=bool(config.allow_stop_action and config.phase != "GROWTH"),
            max_candidates=config.macro_num_candidates,
            current_vol_frac=current_vol_frac,
            remove_min_vol_frac=config.remove_min_vol_frac,
            remove_floor_vol_frac=config.remove_floor_vol_frac,
        )
        return

    node.expand(
        policy_add=policy_add,
        policy_remove=policy_remove,
        valid_add=valid_add,
        valid_remove=valid_remove,
        top_k=config.top_k_expansion,
        min_prior_threshold=config.min_prior_threshold,
        temperature=config.policy_temperature,
        min_remove_candidates=config.min_remove_candidates,
        remove_prior_floor=config.remove_prior_floor,
        factored_policy=config.factored_policy,
        op_probs=op_probs,
        allow_stop=bool(config.allow_stop_action and config.phase != "GROWTH"),
    )


def backup(path: List[MCTSNode], value: float) -> None:
    """Propagate leaf evaluation up the tree/DAG path."""
    for node in path:
        node.update(value)


def run_mcts_search(
    root_density: np.ndarray,
    bc_masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
    forces: Tuple[np.ndarray, np.ndarray, np.ndarray],
    predict_fn: Callable[[np.ndarray], Tuple[float, np.ndarray, np.ndarray]],
    config: MCTSConfig = MCTSConfig(),
    root_node: Optional[MCTSNode] = None,
    eval_cache: Optional[Dict[str, Tuple[float, np.ndarray, np.ndarray]]] = None,
    predict_batch_fn: Optional[
        Callable[[List[np.ndarray]], Tuple[List[float], List[np.ndarray], List[np.ndarray]]]
    ] = None,
    domain_mask: Optional[np.ndarray] = None,
    stop_value: Optional[float] = None,
    root_op_probs: Optional[Tuple[float, float, float]] = None,
    dag_nodes: Optional[Dict[str, MCTSNode]] = None,
    dag_stats: Optional[Dict[str, int]] = None,
) -> SearchResult:
    """Run batch or sequential MCTS simulations from root state."""
    if eval_cache is None:
        eval_cache = {}
    if dag_nodes is None and config.use_transposition_dag:
        dag_nodes = {}

    context_key = _compute_context_key(bc_masks, forces)
    inc_load = bool(config.bidirectional_p1 and config.phase == "GROWTH")
    surf_rem = bool(config.surface_only_remove and config.phase != "GROWTH")

    def _eval_single(density: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        sha = compute_state_hash(density, phase=config.phase, context_key=context_key)
        cached = eval_cache.get(sha)
        if cached is not None:
            return cached
        state_tensor = build_state_tensor(density, bc_masks, forces)
        val, p_a, p_r = predict_fn(state_tensor)
        eval_cache[sha] = (val, p_a, p_r)
        return val, p_a, p_r

    def _legal(density: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return get_legal_moves(
            density,
            config.density_threshold,
            bc_masks=bc_masks,
            forces=forces,
            domain_mask=domain_mask,
            surface_only_remove=surf_rem,
            include_load_seeds=inc_load,
        )

    root_hash = compute_state_hash(root_density, phase=config.phase, context_key=context_key)
    root_vol_frac = float(np.mean(root_density > config.density_threshold))
    if root_node is not None:
        root = root_node
        if not root.is_expanded:
            value, p_add, p_remove = _eval_single(root_density)
            valid_add, valid_remove = _legal(root_density)
            expand_node(
                root,
                p_add,
                p_remove,
                valid_add,
                valid_remove,
                config,
                op_probs=root_op_probs,
                current_vol_frac=root_vol_frac,
            )
            root.update(value)
    else:
        root_value, policy_add, policy_remove = _eval_single(root_density)
        valid_add, valid_remove = _legal(root_density)
        root = MCTSNode(prior=1.0)
        expand_node(
            root,
            policy_add,
            policy_remove,
            valid_add,
            valid_remove,
            config,
            op_probs=root_op_probs,
            current_vol_frac=root_vol_frac,
        )
        root.update(root_value)

    if config.use_transposition_dag and dag_nodes is not None:
        dag_nodes.setdefault(root_hash, root)

    eff_stop_val = float(stop_value) if stop_value is not None else float(root.mean_value)

    if config.add_root_noise and root.is_expanded:
        children = list(root.children.values())
        if children:
            prior_sum = sum(c.prior for c in children)
            if prior_sum > 0.0:
                for c in children:
                    c.prior /= prior_sum
            else:
                uni = 1.0 / len(children)
                for c in children:
                    c.prior = uni

            noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(children))
            frac = config.root_exploration_fraction
            for child, n in zip(children, noise):
                child.prior = child.prior * (1.0 - frac) + float(n) * frac

    def _handle_stop_leaf(leaf: MCTSNode, path_actions: List[TreeAction], path_nodes: List[MCTSNode]) -> float:
        leaf.is_expanded = True
        if len(path_actions) <= 1:
            return eff_stop_val
        parent_actions = path_actions[:-1]
        parent_flat = [a for ta in parent_actions for a in flatten_tree_action(ta)]
        st_dens = apply_actions(root_density, parent_flat)
        val, _, _ = _eval_single(st_dens)
        return float(val)

    def _maybe_dag_merge(
        leaf: MCTSNode,
        path_nodes: List[MCTSNode],
        path_actions: List[TreeAction],
        state_hash: str,
        with_virtual_loss: bool,
    ) -> Tuple[MCTSNode, bool]:
        if not (config.use_transposition_dag and dag_nodes is not None and len(path_nodes) >= 2):
            return leaf, False
        existing = dag_nodes.get(state_hash)
        ancestor_ids = {id(n) for n in path_nodes[:-1]}
        if existing is not None and existing is not leaf and id(existing) not in ancestor_ids:
            parent_node = path_nodes[-2]
            last_act = path_actions[-1]
            if with_virtual_loss:
                leaf.revert_virtual_loss()
                existing.apply_virtual_loss()
            parent_node.children[last_act] = existing
            path_nodes[-1] = existing
            if dag_stats is not None:
                dag_stats["hits"] = dag_stats.get("hits", 0) + 1
            return existing, True
        dag_nodes[state_hash] = leaf
        return leaf, False

    if predict_batch_fn is not None and config.batch_size > 1:
        num_batches = (config.num_simulations + config.batch_size - 1) // config.batch_size
        for _ in range(num_batches):
            batch_leaves: List[MCTSNode] = []
            batch_paths: List[List[MCTSNode]] = []
            batch_states: List[np.ndarray] = []
            batch_densities: List[np.ndarray] = []
            batch_state_hashes: List[str] = []
            ready_to_backup: List[Tuple[List[MCTSNode], float]] = []

            for _ in range(config.batch_size):
                leaf, path, path_actions = select_leaf_with_actions(
                    root,
                    config.c_puct,
                    mixmax_beta=config.mixmax_beta,
                    fpu_reduction=config.fpu_reduction,
                )
                for node in path:
                    node.apply_virtual_loss()

                if path_actions and is_stop_action(path_actions[-1]):
                    val = _handle_stop_leaf(leaf, path_actions, path)
                    ready_to_backup.append((path, val))
                    continue

                flat_actions = [a for ta in path_actions for a in flatten_tree_action(ta)]
                leaf_density = apply_actions(root_density, flat_actions)

                is_terminal, reason = is_terminal_state(
                    leaf_density,
                    max_volume_fraction=config.max_volume_fraction,
                    min_volume_fraction=config.min_volume_fraction,
                    threshold=config.density_threshold,
                    bc_masks=bc_masks,
                    forces=forces,
                    domain_mask=domain_mask,
                )

                if is_terminal:
                    if reason == "max_volume_exceeded":
                        value = -0.5
                    elif reason == "structure_too_small":
                        value = -1.0
                    else:
                        value = 0.0
                    ready_to_backup.append((path, value))
                    continue

                state_hash = compute_state_hash(leaf_density, phase=config.phase, context_key=context_key)
                leaf, merged = _maybe_dag_merge(leaf, path, path_actions, state_hash, with_virtual_loss=True)
                if merged:
                    if state_hash in eval_cache:
                        val = leaf.mean_value if leaf.visit_count > 0 else float(eval_cache[state_hash][0])
                        ready_to_backup.append((path, val))
                        continue

                if state_hash in eval_cache:
                    value, p_add, p_remove = eval_cache[state_hash]
                    if not leaf.is_expanded:
                        valid_add, valid_remove = _legal(leaf_density)
                        expand_node(
                            leaf,
                            p_add,
                            p_remove,
                            valid_add,
                            valid_remove,
                            config,
                            current_vol_frac=float(np.mean(leaf_density > config.density_threshold)),
                        )
                    ready_to_backup.append((path, value))
                else:
                    leaf_state = build_state_tensor(leaf_density, bc_masks, forces)
                    batch_leaves.append(leaf)
                    batch_paths.append(path)
                    batch_states.append(leaf_state)
                    batch_densities.append(leaf_density)
                    batch_state_hashes.append(state_hash)

            if batch_states:
                values, pols_add, pols_remove = predict_batch_fn(batch_states)
                for leaf, path, sha, leaf_density, val, p_a, p_r in zip(
                    batch_leaves,
                    batch_paths,
                    batch_state_hashes,
                    batch_densities,
                    values,
                    pols_add,
                    pols_remove,
                ):
                    eval_cache[sha] = (val, p_a, p_r)
                    if not leaf.is_expanded:
                        valid_add, valid_remove = _legal(leaf_density)
                        expand_node(
                            leaf,
                            p_a,
                            p_r,
                            valid_add,
                            valid_remove,
                            config,
                            current_vol_frac=float(np.mean(leaf_density > config.density_threshold)),
                        )
                    ready_to_backup.append((path, val))

            for path, value in ready_to_backup:
                for node in path:
                    node.revert_virtual_loss()
                backup(path, value)
    else:
        for _ in range(config.num_simulations):
            leaf, path, path_actions = select_leaf_with_actions(
                root,
                config.c_puct,
                mixmax_beta=config.mixmax_beta,
                fpu_reduction=config.fpu_reduction,
            )

            if path_actions and is_stop_action(path_actions[-1]):
                value = _handle_stop_leaf(leaf, path_actions, path)
                backup(path, value)
                continue

            flat_actions = [a for ta in path_actions for a in flatten_tree_action(ta)]
            leaf_density = apply_actions(root_density, flat_actions)

            is_terminal, reason = is_terminal_state(
                leaf_density,
                max_volume_fraction=config.max_volume_fraction,
                min_volume_fraction=config.min_volume_fraction,
                threshold=config.density_threshold,
                bc_masks=bc_masks,
                forces=forces,
                domain_mask=domain_mask,
            )

            if is_terminal:
                if reason == "max_volume_exceeded":
                    value = -0.5
                elif reason == "structure_too_small":
                    value = -1.0
                else:
                    value = 0.0
            else:
                state_hash = compute_state_hash(leaf_density, phase=config.phase, context_key=context_key)
                leaf, merged = _maybe_dag_merge(leaf, path, path_actions, state_hash, with_virtual_loss=False)
                if merged and leaf.is_expanded:
                    value = leaf.mean_value if leaf.visit_count > 0 else float(eval_cache[state_hash][0])
                else:
                    value, policy_add, policy_remove = _eval_single(leaf_density)
                    valid_add, valid_remove = _legal(leaf_density)
                    expand_node(
                        leaf,
                        policy_add,
                        policy_remove,
                        valid_add,
                        valid_remove,
                        config,
                        current_vol_frac=float(np.mean(leaf_density > config.density_threshold)),
                    )

            backup(path, value)

    if config.root_only_policy_target or config.macro_action_mode:
        visit_dist = root.get_root_voxel_visits()
    else:
        visit_dist = root.collect_subtree_visits()
    top_actions = root.get_top_k_actions(config.action_batch_size)

    return SearchResult(
        actions=top_actions,
        visit_distribution=visit_dist,
        root_value=root.mean_value,
        num_simulations=config.num_simulations,
        root=root,
    )


class AlphaBuilderMCTS:
    """Convenience stateful MCTS wrapper preserving root tree/DAG & phase-scoped eval cache."""

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], Tuple[float, np.ndarray, np.ndarray]],
        num_simulations: int = 80,
        batch_size: int = 8,
        action_batch_size: int = 1,
        c_puct: float = 1.25,
        top_k_expansion: Optional[int] = None,
        predict_batch_fn: Optional[
            Callable[[List[np.ndarray]], Tuple[List[float], List[np.ndarray], List[np.ndarray]]]
        ] = None,
    ):
        self.predict_fn = predict_fn
        self.predict_batch_fn = predict_batch_fn
        self.config = MCTSConfig(
            num_simulations=num_simulations,
            batch_size=batch_size,
            action_batch_size=action_batch_size,
            c_puct=c_puct,
            top_k_expansion=top_k_expansion,
            add_root_noise=True,
        )
        self.root: Optional[MCTSNode] = None
        self.eval_cache: Dict[str, Tuple[float, np.ndarray, np.ndarray]] = {}
        self.dag_nodes: Dict[str, MCTSNode] = {}
        self.last_dag_hits: int = 0
        self.total_dag_hits: int = 0

    def reset(self) -> None:
        """Clear search tree/DAG and evaluation cache."""
        self.root = None
        self.eval_cache = {}
        self.dag_nodes = {}
        self.last_dag_hits = 0

    def clear_tree(self) -> None:
        """Drop current search tree/DAG while retaining evaluation cache."""
        self.root = None
        self.dag_nodes = {}

    def get_action_batch(
        self,
        density: np.ndarray,
        bc_masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
        forces: Tuple[np.ndarray, np.ndarray, np.ndarray],
        domain_mask: Optional[np.ndarray] = None,
        stop_value: Optional[float] = None,
        root_op_probs: Optional[Tuple[float, float, float]] = None,
    ) -> SearchResult:
        dag_stats = {"hits": 0}
        result = run_mcts_search(
            root_density=density,
            bc_masks=bc_masks,
            forces=forces,
            predict_fn=self.predict_fn,
            config=self.config,
            root_node=self.root,
            eval_cache=self.eval_cache,
            predict_batch_fn=self.predict_batch_fn,
            domain_mask=domain_mask,
            stop_value=stop_value,
            root_op_probs=root_op_probs,
            dag_nodes=self.dag_nodes if self.config.use_transposition_dag else None,
            dag_stats=dag_stats,
        )
        self.root = result.root
        self.last_dag_hits = int(dag_stats["hits"])
        self.total_dag_hits += self.last_dag_hits
        return result

    def step(self, action: TreeAction) -> bool:
        """Advance tree root to selected child if present; return True if subtree reused."""
        if self.root is not None and action in self.root.children:
            new_root = self.root.children[action]
            new_root.detach_parent()
            self.root = new_root
            return True
        self.root = None
        self.dag_nodes = {}
        return False

    def search(
        self,
        density: np.ndarray,
        bc_masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
        forces: Tuple[np.ndarray, np.ndarray, np.ndarray],
        domain_mask: Optional[np.ndarray] = None,
        stop_value: Optional[float] = None,
        root_op_probs: Optional[Tuple[float, float, float]] = None,
    ) -> SearchResult:
        """Run MCTS search, forwarding batch predictor and persisting root."""
        return self.get_action_batch(
            density=density,
            bc_masks=bc_masks,
            forces=forces,
            domain_mask=domain_mask,
            stop_value=stop_value,
            root_op_probs=root_op_probs,
        )


def analyze_search_quality(
    result: SearchResult,
    policy_add: np.ndarray,
    policy_remove: np.ndarray,
) -> Dict[str, Any]:
    """Analyze MCTS visit concentration and policy refutation metrics."""
    visit_dist = result.visit_distribution
    if not visit_dist:
        return {"error": "no_visits"}

    total_visits = sum(visit_dist.values())
    sorted_visits = sorted(visit_dist.values(), reverse=True)
    top_8_visits = sum(sorted_visits[:8])
    concentration = top_8_visits / max(total_visits, 1)

    all_probs = np.concatenate([policy_add.flatten(), policy_remove.flatten()])
    best_policy_idx = int(np.argmax(all_probs))
    most_visited_action = max(visit_dist.items(), key=lambda x: x[1])[0]

    spatial_size = policy_add.size
    if best_policy_idx < spatial_size:
        best_policy_action = (0,) + np.unravel_index(best_policy_idx, policy_add.shape)
    else:
        best_policy_action = (1,) + np.unravel_index(best_policy_idx - spatial_size, policy_remove.shape)

    return {
        "total_visits": total_visits,
        "top_8_concentration": concentration,
        "most_visited_action": most_visited_action,
        "highest_policy_action": best_policy_action,
        "refutation": most_visited_action != best_policy_action,
        "root_value": result.root_value,
    }
