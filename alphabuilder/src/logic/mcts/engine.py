"""
MCTS Engine for AlphaBuilder v3.3.

Implements batch/sequential Monte Carlo Tree Search with Virtual Loss,
protected BC/load removal masks, post-pruning prior renormalization, MixMax backup,
and Principal Variation branch extraction.
"""

import hashlib
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
import numpy as np

from .legal_moves import get_legal_moves, is_terminal_state
from .node import Action, MCTSNode


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


# Phase 1 (Growth): allow empty initial grid (min_volume_fraction = 0.0) and MixMax backup
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

# Phase 2 (Refinement): finer micro-batch (action_batch_size = 8) to avoid brittle disconnections
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


class SearchResult(NamedTuple):
    """Result from MCTS search."""

    actions: List[Action]
    visit_distribution: Dict[Action, int]
    root_value: float
    num_simulations: int
    root: Any


def apply_action(density: np.ndarray, action: Action) -> np.ndarray:
    """Pure function applying a single (channel, x, y, z) action."""
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
    """Build 7-channel network state tensor (7, D, H, W)."""
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
) -> Tuple[MCTSNode, List[MCTSNode]]:
    """Select an unexpanded leaf by traversing children via PUCT."""
    node = root
    path = [node]
    while node.is_expanded and node.children:
        _, node = node.select_child(c_puct, mixmax_beta=mixmax_beta)
        path.append(node)
    return node, path


def select_pv_branches(
    root: MCTSNode,
    max_actions: int = 32,
    max_depth: int = 4,
) -> List[List[Tuple[Action, int, float]]]:
    """
    Extract Principal Variation branches with spatial voxel locking.

    Returns a list of branches; each branch is a list of
    `(action, visit_count, mean_value)` read directly from the corresponding tree nodes.
    """
    if not getattr(root, "children", None):
        return []

    root_children_sorted = sorted(
        root.children.items(),
        key=lambda item: (item[1].visit_count, item[1].mean_value),
        reverse=True,
    )

    branches: List[List[Tuple[Action, int, float]]] = []
    total_actions = 0
    locked_voxels = set()

    def is_locked(act: Action) -> bool:
        return (act[1], act[2], act[3]) in locked_voxels

    for action, child in root_children_sorted:
        if total_actions >= max_actions:
            break
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
            for child_act, child_node in sorted_children:
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


def expand_node(
    node: MCTSNode,
    policy_add: np.ndarray,
    policy_remove: np.ndarray,
    valid_add: np.ndarray,
    valid_remove: np.ndarray,
    config: MCTSConfig,
) -> None:
    """Expand a leaf node using config parameters."""
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
    )


def backup(path: List[MCTSNode], value: float) -> None:
    """Propagate leaf evaluation up the tree path."""
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
) -> SearchResult:
    """Run batch or sequential MCTS simulations from root state."""
    if eval_cache is None:
        eval_cache = {}

    context_key = _compute_context_key(bc_masks, forces)

    def _eval_single(density: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        sha = compute_state_hash(density, phase=config.phase, context_key=context_key)
        cached = eval_cache.get(sha)
        if cached is not None:
            return cached
        state_tensor = build_state_tensor(density, bc_masks, forces)
        val, p_a, p_r = predict_fn(state_tensor)
        eval_cache[sha] = (val, p_a, p_r)
        return val, p_a, p_r

    if root_node is not None:
        root = root_node
        if not root.is_expanded:
            value, p_add, p_remove = _eval_single(root_density)
            valid_add, valid_remove = get_legal_moves(
                root_density, config.density_threshold, bc_masks=bc_masks, forces=forces, domain_mask=domain_mask
            )
            expand_node(root, p_add, p_remove, valid_add, valid_remove, config)
            root.update(value)
    else:
        root_value, policy_add, policy_remove = _eval_single(root_density)
        valid_add, valid_remove = get_legal_moves(
            root_density, config.density_threshold, bc_masks=bc_masks, forces=forces, domain_mask=domain_mask
        )
        root = MCTSNode(prior=1.0)
        expand_node(root, policy_add, policy_remove, valid_add, valid_remove, config)
        root.update(root_value)

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
                leaf, path = select_leaf(root, config.c_puct, mixmax_beta=config.mixmax_beta)
                for node in path:
                    node.apply_virtual_loss()

                leaf_density = apply_actions(root_density, leaf.actions_from_root())
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
                if state_hash in eval_cache:
                    value, p_add, p_remove = eval_cache[state_hash]
                    if not leaf.is_expanded:
                        valid_add, valid_remove = get_legal_moves(
                            leaf_density,
                            config.density_threshold,
                            bc_masks=bc_masks,
                            forces=forces,
                            domain_mask=domain_mask,
                        )
                        expand_node(leaf, p_add, p_remove, valid_add, valid_remove, config)
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
                        valid_add, valid_remove = get_legal_moves(
                            leaf_density,
                            config.density_threshold,
                            bc_masks=bc_masks,
                            forces=forces,
                            domain_mask=domain_mask,
                        )
                        expand_node(leaf, p_a, p_r, valid_add, valid_remove, config)
                    ready_to_backup.append((path, val))

            for path, value in ready_to_backup:
                for node in path:
                    node.revert_virtual_loss()
                backup(path, value)
    else:
        for _ in range(config.num_simulations):
            leaf, path = select_leaf(root, config.c_puct, mixmax_beta=config.mixmax_beta)
            leaf_density = apply_actions(root_density, leaf.actions_from_root())

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
                value, policy_add, policy_remove = _eval_single(leaf_density)
                valid_add, valid_remove = get_legal_moves(
                    leaf_density,
                    config.density_threshold,
                    bc_masks=bc_masks,
                    forces=forces,
                    domain_mask=domain_mask,
                )
                expand_node(leaf, policy_add, policy_remove, valid_add, valid_remove, config)

            backup(path, value)

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
    """Convenience stateful MCTS wrapper preserving root tree & phase-scoped eval cache."""

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

    def reset(self) -> None:
        """Clear search tree and evaluation cache."""
        self.root = None
        self.eval_cache = {}

    def clear_tree(self) -> None:
        """Drop current search tree while retaining evaluation cache."""
        self.root = None

    def get_action_batch(
        self,
        density: np.ndarray,
        bc_masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
        forces: Tuple[np.ndarray, np.ndarray, np.ndarray],
        domain_mask: Optional[np.ndarray] = None,
    ) -> SearchResult:
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
        )
        self.root = result.root
        return result

    def step(self, action: Action) -> None:
        """Advance tree root to selected child if present."""
        if self.root is not None and action in self.root.children:
            new_root = self.root.children[action]
            new_root.detach_parent()
            self.root = new_root
        else:
            self.root = None

    def search(
        self,
        density: np.ndarray,
        bc_masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
        forces: Tuple[np.ndarray, np.ndarray, np.ndarray],
        domain_mask: Optional[np.ndarray] = None,
    ) -> SearchResult:
        """Run MCTS search, forwarding batch predictor and persisting root."""
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
        )
        self.root = result.root
        return result


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
