"""
MCTS Node implementation for AlphaBuilder v3.3.

Each node stores lightweight PUCT statistics:
- visit_count (N)
- value_sum (W)
- max_value (for optional MixMax backup in sparse-reward Phase 1)
- prior (P, strictly normalized over expanded children)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# Action type: (channel, x, y, z) where channel = 0 for Add, 1 for Remove
Action = Tuple[int, int, int, int]


@dataclass
class MCTSNode:
    """Tree node for MCTS with PUCT statistics."""

    prior: float = 0.0
    action_to_parent: Optional[Action] = None
    parent: Optional["MCTSNode"] = None
    visit_count: int = 0
    value_sum: float = 0.0
    max_value: float = float("-inf")
    children: Dict[Action, "MCTSNode"] = field(default_factory=dict)
    is_expanded: bool = False
    virtual_loss: int = 0

    def detach_parent(self) -> None:
        """Sever link to parent when promoting a child to root."""
        self.parent = None
        self.action_to_parent = None

    @property
    def mean_value(self) -> float:
        """Mean value Q = W / N in [-1, 1], returning 0.0 when unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def Q(self) -> float:
        return self.mean_value

    @property
    def N(self) -> int:
        return self.visit_count

    @property
    def P(self) -> float:
        return self.prior

    def puct_score(
        self,
        c_puct: float,
        parent_visits: int,
        mixmax_beta: float = 1.0,
    ) -> float:
        """
        Compute PUCT score with optional MixMax value blending (AlphaTruss)
        and batch MCTS virtual loss penalty.
        """
        eff_n = self.visit_count + self.virtual_loss

        if eff_n == 0:
            q_value = 0.0
        elif mixmax_beta < 1.0 and self.visit_count > 0 and math.isfinite(self.max_value):
            mean_q = self.value_sum / self.visit_count
            q_base = mixmax_beta * mean_q + (1.0 - mixmax_beta) * self.max_value
            q_value = (q_base * self.visit_count - self.virtual_loss) / eff_n
        else:
            eff_w = self.value_sum - self.virtual_loss
            q_value = eff_w / eff_n

        exploration = c_puct * self.prior * math.sqrt(max(0, parent_visits)) / (1 + eff_n)
        return q_value + exploration

    def apply_virtual_loss(self) -> None:
        self.virtual_loss += 1

    def revert_virtual_loss(self) -> None:
        self.virtual_loss -= 1

    def select_child(
        self,
        c_puct: float,
        mixmax_beta: float = 1.0,
    ) -> Tuple[Action, "MCTSNode"]:
        """Select child with highest PUCT score."""
        if not self.children:
            raise ValueError("Cannot select child from node with no children")

        total_visits = sum(child.visit_count for child in self.children.values())
        best_action = None
        best_score = float("-inf")
        best_child = None

        for action, child in self.children.items():
            score = child.puct_score(c_puct, total_visits, mixmax_beta=mixmax_beta)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child  # type: ignore[return-value]

    def expand(
        self,
        policy_add: np.ndarray,
        policy_remove: np.ndarray,
        valid_add: np.ndarray,
        valid_remove: np.ndarray,
        top_k: Optional[int] = None,
        min_prior_threshold: float = 1e-6,
        temperature: float = 1.0,
    ) -> None:
        """
        Expand leaf node using masked joint softmax over legal actions.
        Priors are strictly renormalized after top_k / threshold pruning so sum == 1.0.
        """
        if self.is_expanded:
            return

        temp = max(float(temperature), 1e-6)
        masked_add = np.where(valid_add > 0, policy_add / temp, -np.inf)
        masked_remove = np.where(valid_remove > 0, policy_remove / temp, -np.inf)

        flat_add = masked_add.flatten()
        flat_remove = masked_remove.flatten()
        combined = np.concatenate([flat_add, flat_remove])

        finite_mask = np.isfinite(combined)
        if not np.any(finite_mask):
            self.is_expanded = True
            return

        max_val = np.max(combined[finite_mask])
        exp_combined = np.exp(combined - max_val)
        exp_combined[~finite_mask] = 0.0
        total_exp = exp_combined.sum()
        if total_exp <= 0.0:
            self.is_expanded = True
            return

        priors = exp_combined / total_exp

        spatial_size = flat_add.size
        add_priors = priors[:spatial_size].reshape(policy_add.shape)
        remove_priors = priors[spatial_size:].reshape(policy_remove.shape)

        candidates_list = _expand_candidates_jit(
            add_priors, remove_priors, valid_add, valid_remove, min_prior_threshold
        )

        # Fallback if threshold pruned everything while finite legal moves existed
        if not candidates_list:
            candidates_list = _expand_candidates_jit(
                add_priors, remove_priors, valid_add, valid_remove, -1.0
            )

        if top_k is not None and len(candidates_list) > top_k:
            candidates_list.sort(key=lambda x: x[1], reverse=True)
            candidates_list = candidates_list[:top_k]

        # Mandatory post-pruning prior renormalization before Dirichlet noise
        kept_sum = float(sum(float(p) for _, p in candidates_list))
        for action_tuple, prior in candidates_list:
            norm_prior = float(prior) / kept_sum if kept_sum > 0.0 else 1.0 / len(candidates_list)
            self.children[action_tuple] = MCTSNode(
                prior=norm_prior,
                action_to_parent=action_tuple,
                parent=self,
            )

        self.is_expanded = True

    def update(self, value: float) -> None:
        """Update node visit count, value sum, and running maximum."""
        self.visit_count += 1
        self.value_sum += value
        if value > self.max_value:
            self.max_value = value

    def get_visit_distribution(self) -> Dict[Action, int]:
        """Shallow visit count distribution over direct children."""
        return {action: child.visit_count for action, child in self.children.items()}

    def collect_subtree_visits(self, accumulator: Optional[Dict[Action, int]] = None) -> Dict[Action, int]:
        """Recursively collect visit counts across the subtree."""
        if accumulator is None:
            accumulator = {}
        for action, child in self.children.items():
            accumulator[action] = accumulator.get(action, 0) + child.visit_count
            child.collect_subtree_visits(accumulator)
        return accumulator

    def tree_stats(self) -> Tuple[int, int]:
        """Return (total_expanded_nodes, max_depth) in this subtree."""
        if not self.children:
            return (1 if self.is_expanded else 0), 0
        total_exp = 1 if self.is_expanded else 0
        max_d = 0
        for child in self.children.values():
            c_exp, c_d = child.tree_stats()
            total_exp += c_exp
            if c_d + 1 > max_d:
                max_d = c_d + 1
        return total_exp, max_d

    def get_top_k_actions(self, k: int) -> List[Action]:
        """Return top-K direct child actions sorted by visit count descending."""
        sorted_children = sorted(
            self.children.items(),
            key=lambda x: (x[1].visit_count, x[1].mean_value),
            reverse=True,
        )
        return [action for action, _ in sorted_children[:k]]

    def is_leaf(self) -> bool:
        return not self.is_expanded

    def is_root(self) -> bool:
        return self.parent is None

    def path_from_root(self) -> List["MCTSNode"]:
        path = []
        node: Optional[MCTSNode] = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def actions_from_root(self) -> List[Action]:
        path = self.path_from_root()
        return [node.action_to_parent for node in path[1:] if node.action_to_parent is not None]

    def __repr__(self) -> str:
        return (
            f"MCTSNode(N={self.visit_count}, Q={self.mean_value:.3f}, "
            f"P={self.prior:.4f}, children={len(self.children)}, "
            f"expanded={self.is_expanded})"
        )


try:
    from numba import jit

    @jit(nopython=True, cache=True)
    def _expand_candidates_jit(add_priors, remove_priors, valid_add, valid_remove, threshold):
        """Extract legal actions with prior > threshold."""
        candidates = []
        D, H, W = add_priors.shape

        for x in range(D):
            for y in range(H):
                for z in range(W):
                    if valid_add[x, y, z] > 0 and add_priors[x, y, z] > threshold:
                        candidates.append(((0, x, y, z), add_priors[x, y, z]))

        for x in range(D):
            for y in range(H):
                for z in range(W):
                    if valid_remove[x, y, z] > 0 and remove_priors[x, y, z] > threshold:
                        candidates.append(((1, x, y, z), remove_priors[x, y, z]))

        return candidates

except ImportError:

    def _expand_candidates_jit(add_priors, remove_priors, valid_add, valid_remove, threshold):
        candidates = []
        for idx in np.ndindex(add_priors.shape):
            if valid_add[idx] > 0 and add_priors[idx] > threshold:
                candidates.append(((0, idx[0], idx[1], idx[2]), float(add_priors[idx])))
        for idx in np.ndindex(remove_priors.shape):
            if valid_remove[idx] > 0 and remove_priors[idx] > threshold:
                candidates.append(((1, idx[0], idx[1], idx[2]), float(remove_priors[idx])))
        return candidates
