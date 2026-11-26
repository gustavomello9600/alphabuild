"""
MCTS Node implementation for AlphaBuilder v3.3 / v4.1 (MCTS-DAG + Macro/SWAP/STOP).

Each node stores lightweight PUCT statistics:
- visit_count (N)
- value_sum (W)
- max_value (for optional MixMax backup in sparse-reward Phase 1)
- prior (P, strictly normalized over expanded children)

Supports both legacy unitary voxel edges and v4.1 upgrades:
- Relative First Play Urgency (`fpu_reduction`: parent_Q - fpu_reduction)
- First-batch virtual-loss prior tie-breaking
- Factored policy `P(op) * P(voxel | op)`
- Composite macro-actions (ADD, REMOVE, mass-conserving SWAP, STOP)
- Root-only voxel visit projection without depth 2..4 illegal leakage
- Transposition-DAG safe `tree_stats()`
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np


# Action type: (channel, x, y, z) where channel = 0 for Add, 1 for Remove
Action = Tuple[int, int, int, int]
STOP_ACTION: Tuple[str] = ("STOP",)
TreeAction = Union[Action, Tuple[Action, ...], Tuple[str]]


def is_stop_action(tree_action: Any) -> bool:
    """Return True if edge represents voluntary termination STOP."""
    return tree_action == STOP_ACTION


def is_single_voxel_action(tree_action: Any) -> bool:
    """Return True if edge represents a single unitary `(ch, x, y, z)` edit."""
    return (
        isinstance(tree_action, tuple)
        and len(tree_action) == 4
        and isinstance(tree_action[0], (int, np.integer))
    )


def flatten_tree_action(tree_action: Any) -> List[Action]:
    """Flatten a tree edge (single voxel, macro tuple, or STOP) into unitary edits."""
    if is_stop_action(tree_action) or tree_action is None:
        return []
    if is_single_voxel_action(tree_action):
        ch, x, y, z = tree_action
        return [(int(ch), int(x), int(y), int(z))]
    out: List[Action] = []
    for item in tree_action:
        if is_single_voxel_action(item):
            ch, x, y, z = item
            out.append((int(ch), int(x), int(y), int(z)))
    return out


def _single_channel_softmax(logits: np.ndarray, valid_mask: np.ndarray, temperature: float) -> Tuple[np.ndarray, bool]:
    valid_b = valid_mask > 0
    if not np.any(valid_b):
        return np.zeros_like(logits, dtype=np.float32), False
    temp = max(float(temperature), 1e-6)
    vals = logits.astype(np.float64) / temp
    max_v = float(np.max(vals[valid_b]))
    exp_v = np.zeros_like(vals, dtype=np.float64)
    exp_v[valid_b] = np.exp(vals[valid_b] - max_v)
    tot = float(exp_v.sum())
    if tot <= 1e-12:
        probs = np.zeros_like(vals, dtype=np.float32)
        probs[valid_b] = 1.0 / float(np.sum(valid_b))
        return probs, True
    return (exp_v / tot).astype(np.float32), True


@dataclass
class MCTSNode:
    """Tree/DAG node for MCTS with PUCT statistics."""

    prior: float = 0.0
    action_to_parent: Optional[TreeAction] = None
    parent: Optional["MCTSNode"] = None
    visit_count: int = 0
    value_sum: float = 0.0
    max_value: float = float("-inf")
    children: Dict[TreeAction, "MCTSNode"] = field(default_factory=dict)
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
        fpu_value: float = 0.0,
    ) -> float:
        """
        Compute PUCT score with optional MixMax value blending (AlphaTruss),
        configurable First Play Urgency (`fpu_value`), and Virtual Loss penalty.
        """
        eff_n = self.visit_count + self.virtual_loss

        if eff_n == 0:
            q_value = float(fpu_value)
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
        self.virtual_loss = max(0, self.virtual_loss - 1)

    def select_child(
        self,
        c_puct: float,
        mixmax_beta: float = 1.0,
        fpu_reduction: Optional[float] = None,
        exclude_ids: Optional[Set[int]] = None,
    ) -> Tuple[TreeAction, "MCTSNode"]:
        """Select child with highest PUCT score, supporting relative FPU and DAG acyclicity."""
        if not self.children:
            raise ValueError("Cannot select child from node with no children")

        total_visits = sum(child.visit_count + child.virtual_loss for child in self.children.values())
        has_vl = any(child.virtual_loss > 0 for child in self.children.values())
        # Ensure sim 0 / batch 0 break ties by prior whenever FPU or virtual loss is active
        parent_eff = max(1, total_visits) if (fpu_reduction is not None or has_vl) else total_visits

        fpu_val = 0.0
        if fpu_reduction is not None and self.visit_count > 0:
            fpu_val = max(-1.0, self.mean_value - float(fpu_reduction))

        best_action = None
        best_score = float("-inf")
        best_child = None

        for action, child in self.children.items():
            if exclude_ids and id(child) in exclude_ids:
                continue
            score = child.puct_score(
                c_puct,
                parent_eff,
                mixmax_beta=mixmax_beta,
                fpu_value=fpu_val,
            )
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_child is None:
            action, child = next(iter(self.children.items()))
            return action, child

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
        min_remove_candidates: int = 0,
        remove_prior_floor: float = 0.0,
        factored_policy: bool = False,
        op_probs: Optional[Tuple[float, float, float]] = None,
        allow_stop: bool = False,
    ) -> None:
        """
        Expand leaf node using masked softmax over legal actions.

        When `factored_policy=True`, spatial priors over ADD and REMOVE are normalized
        independently (`P(v | op)`) and weighted by operation probabilities `P(op)`.
        Priors are strictly renormalized after pruning so sum == 1.0.
        """
        if self.is_expanded:
            return

        temp = max(float(temperature), 1e-6)
        if factored_policy:
            add_probs_only, has_add = _single_channel_softmax(policy_add, valid_add, temp)
            rem_probs_only, has_rem = _single_channel_softmax(policy_remove, valid_remove, temp)
            if not has_add and not has_rem:
                self.is_expanded = True
                return
            if has_add and has_rem:
                if op_probs is not None:
                    p_a_op = max(1e-4, float(op_probs[0]))
                    p_r_op = max(1e-4, float(op_probs[1]))
                    s_op = p_a_op + p_r_op
                    w_add, w_rem = p_a_op / s_op, p_r_op / s_op
                else:
                    w_add, w_rem = 0.5, 0.5
                if remove_prior_floor > 0.0:
                    w_rem = max(w_rem, float(min(max(remove_prior_floor, 0.0), 0.90)))
                    w_add = 1.0 - w_rem
            elif has_add:
                w_add, w_rem = 1.0, 0.0
            else:
                w_add, w_rem = 0.0, 1.0
            add_priors = add_probs_only * np.float32(w_add)
            remove_priors = rem_probs_only * np.float32(w_rem)
        else:
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

        if min_remove_candidates > 0:
            rem_coords = np.argwhere(valid_remove > 0)
            rem_cands = [
                ((1, int(c[0]), int(c[1]), int(c[2])), float(remove_priors[c[0], c[1], c[2]]))
                for c in rem_coords
            ]
            rem_cands.sort(key=lambda x: x[1], reverse=True)
            add_cands = [item for item in candidates_list if item[0][0] == 0]
            add_cands.sort(key=lambda x: x[1], reverse=True)

            if top_k is not None:
                n_rem = min(len(rem_cands), int(min_remove_candidates), int(top_k))
                n_add = min(len(add_cands), int(top_k) - n_rem)
                kept = add_cands[:n_add] + rem_cands[:n_rem]
                if len(kept) < int(top_k) and len(rem_cands) > n_rem:
                    extra = min(len(rem_cands) - n_rem, int(top_k) - len(kept))
                    kept.extend(rem_cands[n_rem : n_rem + extra])
                candidates_list = kept
            else:
                kept_rem_set = {a for a, _ in candidates_list if a[0] == 1}
                for act, pr in rem_cands[: int(min_remove_candidates)]:
                    if act not in kept_rem_set:
                        candidates_list.append((act, pr))
        elif top_k is not None and len(candidates_list) > top_k:
            candidates_list.sort(key=lambda x: x[1], reverse=True)
            candidates_list = candidates_list[:top_k]

        if not candidates_list and not allow_stop:
            self.is_expanded = True
            return

        if remove_prior_floor > 0.0:
            add_items = [(a, float(p)) for a, p in candidates_list if a[0] == 0]
            rem_items = [(a, float(p)) for a, p in candidates_list if a[0] == 1]
            if add_items and rem_items:
                sum_add = sum(p for _, p in add_items)
                sum_rem = sum(p for _, p in rem_items)
                tot = sum_add + sum_rem
                rem_frac = (sum_rem / tot) if tot > 0.0 else 0.0
                floor = float(min(max(remove_prior_floor, 0.0), 0.95))
                if rem_frac < floor:
                    if sum_rem > 0.0:
                        rem_scaled = [(a, floor * (p / sum_rem)) for a, p in rem_items]
                    else:
                        u_r = floor / len(rem_items)
                        rem_scaled = [(a, u_r) for a, _ in rem_items]
                    add_target = 1.0 - floor
                    if sum_add > 0.0:
                        add_scaled = [(a, add_target * (p / sum_add)) for a, p in add_items]
                    else:
                        u_a = add_target / len(add_items)
                        add_scaled = [(a, u_a) for a, _ in add_items]
                    candidates_list = add_scaled + rem_scaled

        final_entries: List[Tuple[TreeAction, float]] = [(a, float(p)) for a, p in candidates_list]
        if allow_stop:
            stop_w = max(0.05, float(op_probs[2]) if op_probs is not None else 0.05)
            spatial_sum = sum(p for _, p in final_entries)
            if spatial_sum > 0.0:
                scale = (1.0 - stop_w) / spatial_sum
                final_entries = [(a, p * scale) for a, p in final_entries]
            final_entries.append((STOP_ACTION, stop_w))

        kept_sum = float(sum(float(p) for _, p in final_entries))
        for action_key, prior in final_entries:
            norm_prior = float(prior) / kept_sum if kept_sum > 0.0 else 1.0 / len(final_entries)
            self.children[action_key] = MCTSNode(
                prior=norm_prior,
                action_to_parent=action_key,
                parent=self,
            )

        self.is_expanded = True

    def expand_macro(
        self,
        policy_add: np.ndarray,
        policy_remove: np.ndarray,
        valid_add: np.ndarray,
        valid_remove: np.ndarray,
        action_batch_size: int = 8,
        phase: str = "REFINEMENT",
        temperature: float = 1.0,
        remove_prior_floor: float = 0.25,
        op_probs: Optional[Tuple[float, float, float]] = None,
        allow_stop: bool = False,
        max_candidates: int = 16,
        seed: Optional[int] = None,
    ) -> None:
        """
        Expand node with composite macro-actions of size `<= action_batch_size`:
        - Phase 1 (GROWTH): top-K ADD, upper/lower-flange ADD, and exploratory Gumbel ADD
        - Phase 2 (REFINEMENT): pure ADD, pure REMOVE, mass-conserving SWAP (REMOVE+ADD), and STOP
        """
        if self.is_expanded:
            return

        temp = max(float(temperature), 1e-6)
        prob_add, has_add = _single_channel_softmax(policy_add, valid_add, temp)
        prob_rem, has_rem = _single_channel_softmax(policy_remove, valid_remove, temp)
        if not has_add and not has_rem:
            if allow_stop:
                self.children[STOP_ACTION] = MCTSNode(prior=1.0, action_to_parent=STOP_ACTION, parent=self)
            self.is_expanded = True
            return

        ny = policy_add.shape[1]
        K = max(1, int(action_batch_size))
        rng = np.random.default_rng(seed if seed is not None else (abs(hash((self.visit_count, int(valid_add.sum()), int(valid_remove.sum())))) % (2**31)))

        add_coords = np.argwhere(valid_add > 0)
        rem_coords = np.argwhere(valid_remove > 0)

        def _top_pool(coords: np.ndarray, probs: np.ndarray, ch: int, cap: int = 128) -> Tuple[List[Action], np.ndarray]:
            if coords.size == 0:
                return [], np.zeros((0,), dtype=np.float64)
            pvals = probs[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float64)
            order = np.argsort(pvals)[::-1][:cap]
            acts = [(ch, int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])) for i in order]
            return acts, pvals[order]

        add_pool, add_p = _top_pool(add_coords, prob_add, 0, cap=max(96, K * 6))
        rem_pool, rem_p = _top_pool(rem_coords, prob_rem, 1, cap=max(96, K * 6))

        macro_entries: List[Tuple[str, Tuple[Action, ...], float]] = []
        seen_keys: Set[TreeAction] = set()

        def _score_logmean(pvals: np.ndarray, idxs: np.ndarray) -> float:
            if len(idxs) == 0:
                return -20.0
            return float(np.mean(np.log(np.clip(pvals[idxs], 1e-8, 1.0))))

        def _add_macro(kind: str, actions_seq: List[Action], raw_score: float) -> None:
            if not actions_seq:
                return
            key: Tuple[Action, ...] = tuple(sorted(actions_seq))
            if key in seen_keys:
                return
            seen_keys.add(key)
            macro_entries.append((kind, key, raw_score))

        # 1. Pure ADD macro candidates
        if add_pool:
            ka = min(K, len(add_pool))
            idx_top = np.arange(ka)
            _add_macro("ADD", [add_pool[i] for i in idx_top], _score_logmean(add_p, idx_top) + 0.25)

            # Structural flange-directed variants (top chord y >= ny/2, bottom chord y < ny/2)
            ys = np.array([a[2] for a in add_pool], dtype=np.float64)
            y_mid = float(ny - 1) * 0.5
            top_bias = np.log(np.clip(add_p, 1e-8, 1.0)) + 1.6 * np.maximum(0.0, (ys - y_mid) / max(1.0, y_mid))
            bot_bias = np.log(np.clip(add_p, 1e-8, 1.0)) + 1.6 * np.maximum(0.0, (y_mid - ys) / max(1.0, y_mid))
            outer_bias = np.log(np.clip(add_p, 1e-8, 1.0)) + 1.4 * np.abs(ys - y_mid) / max(1.0, y_mid)
            for bias_arr, bonus in ((top_bias, 0.05), (bot_bias, 0.05), (outer_bias, 0.15)):
                idx_b = np.argsort(bias_arr)[::-1][:ka]
                _add_macro("ADD", [add_pool[int(i)] for i in idx_b], _score_logmean(add_p, idx_b) + bonus)

            # Gumbel-Top-K diverse ADD variants
            n_g_add = 4 if phase == "GROWTH" else 2
            log_a = np.log(np.clip(add_p, 1e-8, 1.0))
            for _ in range(n_g_add):
                gumbel = -np.log(-np.log(np.clip(rng.random(len(add_pool)), 1e-6, 1.0 - 1e-6)))
                idx_g = np.argsort(log_a + 0.85 * gumbel)[::-1][:ka]
                _add_macro("ADD", [add_pool[int(i)] for i in idx_g], _score_logmean(add_p, idx_g))

        # 2. Pure REMOVE and SWAP macro candidates in Phase 2 (REFINEMENT)
        if phase != "GROWTH" and rem_pool:
            kr = min(K, len(rem_pool))
            idx_rtop = np.arange(kr)
            _add_macro("REM", [rem_pool[i] for i in idx_rtop], _score_logmean(rem_p, idx_rtop) + 0.25)

            log_r = np.log(np.clip(rem_p, 1e-8, 1.0))
            for _ in range(3):
                gumbel = -np.log(-np.log(np.clip(rng.random(len(rem_pool)), 1e-6, 1.0 - 1e-6)))
                idx_rg = np.argsort(log_r + 0.85 * gumbel)[::-1][:kr]
                _add_macro("REM", [rem_pool[int(i)] for i in idx_rg], _score_logmean(rem_p, idx_rg))

            # 3. Mass-conserving SWAP edges: move low-value/low-SED core mass out to stiff flanges
            if add_pool:
                kr_s = min(max(1, K // 2), len(rem_pool))
                ka_s = min(max(1, K - kr_s), len(add_pool))
                swap_acts = [rem_pool[i] for i in range(kr_s)] + [add_pool[i] for i in range(ka_s)]
                swap_sc = 0.5 * (_score_logmean(rem_p, np.arange(kr_s)) + _score_logmean(add_p, np.arange(ka_s))) + 0.20
                _add_macro("SWAP", swap_acts, swap_sc)

                ys_a = np.array([a[2] for a in add_pool], dtype=np.float64)
                y_mid = float(ny - 1) * 0.5
                outer_a = np.log(np.clip(add_p, 1e-8, 1.0)) + 1.5 * np.abs(ys_a - y_mid) / max(1.0, y_mid)
                ys_r = np.array([a[2] for a in rem_pool], dtype=np.float64)
                inner_r = np.log(np.clip(rem_p, 1e-8, 1.0)) + 1.2 * (1.0 - np.abs(ys_r - y_mid) / max(1.0, y_mid))
                idx_ra = np.argsort(outer_a)[::-1][:ka_s]
                idx_rr = np.argsort(inner_r)[::-1][:kr_s]
                swap_flange = [rem_pool[int(i)] for i in idx_rr] + [add_pool[int(i)] for i in idx_ra]
                swap_flange_sc = 0.5 * (_score_logmean(rem_p, idx_rr) + _score_logmean(add_p, idx_ra)) + 0.15
                _add_macro("SWAP", swap_flange, swap_flange_sc)

                for _ in range(2):
                    g_r = -np.log(-np.log(np.clip(rng.random(len(rem_pool)), 1e-6, 1.0 - 1e-6)))
                    g_a = -np.log(-np.log(np.clip(rng.random(len(add_pool)), 1e-6, 1.0 - 1e-6)))
                    ir = np.argsort(log_r + 0.8 * g_r)[::-1][:kr_s]
                    ia = np.argsort(np.log(np.clip(add_p, 1e-8, 1.0)) + 0.8 * g_a)[::-1][:ka_s]
                    _add_macro(
                        "SWAP",
                        [rem_pool[int(i)] for i in ir] + [add_pool[int(i)] for i in ia],
                        0.5 * (_score_logmean(rem_p, ir) + _score_logmean(add_p, ia)),
                    )

        # Compute factored weights across active macro categories
        if phase == "GROWTH":
            type_budget = {"ADD": 1.0}
        else:
            p_a_op = float(op_probs[0]) if op_probs is not None else 0.5
            p_r_op = float(op_probs[1]) if op_probs is not None else 0.5
            if rem_pool and add_pool:
                tot_ar = max(1e-6, p_a_op + p_r_op)
                w_a = max(0.20, p_a_op / tot_ar)
                w_r = max(float(remove_prior_floor), p_r_op / tot_ar)
                w_sw = max(0.25, math.sqrt(w_a * w_r))
                type_budget = {"ADD": w_a, "REM": w_r, "SWAP": w_sw}
            elif rem_pool:
                type_budget = {"REM": 1.0}
            else:
                type_budget = {"ADD": 1.0}

        # Softmax within each category, then weight by category budget
        by_type: Dict[str, List[Tuple[Tuple[Action, ...], float]]] = {}
        for kind, key, sc in macro_entries:
            by_type.setdefault(kind, []).append((key, sc))

        weighted_items: List[Tuple[TreeAction, float]] = []
        active_budget_sum = sum(type_budget.get(k, 0.0) for k in by_type)
        for kind, items in by_type.items():
            b_w = type_budget.get(kind, 0.25) / max(1e-6, active_budget_sum)
            scs = np.array([sc for _, sc in items], dtype=np.float64)
            ex = np.exp(scs - np.max(scs))
            probs_k = ex / np.maximum(1e-12, ex.sum())
            for (key, _), pr in zip(items, probs_k):
                weighted_items.append((key, float(b_w * pr)))

        weighted_items.sort(key=lambda x: x[1], reverse=True)
        cap = max(4, int(max_candidates) - (1 if allow_stop and phase != "GROWTH" else 0))
        weighted_items = weighted_items[:cap]

        if allow_stop and phase != "GROWTH":
            stop_w = max(0.05, min(0.45, float(op_probs[2]) if op_probs is not None else 0.06))
            w_sum = sum(p for _, p in weighted_items)
            if w_sum > 0.0:
                weighted_items = [(act, p * (1.0 - stop_w) / w_sum) for act, p in weighted_items]
            weighted_items.append((STOP_ACTION, stop_w))

        tot_p = float(sum(p for _, p in weighted_items))
        for action_key, p in weighted_items:
            norm_p = float(p) / tot_p if tot_p > 0.0 else 1.0 / len(weighted_items)
            self.children[action_key] = MCTSNode(
                prior=norm_p,
                action_to_parent=action_key,
                parent=self,
            )

        self.is_expanded = True

    def update(self, value: float) -> None:
        """Update node visit count, value sum, and running maximum."""
        self.visit_count += 1
        self.value_sum += value
        if value > self.max_value:
            self.max_value = value

    def get_visit_distribution(self) -> Dict[TreeAction, int]:
        """Shallow visit count distribution over direct children."""
        return {action: child.visit_count for action, child in self.children.items()}

    def get_root_voxel_visits(self) -> Dict[Action, int]:
        """
        Project ONLY direct root children onto legal single-voxel actions `(ch, x, y, z)`,
        eliminating depth 2..4 illegal-action leakage from policy training targets.
        """
        dist: Dict[Action, float] = {}
        for tree_act, child in self.children.items():
            if child.visit_count <= 0 or is_stop_action(tree_act):
                continue
            flat = flatten_tree_action(tree_act)
            if not flat:
                continue
            weight = float(child.visit_count) / float(len(flat))
            for act in flat:
                dist[act] = dist.get(act, 0.0) + weight
        return {act: max(1, int(round(v))) for act, v in dist.items()}

    def collect_subtree_visits(
        self,
        accumulator: Optional[Dict[Action, int]] = None,
        _visited: Optional[Set[int]] = None,
    ) -> Dict[Action, int]:
        """Recursively collect visit counts across the subtree/DAG (cycle-safe)."""
        if accumulator is None:
            accumulator = {}
        if _visited is None:
            _visited = set()
        nid = id(self)
        if nid in _visited:
            return accumulator
        _visited.add(nid)

        for action, child in self.children.items():
            if child.visit_count > 0 and not is_stop_action(action):
                flat = flatten_tree_action(action)
                if len(flat) == 1:
                    act = flat[0]
                    accumulator[act] = accumulator.get(act, 0) + child.visit_count
                elif len(flat) > 1:
                    w = max(1, int(round(child.visit_count / len(flat))))
                    for act in flat:
                        accumulator[act] = accumulator.get(act, 0) + w
            child.collect_subtree_visits(accumulator, _visited)
        return accumulator

    def tree_stats(self, _visited: Optional[Set[int]] = None) -> Tuple[int, int]:
        """Return (unique_expanded_nodes, max_depth) in this tree/DAG."""
        if _visited is None:
            _visited = set()
        nid = id(self)
        if nid in _visited:
            return 0, 0
        _visited.add(nid)

        if not self.children:
            return (1 if self.is_expanded else 0), 0
        total_exp = 1 if self.is_expanded else 0
        max_d = 0
        for child in self.children.values():
            c_exp, c_d = child.tree_stats(_visited)
            total_exp += c_exp
            if c_d + 1 > max_d:
                max_d = c_d + 1
        return total_exp, max_d

    def get_best_child(self) -> Optional[Tuple[TreeAction, "MCTSNode"]]:
        """Return the highest-visit direct child `(tree_action, node)`."""
        if not self.children:
            return None
        return max(
            self.children.items(),
            key=lambda x: (x[1].visit_count, x[1].mean_value),
        )

    def get_top_k_actions(self, k: int) -> List[Action]:
        """Return top-K direct child actions (flattened if macro) sorted by visit count descending."""
        sorted_children = sorted(
            self.children.items(),
            key=lambda x: (x[1].visit_count, x[1].mean_value),
            reverse=True,
        )
        out: List[Action] = []
        seen: Set[Action] = set()
        for tree_act, _ in sorted_children:
            if is_stop_action(tree_act):
                if not out:
                    return []
                continue
            for act in flatten_tree_action(tree_act):
                if act not in seen:
                    seen.add(act)
                    out.append(act)
                    if is_single_voxel_action(tree_act) and len(out) >= k:
                        return out
            if not is_single_voxel_action(tree_act):
                return out[:k]
        return out[:k]

    def is_leaf(self) -> bool:
        return not self.is_expanded

    def is_root(self) -> bool:
        return self.parent is None

    def path_from_root(self) -> List["MCTSNode"]:
        path = []
        node: Optional[MCTSNode] = self
        seen: Set[int] = set()
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def actions_from_root(self) -> List[Action]:
        path = self.path_from_root()
        out: List[Action] = []
        for node in path[1:]:
            if node.action_to_parent is not None:
                out.extend(flatten_tree_action(node.action_to_parent))
        return out

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
