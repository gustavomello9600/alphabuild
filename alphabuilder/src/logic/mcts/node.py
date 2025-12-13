"""
MCTS Node implementation for AlphaBuilder v3.1.

Each node represents a state reached after a sequence of unitary voxel edits.
Stores only PUCT statistics (lightweight), not the full grid state.

Reference: AlphaBuilder v3.1 MCTS Specification (specs/mcts_spec.md)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Any


# Action type: (channel, x, y, z) where channel = 0 for Add, 1 for Remove
Action = Tuple[int, int, int, int]


@dataclass
class MCTSNode:
    """
    Tree node for MCTS with PUCT statistics.
    
    Stores minimal information for UCB calculation:
    - Visit count (N)
    - Value sum (W) 
    - Prior probability (P) from policy network
    - Reference to parent and children
    
    The actual grid state is NOT stored - only action deltas.
    
    Attributes:
        prior: Initial probability from Policy Head (P)
        action_to_parent: The (channel, x, y, z) action that led here from parent
        parent: Reference to parent node (None for root)
        visit_count: Number of times this node was visited (N)
        value_sum: Sum of values backed up through this node (W)
        children: Dict mapping actions to child nodes
        is_expanded: Whether this node has been expanded with children
    """
    prior: float = 0.0
    action_to_parent: Optional[Action] = None
    parent: Optional['MCTSNode'] = None
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[Action, 'MCTSNode'] = field(default_factory=dict)
    is_expanded: bool = False
    virtual_loss: int = 0
    
    def detach_parent(self) -> None:
        """
        Sever the link to parent.
        
        Usage: call this when promoting a child node to be the new root of the search tree.
        Crucial for garbage collection of the old, unused parts of the tree.
        """
        self.parent = None
        self.action_to_parent = None
    
    @property
    def mean_value(self) -> float:
        """
        Mean value Q = W/N.
        
        Returns 0 if node hasn't been visited (avoids division by zero).
        Range: [-1, 1] (matching Value Head output).
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    @property
    def Q(self) -> float:
        """Alias for mean_value (standard MCTS notation)."""
        return self.mean_value
    
    @property
    def N(self) -> int:
        """Alias for visit_count (standard MCTS notation)."""
        return self.visit_count
    
    @property
    def P(self) -> float:
        """Alias for prior (standard MCTS notation)."""
        return self.prior
    
    def puct_score(self, c_puct: float, parent_visits: int) -> float:
        """
        Calculate PUCT score for action selection.
        
        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
        
        Accounts for Virtual Loss during batch MCTS.
        """
        # Effective counts with virtual loss
        eff_n = self.visit_count + self.virtual_loss
        
        # Effective Q
        # If visited, use actual mean value. 
        # Apply virtual loss penalty to the mean value if virtual loss is active.
        # Standard approach: treat virtual loss as losing visits (value -1 or similar penalty)
        # Here we penalize the value sum directly.
        # W_eff = W - virtual_loss (assuming penalty of 1 per virtual visit)
        
        if eff_n == 0:
            q_value = 0.0
        else:
            # Penalize value sum by virtual loss amount (assuming scale [-1, 1], penalty of 1 is strong)
            eff_w = self.value_sum - self.virtual_loss
            q_value = eff_w / eff_n

        # Exploration term
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + eff_n)
        
        return q_value + exploration
        
    def apply_virtual_loss(self) -> None:
        """Apply virtual loss to discourage re-selection during batching."""
        self.virtual_loss += 1
        
    def revert_virtual_loss(self) -> None:
        """Revert virtual loss after batch evaluation."""
        self.virtual_loss -= 1

    
    def select_child(self, c_puct: float) -> Tuple[Action, 'MCTSNode']:
        """
        Select child with highest PUCT score.
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            Tuple of (action, child_node) with highest PUCT score
            
        Raises:
            ValueError: If node has no children
        """
        if not self.children:
            raise ValueError("Cannot select child from node with no children")
        
        # Total visits across all children (for PUCT calculation)
        total_visits = sum(child.visit_count for child in self.children.values())
        
        best_action = None
        best_score = float('-inf')
        best_child = None
        
        for action, child in self.children.items():
            score = child.puct_score(c_puct, total_visits)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def expand(
        self, 
        policy_add: np.ndarray, 
        policy_remove: np.ndarray,
        valid_add: np.ndarray,
        valid_remove: np.ndarray,
        top_k: Optional[int] = None,
        min_prior_threshold: float = 1e-6
    ) -> None:
        """
        Expand node with children based on policy priors and valid moves.
        
        Creates child nodes for legal actions with non-negligible priors.
        Applies Top-K pruning if specified to reduce branching factor.
        
        Args:
            policy_add: Policy logits for add actions (D, H, W)
            policy_remove: Policy logits for remove actions (D, H, W)
            valid_add: Binary mask of valid add positions (D, H, W)
            valid_remove: Binary mask of valid remove positions (D, H, W)
            top_k: If set, only expand top-K actions by prior (None = all valid)
            min_prior_threshold: Minimum prior to consider action (avoid noise)
        """
        if self.is_expanded:
            return
            
        # Apply masks to policies (set invalid to -inf for softmax)
        masked_add = np.where(valid_add > 0, policy_add, -np.inf)
        masked_remove = np.where(valid_remove > 0, policy_remove, -np.inf)
        
        # Flatten and combine for joint softmax
        # Flatten and combine for joint softmax
        flat_add = masked_add.flatten()
        flat_remove = masked_remove.flatten()
        combined = np.concatenate([flat_add, flat_remove])
        
        # Stable softmax
        max_val = np.max(combined[np.isfinite(combined)])
        exp_combined = np.exp(combined - max_val)
        exp_combined[~np.isfinite(combined)] = 0  # Zero out masked
        priors = exp_combined / (exp_combined.sum() + 1e-10)
        
        # Split back to add/remove
        spatial_size = flat_add.size
        add_priors = priors[:spatial_size].reshape(policy_add.shape)
        remove_priors = priors[spatial_size:].reshape(policy_remove.shape)
        
        # JIT accelerated candidate selection
        candidates_list = _expand_candidates_jit(
            add_priors, remove_priors, valid_add, valid_remove, min_prior_threshold
        )
        
        # Top-K pruning if specified
        if top_k is not None and len(candidates_list) > top_k:
            # Sort by prior (2nd element in tuple)
            # Python's sort is faster than doing this in JIT usually due to object overhead
            candidates_list.sort(key=lambda x: x[1], reverse=True)
            candidates_list = candidates_list[:top_k]
        
        # Create child nodes
        for action_tuple, prior in candidates_list:
            # Convert tuple back to Action format if needed, but jit returns correct shape
            # Action: (channel, x, y, z)
            action = action_tuple
            
            self.children[action] = MCTSNode(
                prior=prior,
                action_to_parent=action,
                parent=self
            )
        
        self.is_expanded = True

    def update(self, value: float) -> None:
        """
        Update node statistics during backpropagation.
        
        N_i <- N_i + 1
        W_i <- W_i + V
        Q_i <- W_i / N_i (computed via property)
        
        Args:
            value: Value estimate from neural network or terminal state
        """
        self.visit_count += 1
        self.value_sum += value
    
    def get_visit_distribution(self) -> Dict[Action, int]:
        """
        Get visit count distribution over children (Legacy: Shallow).
        
        Returns:
            Dict mapping actions to visit counts
        """
        return {action: child.visit_count for action, child in self.children.items()}

    def collect_subtree_visits(self, accumulator: Optional[Dict[Action, int]] = None) -> Dict[Action, int]:
        """
        Recursively collect visit counts from all descendants (Deep).
        
        Args:
            accumulator: Dictionary to accumulate counts (optional)
            
        Returns:
            Dict mapping actions to visit counts across the entire subtree
        """
        if accumulator is None:
            # Use a standard dict instead of defaultdict to avoid import changes if possible, 
            # or just be careful. We'll use standard dict fetch/set.
            accumulator = {}
        
        # Traverse children
        for action, child in self.children.items():
            # Add child's visits
            # Note: We track the action that *led* to the child. 
            # If multiple nodes result from same action (transpositions), this might sum them.
            # But MCTS tree structure is typically strictly hierarchical here (no graph DAG yet).
            
            current_count = accumulator.get(action, 0)
            accumulator[action] = current_count + child.visit_count
            
            # Recurse
            child.collect_subtree_visits(accumulator)
            
        return accumulator
    
    def get_top_k_actions(self, k: int) -> List[Action]:
        """
        Get top-K actions by visit count.
        
        Args:
            k: Number of actions to return
            
        Returns:
            List of actions sorted by visit count (descending)
        """
        sorted_children = sorted(
            self.children.items(), 
            key=lambda x: x[1].visit_count, 
            reverse=True
        )
        return [action for action, _ in sorted_children[:k]]
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (not expanded)."""
        return not self.is_expanded
    
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None
    
    def path_from_root(self) -> List['MCTSNode']:
        """
        Get path from root to this node.
        
        Returns:
            List of nodes from root to self (inclusive)
        """
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))
    
    def actions_from_root(self) -> List[Action]:
        """
        Get actions taken from root to reach this node.
        
        Returns:
            List of actions (excluding root which has no action)
        """
        path = self.path_from_root()
        return [node.action_to_parent for node in path[1:]]  # Skip root
    
    def __repr__(self) -> str:
        return (
            f"MCTSNode(N={self.visit_count}, Q={self.mean_value:.3f}, "
            f"P={self.prior:.4f}, children={len(self.children)}, "
            f"expanded={self.is_expanded})"
        )

# JIT Compiled Helper Functions
# Place at module level

try:
    from numba import jit
    # Use nopython=True for maximum speed, cache=True to avoid recompilation
    @jit(nopython=True, cache=True)
    def _expand_candidates_jit(add_priors, remove_priors, valid_add, valid_remove, threshold):
        """
        Extract valid actions with prior > threshold.
        Returns list of ((channel, x, y, z), prior) tuples.
        """
        # Numba doesn't list very well with heterogeneous types, but tuples of primitives are OK
        # We return a list of tuples: ( (c,x,y,z), prior )
        # Actually numba lists are typed.
        
        candidates = []
        
        # Add actions (channel=0)
        # Assuming 3D shape (D,H,W)
        D, H, W = add_priors.shape
        
        for x in range(D):
            for y in range(H):
                for z in range(W):
                    if valid_add[x, y, z] > 0 and add_priors[x, y, z] > threshold:
                        # Construct action tuple
                        # Note: Numba handles tuple construction
                        act = (0, x, y, z)
                        candidates.append((act, add_priors[x, y, z]))
                        
        # Remove actions (channel=1)
        for x in range(D):
            for y in range(H):
                for z in range(W):
                    if valid_remove[x, y, z] > 0 and remove_priors[x, y, z] > threshold:
                        act = (1, x, y, z)
                        candidates.append((act, remove_priors[x, y, z]))
                        
        return candidates

except ImportError:
    # Fallback if numba not installed
    def _expand_candidates_jit(add_priors, remove_priors, valid_add, valid_remove, threshold):
        candidates = []
        # Add actions
        for idx in np.ndindex(add_priors.shape):
            if valid_add[idx] > 0 and add_priors[idx] > threshold:
                action = (0, idx[0], idx[1], idx[2])
                candidates.append((action, float(add_priors[idx])))
        # Remove actions
        for idx in np.ndindex(remove_priors.shape):
            if valid_remove[idx] > 0 and remove_priors[idx] > threshold:
                action = (1, idx[0], idx[1], idx[2])
                candidates.append((action, float(remove_priors[idx])))
        return candidates
    
    def update(self, value: float) -> None:
        """
        Update node statistics during backpropagation.
        
        N_i <- N_i + 1
        W_i <- W_i + V
        Q_i <- W_i / N_i (computed via property)
        
        Args:
            value: Value estimate from neural network or terminal state
        """
        self.visit_count += 1
        self.value_sum += value
    
    def get_visit_distribution(self) -> Dict[Action, int]:
        """
        Get visit count distribution over children (Legacy: Shallow).
        
        Returns:
            Dict mapping actions to visit counts
        """
        return {action: child.visit_count for action, child in self.children.items()}

    def collect_subtree_visits(self, accumulator: Optional[Dict[Action, int]] = None) -> Dict[Action, int]:
        """
        Recursively collect visit counts from all descendants (Deep).
        
        Args:
            accumulator: Dictionary to accumulate counts (optional)
            
        Returns:
            Dict mapping actions to visit counts across the entire subtree
        """
        if accumulator is None:
            # Use a standard dict instead of defaultdict to avoid import changes if possible, 
            # or just be careful. We'll use standard dict fetch/set.
            accumulator = {}
        
        # Traverse children
        for action, child in self.children.items():
            # Add child's visits
            # Note: We track the action that *led* to the child. 
            # If multiple nodes result from same action (transpositions), this might sum them.
            # But MCTS tree structure is typically strictly hierarchical here (no graph DAG yet).
            
            current_count = accumulator.get(action, 0)
            accumulator[action] = current_count + child.visit_count
            
            # Recurse
            child.collect_subtree_visits(accumulator)
            
        return accumulator
    
    def get_top_k_actions(self, k: int) -> List[Action]:
        """
        Get top-K actions by visit count.
        
        Args:
            k: Number of actions to return
            
        Returns:
            List of actions sorted by visit count (descending)
        """
        sorted_children = sorted(
            self.children.items(), 
            key=lambda x: x[1].visit_count, 
            reverse=True
        )
        return [action for action, _ in sorted_children[:k]]
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (not expanded)."""
        return not self.is_expanded
    
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None
    
    def path_from_root(self) -> List['MCTSNode']:
        """
        Get path from root to this node.
        
        Returns:
            List of nodes from root to self (inclusive)
        """
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))
    
    def actions_from_root(self) -> List[Action]:
        """
        Get actions taken from root to reach this node.
        
        Returns:
            List of actions (excluding root which has no action)
        """
        path = self.path_from_root()
        return [node.action_to_parent for node in path[1:]]  # Skip root
    
    def __repr__(self) -> str:
        return (
            f"MCTSNode(N={self.visit_count}, Q={self.mean_value:.3f}, "
            f"P={self.prior:.4f}, children={len(self.children)}, "
            f"expanded={self.is_expanded})"
        )
