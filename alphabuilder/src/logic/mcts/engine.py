"""
MCTS Engine for AlphaBuilder v3.1.

Implements Monte Carlo Tree Search with neural network guidance.
Uses a functional/imperative approach with pure functions where possible.

Reference: AlphaBuilder v3.1 MCTS Specification (specs/mcts_spec.md)
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Callable, NamedTuple
from functools import lru_cache
from dataclasses import dataclass
import hashlib

from .node import MCTSNode, Action
from .legal_moves import get_legal_moves, is_terminal_state


# =============================================================================
# Type Definitions
# =============================================================================

class MCTSConfig(NamedTuple):
    """Configuration for MCTS search."""
    num_simulations: int = 80
    batch_size: int = 8
    c_puct: float = 1.25
    density_threshold: float = 0.5
    top_k_expansion: Optional[int] = None  # None = expand all valid
    min_prior_threshold: float = 1e-6
    max_volume_fraction: float = 0.3
    min_volume_fraction: float = 0.01


class SearchResult(NamedTuple):
    """Result from MCTS search."""
    actions: List[Action]  # Top-k actions to execute
    visit_distribution: Dict[Action, int]  # Full visit counts
    root_value: float  # Mean value at root
    num_simulations: int  # Actual simulations performed


# =============================================================================
# State Manipulation (Pure Functions)
# =============================================================================

def apply_action(density: np.ndarray, action: Action) -> np.ndarray:
    """
    Apply a single action to density grid.
    
    Pure function - returns new array, doesn't modify input.
    
    Args:
        density: Current density grid (D, H, W)
        action: (channel, x, y, z) where channel=0 is add, channel=1 is remove
        
    Returns:
        New density grid with action applied
    """
    new_density = density.copy()
    channel, x, y, z = action
    
    if channel == 0:  # Add
        new_density[x, y, z] = 1.0
    else:  # Remove
        new_density[x, y, z] = 0.0
    
    return new_density


def apply_actions(density: np.ndarray, actions: List[Action]) -> np.ndarray:
    """
    Apply a batch of actions to density grid.
    
    Pure function - returns new array.
    
    Args:
        density: Current density grid (D, H, W)
        actions: List of (channel, x, y, z) actions
        
    Returns:
        New density grid with all actions applied
    """
    new_density = density.copy()
    
    for channel, x, y, z in actions:
        if channel == 0:  # Add
            new_density[x, y, z] = 1.0
        else:  # Remove
            new_density[x, y, z] = 0.0
    
    return new_density


def compute_state_hash(density: np.ndarray) -> str:
    """
    Compute hash of density grid for cache lookup.
    
    Uses binarized grid for consistent hashing.
    """
    binary = (density > 0.5).astype(np.uint8)
    return hashlib.md5(binary.tobytes()).hexdigest()


def build_state_tensor(
    density: np.ndarray,
    bc_masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
    forces: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    """
    Build 7-channel state tensor for neural network.
    
    Pure function - composes inputs into network format.
    
    Args:
        density: Density grid (D, H, W)
        bc_masks: (mask_x, mask_y, mask_z) boundary condition masks
        forces: (force_x, force_y, force_z) normalized force vectors
        
    Returns:
        State tensor (7, D, H, W)
    """
    return np.stack([
        density,
        bc_masks[0],
        bc_masks[1],
        bc_masks[2],
        forces[0],
        forces[1],
        forces[2]
    ], axis=0).astype(np.float32)


# =============================================================================
# Selection Phase
# =============================================================================

def select_leaf(root: MCTSNode, c_puct: float) -> Tuple[MCTSNode, List[MCTSNode]]:
    """
    Select a leaf node by traversing tree with PUCT.
    
    Descends from root choosing child with highest PUCT score
    until reaching an unexpanded node.
    
    Args:
        root: Root node of search tree
        c_puct: Exploration constant
        
    Returns:
        Tuple of (leaf_node, path_from_root)
    """
    node = root
    path = [node]
    
    while node.is_expanded and node.children:
        _, node = node.select_child(c_puct)
        path.append(node)
    
    return node, path


# =============================================================================
# Expansion Phase
# =============================================================================

def expand_node(
    node: MCTSNode,
    policy_add: np.ndarray,
    policy_remove: np.ndarray,
    valid_add: np.ndarray,
    valid_remove: np.ndarray,
    config: MCTSConfig
) -> None:
    """
    Expand a leaf node with children based on policy and valid moves.
    
    Mutates node in place (pragmatic choice for tree efficiency).
    
    Args:
        node: Leaf node to expand
        policy_add: Policy logits for add actions
        policy_remove: Policy logits for remove actions  
        valid_add: Binary mask of valid add positions
        valid_remove: Binary mask of valid remove positions
        config: MCTS configuration
    """
    node.expand(
        policy_add=policy_add,
        policy_remove=policy_remove,
        valid_add=valid_add,
        valid_remove=valid_remove,
        top_k=config.top_k_expansion,
        min_prior_threshold=config.min_prior_threshold
    )


# =============================================================================
# Backpropagation Phase
# =============================================================================

def backup(path: List[MCTSNode], value: float) -> None:
    """
    Propagate value back up the tree.
    
    For each node in path:
        N_i <- N_i + 1
        W_i <- W_i + V
    
    Note: Single-player game, no sign inversion needed.
    
    Args:
        path: Path from root to leaf
        value: Value estimate to propagate
    """
    for node in path:
        node.update(value)


# =============================================================================
# Main Search Function
# =============================================================================

def run_mcts_search(
    root_density: np.ndarray,
    bc_masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
    forces: Tuple[np.ndarray, np.ndarray, np.ndarray],
    predict_fn: Callable[[np.ndarray], Tuple[float, np.ndarray, np.ndarray]],
    config: MCTSConfig = MCTSConfig()
) -> SearchResult:
    """
    Run MCTS search from given root state.
    
    Executes the 4-phase simulation loop:
    1. Selection: Descend tree with PUCT
    2. Expansion: Add children based on policy
    3. Evaluation: Get value from neural network
    4. Backup: Propagate value up tree
    
    Args:
        root_density: Initial density grid (D, H, W)
        bc_masks: (mask_x, mask_y, mask_z) boundary conditions (constant)
        forces: (force_x, force_y, force_z) normalized forces (constant)
        predict_fn: Neural network predict(state) -> (value, policy_add, policy_remove)
        config: MCTS configuration
        
    Returns:
        SearchResult with top-k actions and statistics
    """
    # Build root state tensor
    root_state = build_state_tensor(root_density, bc_masks, forces)
    
    # Get initial policy and value for root
    root_value, policy_add, policy_remove = predict_fn(root_state)
    
    # Get legal moves for root
    valid_add, valid_remove = get_legal_moves(root_density, config.density_threshold)
    
    # Create and expand root node
    root = MCTSNode(prior=1.0)
    expand_node(root, policy_add, policy_remove, valid_add, valid_remove, config)
    
    # Initial backup for root
    root.update(root_value)
    
    # Cache for neural network evaluations (state_hash -> (value, policy_add, policy_remove))
    eval_cache: Dict[str, Tuple[float, np.ndarray, np.ndarray]] = {}
    
    # Simulation loop
    for sim in range(config.num_simulations):
        # 1. Selection
        leaf, path = select_leaf(root, config.c_puct)
        
        # Compute leaf state by applying actions from root
        actions = leaf.actions_from_root()
        leaf_density = apply_actions(root_density, actions)
        
        # 2. Check terminal
        is_terminal, reason = is_terminal_state(
            leaf_density,
            max_volume_fraction=config.max_volume_fraction,
            min_volume_fraction=config.min_volume_fraction,
            threshold=config.density_threshold
        )
        
        if is_terminal:
            # Terminal value based on reason
            if reason == "max_volume_exceeded":
                value = -0.5  # Penalty but not catastrophic
            elif reason == "structure_too_small":
                value = -1.0  # Catastrophic failure
            else:
                value = 0.0  # Neutral terminal
        else:
            # 3. Expansion & Evaluation
            leaf_state = build_state_tensor(leaf_density, bc_masks, forces)
            state_hash = compute_state_hash(leaf_density)
            
            # Check cache
            if state_hash in eval_cache:
                value, policy_add, policy_remove = eval_cache[state_hash]
            else:
                value, policy_add, policy_remove = predict_fn(leaf_state)
                eval_cache[state_hash] = (value, policy_add, policy_remove)
            
            # Get legal moves for leaf
            valid_add, valid_remove = get_legal_moves(leaf_density, config.density_threshold)
            
            # Expand leaf
            expand_node(leaf, policy_add, policy_remove, valid_add, valid_remove, config)
        
        # 4. Backup
        backup(path, value)
    
    # Extract results
    visit_dist = root.get_visit_distribution()
    top_actions = root.get_top_k_actions(config.batch_size)
    
    return SearchResult(
        actions=top_actions,
        visit_distribution=visit_dist,
        root_value=root.mean_value,
        num_simulations=config.num_simulations
    )


# =============================================================================
# Convenience Wrapper Class (Optional, for compatibility)
# =============================================================================

class AlphaBuilderMCTS:
    """
    Convenience wrapper for MCTS with stored configuration.
    
    Wraps the functional MCTS implementation for easier integration
    with existing OOP-style code.
    """
    
    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], Tuple[float, np.ndarray, np.ndarray]],
        num_simulations: int = 80,
        batch_size: int = 8,
        c_puct: float = 1.25
    ):
        """
        Initialize MCTS wrapper.
        
        Args:
            predict_fn: Neural network prediction function
            num_simulations: Number of MCTS simulations per search
            batch_size: Number of top actions to return (micro-batch size)
            c_puct: PUCT exploration constant
        """
        self.predict_fn = predict_fn
        self.config = MCTSConfig(
            num_simulations=num_simulations,
            batch_size=batch_size,
            c_puct=c_puct
        )
    
    def get_action_batch(
        self,
        density: np.ndarray,
        bc_masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
        forces: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> Tuple[List[Action], Dict[Action, int]]:
        """
        Get micro-batch of actions for current state.
        
        Args:
            density: Current density grid (D, H, W)
            bc_masks: Boundary condition masks
            forces: Force vectors
            
        Returns:
            Tuple of (action_list, visit_distribution)
        """
        result = run_mcts_search(
            root_density=density,
            bc_masks=bc_masks,
            forces=forces,
            predict_fn=self.predict_fn,
            config=self.config
        )
        return result.actions, result.visit_distribution
    
    def search(
        self,
        density: np.ndarray,
        bc_masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
        forces: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> SearchResult:
        """
        Full MCTS search with complete results.
        
        Args:
            density: Current density grid
            bc_masks: Boundary condition masks
            forces: Force vectors
            
        Returns:
            SearchResult with full statistics
        """
        return run_mcts_search(
            root_density=density,
            bc_masks=bc_masks,
            forces=forces,
            predict_fn=self.predict_fn,
            config=self.config
        )


# =============================================================================
# Utility Functions for Analysis
# =============================================================================

def analyze_search_quality(result: SearchResult, policy_add: np.ndarray, policy_remove: np.ndarray) -> Dict:
    """
    Analyze MCTS search quality metrics.
    
    Used to verify success criteria from spec:
    1. Convergence: Top actions should dominate visits
    2. Refutation: Sometimes MCTS should disagree with raw policy
    
    Args:
        result: MCTS search result
        policy_add: Raw policy logits for add
        policy_remove: Raw policy logits for remove
        
    Returns:
        Dict with analysis metrics
    """
    visit_dist = result.visit_distribution
    
    if not visit_dist:
        return {"error": "no_visits"}
    
    # Total visits (excluding root's initial visit)
    total_visits = sum(visit_dist.values())
    
    # Top-8 concentration
    sorted_visits = sorted(visit_dist.values(), reverse=True)
    top_8_visits = sum(sorted_visits[:8])
    concentration = top_8_visits / max(total_visits, 1)
    
    # Find action with highest raw policy probability
    all_probs = np.concatenate([policy_add.flatten(), policy_remove.flatten()])
    best_policy_idx = np.argmax(all_probs)
    
    # Check if most visited action matches highest policy action
    most_visited_action = max(visit_dist.items(), key=lambda x: x[1])[0]
    
    # Convert policy idx to action format for comparison
    spatial_size = policy_add.size
    if best_policy_idx < spatial_size:
        best_policy_action = (0,) + np.unravel_index(best_policy_idx, policy_add.shape)
    else:
        best_policy_action = (1,) + np.unravel_index(best_policy_idx - spatial_size, policy_remove.shape)
    
    refutation = most_visited_action != best_policy_action
    
    return {
        "total_visits": total_visits,
        "top_8_concentration": concentration,
        "most_visited_action": most_visited_action,
        "highest_policy_action": best_policy_action,
        "refutation": refutation,
        "root_value": result.root_value
    }
