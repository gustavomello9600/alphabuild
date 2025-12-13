"""
MCTS Engine for AlphaBuilder v3.1.

Implements Monte Carlo Tree Search with neural network guidance.
Uses a functional/imperative approach with pure functions where possible.

Reference: AlphaBuilder v3.1 MCTS Specification (specs/mcts_spec.md)
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Callable, NamedTuple, Any
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
    action_batch_size: int = 1 # Number of actions to execute/return
    c_puct: float = 1.25
    density_threshold: float = 0.5
    top_k_expansion: Optional[int] = None  # None = expand all valid
    min_prior_threshold: float = 1e-6
    max_volume_fraction: float = 0.3
    min_volume_fraction: float = 0.01
    phase: str = "REFINEMENT"  # "GROWTH" or "REFINEMENT"
    add_root_noise: bool = False
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25


# Phase-specific configuration presets (from MCTS Spec Section 8)
# Optimization: batch_size=8 matches GPU sweet spot
PHASE1_CONFIG = MCTSConfig(
    num_simulations=160, # Increased sims to leverage higher throughput
    c_puct=1.2,
    batch_size=8,
    action_batch_size=32,
    top_k_expansion=32,  # Must be >= action_batch_size to allow full PV extraction
    min_volume_fraction=0.05,
    max_volume_fraction=0.9,
    phase="GROWTH"
)

# Standard Config for Phase 2 (Refinement)
# Uses same micro-batch size (32 actions) and max_depth (4) as Phase 1
PHASE2_CONFIG = MCTSConfig(
    num_simulations=160,
    c_puct=0.8,
    batch_size=8,
    action_batch_size=32,
    top_k_expansion=32,  # Must be >= action_batch_size to allow full PV extraction
    min_volume_fraction=0.05,
    max_volume_fraction=0.9,
    phase="REFINEMENT"
)


class SearchResult(NamedTuple):
    """Result from MCTS search."""
    actions: List[Action]  # Top-k actions to execute
    visit_distribution: Dict[Action, int]  # Full visit counts
    root_value: float  # Mean value at root
    num_simulations: int  # Actual simulations performed
    root: Any # MCTSNode


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


def select_pv_sequences(
    root: MCTSNode, 
    max_actions: int = 32, 
    max_depth: int = 4
) -> List[Action]:
    """
    Selects a micro-batch of actions by extracting Principal Variation sequences.
    
    Algorithm: Breadth-then-Depth
    1. Sort root children by visit count (Breadth).
    2. For each top child (Head), greedily follow the best path up to max_depth (Depth).
    3. Aggregate sequences until max_actions is reached.
    
    Args:
        root: The MCTS root node
        max_actions: Total number of actions to fill in the batch
        max_depth: Maximum depth of any single sequence
        
    Returns:
        List of Actions to be executed sequentially.
    """
    if not root.children:
        return []

    # 1. Identify Heads (Breadth)
    # Sort children by visit count (primary) and Q-value (secondary)
    root_children_sorted = sorted(
        root.children.items(), 
        key=lambda item: (item[1].visit_count, item[1].mean_value), 
        reverse=True
    )
    
    selected_actions: List[Action] = []
    total_actions = 0
    
    # Spatial Lock: Prevent multiple actions on the same voxel in one batch
    # This ensures 32 actions = 32 unique modifications
    locked_voxels = set()
    
    # 2. Extract Deep Sequence for each Head
    for action, child in root_children_sorted:
        if total_actions >= max_actions:
            break
            
        # Helper to check spatial conflict
        # Action format: (channel, x, y, z)
        def is_locked(act):
            coords = (act[1], act[2], act[3])
            return coords in locked_voxels

        # Check Root Action
        if is_locked(action):
            continue
            
        # Start sequence with the root action (Head)
        seq = [action]
        # Lock this voxel for this batch
        locked_voxels.add((action[1], action[2], action[3]))
        
        current_node = child
        
        # Extend sequence deeper (Depth)
        # We start loop 1 level deep, so we go up to max_depth - 1 extensions
        for _ in range(max_depth - 1):
            if not current_node.children:
                break
                
            # Greedy selection: Best child by visit count
            # BUT we must filter out locked voxels first
            
            # Get valid children (not locked)
            # Note: We sort all children, then pick the first valid one.
            sorted_children = sorted(
                current_node.children.items(),
                key=lambda item: (item[1].visit_count, item[1].mean_value),
                reverse=True
            )
            
            best_action = None
            best_child = None
            
            for child_act, child_node in sorted_children:
                if not is_locked(child_act):
                    best_action = child_act
                    best_child = child_node
                    break
            
            # If no valid child found (all locked), stop this sequence branch here
            if best_action is None:
                break
            
            seq.append(best_action)
            locked_voxels.add((best_action[1], best_action[2], best_action[3]))
            current_node = best_child
        
        # Add to batch
        # Truncate if we overflow max_actions
        remaining_slots = max_actions - total_actions
        actions_to_add = seq[:remaining_slots]
        
        # If truncation removes some locked voxels, technically we should unlock them
        # but since we process sequentially and greedily, it doesn't matter for this batch.
        
        selected_actions.extend(actions_to_add)
        total_actions += len(actions_to_add)
        
    return selected_actions



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
    config: MCTSConfig = MCTSConfig(),
    root_node: Optional[MCTSNode] = None,
    eval_cache: Optional[Dict[str, Tuple[float, np.ndarray, np.ndarray]]] = None,
    predict_batch_fn: Optional[Callable[[List[np.ndarray]], Tuple[List[float], List[np.ndarray], List[np.ndarray]]]] = None
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
    # Reuse provided cache or create new one
    if eval_cache is None:
        eval_cache = {}
        
    # Setup Root
    if root_node is not None:
        # Reuse existing tree
        root = root_node
        if not root.is_expanded:
            # Should not happen if passed correctly, but safe fallback
            root_state = build_state_tensor(root_density, bc_masks, forces)
            value, p_add, p_remove = predict_fn(root_state)
            valid_add, valid_remove = get_legal_moves(
                root_density, config.density_threshold, bc_masks=bc_masks
            )
            expand_node(root, p_add, p_remove, valid_add, valid_remove, config)
            root.update(value)
    else:
        # Create fresh tree
        root_state = build_state_tensor(root_density, bc_masks, forces)
        root_value, policy_add, policy_remove = predict_fn(root_state)
        valid_add, valid_remove = get_legal_moves(
            root_density, config.density_threshold, bc_masks=bc_masks
        )
        root = MCTSNode(prior=1.0)
        expand_node(root, policy_add, policy_remove, valid_add, valid_remove, config)
        root.update(root_value)
    
    # Add Dirichlet noise to root (crucial for exploration with reused trees)
    if config.add_root_noise and root.is_expanded:
        # Note: We apply noise to the priors of CHILDREN
        # But MCTSNode stores children in a dict.
        # We need to re-normalize priors.
        # Simplification: Only efficient way is if expand_node supports noise injection,
        # or we hack it here. 
        # Given functional structures, modifying children's priors is OK.
        
        children = list(root.children.values())
        if children:
            noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(children))
            frac = config.root_exploration_fraction
            for child, n in zip(children, noise):
                child.prior = child.prior * (1 - frac) + n * frac
    
    # Cache for neural network evaluations (state_hash -> (value, policy_add, policy_remove))
    # Using the eval_cache passed in argument

    
    # Simulation loop
    if predict_batch_fn is not None and config.batch_size > 1:
        # Batch MCTS
        num_batches = (config.num_simulations + config.batch_size - 1) // config.batch_size
        
        for _ in range(num_batches):
            # 1. Selection Phase (Parallel-ish)
            batch_leaves = []
            batch_paths = []
            batch_states = []
            batch_state_hashes = []
            batch_indices = [] # Indices in the inference batch
            
            # Temporary list to hold items that don't need inference (terminal/cached)
            ready_to_backup = [] # List of (path, value)
            
            current_batch_size = 0
            
            # Select B leaves with Virtual Loss
            for __ in range(config.batch_size):
                leaf, path = select_leaf(root, config.c_puct)
                
                # Apply Virtual Loss immediately to discourage re-selection
                for node in path:
                    node.apply_virtual_loss()
                
                # Compute leaf state
                actions = leaf.actions_from_root()
                leaf_density = apply_actions(root_density, actions)
                
                # Check terminal
                is_terminal, reason = is_terminal_state(
                    leaf_density,
                    max_volume_fraction=config.max_volume_fraction,
                    min_volume_fraction=config.min_volume_fraction,
                    threshold=config.density_threshold,
                    bc_masks=bc_masks
                )
                
                if is_terminal:
                    if reason == "max_volume_exceeded":
                        value = -0.5
                    elif reason == "structure_too_small":
                        value = -1.0
                    else:
                        value = 0.0
                    # Will backup after loop
                    ready_to_backup.append((path, value))
                    continue
                
                # Evaluation Prep
                state_hash = compute_state_hash(leaf_density)
                if state_hash in eval_cache:
                    value, p_add, p_remove = eval_cache[state_hash]
                    
                    # Already evaluated, verify expansion
                    leaf_state = build_state_tensor(leaf_density, bc_masks, forces) # Need this? Maybe not if cached.
                    # But we need valid moves if not expanded.
                    # If in cache, it might still need expansion if we just cached the prediction?
                    # `eval_cache` stores prediction results.
                    # If node is new instance (not expanded), we need to expand it.
                    
                    valid_add, valid_remove = get_legal_moves(
                        leaf_density, config.density_threshold, bc_masks=bc_masks
                    )
                    expand_node(leaf, p_add, p_remove, valid_add, valid_remove, config)
                    
                    ready_to_backup.append((path, value))
                else:
                    # Needs inference
                    leaf_state = build_state_tensor(leaf_density, bc_masks, forces)
                    
                    batch_leaves.append(leaf)
                    batch_paths.append(path)
                    batch_states.append(leaf_state)
                    batch_state_hashes.append(state_hash)
                    
            # 2. Inference Phase
            if batch_states:
                values, pols_add, pols_remove = predict_batch_fn(batch_states)
                
                # 3. Expansion & Backup Phase
                for i, (val, p_a, p_r) in enumerate(zip(values, pols_add, pols_remove)):
                    leaf = batch_leaves[i]
                    path = batch_paths[i]
                    sha = batch_state_hashes[i]
                    
                    # Cache
                    eval_cache[sha] = (val, p_a, p_r)
                    
                    # Reconstruct density for valid moves (expensive? optimized in get_legal_moves?)
                    # We didn't store density to save memory/complexity, need to re-derive or store in temp list
                    # Storing 4-8 density arrays is fine.
                    # Let's re-optimize: store density in the selection loop?
                    # Yes, minimal overhead for small batch.
                    # But for now, re-computing actions is safer than passing mutable arrays around.
                    # Actually, we need density for get_legal_moves.
                    actions = leaf.actions_from_root()
                    leaf_density = apply_actions(root_density, actions)
                    
                    valid_add, valid_remove = get_legal_moves(
                        leaf_density, config.density_threshold, bc_masks=bc_masks
                    )
                    
                    expand_node(leaf, p_a, p_r, valid_add, valid_remove, config)
                    
                    ready_to_backup.append((path, val))
            
            # 4. Finalize Backup & Revert Virtual Loss
            # Note: We must revert VL for ALL selected paths, including terminal ones
            # The ready_to_backup list contains all paths we touched.
            
            for path, value in ready_to_backup:
                for node in path:
                    node.revert_virtual_loss()
                backup(path, value)

    else:
        # Sequential MCTS (Original)
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
                threshold=config.density_threshold,
                bc_masks=bc_masks
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
                valid_add, valid_remove = get_legal_moves(
                    leaf_density, config.density_threshold, bc_masks=bc_masks
                )
                
                # Expand leaf
                expand_node(leaf, policy_add, policy_remove, valid_add, valid_remove, config)
            
            # 4. Backup
            backup(path, value)
    
    # Extract results
    # visit_dist = root.get_visit_distribution() # Legacy: Shallow
    visit_dist = root.collect_subtree_visits() # New: Deep Recursive (All Search Nodes)
    
    # Get top actions (root-level only for 'actions' return, usually)
    # But for 'actions' list we usually want the alternatives.
    # The 'actions' field in SearchResult is typically top-k root actions.
    top_actions = root.get_top_k_actions(config.action_batch_size)
    
    return SearchResult(
        actions=top_actions,
        visit_distribution=visit_dist,
        root_value=root.mean_value,
        num_simulations=config.num_simulations,
        root=root
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
        action_batch_size: int = 1,
        c_puct: float = 1.25,
        top_k_expansion: Optional[int] = None,
        predict_batch_fn: Optional[Callable[[List[np.ndarray]], Tuple[List[float], List[np.ndarray], List[np.ndarray]]]] = None
    ):
        """
        Initialize MCTS wrapper.
        
        Args:
            predict_fn: Neural network prediction function
            num_simulations: Number of MCTS simulations per search
            batch_size: Number of top actions to return (micro-batch size)
            c_puct: PUCT exploration constant
            top_k_expansion: Max children to expand (pruning)
            predict_batch_fn: Optional batch prediction function for parallel search
        """
        self.predict_fn = predict_fn
        self.predict_batch_fn = predict_batch_fn
        self.config = MCTSConfig(
            num_simulations=num_simulations,
            batch_size=batch_size,
            action_batch_size=action_batch_size,
            c_puct=c_puct,
            top_k_expansion=top_k_expansion,
            add_root_noise=True  # Enable noise for tree reuse
        )
        self.root: Optional[MCTSNode] = None
        self.eval_cache: Dict[str, Tuple[float, np.ndarray, np.ndarray]] = {}

    def reset(self):
        """Clear search tree and cache for new episode."""
        self.root = None
        self.eval_cache = {}
    
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
            config=self.config,
            root_node=self.root,
            eval_cache=self.eval_cache
        )
        
        # Persist the root for subsequent calls (e.g. sequence extraction)
        self.root = result.root
        
        return result
        
        # Update our root reference to the (potentially new) root of the search
        # Actually run_mcts_search doesn't return the root explicitly, 
        # but if we passed self.root, it modified it in-place.
        # If self.root was None, we need to know what the new root is.
        # Problem: run_mcts_search creates 'root' local var if None passed.
        # We need to capture it.
        # Solution: Since run_mcts_search returns SearchResult which doesn't include root...
        # We should probably modify run_mcts_search or this wrapper to handle it.
        # BUT: For get_action_batch, we generally don't want to advance the tree yet?
        # Typically this is used for parallel playing.
        # Let's assume for now we don't support tree reuse in get_action_batch 
        # unless we refactor run_mcts_search to return the root.
        
        # ACTUALLY: The best way to use this wrapper is via a explicit 'step' method.
        pass # Placeholder comment
        
        return result.actions, result.visit_distribution
    
    def step(self, action: Action) -> None:
        """
        Advance the tree to the selected child.
        
        Args:
            action: Comparison action (channel, x, y, z)
        """
        if self.root is not None and action in self.root.children:
            new_root = self.root.children[action]
            new_root.detach_parent()
            self.root = new_root
        else:
            # Tree divergence or reset: clear root to start fresh next search
            self.root = None
            
    def search(
        self,
        density: np.ndarray,
        bc_masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
        forces: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> SearchResult:
        """
        Full MCTS search with complete results.
        
        Updates internal state (root) if it was None.
        """
        # If root is None, create it here effectively by passing None
        # But we need to capture the created root.
        # Since run_mcts_search doesn't return root, we have a problem implementing 'reuse' 
        # if we start from scratch.
        # Workaround: Manually create root if None.
        
        if self.root is None:
            # Create fresh root
            root_state = build_state_tensor(density, bc_masks, forces)
            root_value, policy_add, policy_remove = self.predict_fn(root_state)
            valid_add, valid_remove = get_legal_moves(
                density, self.config.density_threshold, bc_masks=bc_masks
            )
            self.root = MCTSNode(prior=1.0)
            expand_node(self.root, policy_add, policy_remove, valid_add, valid_remove, self.config)
            self.root.update(root_value)
            
        result = run_mcts_search(
            root_density=density,
            bc_masks=bc_masks,
            forces=forces,
            predict_fn=self.predict_fn,
            config=self.config,
            root_node=self.root,
            eval_cache=self.eval_cache
        )
        return result


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
