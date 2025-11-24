"""
Exploration strategies for Phase 2 refinement.
Implements semi-guided exploration with multiple heuristics.
"""

import numpy as np
from typing import List, Tuple
from enum import Enum


class ExplorationStrategy(Enum):
    """Available exploration strategies."""
    RANDOM = "random"
    REMOVE_WEAK = "remove_weak"
    ADD_SUPPORT = "add_support"
    SYMMETRY = "symmetry"


def calculate_distance_field(topology: np.ndarray, targets: List[Tuple[int, int]]) -> np.ndarray:
    """
    Calculate distance field from target points using BFS.
    Used to identify "weak" regions far from critical points.
    """
    nx, ny = topology.shape
    dist = np.full((nx, ny), np.inf)
    
    # Initialize with target points
    queue = []
    for x, y in targets:
        if 0 <= x < nx and 0 <= y < ny:
            dist[x, y] = 0
            queue.append((x, y))
    
    # BFS to compute distances
    while queue:
        x, y = queue.pop(0)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx_new, ny_new = x + dx, y + dy
            if 0 <= nx_new < nx and 0 <= ny_new < ny:
                if dist[nx_new, ny_new] > dist[x, y] + 1:
                    dist[nx_new, ny_new] = dist[x, y] + 1
                    queue.append((nx_new, ny_new))
    
    return dist


def score_remove_weak(topology: np.ndarray, coord: Tuple[int, int], 
                      load_points: List[Tuple[int, int]],
                      support_points: List[Tuple[int, int]]) -> float:
    """
    Score for removing material at coord.
    Higher score = more likely to be removable (less structurally important).
    
    Heuristic: Distance from both load and support paths.
    """
    x, y = coord
    
    # Calculate distance from load and support
    dist_load = calculate_distance_field(topology, load_points)
    dist_support = calculate_distance_field(topology, support_points)
    
    # Material far from both load and support = weak
    score = (dist_load[x, y] + dist_support[x, y]) / 2.0
    
    return score


def score_add_support(topology: np.ndarray, coord: Tuple[int, int],
                      load_points: List[Tuple[int, int]]) -> float:
    """
    Score for adding material at coord.
    Higher score = more likely to improve structure.
    
    Heuristic: Close to load points + adjacent to existing material.
    """
    x, y = coord
    nx, ny = topology.shape
    
    # Distance to nearest load
    dist_load = calculate_distance_field(topology, load_points)
    load_proximity = 1.0 / (dist_load[x, y] + 1.0)
    
    # Number of adjacent material cells
    adjacent_material = 0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx_new, ny_new = x + dx, y + dy
        if 0 <= nx_new < nx and 0 <= ny_new < ny:
            if topology[nx_new, ny_new] == 1:
                adjacent_material += 1
    
    # Prefer locations near load with some adjacent support
    score = load_proximity * (adjacent_material + 0.1)
    
    return score


def score_symmetry(topology: np.ndarray, coord: Tuple[int, int], action_type: str) -> float:
    """
    Score for maintaining vertical symmetry.
    Higher score = action maintains or improves symmetry.
    """
    x, y = coord
    nx, ny = topology.shape
    mid_y = ny // 2
    
    # Mirror point
    y_mirror = 2 * mid_y - y
    
    if not (0 <= y_mirror < ny):
        return 0.0
    
    current_value = topology[x, y]
    mirror_value = topology[x, y_mirror]
    
    if action_type == "ADD":
        # Adding material - check if mirror also has material
        if mirror_value == 1:
            return 1.0  # Maintains symmetry
        else:
            return 0.5  # Breaks symmetry
    else:  # REMOVE
        # Removing material - check if mirror is empty
        if mirror_value == 0:
            return 1.0  # Maintains symmetry
        else:
            return 0.5  # Breaks symmetry
    
    return 0.0


def select_action_with_strategy(
    actions: List[Tuple[str, Tuple[int, int]]],
    topology: np.ndarray,
    strategy: str,
    load_points: List[Tuple[int, int]],
    support_points: List[Tuple[int, int]],
    rng: np.random.Generator
) -> Tuple[str, Tuple[int, int]]:
    """
    Select an action using specified strategy.
    
    Args:
        actions: List of (action_type, coord) tuples
        topology: Current topology
        strategy: Strategy name
        load_points: List of load coordinates
        support_points: List of support coordinates
        rng: Random number generator
        
    Returns:
        Selected (action_type, coord)
    """
    if strategy == "random" or len(actions) == 0:
        return actions[rng.integers(0, len(actions))]
    
    # Separate add and remove actions
    add_actions = [(t, c) for t, c in actions if t == "ADD"]
    remove_actions = [(t, c) for t, c in actions if t == "REMOVE"]
    
    if strategy == "remove_weak" and len(remove_actions) > 0:
        # Score all remove actions
        scores = [score_remove_weak(topology, coord, load_points, support_points) 
                  for _, coord in remove_actions]
        # Select with softmax probability
        if max(scores) > 0:
            exp_scores = np.exp(np.array(scores) - max(scores))
            probs = exp_scores / exp_scores.sum()
            idx = rng.choice(len(remove_actions), p=probs)
            return remove_actions[idx]
        else:
            return remove_actions[rng.integers(0, len(remove_actions))]
    
    elif strategy == "add_support" and len(add_actions) > 0:
        # Score all add actions
        scores = [score_add_support(topology, coord, load_points)
                  for _, coord in add_actions]
        # Select with softmax probability
        if max(scores) > 0:
            exp_scores = np.exp(np.array(scores) - max(scores))
            probs = exp_scores / exp_scores.sum()
            idx = rng.choice(len(add_actions), p=probs)
            return add_actions[idx]
        else:
            return add_actions[rng.integers(0, len(add_actions))]
    
    elif strategy == "symmetry":
        # Score all actions for symmetry
        scores = [score_symmetry(topology, coord, action_type)
                  for action_type, coord in actions]
        # Select with softmax probability
        if max(scores) > 0:
            exp_scores = np.exp(np.array(scores) - max(scores))
            probs = exp_scores / exp_scores.sum()
            idx = rng.choice(len(actions), p=probs)
            return actions[idx]
        else:
            return actions[rng.integers(0, len(actions))]
    
    # Fallback to random
    return actions[rng.integers(0, len(actions))]
