"""
Advanced pathfinding and exploration strategies for AlphaBuilder.

Implements A* pathfinding, heuristic exploration, and pattern generation.
"""

import numpy as np
import heapq
from typing import Tuple, List, Set, Dict, Optional
from dataclasses import dataclass


@dataclass
class Node:
    """Node for A* pathfinding."""
    pos: Tuple[int, int]
    g: float  # Cost from start
    h: float  # Heuristic to goal
    parent: Optional['Node'] = None
    
    @property
    def f(self) -> float:
        return self.g + self.h
    
    def __lt__(self, other):
        return self.f < other.f


def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Euclidean distance heuristic."""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def astar_pathfind(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    shape: Tuple[int, int],
    diagonal_penalty: float = 0.5
) -> List[Tuple[int, int]]:
    """
    A* pathfinding with structural bias.
    
    Args:
        start: Starting position (x, y)
        goal: Goal position (x, y)
        shape: Grid shape (nx, ny)
        diagonal_penalty: Penalty for non-diagonal moves (lower = prefer diagonal)
    
    Returns:
        List of (x, y) positions forming the path.
    """
    nx, ny = shape
    
    def get_neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbors with 8-connectivity."""
        x, y = pos
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx_new, ny_new = x + dx, y + dy
                if 0 <= nx_new < nx and 0 <= ny_new < ny:
                    neighbors.append((nx_new, ny_new))
        return neighbors
    
    def cost(current: Tuple[int, int], neighbor: Tuple[int, int]) -> float:
        """Cost function favoring diagonal/straight paths with noise."""
        dx = abs(neighbor[0] - current[0])
        dy = abs(neighbor[1] - current[1])
        
        base_cost = 0.0
        # Diagonal move
        if dx == 1 and dy == 1:
            base_cost = 1.414  # sqrt(2)
        # Straight move
        else:
            base_cost = 1.0 + diagonal_penalty
            
        # Add small random noise to break symmetry/determinism
        # Deterministic noise based on coordinates to keep A* consistent within one run
        noise = (hash((current, neighbor)) % 100) / 1000.0  # 0.0 to 0.1
        return base_cost + noise
    
    # Initialize
    open_set = []
    start_node = Node(start, g=0, h=euclidean_distance(start, goal))
    heapq.heappush(open_set, start_node)
    
    visited: Set[Tuple[int, int]] = set()
    g_scores: Dict[Tuple[int, int], float] = {start: 0}
    
    while open_set:
        current = heapq.heappop(open_set)
        
        if current.pos in visited:
            continue
            
        visited.add(current.pos)
        
        # Goal reached
        if current.pos == goal:
            path = []
            node = current
            while node is not None:
                path.append(node.pos)
                node = node.parent
            return list(reversed(path))
        
        # Explore neighbors
        for neighbor_pos in get_neighbors(current.pos):
            if neighbor_pos in visited:
                continue
            
            tentative_g = current.g + cost(current.pos, neighbor_pos)
            
            if tentative_g < g_scores.get(neighbor_pos, float('inf')):
                g_scores[neighbor_pos] = tentative_g
                neighbor_node = Node(
                    pos=neighbor_pos,
                    g=tentative_g,
                    h=euclidean_distance(neighbor_pos, goal),
                    parent=current
                )
                heapq.heappush(open_set, neighbor_node)
    
    # No path found - return straight line
    return [start, goal]


def create_astar_topology(
    topology: np.ndarray,
    starts: List[Tuple[int, int]],
    goals: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Create topology using A* paths from multiple start/goal pairs.
    
    Args:
        topology: Empty topology array (ny, nx)
        starts: List of start positions
        goals: List of goal positions
    
    Returns:
        Topology with A* paths filled.
    """
    result = topology.copy()
    ny, nx = topology.shape  # Fixed: shape is (rows, cols) -> (ny, nx)
    
    for start, goal in zip(starts, goals):
        path = astar_pathfind(start, goal, (nx, ny)) # Pathfinding expects (width, height) usually? 
        # Wait, astar_pathfind likely expects grid coordinates.
        # If grid is (ny, nx), then x is col (0..nx-1), y is row (0..ny-1).
        # Let's check astar_pathfind usage.
        
        for c, r in path: # Path usually returns (x, y) = (col, row)
            if 0 <= r < ny and 0 <= c < nx:
                result[r, c] = 1 # result[row, col]
            
    return result


def create_smart_heuristic_topology(topology: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """
    Create arch/truss-like topology using simple heuristics.
    
    Creates two diagonal braces forming an arch shape.
    """
    result = topology.copy()
    
    # Left edge (support)
    result[:, 0] = 1 # Changed from result[0, :] to result[:, 0] for left edge support
    
    # Upper diagonal brace
    for x in range(nx):
        y = int(ny * 0.75 - (ny * 0.5 * x / nx))  # Diagonal from top-left to mid-right
        if 0 <= y < ny:
            result[y, x] = 1 # Fixed: result[row, col]
    
    # Lower diagonal brace  
    for x in range(nx):
        y = int(ny * 0.25 + (ny * 0.5 * x / nx))  # Diagonal from bottom-left to mid-right
        if 0 <= y < ny:
            result[y, x] = 1 # Fixed: result[row, col]
    
    # Connect at right end
    mid_y = ny // 2
    result[max(0, mid_y-1):min(ny, mid_y+2), nx-1] = 1 # Fixed: result[row_slice, col]
    
    return result


def create_random_pattern(
    topology: np.ndarray, 
    ny: int, 
    nx: int, 
    seed: int
) -> np.ndarray:
    """
    Create a random structural pattern based on seed.
    
    Patterns:
    - 0: Straight line (baseline)
    - 1: Diagonal brace (X shape)
    - 2: Arch (curved)
    - 3: Double truss (parallel diagonals)
    """
    result = topology.copy()
    pattern_type = seed % 4
    
    # Note: In numpy, shape is (ny, nx). 
    # Loops usually go: for x in range(nx): result[y, x] = 1 (or result[row, col])
    
    if pattern_type == 0:
        # Straight line
        result[:, 0] = 1 # Left support (fixed from result[0, :])
        mid_y = ny // 2
        for x in range(nx):
            result[mid_y, x] = 1 # Fixed: result[row, col]
    
    elif pattern_type == 1:
        # X-shaped diagonal brace
        result[:, 0] = 1 # Left support (fixed from result[0, :])
        
        for x in range(nx):
            # Upper diagonal
            y1 = int(ny - (ny * x / nx))
            if 0 <= y1 < ny:
                result[y1, x] = 1 # Fixed: result[row, col]
            # Lower diagonal
            y2 = int(ny * x / nx)
            if 0 <= y2 < ny:
                result[y2, x] = 1 # Fixed: result[row, col]
                
    elif pattern_type == 2:
        # Arch
        result[:, 0] = 1 # Left support (fixed from result[0, :])
        for x in range(nx):
            # Parabolic curve
            y = int(ny * 0.5 + ny * 0.3 * np.sin(np.pi * x / nx))
            if 0 <= y < ny:
                result[y, x] = 1 # Fixed: result[row, col]
                # Thicken for stability (retained from original)
                if y > 0:
                    result[y-1, x] = 1 # Fixed: result[row, col]
                if y < ny - 1:
                    result[y+1, x] = 1 # Fixed: result[row, col]
    
    elif pattern_type == 3:
        # Double parallel diagonal truss
        result[:, 0] = 1 # Left support (fixed from result[0, :])
        offset = ny // 4
        for x in range(nx):
            # Top diagonal
            y1 = int(ny * 0.75 - (ny * 0.5 * x / nx))
            if 0 <= y1 < ny:
                result[y1, x] = 1 # Fixed: result[row, col]
            # Bottom diagonal
            y2 = int(ny * 0.25 + (ny * 0.5 * x / nx))
            if 0 <= y2 < ny:
                result[x, y2] = 1
    
    return result


def calculate_structural_metrics(topology: np.ndarray) -> Dict[str, float]:
    """
    Calculate additional structural metrics for logging.
    
    Returns:
        Dictionary with volume_fraction, pattern_entropy, connectivity_score.
    """
    # Volume fraction
    volume_fraction = float(np.sum(topology)) / topology.size
    
    # Pattern entropy (spatial distribution)
    if np.sum(topology) > 0:
        # Normalize topology to probability distribution
        topology_norm = topology / np.sum(topology)
        # Remove zeros for log
        nonzero_mask = topology_norm > 0
        entropy = -np.sum(topology_norm[nonzero_mask] * np.log(topology_norm[nonzero_mask] + 1e-10))
    else:
        entropy = 0.0
    
    # Connectivity score (rough estimate: ratio of edges to nodes)
    # Count 4-connected neighbors
    material_pixels = np.argwhere(topology == 1)
    if len(material_pixels) > 0:
        edge_count = 0
        for r, c in material_pixels:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < topology.shape[0] and 0 <= nc < topology.shape[1]:
                    if topology[nr, nc] == 1:
                        edge_count += 1
        connectivity_score = edge_count / (2.0 * len(material_pixels))  # Divide by 2 (each edge counted twice)
    else:
        connectivity_score = 0.0
    
    return {
        "volume_fraction": volume_fraction,
        "pattern_entropy": entropy,
        "connectivity_score": connectivity_score
    }
