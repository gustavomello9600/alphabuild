"""
A* Pathfinding for 3D Grid Connectivity (Phase 1: GROWTH)

This module provides pathfinding algorithms to create connected structures
from load points to support points in a 3D voxel grid.
"""

import heapq
import numpy as np
from typing import List, Tuple, Set, Optional

Coord = Tuple[int, int, int]


def find_path_3d(
    start: Coord,
    goal: Coord,
    grid_shape: Tuple[int, int, int],
    obstacles: Optional[Set[Coord]] = None
) -> List[Coord]:
    """
    Find shortest path from start to goal using A* algorithm.
    
    Args:
        start: Starting coordinate (d, h, w)
        goal: Goal coordinate (d, h, w)
        grid_shape: Grid dimensions (D, H, W)
        obstacles: Optional set of blocked coordinates
        
    Returns:
        List of coordinates from start to goal (inclusive)
    """
    if obstacles is None:
        obstacles = set()
    
    D, H, W = grid_shape
    
    # Priority queue: (f_score, counter, coord, path)
    counter = 0
    open_set = [(0, counter, start, [start])]
    closed_set = set()
    
    def heuristic(coord: Coord) -> float:
        """Euclidean distance heuristic (admissible)"""
        return np.sqrt(sum((a - b)**2 for a, b in zip(coord, goal)))
    
    def get_neighbors(coord: Coord) -> List[Coord]:
        """Get 6-connected neighbors (±x, ±y, ±z)"""
        d, h, w = coord
        neighbors = []
        for dd, dh, dw in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            nd, nh, nw = d + dd, h + dh, w + dw
            if 0 <= nd < D and 0 <= nh < H and 0 <= nw < W:
                neighbor = (nd, nh, nw)
                if neighbor not in obstacles:
                    neighbors.append(neighbor)
        return neighbors
    
    while open_set:
        f_score, _, current, path = heapq.heappop(open_set)
        
        if current == goal:
            return path
        
        if current in closed_set:
            continue
            
        closed_set.add(current)
        
        g_score = len(path) - 1  # Distance from start
        
        for neighbor in get_neighbors(current):
            if neighbor in closed_set:
                continue
            
            new_g = g_score + 1
            h = heuristic(neighbor)
            f = new_g + h
            
            new_path = path + [neighbor]
            counter += 1
            heapq.heappush(open_set, (f, counter, neighbor, new_path))
    
    # No path found
    return []


def build_connectivity_backbone(
    load_points: List[Coord],
    support_points: List[Coord],
    grid_shape: Tuple[int, int, int]
) -> List[Coord]:
    """
    Build a connected backbone structure from all loads to supports.
    
    Strategy: Connect each load to the nearest support point.
    
    Args:
        load_points: List of load coordinates
        support_points: List of support coordinates
        grid_shape: Grid dimensions (D, H, W)
        
    Returns:
        List of all coordinates in the backbone (union of all paths)
    """
    backbone_coords = set()
    
    # Add support points to backbone
    for support in support_points:
        backbone_coords.add(support)
    
    # For each load, find path to nearest support
    for load in load_points:
        # Find nearest support
        min_dist = float('inf')
        nearest_support = None
        
        for support in support_points:
            dist = np.sqrt(sum((a - b)**2 for a, b in zip(load, support)))
            if dist < min_dist:
                min_dist = dist
                nearest_support = support
        
        # Find path from load to nearest support
        # Don't treat existing backbone as obstacles - allow path merging
        path = find_path_3d(load, nearest_support, grid_shape, obstacles=None)
        
        # Add path to backbone
        for coord in path:
            backbone_coords.add(coord)
    
    return list(backbone_coords)


def extract_load_points(tensor: np.ndarray) -> List[Coord]:
    """
    Extract load point coordinates from state tensor.
    
    Args:
        tensor: State tensor (5, D, H, W)
        
    Returns:
        List of coordinates where loads are applied
    """
    # Loads are in channels 2, 3, 4 (Fx, Fy, Fz)
    # Any non-zero force indicates a load point
    force_magnitude = np.sqrt(
        tensor[2]**2 + tensor[3]**2 + tensor[4]**2
    )
    
    coords = np.argwhere(force_magnitude > 0)
    return [(int(d), int(h), int(w)) for d, h, w in coords]


def extract_support_points(tensor: np.ndarray) -> List[Coord]:
    """
    Extract support point coordinates from state tensor.
    
    Args:
        tensor: State tensor (5, D, H, W)
        
    Returns:
        List of coordinates where supports are fixed
    """
    # Supports are in channel 1 (Mask)
    coords = np.argwhere(tensor[1] > 0)
    return [(int(d), int(h), int(w)) for d, h, w in coords]


import scipy.ndimage as nd

def thicken_backbone(
    backbone_coords: List[Coord],
    grid_shape: Tuple[int, int, int],
    thickness: int = 1  # Base iterations
) -> List[Coord]:
    """
    Thicken a 1-voxel backbone using morphological dilation.
    
    This creates structurally robust paths instead of thin lines,
    solving the singular matrix issue in FEM/SIMP.
    
    Args:
        backbone_coords: List of coordinates in the backbone
        grid_shape: Grid dimensions (D, H, W)
        thickness: Base number of dilation iterations (default: 1)
        
    Returns:
        List of coordinates including original + thickened neighbors
    """
    D, H, W = grid_shape
    
    # Create binary grid from coords
    grid = np.zeros(grid_shape, dtype=bool)
    for d, h, w in backbone_coords:
        grid[d, h, w] = True
        
    # Randomize iterations for data augmentation (1 to 2)
    # As per expert directive
    iterations = np.random.randint(1, 3)
    
    # Apply 3D binary dilation
    # structure=None uses default 3x3x3 connectivity
    thickened_grid = nd.binary_dilation(grid, iterations=iterations)
    
    # Extract coordinates
    coords = np.argwhere(thickened_grid)
    return [(int(d), int(h), int(w)) for d, h, w in coords]
