import numpy as np
from typing import Tuple, List, Set, NamedTuple
from dataclasses import dataclass

# Type Definitions
Coord = Tuple[int, int, int] # (z, y, x) or (d, h, w)

@dataclass(frozen=True)
class GameState:
    tensor: np.ndarray # (5, D, H, W)
    phase: str # 'GROWTH' or 'REFINEMENT'
    step_count: int
    
    @property
    def density(self) -> np.ndarray:
        return self.tensor[0]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.tensor.shape[1:]

def get_neighbors_3d(coord: Coord, shape: Tuple[int, int, int]) -> List[Coord]:
    """Return 6-connected neighbors."""
    d, h, w = coord
    D, H, W = shape
    neighbors = []
    for dd, dh, dw in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
        nd, nh, nw = d+dd, h+dh, w+dw
        if 0 <= nd < D and 0 <= nh < H and 0 <= nw < W:
            neighbors.append((nd, nh, nw))
    return neighbors

def check_connectivity_3d(density: np.ndarray, starts: List[Coord], ends: List[Coord]) -> bool:
    """
    Check if any start is connected to any end via material (1s).
    """
    if not starts or not ends:
        return False
        
    shape = density.shape
    visited = set()
    queue = list(starts)
    visited.update(starts)
    
    found_end = False
    
    # Optimization: Convert ends to set for O(1) lookup
    end_set = set(ends)
    
    while queue:
        curr = queue.pop(0)
        
        if curr in end_set:
            return True
            
        for n in get_neighbors_3d(curr, shape):
            if density[n] == 1 and n not in visited:
                visited.add(n)
                queue.append(n)
                
    return False

def get_legal_actions_3d(state: GameState) -> List[Tuple[str, Coord]]:
    """
    Get legal actions based on phase.
    """
    actions = []
    D, H, W = state.shape
    density = state.density
    
    # GROWTH: Add material adjacent to existing material or seeds
    if state.phase == 'GROWTH':
        # Find all 0s adjacent to 1s OR Seeds (Supports/Loads)
        
        # 1. Material Indices
        mat_indices = np.argwhere(density == 1)
        
        # 2. Support Indices (Ch1)
        support_indices = np.argwhere(state.tensor[1] == 1)
        
        # 3. Load Indices (Ch2-4)
        load_indices = np.argwhere(state.tensor[2:].sum(axis=0) != 0)
        
        # Combine all "Active" voxels
        # We use a set of tuples for uniqueness
        active_voxels = set()
        for idx in mat_indices: active_voxels.add(tuple(idx))
        for idx in support_indices: active_voxels.add(tuple(idx))
        for idx in load_indices: active_voxels.add(tuple(idx))
        
        potential_adds = set()
        for coord in active_voxels:
            for n in get_neighbors_3d(coord, state.shape):
                if density[n] == 0:
                    potential_adds.add(n)
                    
        for coord in potential_adds:
            actions.append(('ADD', coord))
            
    # REFINEMENT: Add or Remove
    elif state.phase == 'REFINEMENT':
        # ADD: Same as above (Perimeter)
        mat_indices = np.argwhere(density == 1)
        potential_adds = set()
        for idx in mat_indices:
            coord = tuple(idx)
            for n in get_neighbors_3d(coord, state.shape):
                if density[n] == 0:
                    potential_adds.add(n)
        for coord in potential_adds:
            actions.append(('ADD', coord))
            
        # REMOVE: Any material that is NOT a support or load
        # And does not break connectivity (expensive check!)
        # For MCTS, we might allow breaking and punish with reward?
        # Or strictly forbid.
        
        # Supports: Ch1
        supports = np.argwhere(state.tensor[1] == 1)
        support_set = set(map(tuple, supports))
        
        # Loads: Ch2-4
        loads = np.argwhere(state.tensor[2:].sum(axis=0) != 0)
        load_set = set(map(tuple, loads))
        
        for idx in mat_indices:
            coord = tuple(idx)
            if coord in support_set or coord in load_set:
                continue
                
            # Check Articulation Point (Optional/Expensive)
            # For now, allow removal. If it breaks, the Physics Solver 
            # will return 0 fitness (Singular Matrix) or we check connectivity after.
            actions.append(('REMOVE', coord))
            
    return actions
