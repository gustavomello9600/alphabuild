import numpy as np
from typing import Tuple, FrozenSet, Set
from alphabuilder.src.logic.game_types import DesignState, GameAction, Coord, PhaseType
from alphabuilder.src.logic.graph_ops import check_global_connectivity, get_articulation_points

def get_perimeter(grid: np.ndarray) -> FrozenSet[Coord]:
    """
    Identifies all empty cells adjacent to at least one material cell.
    """
    rows, cols = grid.shape
    perimeter = set()
    
    # Get all material indices
    material_indices = np.argwhere(grid == 1)
    
    for r, c in material_indices:
        neighbors = [
            (r-1, c), (r+1, c), 
            (r, c-1), (r, c+1)
        ]
        for nr, nc in neighbors:
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] == 0:
                    perimeter.add((nr, nc))
                    
    return frozenset(perimeter)

def get_legal_actions(state: DesignState) -> Tuple[GameAction, ...]:
    """
    Returns a tuple of all legal actions for the current state.
    """
    actions = []
    
    # ADD Actions: Always allowed on perimeter
    # Heuristic ordering could be applied here, but we'll keep it simple for now
    for coord in state.perimeter:
        actions.append(GameAction(type='ADD', coord=coord))
        
    if state.phase == 'REFINEMENT':
        # REMOVE Actions: Allowed on material, with exceptions
        
        # 1. Calculate Articulation Points (expensive, only if needed)
        # We only need to check this if we are considering removing material
        articulation_points = get_articulation_points(state.grid)
        
        # Get all material coordinates
        rows, cols = state.grid.shape
        for r in range(rows):
            for c in range(cols):
                if state.grid[r, c] == 1:
                    coord = (r, c)
                    
                    # Constraint 1: Cannot remove fixed supports
                    if coord in state.supports:
                        continue
                        
                    # Constraint 2: Cannot remove load points
                    if coord in state.loads:
                        continue
                        
                    # Constraint 3: Cannot remove articulation points (would disconnect graph)
                    if coord in articulation_points:
                        continue
                        
                    actions.append(GameAction(type='REMOVE', coord=coord))
                    
    return tuple(actions)

def apply_action(state: DesignState, action: GameAction) -> DesignState:
    """
    Applies an action to the state and returns a new immutable state.
    """
    new_grid = state.grid.copy()
    r, c = action.coord
    
    if action.type == 'ADD':
        new_grid[r, c] = 1
    elif action.type == 'REMOVE':
        new_grid[r, c] = 0
        
    # Recalculate derived metadata
    new_perimeter = get_perimeter(new_grid)
    new_volume = int(np.sum(new_grid))
    
    # Check connectivity
    # Optimization: If phase is GROWTH, we check if we just connected everything.
    # If phase is REFINEMENT, we assume it stays connected because we forbid removing articulation points.
    # However, 'check_global_connectivity' is the ground truth.
    
    is_connected = check_global_connectivity(new_grid, state.supports, state.loads)
    
    # Determine Phase Transition
    new_phase = state.phase
    if state.phase == 'GROWTH' and is_connected:
        new_phase = 'REFINEMENT'
        
    return DesignState(
        grid=new_grid,
        supports=state.supports,
        loads=state.loads,
        phase=new_phase,
        is_connected=is_connected,
        volume=new_volume,
        perimeter=new_perimeter
    )
