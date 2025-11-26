import pytest
import numpy as np
from alphabuilder.src.logic.game_rules import GameState, get_legal_actions_3d, get_neighbors_3d, check_connectivity_3d

@pytest.fixture
def empty_state_3d():
    # 5D Tensor: (5, 4, 4, 4)
    tensor = np.zeros((5, 4, 4, 4), dtype=np.float32)
    # Add Support at (0,0,0)
    tensor[1, 0, 0, 0] = 1.0
    # Add Load at (3,3,3)
    tensor[3, 3, 3, 3] = -1.0
    return GameState(tensor=tensor, phase='GROWTH', step_count=0)

def test_get_neighbors_3d():
    coord = (1, 1, 1)
    shape = (3, 3, 3)
    neighbors = get_neighbors_3d(coord, shape)
    assert len(neighbors) == 6
    assert (0, 1, 1) in neighbors
    assert (2, 1, 1) in neighbors

def test_get_neighbors_boundary():
    coord = (0, 0, 0)
    shape = (3, 3, 3)
    neighbors = get_neighbors_3d(coord, shape)
    assert len(neighbors) == 3
    assert (1, 0, 0) in neighbors
    assert (0, 1, 0) in neighbors
    assert (0, 0, 1) in neighbors

def test_legal_actions_growth_initial(empty_state_3d):
    # Should allow adding near Support (0,0,0) and Load (3,3,3)
    actions = get_legal_actions_3d(empty_state_3d)
    
    # Support Neighbors: (1,0,0), (0,1,0), (0,0,1)
    # Load Neighbors: (2,3,3), (3,2,3), (3,3,2)
    
    coords = [a[1] for a in actions]
    assert (1, 0, 0) in coords
    assert (2, 3, 3) in coords
    assert len(actions) == 6 # 3 from support + 3 from load

def test_legal_actions_refinement():
    tensor = np.zeros((5, 3, 3, 3), dtype=np.float32)
    tensor[0, 1, 1, 1] = 1.0 # Material center
    tensor[1, 0, 0, 0] = 1.0 # Support
    
    state = GameState(tensor=tensor, phase='REFINEMENT', step_count=0)
    actions = get_legal_actions_3d(state)
    
    # ADD: Neighbors of (1,1,1)
    add_actions = [a for a in actions if a[0] == 'ADD']
    assert len(add_actions) == 6
    
    # REMOVE: (1,1,1)
    rem_actions = [a for a in actions if a[0] == 'REMOVE']
    assert len(rem_actions) == 1
    assert rem_actions[0][1] == (1, 1, 1)

def test_connectivity():
    # Line of material: (0,0,0) -> (0,0,1) -> (0,0,2)
    density = np.zeros((3, 3, 3))
    density[0, 0, 0] = 1
    density[0, 0, 1] = 1
    density[0, 0, 2] = 1
    
    starts = [(0, 0, 0)]
    ends = [(0, 0, 2)]
    
    assert check_connectivity_3d(density, starts, ends) == True
    
    # Break link
    density[0, 0, 1] = 0
    assert check_connectivity_3d(density, starts, ends) == False
