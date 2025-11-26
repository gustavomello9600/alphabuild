import pytest
import numpy as np
from unittest.mock import MagicMock
from alphabuilder.src.logic.mcts import MCTSAgent, MCTSNode
from alphabuilder.src.logic.game_rules import GameState

@pytest.fixture
def mock_state():
    tensor = np.zeros((5, 4, 4, 4), dtype=np.float32)
    tensor[1, 0, 0, 0] = 1.0 # Support
    return GameState(tensor=tensor, phase='GROWTH', step_count=0)

def test_mcts_initialization():
    agent = MCTSAgent(model=None, num_simulations=10)
    assert agent.num_simulations == 10

def test_mcts_search_mock_model(mock_state):
    # Mock Model Output
    # Not used yet in current MCTS implementation (it handles None)
    agent = MCTSAgent(model=None, num_simulations=5)
    
    action_type, coord = agent.search(mock_state)
    
    assert action_type in ['ADD', 'REMOVE']
    assert isinstance(coord, tuple)
    assert len(coord) == 3

def test_mcts_node_expansion(mock_state):
    agent = MCTSAgent(model=None)
    root = MCTSNode(mock_state)
    
    value = agent._expand(root)
    
    assert len(root.children) > 0
    assert value == 0.5 # Default mock value

def test_mcts_backprop():
    agent = MCTSAgent(model=None)
    root = MCTSNode(None)
    child = MCTSNode(None, parent=root)
    
    agent._backpropagate(child, 1.0)
    
    assert child.visit_count == 1
    assert child.value_sum == 1.0
    assert root.visit_count == 1
    assert root.value_sum == 1.0
