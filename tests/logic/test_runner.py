import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from alphabuilder.src.logic.runner import (
    EpisodeConfig,
    run_episode_v1_1
)
from alphabuilder.src.core.physics_model import SimulationResult

def test_episode_config_defaults():
    """Test EpisodeConfig default values."""
    config = EpisodeConfig()
    assert config.resolution == (64, 32, 32)
    assert config.max_refinement_steps == 100
    assert config.stagnation_threshold == 1e-4
    assert config.stagnation_patience == 20

@patch("alphabuilder.src.logic.runner.solve_topology_3d")
@patch("alphabuilder.src.logic.runner.initialize_cantilever_context")
@patch("alphabuilder.src.logic.runner.MCTSAgent")
def test_run_episode_v1_1(mock_agent_cls, mock_init_ctx, mock_solve, temp_db):
    """Test run_episode_v1_1 execution flow."""
    # Mock Context
    mock_ctx = MagicMock()
    mock_init_ctx.return_value = mock_ctx
    
    # Mock Solver
    mock_solve.return_value = SimulationResult(
        fitness=0.5,
        max_displacement=1.0,
        compliance=10.0,
        valid=True,
        displacement_array=np.zeros(10)
    )
    
    # Mock Agent
    mock_agent = MagicMock()
    mock_agent.search.return_value = ('ADD', (1, 1, 1))
    mock_agent_cls.return_value = mock_agent
    
    config = EpisodeConfig(
        resolution=(10, 10, 10),
        max_refinement_steps=2
    )
    
    run_episode_v1_1(
        db_path=temp_db,
        max_steps=2,
        config=config
    )
    
    # Verify calls
    assert mock_init_ctx.called
    assert mock_agent.search.call_count == 2
    assert mock_solve.call_count == 2
