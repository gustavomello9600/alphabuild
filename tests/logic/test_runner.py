import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from alphabuilder.src.logic.runner import (
    EpisodeConfig,
    find_boundary_cells,
    is_connected_bfs,
    phase1_growth,
    phase2_refinement,
    run_episode
)
from alphabuilder.src.core.physics_model import SimulationResult

def test_episode_config_defaults():
    """Test EpisodeConfig default values."""
    config = EpisodeConfig()
    assert config.resolution == (16, 32)
    assert config.max_refinement_steps == 50
    assert config.stagnation_threshold == 1e-4
    assert config.stagnation_patience == 20

def test_find_boundary_cells():
    """Test boundary cell detection."""
    topology = np.zeros((5, 5), dtype=np.int32)
    topology[2, 2] = 1 # Center pixel
    
    boundary = find_boundary_cells(topology)
    assert (2, 2) in boundary
    
    # Fill 3x3 block
    topology[1:4, 1:4] = 1
    boundary = find_boundary_cells(topology)
    
    # Center (2,2) is now internal, not boundary
    assert (2, 2) not in boundary
    # Corners (1,1) etc are boundary
    assert (1, 1) in boundary

def test_is_connected_bfs():
    """Test connectivity check."""
    topology = np.zeros((5, 5), dtype=np.int32)
    
    # Disconnected
    topology[2, 0] = 1
    topology[2, 4] = 1
    assert not is_connected_bfs(topology)
    
    # Connected line
    topology[2, :] = 1
    assert is_connected_bfs(topology)

def test_phase1_growth():
    """Test Phase 1 growth."""
    config = EpisodeConfig(resolution=(10, 20), growth_strategy="straight")
    initial = np.zeros((20, 10), dtype=np.int32) # (nx, ny)
    
    result = phase1_growth(initial, config)
    
    assert result.shape == (20, 10)
    assert np.sum(result) > 0
    # Check left edge support
    assert np.all(result[0, :] == 1)

@patch("alphabuilder.src.logic.runner.solve_topology")
def test_phase2_refinement(mock_solve, mock_fem_context, sample_props, temp_db):
    """Test Phase 2 refinement loop."""
    # Mock solver return
    mock_solve.return_value = SimulationResult(
        fitness=0.5,
        max_displacement=1.0,
        compliance=10.0,
        valid=True
    )
    
    config = EpisodeConfig(
        resolution=(16, 32),
        max_refinement_steps=5,
        stagnation_patience=10
    )
    
    topology = np.ones((16, 32), dtype=np.int32)
    rng = np.random.default_rng(42)
    
    final_topo, history = phase2_refinement(
        topology,
        mock_fem_context,
        sample_props,
        config,
        temp_db,
        "ep1",
        rng
    )
    
    assert len(history) == 5
    assert mock_solve.call_count == 5
