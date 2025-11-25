import pytest
import numpy as np
from alphabuilder.src.logic.pathfinding import (
    create_astar_topology,
    create_random_pattern,
    calculate_structural_metrics
)

def test_create_astar_topology():
    """Test A* path generation."""
    ny, nx = 16, 32
    topology = np.zeros((ny, nx), dtype=np.int32)
    
    starts = [(0, ny // 2)]
    goals = [(nx - 1, ny // 2)]
    
    result = create_astar_topology(topology, starts, goals)
    
    # Check connectivity (start and end should be filled)
    # Start was (0, 8) -> result[8, 0] should be 1
    assert result[ny // 2, 0] == 1
    # Goal was (31, 8) -> result[8, 31] should be 1
    assert result[ny // 2, nx - 1] == 1
    
    # Check that a path exists (sum > 0)
    assert np.sum(result) >= nx 

def test_create_random_pattern():
    """Test random pattern generation."""
    ny, nx = 16, 32
    topology = np.zeros((ny, nx), dtype=np.int32)
    
    result = create_random_pattern(topology, ny, nx, seed=42)
    
    assert result.shape == (ny, nx)
    assert np.sum(result) > 0
    
    # Test reproducibility
    result2 = create_random_pattern(np.zeros((ny, nx), dtype=np.int32), ny, nx, seed=42)
    np.testing.assert_array_equal(result, result2)

def test_calculate_structural_metrics():
    """Test calculation of structural metrics."""
    ny, nx = 10, 10
    topology = np.zeros((ny, nx), dtype=np.int32)
    
    # Fill half
    topology[:, :5] = 1
    
    metrics = calculate_structural_metrics(topology)
    
    assert "volume_fraction" in metrics
    assert metrics["volume_fraction"] == 0.5
    assert "pattern_entropy" in metrics
    assert "connectivity_score" in metrics
