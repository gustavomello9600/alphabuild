import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys

# Mock dolfinx before importing simp_generator if possible, 
# but it's already imported in conftest or other tests.
# We rely on patch.

from alphabuilder.src.logic.simp_generator import (
    run_simp_optimization_3d,
    SIMPConfig
)
from alphabuilder.src.core.physics_model import SimulationResult

@patch("alphabuilder.src.logic.simp_generator.fem")
@patch("alphabuilder.src.logic.simp_generator.dolfinx")
def test_run_simp_optimization_3d(mock_dolfinx, mock_fem, sample_props):
    """Test SIMP optimization loop (Mocked Physics)."""
    # Mock Context
    mock_ctx = MagicMock()
    # Mock material field array
    mock_ctx.material_field.x.array = np.ones(1000)
    mock_ctx.domain.topology.dim = 3
    mock_ctx.domain.topology.index_map.return_value.size_local = 1000
    mock_ctx.domain.geometry.x.max.return_value = np.array([2.0, 1.0, 1.0])
    
    # Mock Solver
    mock_solver = mock_ctx.problem.solver
    mock_solver.solve.return_value = None
    
    # Mock b dot u (Compliance)
    mock_b = mock_ctx.problem.b
    mock_u = mock_ctx.u_sol
    mock_b.dot.return_value = 10.0
    
    # Mock Displacement
    mock_u.x.array.reshape.return_value = np.zeros((1000, 3))
    
    # Mock locate_dofs_geometrical
    mock_fem.locate_dofs_geometrical.return_value = [0, 1, 2]
    
    # Mock Block Size
    mock_ctx.V.dofmap.index_map_bs = 3
    
    # Mock Energy Values (Strain Energy Density)
    # This is accessed via ctx.energy_vals.x.array
    # We need to ensure it returns a numpy array of shape (1000,)
    mock_ctx.energy_vals.x.array = np.ones(1000)
    
    # Initial Density
    initial_density = np.ones(1000)
    
    config = SIMPConfig(
        vol_frac=0.5,
        max_iter=2, # Short run
        penal=3.0
    )
    
    resolution = (10, 10, 10)
    
    history = run_simp_optimization_3d(
        mock_ctx,
        sample_props,
        config,
        resolution,
        initial_density=initial_density
    )
    
    assert len(history) > 0
    
    # Check history structure
    record = history[0]
    assert isinstance(record, dict)
    assert record['step'] > 0
    assert record['density_map'].shape == (10, 10, 10)
    assert 'compliance' in record
    assert 'max_displacement' in record

def test_simp_config_defaults():
    config = SIMPConfig()
    assert config.vol_frac == 0.5
    assert config.max_iter == 50
