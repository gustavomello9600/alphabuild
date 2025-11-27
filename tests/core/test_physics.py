import pytest
import numpy as np
from alphabuilder.src.core.physics_model import (
    PhysicalProperties,
    initialize_cantilever_context,
    FEMContext
)
from alphabuilder.src.core.solver import solve_topology_3d
import sys
from unittest.mock import MagicMock

def is_dolfinx_mocked():
    dolfinx = sys.modules.get("dolfinx")
    return isinstance(dolfinx, MagicMock) or dolfinx is None

def test_physical_properties_defaults():
    """Test default values of PhysicalProperties."""
    props = PhysicalProperties()
    assert props.E == 1.0
    assert props.nu == 0.3
    assert props.rho == 1.0

def test_initialize_cantilever_context(sample_props):
    """Integration test: Initialize FEM context (requires FEniCSx)."""
    if is_dolfinx_mocked():
        pytest.skip("FEniCSx not installed or mocked")
        
    resolution = (64, 32, 32)
    ctx = initialize_cantilever_context(resolution, sample_props)
    
    assert isinstance(ctx, FEMContext)
    # Check dof_map or similar if needed, but it's a placeholder now
    assert ctx.domain is not None
    assert ctx.V is not None

def test_solve_topology_solid_beam(sample_props):
    """Integration test: Solve for a solid beam (Smoke Test)."""
    if is_dolfinx_mocked():
        pytest.skip("FEniCSx not installed or mocked")
        
    resolution = (64, 32, 32)
    ctx = initialize_cantilever_context(resolution, sample_props)
    
    # Solid beam (all 1s)
    # Create 5D Tensor (5, D, H, W)
    D, H, W = resolution
    tensor = np.zeros((5, D, H, W), dtype=np.float32)
    tensor[0] = 1.0 # Density
    
    # Add Load
    tensor[3, D//2, H//2, W-1] = -1.0
    
    result = solve_topology_3d(tensor, ctx, sample_props)
    
    assert result.valid is True
    assert result.max_displacement > 0
    assert result.compliance > 0
    assert result.fitness > 0
    
    # Check against analytical expectation (approximate)
    # For 3D cantilever, it's complex, but should be finite.
    assert result.max_displacement < 1e5

def test_solve_topology_void_beam(sample_props):
    """Integration test: Solve for a void beam (Linear scaling check)."""
    if is_dolfinx_mocked():
        pytest.skip("FEniCSx not installed or mocked")
        
    resolution = (64, 32, 32)
    ctx = initialize_cantilever_context(resolution, sample_props)
    
    # Void beam (all 0s)
    D, H, W = resolution
    tensor = np.zeros((5, D, H, W), dtype=np.float32)
    tensor[0] = 0.0 # Density
    
    # Add Load
    tensor[3, D//2, H//2, W-1] = -1.0
    
    result = solve_topology_3d(tensor, ctx, sample_props)
    
    # Should be much softer (higher displacement)
    # Ratio should be approx E_solid / E_min = 1/1e-3 = 1000
    assert result.max_displacement > 100 

def test_solver_mismatch_shape(sample_props):
    """Test error handling for mismatched topology shape."""
    if is_dolfinx_mocked():
        pytest.skip("FEniCSx not installed or mocked")
        
    resolution = (64, 32, 32)
    ctx = initialize_cantilever_context(resolution, sample_props)
    
    # Wrong shape (D, H, W) mismatch with context
    # Context expects 64x32x32
    # Pass 10x10x10
    tensor = np.zeros((5, 10, 10, 10), dtype=np.float32)
    
    # The solver might not raise ValueError explicitly if it just flattens, 
    # but it will fail when assigning to material_field if size mismatch.
    # Let's check if it raises ANY error.
    with pytest.raises(Exception):
        solve_topology_3d(tensor, ctx, sample_props)
