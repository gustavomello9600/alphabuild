import pytest
import numpy as np
from alphabuilder.src.core.physics_model import (
    PhysicalProperties,
    initialize_cantilever_context,
    FEMContext
)
from alphabuilder.src.core.solver import solve_topology

def test_physical_properties_defaults():
    """Test default values of PhysicalProperties."""
    props = PhysicalProperties()
    assert props.E_solid == 1.0
    assert props.E_void == 1e-6
    assert props.nu == 0.3
    assert props.penalty_alpha == 0.5

def test_initialize_cantilever_context(sample_props):
    """Integration test: Initialize FEM context (requires FEniCSx)."""
    try:
        import dolfinx
    except ImportError:
        pytest.skip("FEniCSx not installed")
        
    resolution = (32, 16)
    ctx = initialize_cantilever_context(resolution, sample_props)
    
    assert isinstance(ctx, FEMContext)
    assert ctx.dof_map.shape == (16, 32) # (ny, nx)
    assert ctx.mesh is not None
    assert ctx.V is not None
    assert ctx.D is not None

def test_solve_topology_solid_beam(sample_props):
    """Integration test: Solve for a solid beam (Smoke Test)."""
    try:
        import dolfinx
    except ImportError:
        pytest.skip("FEniCSx not installed")
        
    resolution = (32, 16)
    ctx = initialize_cantilever_context(resolution, sample_props)
    
    # Solid beam (all 1s)
    topology = np.ones((16, 32), dtype=np.int32)
    
    result = solve_topology(topology, ctx, sample_props)
    
    assert result.valid is True
    assert result.max_displacement > 0
    assert result.compliance > 0
    assert result.fitness > 0
    
    # Check against analytical expectation (approximate)
    # Analytical disp ~ 3200.0, FEM ~ 3583.5
    assert 3000.0 < result.max_displacement < 4000.0

def test_solve_topology_void_beam(sample_props):
    """Integration test: Solve for a void beam (Linear scaling check)."""
    try:
        import dolfinx
    except ImportError:
        pytest.skip("FEniCSx not installed")
        
    resolution = (32, 16)
    ctx = initialize_cantilever_context(resolution, sample_props)
    
    # Void beam (all 0s)
    topology = np.zeros((16, 32), dtype=np.int32)
    
    result = solve_topology(topology, ctx, sample_props)
    
    # Should be much softer (higher displacement)
    # Ratio should be approx E_solid / E_void = 1e6
    # We can't easily check exact ratio without running solid first, 
    # but we can check magnitude.
    assert result.max_displacement > 1e8 

def test_solver_mismatch_shape(sample_props):
    """Test error handling for mismatched topology shape."""
    try:
        import dolfinx
    except ImportError:
        pytest.skip("FEniCSx not installed")
        
    resolution = (32, 16)
    ctx = initialize_cantilever_context(resolution, sample_props)
    
    # Wrong shape
    topology = np.ones((10, 10), dtype=np.int32)
    
    with pytest.raises(ValueError, match="Topology shape"):
        solve_topology(topology, ctx, sample_props)
