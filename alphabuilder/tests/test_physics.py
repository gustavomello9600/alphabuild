import pytest
import numpy as np
from alphabuilder.src.core.physics_model import initialize_cantilever_context, PhysicalProperties, SimulationResult
from alphabuilder.src.core.solver import solve_topology_3d

def test_initialize_context():
    ctx = initialize_cantilever_context()
    assert ctx is not None
    # Verify domain is a Mesh
    import dolfinx.mesh
    assert isinstance(ctx.domain, dolfinx.mesh.Mesh)

def test_solve_topology_3d():
    # Setup Real Context
    # Use small resolution for speed
    ctx = initialize_cantilever_context(resolution=(16, 8, 8))
    props = PhysicalProperties()
    
    # Setup Input Tensor (5, 16, 8, 8)
    tensor = np.zeros((5, 16, 8, 8), dtype=np.float32)
    tensor[0, :, :, :] = 1.0 # Full density
    tensor[1, :, :, 0] = 1.0 # Support
    tensor[3, 8, 4, 7] = -1.0 # Load
    
    # Run Solver
    result = solve_topology_3d(tensor, ctx, props)
    
    assert isinstance(result, SimulationResult)
    assert result.valid == True
    assert result.compliance > 0.0
