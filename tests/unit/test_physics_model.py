"""
Unit tests for physics_model module.
"""
import pytest
import numpy as np

# Skip if dolfinx is not available (e.g. in CI)
dolfinx = pytest.importorskip("dolfinx")

from alphabuilder.src.core.physics_model import (
    PhysicalProperties,
    initialize_cantilever_context,
    FEMContext
)

class TestPhysicalProperties:
    """Tests for PhysicalProperties dataclass."""
    
    def test_default_values(self):
        """Should have correct default values."""
        props = PhysicalProperties()
        assert props.E == 1.0
        assert props.nu == 0.3
        assert props.rho == 1.0
        assert props.disp_limit == 100.0
        assert props.penalty_epsilon == 0.1
        assert props.penalty_alpha == 10.0

    def test_custom_values(self):
        """Should accept custom values."""
        props = PhysicalProperties(E=2.0, nu=0.4)
        assert props.E == 2.0
        assert props.nu == 0.4


class TestCantileverContext:
    """Tests for initialize_cantilever_context."""
    
    def test_initialize_small_resolution(self):
        """Should initialize context with small resolution."""
        # Use very small resolution for speed and memory
        resolution = (4, 2, 2)
        
        try:
            context = initialize_cantilever_context(resolution=resolution)
        except ImportError:
            pytest.skip("FEniCSx/dolfinx not installed or MPI issue")
        except Exception as e:
            pytest.fail(f"Initialization failed: {e}")
            
        assert isinstance(context, FEMContext)
        assert context.domain.topology.dim == 3
        assert context.V is not None
        assert context.problem is not None
        assert context.material_field is not None
        
        # Check material field size (DG0)
        # Should match number of cells
        # BoxMesh(nx, ny, nz) has nx*ny*nz cells?
        # dolfinx.mesh.create_box creates cells.
        # Check if material_field array size is plausible
        assert context.material_field.x.array.size > 0
