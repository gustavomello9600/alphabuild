import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock
import sys
import os
from unittest.mock import MagicMock

# Try to import dolfinx, if failing OR if MOCK_DOLFINX is set, mock it
mock_dolfinx = os.environ.get("MOCK_DOLFINX", "0") == "1"

if mock_dolfinx:
    mock_dolfinx = MagicMock()
    sys.modules["dolfinx"] = mock_dolfinx
    mock_dolfinx.default_scalar_type = np.float64
    
    mock_mesh = MagicMock()
    mock_mesh.CellType.quadrilateral = "quadrilateral"
    sys.modules["dolfinx.mesh"] = mock_mesh
    mock_dolfinx.mesh = mock_mesh
    
    # Configure FunctionSpace mock
    mock_fs = MagicMock()
    dummy_coords = np.zeros((512, 3))
    mock_fs.tabulate_dof_coordinates.return_value = dummy_coords
    mock_fs.dofmap.index_map.size_local = 512
    
    mock_fem = MagicMock()
    mock_fem.functionspace.return_value = mock_fs
    sys.modules["dolfinx.fem"] = mock_fem
    mock_dolfinx.fem = mock_fem
    
    mock_fem_petsc = MagicMock()
    
    # Configure LinearProblem
    mock_problem = MagicMock()
    # Mock solution vector u
    mock_u = MagicMock()
    # Array must be large enough for reshape if needed, or just flat
    # Physics model might reshape it?
    # "Calcule D_max = max ||u||_L2" usually implies reshaping to (N, 2) or similar?
    # Or just norm of the vector?
    # If it's a flat vector of size 2*N, norm is scalar.
    # But if code does .reshape(-1, 2), we need correct size.
    # Let's assume 512 nodes * 2 dofs = 1024
    mock_u.x.array = np.zeros(1024) 
    mock_problem.u = mock_u
    
    # Mock RHS vector b
    mock_b = MagicMock()
    mock_b.dot.return_value = 1.0
    mock_b.norm.return_value = 1.0
    mock_problem.b = mock_b
    
    mock_fem_petsc.LinearProblem.return_value = mock_problem
    sys.modules["dolfinx.fem.petsc"] = mock_fem_petsc
    mock_fem.petsc = mock_fem_petsc
    
    sys.modules["mpi4py"] = MagicMock()
    sys.modules["petsc4py"] = MagicMock()
    sys.modules["ufl"] = MagicMock()
    sys.modules["ufl"] = MagicMock()
else:
    try:
        import dolfinx
    except ImportError:
        # Mock dolfinx components if not installed
        sys.modules["dolfinx"] = MagicMock()
        sys.modules["dolfinx.mesh"] = MagicMock()
        sys.modules["dolfinx.fem"] = MagicMock()
        sys.modules["dolfinx.fem.petsc"] = MagicMock()
        sys.modules["mpi4py"] = MagicMock()
        sys.modules["petsc4py"] = MagicMock()
        sys.modules["ufl"] = MagicMock()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphabuilder.src.core.physics_model import PhysicalProperties, FEMContext, SimulationResult
from alphabuilder.src.logic.storage import initialize_database

@pytest.fixture
def mock_fem_context():
    """
    Creates a mock FEMContext to avoid FEniCSx initialization overhead in logic tests.
    """
    mock_ctx = MagicMock(spec=FEMContext)
    # Mock attributes that might be accessed
    mock_ctx.dof_map = np.zeros((32, 16), dtype=np.int32)
    return mock_ctx

@pytest.fixture
def temp_db():
    """
    Creates a temporary SQLite database for testing storage.
    """
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_path = Path(path)
    initialize_database(db_path)
    yield db_path
    # Cleanup
    if db_path.exists():
        os.unlink(db_path)

@pytest.fixture
def sample_topology():
    """
    Returns a simple 16x32 topology matrix (all zeros).
    """
    return np.zeros((32, 16), dtype=np.int32)

@pytest.fixture
def sample_props():
    """
    Returns standard PhysicalProperties.
    """
    return PhysicalProperties()
