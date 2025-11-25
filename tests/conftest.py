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
    sys.modules["dolfinx"] = MagicMock()
    sys.modules["dolfinx.mesh"] = MagicMock()
    sys.modules["dolfinx.fem"] = MagicMock()
    sys.modules["dolfinx.fem.petsc"] = MagicMock()
    sys.modules["mpi4py"] = MagicMock()
    sys.modules["petsc4py"] = MagicMock()
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
