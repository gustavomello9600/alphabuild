"""
Pytest configuration and shared fixtures for AlphaBuilder v3.1 tests.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test database path
TEST_DB_PATH = Path(__file__).parent / "data" / "episodios_de_testes_de_integracao.db"


@pytest.fixture(scope="session")
def test_db_path():
    """Return path to test database."""
    return TEST_DB_PATH


@pytest.fixture(scope="session")
def resolution():
    """Standard test resolution."""
    return (64, 32, 8)


@pytest.fixture(scope="session")
def device():
    """Get available device for neural network."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

