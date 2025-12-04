import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys

# Mock dolfinx and mpi4py in sys.modules BEFORE importing modules that depend on them
# This ensures tests run even if these heavy dependencies are missing (e.g. in CI)
mock_dolfinx = MagicMock()
mock_mpi4py = MagicMock()
mock_basix = MagicMock()

sys.modules['dolfinx'] = mock_dolfinx
sys.modules['dolfinx.mesh'] = mock_dolfinx.mesh
sys.modules['dolfinx.fem'] = mock_dolfinx.fem
sys.modules['mpi4py'] = mock_mpi4py
sys.modules['basix'] = mock_basix
sys.modules['basix.ufl'] = mock_basix.ufl

from alphabuilder.src.logic.harvest.config import SIMPConfig, LOG_SQUASH_ALPHA
from alphabuilder.src.logic.harvest.generators import (
    generate_bezier_structure,
    generate_seeded_cantilever,
    generate_random_load_config,
    generate_bc_masks,
    quadratic_bezier
)
from alphabuilder.src.logic.harvest.processing import (
    compute_normalized_value,
    check_connectivity,
    generate_phase1_slices,
    generate_refinement_targets,
    compute_boundary_mask,
    compute_filled_mask
)
from alphabuilder.src.logic.storage import Phase

# --- Tests for Config ---
def test_simp_config_defaults():
    config = SIMPConfig()
    assert config.vol_frac == 0.15
    assert config.max_iter == 120
    assert config.r_min == 1.5

# --- Tests for Generators ---
def test_quadratic_bezier():
    p0 = np.array([0, 0, 0])
    p1 = np.array([1, 1, 1])
    p2 = np.array([2, 0, 0])
    
    # t=0 -> p0
    assert np.allclose(quadratic_bezier(p0, p1, p2, 0.0), p0)
    # t=1 -> p2
    assert np.allclose(quadratic_bezier(p0, p1, p2, 1.0), p2)
    # t=0.5 -> (0.25*p0 + 0.5*p1 + 0.25*p2) = (0.5, 0.5, 0.5) + (0.5, 0, 0) = (1, 0.5, 0.5)
    # Wait: (1-0.5)^2 * p0 + 2*(1-0.5)*0.5 * p1 + 0.5^2 * p2
    # = 0.25 * [0,0,0] + 0.5 * [1,1,1] + 0.25 * [2,0,0]
    # = [0,0,0] + [0.5, 0.5, 0.5] + [0.5, 0, 0] = [1.0, 0.5, 0.5]
    expected = np.array([1.0, 0.5, 0.5])
    assert np.allclose(quadratic_bezier(p0, p1, p2, 0.5), expected)

def test_generate_random_load_config():
    res = (64, 32, 32)
    config = generate_random_load_config(res)
    assert config['x'] == 63
    assert 0 <= config['y'] < 32
    assert 0 <= config['z_start'] < config['z_end'] <= 32
    assert config['bc_type'] in ['FULL_CLAMP', 'RAIL_XY']

def test_generate_bc_masks():
    res = (10, 10, 10)
    
    # FULL_CLAMP
    mx, my, mz = generate_bc_masks(res, 'FULL_CLAMP')
    assert mx.shape == res
    assert mx[0, 5, 5] == 1.0
    assert my[0, 5, 5] == 1.0
    assert mz[0, 5, 5] == 1.0
    assert mx[1, 5, 5] == 0.0 # Only X=0 is fixed
    
    # RAIL_XY
    mx, my, mz = generate_bc_masks(res, 'RAIL_XY')
    assert mx[0, 5, 5] == 1.0
    assert my[0, 5, 5] == 1.0
    assert mz[0, 5, 5] == 0.0 # Z is free

def test_generate_bezier_structure():
    res = (32, 16, 16)
    load_config = {'x': 31, 'y': 8, 'z_start': 7, 'z_end': 9}
    grid = generate_bezier_structure(res, load_config)
    
    assert grid.shape == res
    assert grid.dtype == np.float32
    # Check if wall is connected (X=0)
    assert np.any(grid[0, :, :] > 0.5)
    # Check if load is connected (X=31)
    assert np.any(grid[31, :, :] > 0.5)

def test_bezier_structure_3_curves_connectivity():
    """
    Regression test: Verify that when 3 Bezier curves are generated,
    ALL of them connect the support (X=0) to the load region.
    
    This test sets a fixed seed that produces 3 curves and verifies
    the structure is fully connected.
    """
    import random
    
    res = (64, 32, 8)
    load_config = {'x': 63, 'y': 16, 'z_start': 3, 'z_end': 5}
    
    # Run multiple times with different seeds to stress-test
    connection_failures = 0
    seeds_tested = []
    
    for seed in range(100):
        random.seed(seed)
        np.random.seed(seed)
        
        # Force 3 curves by mocking or just checking after
        grid = generate_bezier_structure(res, load_config)
        
        # Check connectivity using the same logic as the harvest script
        is_conn, _ = check_connectivity(grid, threshold=0.5, load_cfg=load_config)
        
        if not is_conn:
            # Try lower threshold
            is_conn_low, _ = check_connectivity(grid, threshold=0.1, load_cfg=load_config)
            if not is_conn_low:
                connection_failures += 1
                seeds_tested.append(seed)
    
    # Assert that all structures are connected
    assert connection_failures == 0, f"Disconnected structures found at seeds: {seeds_tested}"

def test_generate_seeded_cantilever():
    res = (32, 16, 16)
    load_config = {'x': 31, 'y': 8, 'z_start': 7, 'z_end': 9}
    grid = generate_seeded_cantilever(res, load_config)
    
    assert grid.shape == res
    # Should have gray background
    assert np.isclose(grid[0, 0, 0], 0.35)
    # Should have solid bar
    assert np.max(grid) == 1.0

# --- Tests for Processing ---

def test_compute_normalized_value():
    # High compliance (bad) -> close to -1
    val_bad = compute_normalized_value(1000.0, 0.5)
    # Low compliance (good) -> close to 1
    val_good = compute_normalized_value(10.0, 0.1)
    
    assert val_bad < val_good
    assert -1.0 <= val_bad <= 1.0
    assert -1.0 <= val_good <= 1.0

def test_check_connectivity():
    # Create a disconnected grid
    grid = np.zeros((10, 10, 10))
    # Support at X=0
    grid[0, 5, 5] = 1.0
    # Load at X=9
    grid[9, 5, 5] = 1.0
    
    load_config = {'x': 9, 'y': 5, 'z_start': 4, 'z_end': 6}
    
    is_conn, _ = check_connectivity(grid, 0.5, load_config)
    assert not is_conn
    
    # Connect them
    grid[:, 5, 5] = 1.0
    is_conn, _ = check_connectivity(grid, 0.5, load_config)
    assert is_conn

def test_compute_boundary_mask():
    # 3x3x3 grid
    density = np.zeros((3, 3, 3))
    # Center is solid
    density[1, 1, 1] = 1.0
    
    # Boundary mask should be 1 at neighbors of center
    mask = compute_boundary_mask(density)
    
    # Center itself should be 0 (it's filled, not boundary)
    assert mask[1, 1, 1] == 0.0
    
    # Neighbor (1, 1, 2) should be 1
    assert mask[1, 1, 2] == 1.0
    
    # Far corner (0, 0, 0) should be 0 (not neighbor)
    assert mask[0, 0, 0] == 0.0

def test_compute_filled_mask():
    density = np.array([0.0, 0.2, 0.8, 1.0])
    mask = compute_filled_mask(density)
    # Assuming threshold 0.1
    expected = np.array([0.0, 1.0, 1.0, 1.0])
    assert np.allclose(mask, expected)

def test_generate_refinement_targets():
    # Current: Center solid
    curr = np.zeros((3, 3, 3))
    curr[1, 1, 1] = 1.0
    
    # Next: Center solid + Neighbor solid (Growth)
    next_step = curr.copy()
    next_step[1, 1, 2] = 1.0
    
    add, remove = generate_refinement_targets(curr, next_step)
    
    # Should ADD at (1, 1, 2)
    assert add[1, 1, 2] == 1.0
    # Should NOT REMOVE anywhere
    assert np.all(remove == 0.0)
    
    # Case: Shrinkage
    # Current: Center + Neighbor
    curr2 = next_step.copy()
    # Next: Only Center
    next2 = np.zeros((3, 3, 3))
    next2[1, 1, 1] = 1.0
    
    add, remove = generate_refinement_targets(curr2, next2)
    
    # Should REMOVE at (1, 1, 2)
    assert remove[1, 1, 2] == 1.0
    # Should NOT ADD
    assert np.all(add == 0.0)

def test_generate_phase1_slices():
    # Simple bar 5x1x1
    final_mask = np.zeros((5, 3, 3))
    final_mask[:, 1, 1] = 1.0
    
    # Should generate slices
    records = generate_phase1_slices(final_mask, target_value=0.5, num_steps=5)
    
    assert len(records) == 5
    assert records[0]['phase'] == Phase.GROWTH
    # First slice should be near X=0
    assert records[0]['input_state'][0, 1, 1] == 1.0
    assert records[0]['input_state'][4, 1, 1] == 0.0
    
    # Last slice should be full
    assert records[-1]['input_state'][4, 1, 1] == 1.0

# --- Tests for Optimization (Mocked) ---

@patch('alphabuilder.src.logic.harvest.optimization.topopt')
@patch('alphabuilder.src.logic.harvest.optimization.create_box')
@patch('alphabuilder.src.logic.harvest.optimization.functionspace')
@patch('alphabuilder.src.logic.harvest.optimization.MPI')
def test_run_fenitop_optimization(mock_mpi, mock_fs, mock_box, mock_topopt):
    from alphabuilder.src.logic.harvest.optimization import run_fenitop_optimization
    from alphabuilder.src.core.physics_model import PhysicalProperties
    
    # Setup mocks
    mock_comm = MagicMock()
    mock_comm.rank = 0
    mock_mpi.COMM_WORLD = mock_comm
    
    # Mock topopt callback execution
    def side_effect(fem, opt, initial_density, callback):
        # Simulate one step
        data = {
            'iter': 1,
            'density': np.zeros(8), # Matches 2x2x2 grid (8 nodes)
            'compliance': 100.0,
            'vol_frac': 0.5,
            'beta': 1
        }
        # Call the callback to test record_step
        callback(data)

    mock_topopt.side_effect = side_effect
    
    # Mock functionspace for grid mapper
    mock_V = MagicMock()
    # Return dummy coords for 2x2x2 grid (8 nodes)
    # Order: (0,0,0), (1,0,0), ...
    mock_V.tabulate_dof_coordinates.return_value = np.array([
        [0,0,0], [1,0,0], [0,1,0], [0,0,1],
        [1,1,0], [1,0,1], [0,1,1], [1,1,1]
    ])
    mock_fs.return_value = mock_V
    
    res = (2, 2, 2)
    props = PhysicalProperties()
    simp_config = SIMPConfig(load_config={'x':1, 'y':1, 'z_start':0, 'z_end':2})
    
    history = run_fenitop_optimization(res, props, simp_config)
    
    assert mock_topopt.called
    args, kwargs = mock_topopt.call_args
    
    # args[0] is fem dict
    fem_arg = args[0]
    assert 'mesh' in fem_arg
    assert fem_arg["young's modulus"] == 100
    
    # args[1] is opt dict
    opt_arg = args[1]
    assert opt_arg['max_iter'] == 100 # BEZIER default
    
    # Check if history was populated by callback
    assert len(history) == 1
    assert history[0]['compliance'] == 100.0
