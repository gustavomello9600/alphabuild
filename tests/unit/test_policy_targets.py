import pytest
import numpy as np
from alphabuilder.src.logic.harvest.processing import (
    compute_boundary_mask,
    compute_filled_mask,
    generate_refinement_targets
)

def test_compute_boundary_mask():
    # 3x3x3 grid
    density = np.zeros((3, 3, 3), dtype=np.float32)
    # Center voxel is solid
    density[1, 1, 1] = 1.0
    
    # Boundary mask should be the 6 neighbors of the center
    mask = compute_boundary_mask(density)
    
    assert mask[1, 1, 1] == 0.0  # Center is solid, so not boundary
    assert mask[0, 1, 1] == 1.0  # Face neighbor
    assert mask[2, 1, 1] == 1.0
    assert mask[1, 0, 1] == 1.0
    assert mask[1, 2, 1] == 1.0
    assert mask[1, 1, 0] == 1.0
    assert mask[1, 1, 2] == 1.0
    
    # Corners should be 0 (6-connectivity)
    assert mask[0, 0, 0] == 0.0

def test_compute_filled_mask():
    density = np.zeros((3, 3, 3), dtype=np.float32)
    density[1, 1, 1] = 1.0
    
    mask = compute_filled_mask(density)
    
    assert mask[1, 1, 1] == 1.0
    assert mask[0, 0, 0] == 0.0

def test_generate_refinement_targets_add():
    # Current: Center solid
    curr_dens = np.zeros((3, 3, 3), dtype=np.float32)
    curr_dens[1, 1, 1] = 1.0
    
    # Next: Center + Neighbor solid (Added material)
    next_dens = curr_dens.copy()
    next_dens[1, 2, 1] = 0.5  # Added 0.5 to neighbor
    
    # Binary mask of current state
    curr_binary = (curr_dens > 0.5).astype(np.float32)
    
    target_add, target_remove = generate_refinement_targets(curr_dens, next_dens, curr_binary)
    
    # Should have ADD target at [1, 2, 1]
    # Value should be 1.0 due to Max-Scaling (0.5 / 0.5)
    assert target_add[1, 2, 1] == 1.0
    assert target_remove[1, 2, 1] == 0.0
    
    # Boundary check: Try adding far away (should be masked out)
    next_dens_far = curr_dens.copy()
    next_dens_far[0, 0, 0] = 0.5
    
    target_add_far, _ = generate_refinement_targets(curr_dens, next_dens_far, curr_binary)
    assert target_add_far[0, 0, 0] == 0.0  # Masked out because not neighbor

def test_generate_refinement_targets_remove():
    # Current: Center solid
    curr_dens = np.zeros((3, 3, 3), dtype=np.float32)
    curr_dens[1, 1, 1] = 1.0
    
    # Next: Center reduced (Removed material)
    next_dens = curr_dens.copy()
    next_dens[1, 1, 1] = 0.2  # Reduced by 0.8
    
    curr_binary = (curr_dens > 0.5).astype(np.float32)
    
    target_add, target_remove = generate_refinement_targets(curr_dens, next_dens, curr_binary)
    
    # Should have REMOVE target at [1, 1, 1]
    # Value should be 1.0 due to Max-Scaling
    assert target_remove[1, 1, 1] == 1.0
    assert target_add[1, 1, 1] == 0.0

def test_max_scaling():
    curr_dens = np.zeros((3, 3, 3), dtype=np.float32)
    curr_dens[1, 1, 1] = 1.0
    
    next_dens = curr_dens.copy()
    # Add small amount to neighbor 1
    next_dens[1, 2, 1] = 0.1
    # Add large amount to neighbor 2
    next_dens[1, 0, 1] = 0.5
    
    curr_binary = (curr_dens > 0.5).astype(np.float32)
    
    target_add, _ = generate_refinement_targets(curr_dens, next_dens, curr_binary)
    
    # Max change is 0.5.
    # Neighbor 2 (0.5) should be scaled to 1.0
    # Neighbor 1 (0.1) should be scaled to 0.2
    assert np.isclose(target_add[1, 0, 1], 1.0)
    assert np.isclose(target_add[1, 2, 1], 0.2)
