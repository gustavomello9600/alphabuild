"""
Unit tests for data augmentation module.

Performance target: < 5ms per sample (200+ samples/second)
"""
import pytest
import numpy as np
import time

from alphabuilder.src.neural.augmentation import (
    rotate_90_z,
    flip_y,
    erosion_attack,
    load_multiplier,
    sabotage,
    saboteur,
    EROSION_KERNEL
)


@pytest.fixture
def sample_state():
    """Create a sample 7-channel state tensor."""
    state = np.zeros((7, 64, 32, 8), dtype=np.float32)
    # Add some density
    state[0, 10:50, 10:20, 2:6] = 1.0
    # Add support masks at X=0
    state[1, 0, :, :] = 1.0
    state[2, 0, :, :] = 1.0
    state[3, 0, :, :] = 1.0
    # Add force at X=63
    state[5, 63, 16, 4] = -1.0
    return state


@pytest.fixture
def sample_policy():
    """Create a sample 2-channel policy tensor."""
    policy = np.zeros((2, 64, 32, 8), dtype=np.float32)
    # Some add targets
    policy[0, 20:30, 10:15, 2:5] = 1.0
    return policy


class TestRotate90Z:
    """Tests for 90° rotation around Z axis."""
    
    def test_output_shapes(self, sample_state, sample_policy):
        """Rotation should preserve number of channels."""
        state_rot, policy_rot = rotate_90_z(sample_state, sample_policy)
        
        assert state_rot.shape[0] == 7  # 7 channels
        assert policy_rot.shape[0] == 2  # 2 channels
    
    def test_spatial_rotation(self, sample_state, sample_policy):
        """Spatial dimensions should be rotated."""
        state_rot, policy_rot = rotate_90_z(sample_state, sample_policy)
        
        # After rotation, D and H swap (64, 32) -> (32, 64)
        assert state_rot.shape[1:] == (32, 64, 8)
        assert policy_rot.shape[1:] == (32, 64, 8)
    
    def test_force_vector_rotation(self, sample_state, sample_policy):
        """Force vectors should be rotated correctly."""
        # Set known force
        state = sample_state.copy()
        state[4, :, :, :] = 0  # Clear Fx
        state[5, :, :, :] = 0  # Clear Fy
        state[4, 30, 16, 4] = 1.0  # Fx = 1
        state[5, 30, 16, 4] = 0.0  # Fy = 0
        
        state_rot, _ = rotate_90_z(state, sample_policy)
        
        # After 90° CCW: (Fx, Fy) -> (Fy, -Fx)
        # New Fx should be old Fy (0), new Fy should be -old Fx (-1)
        # Note: spatial position also rotates
        assert state_rot[5].min() == -1.0  # Fy has -1
    
    def test_mask_rotation(self, sample_state, sample_policy):
        """Mask channels should be rotated (without sign change)."""
        state_rot, _ = rotate_90_z(sample_state, sample_policy)
        
        # Masks should be swapped but not negated
        # Original: mask_x at X=0 -> after rotation, should be at different position
        assert state_rot[1].sum() > 0  # Still has values
        assert state_rot[2].sum() > 0
    
    def test_returns_copies(self, sample_state, sample_policy):
        """Should return copies, not views."""
        state_rot, policy_rot = rotate_90_z(sample_state, sample_policy)
        
        state_rot[0, 0, 0, 0] = 999
        assert sample_state[0, 0, 0, 0] != 999


class TestFlipY:
    """Tests for Y-axis flip."""
    
    def test_preserves_shape(self, sample_state, sample_policy):
        """Flip should preserve shape."""
        state_flip, policy_flip = flip_y(sample_state, sample_policy)
        
        assert state_flip.shape == sample_state.shape
        assert policy_flip.shape == sample_policy.shape
    
    def test_flips_spatial_y(self, sample_state, sample_policy):
        """Should flip along Y (H) dimension."""
        # Set marker at Y=0
        state = sample_state.copy()
        state[0, 30, 0, 4] = 0.5
        state[0, 30, 31, 4] = 0.0
        
        state_flip, _ = flip_y(state, sample_policy)
        
        # Marker should now be at Y=31
        assert state_flip[0, 30, 31, 4] == 0.5
    
    def test_inverts_fy(self, sample_state, sample_policy):
        """Fy should be inverted."""
        state = sample_state.copy()
        state[5, 63, 16, 4] = -1.0  # Original Fy = -1
        
        state_flip, _ = flip_y(state, sample_policy)
        
        # After flip, Fy should be +1 at flipped position
        assert state_flip[5].max() == 1.0
    
    def test_fx_fz_unchanged(self, sample_state, sample_policy):
        """Fx and Fz should not be inverted."""
        state = sample_state.copy()
        state[4, 30, 16, 4] = 1.0  # Fx
        state[6, 30, 16, 4] = 0.5  # Fz
        
        state_flip, _ = flip_y(state, sample_policy)
        
        # Fx and Fz should have same sign (just moved position)
        assert state_flip[4].max() == 1.0
        assert state_flip[6].max() == 0.5


class TestErosionAttack:
    """Tests for erosion attack (negative sampling)."""
    
    def test_returns_negative_value(self, sample_state, sample_policy):
        """Should return value = -1.0."""
        _, _, value = erosion_attack(sample_state, sample_policy, 0.5)
        
        assert value == -1.0
    
    def test_reduces_density(self, sample_state, sample_policy):
        """Erosion should reduce material volume."""
        original_volume = sample_state[0].sum()
        
        state_eroded, _, _ = erosion_attack(sample_state, sample_policy, 0.5)
        
        eroded_volume = state_eroded[0].sum()
        assert eroded_volume < original_volume
    
    def test_policy_indicates_repair(self, sample_state, sample_policy):
        """Policy should indicate where to add material (repair)."""
        state_eroded, policy_repair, _ = erosion_attack(sample_state, sample_policy, 0.5)
        
        # Repair policy add channel should have values where material was removed
        removed = sample_state[0] - state_eroded[0]
        np.testing.assert_array_equal(policy_repair[0], np.maximum(0, removed))
    
    def test_remove_channel_zero(self, sample_state, sample_policy):
        """Remove channel should be zero (nothing to remove after erosion)."""
        _, policy_repair, _ = erosion_attack(sample_state, sample_policy, 0.5)
        
        assert policy_repair[1].sum() == 0


class TestLoadMultiplier:
    """Tests for load multiplier (stress test)."""
    
    def test_multiplies_forces(self, sample_state, sample_policy):
        """Forces should be multiplied by k."""
        k = 3.0
        original_fy = sample_state[5, 63, 16, 4]
        
        state_stressed, _, _ = load_multiplier(sample_state, sample_policy, 0.5, k=k)
        
        assert state_stressed[5, 63, 16, 4] == original_fy * k
    
    def test_returns_negative_value(self, sample_state, sample_policy):
        """Should return value = -0.8."""
        _, _, value = load_multiplier(sample_state, sample_policy, 0.5)
        
        assert value == -0.8
    
    def test_policy_suggests_reinforcement(self, sample_state, sample_policy):
        """Policy should suggest adding material near load."""
        _, policy, _ = load_multiplier(sample_state, sample_policy, 0.5)
        
        # Add channel should have values (reinforcement suggestions)
        assert policy[0].sum() > 0


class TestSabotage:
    """Tests for sabotage (remove near support)."""
    
    @pytest.fixture
    def state_with_support_material(self):
        """Create state with material near support for sabotage testing."""
        state = np.zeros((7, 64, 32, 8), dtype=np.float32)
        # Add material near support (X=0 to X=5)
        state[0, 0:5, 10:20, 2:6] = 1.0
        # Add support masks
        state[1, 0, :, :] = 1.0
        state[2, 0, :, :] = 1.0
        state[3, 0, :, :] = 1.0
        return state
    
    def test_returns_negative_value(self, state_with_support_material, sample_policy):
        """Should return value = -1.0."""
        _, _, value = sabotage(state_with_support_material, sample_policy, 0.5)
        
        assert value == -1.0
    
    def test_removes_near_support(self, state_with_support_material, sample_policy):
        """Should remove material near X=0."""
        state = state_with_support_material
        
        state_sab, _, _ = sabotage(state, sample_policy, 0.5)
        
        # Some material should be removed near support
        original_near_support = state[0, 0:5, :, :].sum()
        sabotaged_near_support = state_sab[0, 0:5, :, :].sum()
        assert sabotaged_near_support < original_near_support
    
    def test_policy_indicates_repair(self, state_with_support_material, sample_policy):
        """Policy should indicate where to repair."""
        state_sab, policy_repair, _ = sabotage(state_with_support_material, sample_policy, 0.5)
        
        # Add channel should indicate removed voxels
        assert policy_repair[0].sum() > 0


class TestSaboteur:
    """Tests for saboteur (random cube removal)."""
    
    def test_returns_negative_value(self, sample_state, sample_policy):
        """Should return value = -1.0."""
        _, _, value = saboteur(sample_state, sample_policy, 0.5)
        
        assert value == -1.0
    
    def test_removes_cube(self, sample_state, sample_policy):
        """Should remove a cube of material."""
        original_volume = sample_state[0].sum()
        
        state_sab, _, _ = saboteur(sample_state, sample_policy, 0.5, cube_size=3)
        
        sabotaged_volume = state_sab[0].sum()
        assert sabotaged_volume < original_volume
    
    def test_handles_empty_state(self):
        """Should handle state with no material gracefully."""
        empty_state = np.zeros((7, 64, 32, 8), dtype=np.float32)
        empty_policy = np.zeros((2, 64, 32, 8), dtype=np.float32)
        
        state_out, policy_out, value_out = saboteur(empty_state, empty_policy, 0.5)
        
        # Should return unchanged
        np.testing.assert_array_equal(state_out, empty_state)
        np.testing.assert_array_equal(policy_out, empty_policy)
        assert value_out == 0.5


class TestPerformance:
    """Performance tests - must be fast for training."""
    
    @pytest.fixture
    def large_sample(self):
        """Create larger sample for performance testing."""
        state = np.random.rand(7, 64, 32, 8).astype(np.float32)
        state[0] = (state[0] > 0.5).astype(np.float32)  # Binary density
        policy = np.random.rand(2, 64, 32, 8).astype(np.float32)
        return state, policy
    
    def test_rotate_performance(self, large_sample):
        """Rotation should be < 5ms."""
        state, policy = large_sample
        
        start = time.perf_counter()
        for _ in range(100):
            rotate_90_z(state, policy)
        elapsed = (time.perf_counter() - start) / 100 * 1000
        
        assert elapsed < 5, f"Rotation took {elapsed:.2f}ms, should be < 5ms"
    
    def test_flip_performance(self, large_sample):
        """Flip should be < 5ms."""
        state, policy = large_sample
        
        start = time.perf_counter()
        for _ in range(100):
            flip_y(state, policy)
        elapsed = (time.perf_counter() - start) / 100 * 1000
        
        assert elapsed < 5, f"Flip took {elapsed:.2f}ms, should be < 5ms"
    
    def test_erosion_performance(self, large_sample):
        """Erosion should be < 10ms (morphological ops are slower)."""
        state, policy = large_sample
        
        start = time.perf_counter()
        for _ in range(50):
            erosion_attack(state, policy, 0.5)
        elapsed = (time.perf_counter() - start) / 50 * 1000
        
        assert elapsed < 10, f"Erosion took {elapsed:.2f}ms, should be < 10ms"

