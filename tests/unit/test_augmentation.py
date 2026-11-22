"""
Unit tests for 16x rigid symmetries and physics-consistent negative sampling.
"""
import time
import numpy as np
import pytest

from alphabuilder.src.neural.augmentation import (
    EROSION_KERNEL,
    erosion_attack,
    flip_x,
    flip_y,
    flip_z,
    load_multiplier,
    rotate_90_z,
    sabotage,
    saboteur,
)


@pytest.fixture
def sample_state():
    state = np.zeros((7, 64, 32, 8), dtype=np.float32)
    state[0, 10:50, 10:20, 2:6] = 1.0
    state[1, 0, :, :] = 1.0
    state[2, 0, :, :] = 1.0
    state[3, 0, :, :] = 1.0
    state[5, 63, 16, 4] = -1.0
    return state


@pytest.fixture
def sample_policy():
    policy = np.zeros((2, 64, 32, 8), dtype=np.float32)
    policy[0, 20:30, 10:15, 2:5] = 1.0
    return policy


class TestRotate90Z:
    def test_output_shapes_and_blueprint(self, sample_state, sample_policy):
        bp = np.ones((1, 64, 32, 8), dtype=np.float32)
        state_rot, policy_rot, bp_rot = rotate_90_z(sample_state, sample_policy, bp)
        assert state_rot.shape == (7, 32, 64, 8)
        assert policy_rot.shape == (2, 32, 64, 8)
        assert bp_rot.shape == (1, 32, 64, 8)

    def test_force_vector_rotation_exact_sign(self, sample_state, sample_policy):
        """
        Under np.rot90(..., k=1, axes=(1, 2)): (x, y) -> (H - 1 - y, x)
        so vector components transform as (Fx, Fy) -> (-Fy, +Fx).
        """
        state = np.zeros((7, 8, 6, 2), dtype=np.float32)
        pol = np.zeros((2, 8, 6, 2), dtype=np.float32)
        # Voxel (2, 1, 0) carries Fx=+2.0, Fy=-3.0
        state[4, 2, 1, 0] = 2.0
        state[5, 2, 1, 0] = -3.0

        state_rot, _ = rotate_90_z(state, pol)
        # Rotated coords: (H - 1 - y, x, z) = (6 - 1 - 1, 2, 0) = (4, 2, 0)
        assert state_rot[4, 4, 2, 0] == pytest.approx(3.0)   # new Fx = -Fy = +3.0
        assert state_rot[5, 4, 2, 0] == pytest.approx(2.0)   # new Fy = +Fx = +2.0

    def test_four_rotations_return_identity(self, sample_state, sample_policy):
        s, p = sample_state.copy(), sample_policy.copy()
        for _ in range(4):
            s, p = rotate_90_z(s, p)
        np.testing.assert_allclose(s, sample_state)
        np.testing.assert_allclose(p, sample_policy)

    def test_mask_rotation(self, sample_state, sample_policy):
        """Support on X=0 (fixed X & Z only) swaps Mask_X -> Mask_Y after rot90."""
        state = sample_state.copy()
        state[1, 0, :, :] = 1.0
        state[2, :, :, :] = 0.0  # Asymmetric BC mask between X and Y
        state_rot, _ = rotate_90_z(state, sample_policy)
        assert state_rot[1].sum() == 0.0
        assert state_rot[2].sum() == state[1].sum()


class TestFlips:
    def test_flip_x_inverts_fx_only(self, sample_state, sample_policy):
        state = sample_state.copy()
        state[0, 5, 10, 4] = 0.5
        state[4, 5, 10, 4] = 2.0
        state[5, 5, 10, 4] = -1.0
        state[6, 5, 10, 4] = 0.7
        bp = state[0:1].copy()

        sf, pf, bpf = flip_x(state, sample_policy, bp)
        nx = state.shape[1]
        assert sf[0, nx - 1 - 5, 10, 4] == 0.5
        assert bpf[0, nx - 1 - 5, 10, 4] == 0.5
        assert sf[4, nx - 1 - 5, 10, 4] == -2.0
        assert sf[5, nx - 1 - 5, 10, 4] == -1.0
        assert sf[6, nx - 1 - 5, 10, 4] == 0.7

    def test_flip_y_inverts_fy_only(self, sample_state, sample_policy):
        state = sample_state.copy()
        state[0, 30, 0, 4] = 0.5
        state[4, 30, 0, 4] = 1.0
        state[5, 30, 0, 4] = -1.0
        state[6, 30, 0, 4] = 0.5
        sf, _ = flip_y(state, sample_policy)
        ny = state.shape[2]
        assert sf[0, 30, ny - 1, 4] == 0.5
        assert sf[4, 30, ny - 1, 4] == 1.0
        assert sf[5, 30, ny - 1, 4] == 1.0
        assert sf[6, 30, ny - 1, 4] == 0.5

    def test_flip_z_inverts_fz_only(self, sample_state, sample_policy):
        state = sample_state.copy()
        state[4, 30, 16, 1] = 1.0
        state[5, 30, 16, 1] = -2.0
        state[6, 30, 16, 1] = 3.0
        bp = state[0:1].copy()
        sf, _, bpf = flip_z(state, sample_policy, bp)
        nz = state.shape[3]
        assert sf[4, 30, 16, nz - 2] == 1.0
        assert sf[5, 30, 16, nz - 2] == -2.0
        assert sf[6, 30, 16, nz - 2] == -3.0
        assert bpf.shape == bp.shape

    def test_d4_x_z2_has_16_distinct_orientations(self):
        """8 flips (fx, fy, fz) x 2 rot90 yield 16 unique physical configurations."""
        s0 = np.zeros((7, 8, 8, 8), dtype=np.float32)
        p0 = np.zeros((2, 8, 8, 8), dtype=np.float32)
        # Asymmetric voxel & force off-center so all 16 symmetries differ
        s0[0, 1, 2, 1] = 1.0
        s0[4, 1, 2, 1] = 1.0
        s0[5, 1, 2, 1] = 2.0
        s0[6, 1, 2, 1] = 3.0
        p0[0, 1, 2, 1] = 1.0

        seen = set()
        for r in (0, 1):
            for fx_b in (0, 1):
                for fy_b in (0, 1):
                    for fz_b in (0, 1):
                        s, p = s0.copy(), p0.copy()
                        if fx_b:
                            s, p = flip_x(s, p)
                        if fy_b:
                            s, p = flip_y(s, p)
                        if fz_b:
                            s, p = flip_z(s, p)
                        if r:
                            s, p = rotate_90_z(s, p)
                        seen.add(s.tobytes())
        assert len(seen) == 16


class TestNegativeSampling:
    def test_erosion_attack(self, sample_state, sample_policy):
        original_volume = sample_state[0].sum()
        state_eroded, policy_repair, value = erosion_attack(sample_state, sample_policy, 0.5)
        assert value == -1.0
        assert state_eroded[0].sum() < original_volume
        np.testing.assert_array_equal(policy_repair[0], np.maximum(0.0, sample_state[0] - state_eroded[0]))
        assert policy_repair[1].sum() == 0

    def test_load_multiplier_never_adds_on_solid(self, sample_state, sample_policy):
        k = 3.0
        # Ensure solid exists adjacent to force
        state = sample_state.copy()
        state[0, 62, 16, 4] = 1.0
        state_stressed, policy, value = load_multiplier(state, sample_policy, 0.5, k=k)
        assert value == -0.8
        assert state_stressed[5, 63, 16, 4] == state[5, 63, 16, 4] * k
        assert policy[0].sum() > 0
        # Crucial Bloco B fix: ADD targets cannot overlap already-solid voxels
        solid = state[0] > 0.5
        assert np.sum(policy[0][solid]) == 0.0

    def test_sabotage_removes_and_guards_empty(self, sample_policy):
        state = np.zeros((7, 64, 32, 8), dtype=np.float32)
        state[1:4, 0, :, :] = 1.0
        # Case 1: no material near support -> must leave state/value unchanged
        s_unchanged, p_unchanged, v_unchanged = sabotage(state, sample_policy, 0.6)
        assert v_unchanged == 0.6
        np.testing.assert_array_equal(s_unchanged, state)

        # Case 2: material near support -> removes >0 and labels -1.0
        state[0, 0:4, 10:20, 2:6] = 1.0
        state_sab, pol_rep, val = sabotage(state, sample_policy, 0.6)
        assert val == -1.0
        assert state_sab[0].sum() < state[0].sum()
        assert pol_rep[0].sum() > 0

    def test_saboteur(self, sample_state, sample_policy):
        state_sab, pol_rep, val = saboteur(sample_state, sample_policy, 0.5, cube_size=3)
        assert val == -1.0
        assert state_sab[0].sum() < sample_state[0].sum()
        assert pol_rep[0].sum() > 0

        empty_s = np.zeros((7, 64, 32, 8), dtype=np.float32)
        empty_p = np.zeros((2, 64, 32, 8), dtype=np.float32)
        so, po, vo = saboteur(empty_s, empty_p, 0.5)
        assert vo == 0.5
        np.testing.assert_array_equal(so, empty_s)


class TestPerformance:
    @pytest.fixture
    def large_sample(self):
        state = np.random.rand(7, 64, 32, 8).astype(np.float32)
        state[0] = (state[0] > 0.5).astype(np.float32)
        policy = np.random.rand(2, 64, 32, 8).astype(np.float32)
        return state, policy

    def test_rotate_performance(self, large_sample):
        state, policy = large_sample
        t0 = time.perf_counter()
        for _ in range(100):
            rotate_90_z(state, policy)
        ms = (time.perf_counter() - t0) / 100 * 1000
        assert ms < 5, f"rotate_90_z took {ms:.2f}ms"

    def test_flip_performance(self, large_sample):
        state, policy = large_sample
        t0 = time.perf_counter()
        for _ in range(100):
            flip_y(state, policy)
        ms = (time.perf_counter() - t0) / 100 * 1000
        assert ms < 5, f"flip_y took {ms:.2f}ms"
