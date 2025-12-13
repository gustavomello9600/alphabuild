
import pytest
import numpy as np
from alphabuilder.src.logic.selfplay.reward import (
    calculate_raw_score,
    calculate_reward,
    calculate_compliance_score,
    calculate_volume_bonus,
    get_phase1_reward,
    check_structure_connectivity,
    get_phase2_terminal_reward,
    estimate_reward_components,
    analyze_structure_islands,
    calculate_island_penalty,
    get_reward_with_island_penalty,
    # New constants
    COMPLIANCE_BASE,
    COMPLIANCE_SLOPE,
    VOLUME_REFERENCE,
    VOLUME_SENSITIVITY,
    # Legacy constants
    MU_SCORE,
    SIGMA_SCORE,
    ALPHA_VOL
)

class TestRewardFunctions:

    def test_calculate_compliance_score(self):
        """Test log10-based compliance scoring."""
        # C=10 (log10=1) should give 0.80
        assert calculate_compliance_score(10) == pytest.approx(0.80, abs=0.01)
        # C=1000 (log10=3) should give 0.48
        assert calculate_compliance_score(1000) == pytest.approx(0.48, abs=0.01)
        # C=1M (log10=6) should give 0.00
        assert calculate_compliance_score(1_000_000) == pytest.approx(0.00, abs=0.01)
        # C=0 or negative should return max score
        assert calculate_compliance_score(0) == pytest.approx(0.85, abs=0.01)

    def test_calculate_volume_bonus(self):
        """Test volume bonus/penalty calculation."""
        # V=0.10 (reference) should give 0.0
        assert calculate_volume_bonus(0.10) == pytest.approx(0.0, abs=0.01)
        # V=0.05 should give +0.10 bonus
        assert calculate_volume_bonus(0.05) == pytest.approx(0.10, abs=0.01)
        # V=0.20 should give -0.20 penalty
        assert calculate_volume_bonus(0.20) == pytest.approx(-0.20, abs=0.01)

    def test_calculate_reward_targets(self):
        """Test that reward matches user targets at V=0.1."""
        targets = [
            (10, 0.80),
            (100, 0.70),
            (1000, 0.50),
            (10000, 0.30),
            (100000, 0.10),
            (1000000, 0.00)
        ]
        for compliance, target in targets:
            reward = calculate_reward(compliance, 0.1, is_valid=True)
            assert reward == pytest.approx(target, abs=0.10), f"C={compliance} failed"

    def test_calculate_reward_volume_monotonic(self):
        """Lower volume should ALWAYS give higher reward."""
        for c in [100, 1000, 10000]:
            r_low = calculate_reward(c, 0.05, True)
            r_mid = calculate_reward(c, 0.10, True)
            r_high = calculate_reward(c, 0.20, True)
            assert r_low > r_mid > r_high, f"Volume should be inversely proportional at C={c}"

    def test_calculate_reward_invalid_structure(self):
        """Test penalty for invalid structure."""
        reward = calculate_reward(0.5, 0.2, is_valid=False)
        assert reward == -1.0

    def test_calculate_reward_collapse(self):
        """Test penalty for structural collapse."""
        reward = calculate_reward(0.5, 0.2, is_valid=True, max_displacement=100.0, displacement_limit=10.0)
        assert reward == -1.0

    def test_get_phase1_reward(self):
        """Test Phase 1 reward logic."""
        # Not finished, not connected
        reward, reason = get_phase1_reward(False, False, 10, 20)
        assert reward == -0.01  # living penalty
        assert reason == "CONTINUE"

        # Connected (Success)
        reward, reason = get_phase1_reward(True, True, 10, 20)
        assert reward == 0.5
        assert reason == "CONNECTION_SUCCESS"

        # Timeout
        reward, reason = get_phase1_reward(False, False, 20, 20)
        assert reward == -1.0
        assert reason == "MAX_STEPS_REACHED"

    def test_legacy_raw_score(self):
        """Test legacy raw score for backward compatibility."""
        compliance = 0.5
        vol_frac = 0.2
        expected_score = -np.log(compliance) - ALPHA_VOL * vol_frac
        assert calculate_raw_score(compliance, vol_frac) == pytest.approx(expected_score)

class TestIslandAnalysis:

    def test_analyze_structure_islands_single(self):
        """Test analysis of a single connected component."""
        density = np.zeros((5, 5, 5))
        # Create a bar connecting support (x=0) to load (assume load at x=4)
        density[:, 2, 2] = 1.0 
        
        # Fill the load pad area (3x3 at x=4) to satisfy strict check
        density[4, 1:4, 1:4] = 1.0
        
        load_config = {'x': 4, 'y': 2, 'z_start': 0, 'z_end': 5}
        
        analysis = analyze_structure_islands(density, load_config)
        
        assert analysis['n_islands'] == 1
        assert analysis['is_connected'] == True
        assert analysis['loose_voxels'] == 0

    def test_analyze_structure_islands_disconnected(self):
        """Test disconnected structure."""
        density = np.zeros((5, 5, 5))
        density[0, 2, 2] = 1.0 # Support
        
        # Load pad filled (so we fail on disconnect, not on empty load)
        density[4, 1:4, 1:4] = 1.0 
        
        load_config = {'x': 4, 'y': 2, 'z_start': 0, 'z_end': 5}
        
        analysis = analyze_structure_islands(density, load_config)
        
        # Load pad is one component (size 9). Support is another (size 1).
        assert analysis['n_islands'] == 2
        assert analysis['is_connected'] == False
        assert analysis['loose_voxels'] > 0

    def test_calculate_island_penalty(self):
        """Test penalty calculation."""
        # 1 island, no loose -> 0 penalty
        assert calculate_island_penalty(1, 0, 100) == 0.0
        
        # 2 islands -> penalty
        penalty = calculate_island_penalty(2, 0, 100, penalty_per_island=0.02)
        assert penalty == 0.02
        
        # Loose voxels penalty
        penalty = calculate_island_penalty(2, 10, 100, penalty_per_island=0.0, penalty_per_loose_voxel_frac=0.1)
        # Using n_islands=2 to bypass the n_islands<=1 check.
        # Set penalty_per_island=0.0 to isolate loose voxel penalty.
        # 10/100 = 0.1 fraction * 0.1 weight = 0.01
        assert penalty == pytest.approx(0.01)

    def test_get_reward_with_island_penalty(self):
        """Test application of penalty to reward."""
        base_reward = 0.8
        
        # Create a density with 2 islands
        density = np.zeros((5, 5, 5))
        density[0, 0, 0] = 1.0 # Island 1
        density[4, 4, 4] = 1.0 # Island 2
        
        load_config = {'x': 4, 'y': 4, 'z_start': 0, 'z_end': 5}
        
        final_reward, analysis = get_reward_with_island_penalty(base_reward, density, load_config)
        
        assert analysis['island_penalty'] > 0
        assert final_reward < base_reward
        assert final_reward == base_reward - analysis['island_penalty']
