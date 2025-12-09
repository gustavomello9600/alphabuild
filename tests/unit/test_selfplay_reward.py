
import pytest
import numpy as np
from alphabuilder.src.logic.selfplay.reward import (
    calculate_raw_score,
    calculate_reward,
    get_phase1_reward,
    check_structure_connectivity,
    get_phase2_terminal_reward,
    estimate_reward_components,
    analyze_structure_islands,
    calculate_island_penalty,
    get_reward_with_island_penalty,
    MU_SCORE,
    SIGMA_SCORE,
    ALPHA_VOL
)

class TestRewardFunctions:

    def test_calculate_raw_score(self):
        """Test random score calculation."""
        compliance = 0.5
        vol_frac = 0.2
        expected_score = -np.log(compliance) - ALPHA_VOL * vol_frac
        assert calculate_raw_score(compliance, vol_frac) == pytest.approx(expected_score)

    def test_calculate_reward_valid_structure(self):
        """Test normalized reward for a valid structure."""
        compliance = 0.5
        vol_frac = 0.2
        raw_score = calculate_raw_score(compliance, vol_frac)
        expected_reward = np.tanh((raw_score - MU_SCORE) / SIGMA_SCORE)
        
        reward = calculate_reward(compliance, vol_frac, is_valid=True)
        assert reward == pytest.approx(expected_reward)

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

    def test_estimate_reward_components(self):
        """Test reverse estimation of components."""
        value = 0.5
        vol_frac = 0.3
        
        components = estimate_reward_components(value, vol_frac)
        
        # Verify if recalculating forward gives back the value
        est_raw = components['estimated_raw_score']
        recalc_value = np.tanh((est_raw - MU_SCORE) / SIGMA_SCORE)
        assert recalc_value == pytest.approx(value)

class TestIslandAnalysis:

    def test_analyze_structure_islands_single(self):
        """Test analysis of a single connected component."""
        density = np.zeros((5, 5, 5))
        # Create a bar connecting support (x=0) to load (assume load at x=4)
        density[:, 2, 2] = 1.0 
        
        load_config = {'x': 4, 'y': 2, 'z_start': 0, 'z_end': 5}
        
        analysis = analyze_structure_islands(density, load_config)
        
        assert analysis['n_islands'] == 1
        assert analysis['is_connected'] == True
        assert analysis['loose_voxels'] == 0

    def test_analyze_structure_islands_disconnected(self):
        """Test disconnected structure."""
        density = np.zeros((5, 5, 5))
        density[0, 2, 2] = 1.0 # Support
        density[4, 2, 2] = 1.0 # Load
        # Missing middle
        
        load_config = {'x': 4, 'y': 2, 'z_start': 0, 'z_end': 5}
        
        analysis = analyze_structure_islands(density, load_config)
        
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
