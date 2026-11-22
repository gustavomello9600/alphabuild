"""
Unit tests for v3.3 reward calibration, displacement margin penalty, PBRS helpers,
island analysis, value backpropagation, and FEniCSx evaluator parity.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from alphabuilder.src.logic.harvest.generators import generate_bc_masks, get_load_region_bounds
from alphabuilder.src.logic.selfplay.evaluator import HAS_FENICS, evaluate_fem
from alphabuilder.src.logic.selfplay.reward import (
    ALPHA_VOL,
    COMPLIANCE_BASE,
    COMPLIANCE_SLOPE,
    DISPLACEMENT_LIMIT_RAW,
    MU_SCORE,
    SIGMA_SCORE,
    VOLUME_REFERENCE,
    VOLUME_SENSITIVITY,
    analyze_structure_islands,
    calculate_compliance_score,
    calculate_displacement_penalty,
    calculate_island_penalty,
    calculate_pbrs_potential,
    calculate_raw_score,
    calculate_reward,
    calculate_volume_bonus,
    get_guidance_alpha,
    get_phase1_reward,
    get_reward_with_island_penalty,
)
from alphabuilder.src.logic.selfplay.runner import EpisodeState
from alphabuilder.src.logic.selfplay.storage import (
    GameInfo,
    backpropagate_value_targets,
    initialize_selfplay_db,
    load_game_step,
    record_step,
    save_game,
)
from alphabuilder.src.logic.mcts.engine import SearchResult
from alphabuilder.src.logic.storage import Phase


class TestRewardFunctions:
    def test_calculate_compliance_score(self):
        assert calculate_compliance_score(10) == pytest.approx(0.90, abs=0.01)
        assert calculate_compliance_score(100) == pytest.approx(0.80, abs=0.01)
        assert calculate_compliance_score(1000) == pytest.approx(0.45, abs=0.01)
        assert calculate_compliance_score(1_000_000) == pytest.approx(-0.60, abs=0.01)
        assert calculate_compliance_score(0) == pytest.approx(0.90, abs=0.01)

    def test_calculate_volume_bonus(self):
        assert calculate_volume_bonus(0.10) == pytest.approx(0.0, abs=0.01)
        assert calculate_volume_bonus(0.05) == pytest.approx(0.125, abs=0.01)
        assert calculate_volume_bonus(0.20) == pytest.approx(-0.25, abs=0.01)

    def test_calculate_reward_targets(self):
        targets = [
            (10, 0.90),
            (100, 0.80),
            (1000, 0.45),
            (10000, 0.10),
            (100000, -0.25),
            (1000000, -0.60),
        ]
        for compliance, target in targets:
            reward = calculate_reward(compliance, 0.1, is_valid=True)
            assert reward == pytest.approx(target, abs=0.02)

    def test_displacement_margin_penalty(self):
        limit = 100.0
        assert calculate_displacement_penalty(70.0, limit) == pytest.approx(0.0)
        assert calculate_displacement_penalty(85.0, limit) == pytest.approx(0.25, abs=1e-4)
        assert calculate_displacement_penalty(100.0, limit) == pytest.approx(0.50, abs=1e-4)

        r_safe = calculate_reward(100.0, 0.10, True, max_displacement=50.0, displacement_limit=limit)
        r_warn = calculate_reward(100.0, 0.10, True, max_displacement=85.0, displacement_limit=limit)
        r_collapse = calculate_reward(100.0, 0.10, True, max_displacement=101.0, displacement_limit=limit)
        assert r_safe > r_warn > r_collapse == -1.0

    def test_calculate_reward_volume_monotonic(self):
        for c in [100, 1000, 10000]:
            r_low = calculate_reward(c, 0.05, True)
            r_mid = calculate_reward(c, 0.10, True)
            r_high = calculate_reward(c, 0.20, True)
            assert r_low > r_mid > r_high

    def test_invalid_and_legacy(self):
        assert calculate_reward(0.5, 0.2, is_valid=False) == -1.0
        assert calculate_raw_score(0.5, 0.2) == pytest.approx(
            calculate_compliance_score(0.5) + calculate_volume_bonus(0.2)
        )
        r, reason = get_phase1_reward(True, True, 10, 20)
        assert r == 0.5 and reason == "CONNECTION_SUCCESS"


class TestIslandAnalysisAndPBRS:
    def test_analyze_structure_islands_single(self):
        density = np.zeros((5, 5, 5))
        density[:, 2, 2] = 1.0
        density[4, 1:4, 1:4] = 1.0
        load_config = {"x": 4, "y": 2, "z_start": 0, "z_end": 5}
        analysis = analyze_structure_islands(density, load_config)
        assert analysis["n_islands"] == 1
        assert analysis["is_connected"] is True
        assert analysis["loose_voxels"] == 0

    def test_analyze_phase1_attached_cantilever_has_zero_loose_voxels(self):
        """Partial beam anchored to X=0 not yet hitting load X=9 must have loose_voxels=0."""
        density = np.zeros((10, 6, 4))
        density[0:5, 2:4, 1:3] = 1.0
        load_config = {"x": 9, "y": 3, "z_start": 1, "z_end": 3}
        analysis = analyze_structure_islands(density, load_config)
        assert analysis["n_islands"] == 1
        assert analysis["is_connected"] is False
        assert analysis["loose_voxels"] == 0

    def test_analyze_disconnected_island(self):
        density = np.zeros((5, 5, 5))
        density[0, 2, 2] = 1.0
        density[4, 1:4, 1:4] = 1.0
        load_config = {"x": 4, "y": 2, "z_start": 0, "z_end": 5}
        analysis = analyze_structure_islands(density, load_config)
        assert analysis["n_islands"] == 2
        assert analysis["is_connected"] is False
        assert analysis["loose_voxels"] == 9

    def test_pbrs_and_guidance_decay(self):
        assert get_guidance_alpha(0, 5, 0.30) == pytest.approx(0.30)
        assert get_guidance_alpha(5, 5, 0.30) == pytest.approx(0.0)

        res = (8, 6, 4)
        bc = generate_bc_masks(res, "FULL_CLAMP")
        forces = (np.zeros(res, dtype=np.float32), np.zeros(res, dtype=np.float32), np.zeros(res, dtype=np.float32))
        lcfg = {"x": 7, "y": 3, "z_start": 1, "z_end": 3}
        lx, ym, yx, zm, zx = get_load_region_bounds(res, lcfg)
        forces[1][lx, ym:yx, zm:zx] = -1.0

        d_conn = np.zeros(res, dtype=np.float32)
        d_conn[:, ym:yx, zm:zx] = 1.0
        phi_conn, info_conn = calculate_pbrs_potential(d_conn, bc, forces, lcfg)
        assert phi_conn == pytest.approx(0.50)
        assert info_conn["is_connected"] is True

    def test_backpropagate_respects_collapse_and_gamma(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "bp.db"
            initialize_selfplay_db(db_path)
            res = (4, 4, 4)
            bc = np.zeros((3, *res), dtype=np.float32)
            forces = np.zeros((3, *res), dtype=np.float32)
            save_game(db_path, GameInfo("g_col", "simple", "v3.3", bc, forces, {"x": 3, "y": 2, "z_start": 1, "z_end": 3}, "FULL_CLAMP", res))

            dummy_pol = np.zeros(res, dtype=np.float32)
            sr = SearchResult(actions=[], visit_distribution={}, root_value=0.0, num_simulations=1, root=None)

            # Step 0: GROWTH -> Step 1: REFINEMENT healthy -> Step 2: REFINEMENT collapse
            for step, phase, conn, disp, comp, fem_r in [
                (0, Phase.GROWTH, False, None, None, None),
                (1, Phase.REFINEMENT, True, 100.0, 200.0, 0.60),
                (2, Phase.REFINEMENT, True, 200000.0, 50.0, -1.0),
            ]:
                st = EpisodeState(density=np.ones(res, dtype=np.float32) * 0.1, bc_masks=bc, forces=forces, load_config={}, current_step=step, phase=phase, game_id="g_col")
                record_step(
                    db_path, st, sr, value=0.0, policy_add=dummy_pol, policy_remove=dummy_pol,
                    island_analysis={"n_islands": 1, "loose_voxels": 0, "is_connected": conn},
                    compliance_fem=comp, max_displacement=disp, volume_fraction=0.1, fem_reward=fem_r,
                )

            backpropagate_value_targets(db_path, "g_col", gamma=0.99)
            s0 = load_game_step(db_path, "g_col", 0, res)
            s1 = load_game_step(db_path, "g_col", 1, res)
            s2 = load_game_step(db_path, "g_col", 2, res)
            assert s2.value_target == pytest.approx(-1.0)
            assert s1.value_target == pytest.approx(0.60)
            assert s0.value_target == pytest.approx(-0.99)


@pytest.mark.skipif(not HAS_FENICS, reason="FEniCSx not available")
class TestEvaluatorFEniCSx:
    def test_interior_load_and_bc_separation(self):
        res = (16, 8, 4)
        density = np.zeros(res, dtype=np.float32)
        density[:, 2:6, 1:3] = 1.0
        load_cfg = {"x": 12, "y": 4, "z_start": 1, "z_end": 3}

        res_clamp = evaluate_fem(density, load_cfg, bc_type="FULL_CLAMP")
        res_rail = evaluate_fem(density, load_cfg, bc_type="RAIL_XY")

        assert res_clamp.valid and np.isfinite(res_clamp.compliance) and res_clamp.compliance > 0.0
        assert res_rail.valid and np.isfinite(res_rail.compliance) and res_rail.compliance > 0.0
        # Relaxing Z at X=0 (RAIL_XY) cannot be stiffer than FULL_CLAMP
        assert res_rail.compliance > res_clamp.compliance
