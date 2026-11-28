"""Unit tests verifying Stage 6 v4.2 upgrades (volume-aware expansion, bidirectional SED, trust region)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphabuilder.src.logic.harvest.generators import generate_bc_masks
from alphabuilder.src.logic.mcts.engine import (
    PHASE1_V42_CONFIG,
    PHASE2_V41_CONFIG,
    PHASE2_V42_CONFIG,
    select_non_cycling_root_action,
)
from alphabuilder.src.logic.mcts.legal_moves import get_legal_moves
from alphabuilder.src.logic.mcts.node import MCTSNode, STOP_ACTION
from alphabuilder.src.logic.selfplay.p2_control import (
    build_scaled_p2_schedule,
    is_trust_region_accepted,
    select_sed_probe_mode,
)
from alphabuilder.src.logic.selfplay.runner import _build_sed_or_policy_probe_batch


def _make_cantilever_setup(shape=(12, 8, 4)):
    nx, ny, nz = shape
    density = np.zeros(shape, dtype=np.float32)
    density[:, 3:5, 1:3] = 1.0  # thin 2x2xL central beam along X
    mx, my, mz = generate_bc_masks(shape, "FULL_CLAMP")
    bc = (mx, my, mz)
    forces = (np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32))
    forces[1][nx - 1, 3:5, 1:3] = -1.0
    lcfg = {"x": nx - 1, "y": 4, "z_start": 1, "z_end": 3, "bc_type": "FULL_CLAMP"}
    return density, bc, forces, lcfg


def test_v42_expand_macro_suppresses_pure_remove_below_min_vol():
    """Below remove_min_vol_frac=0.080, pure REM macros are suppressed and SWAP is net-positive."""
    shape = (12, 8, 4)
    density, bc, forces, _ = _make_cantilever_setup(shape)
    valid_add, valid_rem = get_legal_moves(
        density, 0.5, bc_masks=bc, forces=forces, surface_only_remove=True
    )

    # Even if op_head prefers REM (0.20 ADD, 0.70 REM, 0.10 STOP), low volume locks out pure REM
    pa = np.ones(shape, dtype=np.float32)
    pr = np.full(shape, 5.0, dtype=np.float32)
    node = MCTSNode(prior=1.0)
    node.expand_macro(
        policy_add=pa,
        policy_remove=pr,
        valid_add=valid_add,
        valid_remove=valid_rem,
        action_batch_size=16,
        phase="REFINEMENT",
        remove_prior_floor=0.25,
        op_probs=(0.20, 0.70, 0.10),
        allow_stop=True,
        max_candidates=16,
        seed=7,
        current_vol_frac=0.045,
        remove_min_vol_frac=0.080,
        remove_floor_vol_frac=0.105,
    )

    assert node.children
    assert abs(sum(c.prior for c in node.children.values()) - 1.0) < 1e-5
    for edge in node.children:
        if edge == STOP_ACTION:
            continue
        adds = sum(1 for a in edge if a[0] == 0)
        rems = sum(1 for a in edge if a[0] == 1)
        assert rems == 0 or adds > rems, f"Expected pure ADD or net-positive GROW_SWAP, got adds={adds}, rems={rems}"


def test_v42_expand_macro_ramps_remove_and_preserves_v41_compat():
    """Above remove_min_vol_frac pure REM is unlocked; legacy remove_min_vol_frac=0.0 keeps v4.1 behavior."""
    shape = (12, 8, 4)
    density, bc, forces, _ = _make_cantilever_setup(shape)
    valid_add, valid_rem = get_legal_moves(
        density, 0.5, bc_masks=bc, forces=forces, surface_only_remove=True
    )
    pa = np.ones(shape, dtype=np.float32)
    pr = np.ones(shape, dtype=np.float32)

    def _pure_rem_prior(cur_v: float, min_v: float, floor_v: float) -> float:
        n = MCTSNode(prior=1.0)
        n.expand_macro(
            policy_add=pa,
            policy_remove=pr,
            valid_add=valid_add,
            valid_remove=valid_rem,
            action_batch_size=8,
            phase="REFINEMENT",
            remove_prior_floor=0.25,
            op_probs=(0.90, 0.05, 0.05),
            allow_stop=False,
            max_candidates=16,
            seed=11,
            current_vol_frac=cur_v,
            remove_min_vol_frac=min_v,
            remove_floor_vol_frac=floor_v,
        )
        return sum(
            c.prior
            for k, c in n.children.items()
            if k != STOP_ACTION and all(a[0] == 1 for a in k)
        )

    p_low = _pure_rem_prior(0.045, 0.080, 0.105)
    p_mid = _pure_rem_prior(0.092, 0.080, 0.105)
    p_high = _pure_rem_prior(0.120, 0.080, 0.105)
    p_v41 = _pure_rem_prior(0.045, 0.0, 0.0)

    assert p_low == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < p_mid < p_high
    assert p_v41 > 0.10


def test_v42_select_non_cycling_root_action():
    """Verify root selector skips cycling top edge, chooses next non-cycling edge, or STOP if all cycle."""
    import hashlib

    shape = (8, 6, 4)
    dens = np.zeros(shape, dtype=np.float32)
    dens[:, 2:4, 1:3] = 1.0

    e_cycle = ((0, 1, 1, 1),)
    e_fresh = ((0, 2, 1, 1),)
    d_cycle = dens.copy()
    d_cycle[1, 1, 1] = 1.0
    h_cycle = hashlib.md5((d_cycle > 0.5).astype(np.uint8).tobytes()).hexdigest()
    seen = {hashlib.md5((dens > 0.5).astype(np.uint8).tobytes()).hexdigest(), h_cycle}

    root = MCTSNode(prior=1.0, is_expanded=True)
    c1 = MCTSNode(prior=0.6, action_to_parent=e_cycle, parent=root)
    c1.visit_count = 80
    c1.value_sum = 40.0
    c2 = MCTSNode(prior=0.4, action_to_parent=e_fresh, parent=root)
    c2.visit_count = 40
    c2.value_sum = 16.0
    root.children = {e_cycle: c1, e_fresh: c2}

    edge, acts, stats, stopped, avoided = select_non_cycling_root_action(
        root, dens, seen, allow_stop=False, max_actions=16
    )
    assert edge == e_fresh
    assert acts == [(0, 2, 1, 1)]
    assert stats == [((0, 2, 1, 1), 40, pytest.approx(0.4))]
    assert not stopped and avoided

    # If all non-STOP children lead into seen_hashes and STOP is present & allowed -> choose STOP
    d_fresh = dens.copy()
    d_fresh[2, 1, 1] = 1.0
    seen_all = seen | {hashlib.md5((d_fresh > 0.5).astype(np.uint8).tobytes()).hexdigest()}
    c_stop = MCTSNode(prior=0.1, action_to_parent=STOP_ACTION, parent=root)
    c_stop.visit_count = 5
    root.children[STOP_ACTION] = c_stop
    edge2, acts2, _, stopped2, avoided2 = select_non_cycling_root_action(
        root, dens, seen_all, allow_stop=True
    )
    assert edge2 == STOP_ACTION and acts2 == [] and stopped2 and avoided2


def test_v42_bidirectional_sed_probe_modes():
    """Verify SED probe modes across volume regimes and pure outer-SED expansion at V < 0.080."""
    assert select_sed_probe_mode(0.045) == ("sed_expand_topk", 0.0)
    assert select_sed_probe_mode(0.090) == ("sed_grow_swap_topk", 0.25)
    assert select_sed_probe_mode(0.120) == ("sed_hybrid_topk", 0.50)
    assert select_sed_probe_mode(0.160) == ("sed_carve_topk", pytest.approx(2.0 / 3.0))

    shape_low = (16, 32, 4)
    dens = np.zeros(shape_low, dtype=np.float32)
    dens[:, 15:17, :] = 1.0  # 128 / 2048 = 0.0625 < 0.080
    mx_l, my_l, mz_l = generate_bc_masks(shape_low, "FULL_CLAMP")
    bc = (mx_l, my_l, mz_l)
    forces = (np.zeros(shape_low, dtype=np.float32), np.zeros(shape_low, dtype=np.float32), np.zeros(shape_low, dtype=np.float32))
    forces[1][15, 15:17, 1:3] = -1.0
    lcfg = {"x": 15, "y": 16, "z_start": 1, "z_end": 3, "bc_type": "FULL_CLAMP"}
    # Place highest SED along top face y=16 -> only neighbor y=17 has sed_nbr=100
    sed = np.zeros(shape_low, dtype=np.float32)
    sed[:, 16, :] = 100.0
    sed[:, 15, :] = 5.0
    pol_rem = np.ones(shape_low, dtype=np.float32)

    batch_low = _build_sed_or_policy_probe_batch(
        dens,
        bc,
        forces,
        lcfg,
        pol_rem,
        sed_map=sed,
        max_k=12,
        bidirectional_expand=True,
        remove_min_vol=0.080,
        expand_target_vol=0.105,
    )
    assert len(batch_low) == 12
    assert all(ch == 0 for ch, _, _, _ in batch_low)
    assert all(y == 17 for _, _, y, _ in batch_low)

    # Thick beam V >= 0.080 -> grow_swap or hybrid mixes safe REM and highest-SED ADD
    shape = (12, 8, 4)
    _, bc_t, forces_t, lcfg_t = _make_cantilever_setup(shape)
    dens_thick = np.zeros(shape, dtype=np.float32)
    dens_thick[:, 2:6, 1:3] = 1.0  # 25% > 0.145 -> mixed/carve
    sed_thick = np.zeros(shape, dtype=np.float32)
    sed_thick[:, 5, 1:3] = 50.0
    pol_rem_t = np.ones(shape, dtype=np.float32)
    batch_high = _build_sed_or_policy_probe_batch(
        dens_thick,
        bc_t,
        forces_t,
        lcfg_t,
        pol_rem_t,
        sed_map=sed_thick,
        max_k=12,
        bidirectional_expand=True,
        remove_min_vol=0.080,
        expand_target_vol=0.105,
    )
    n_rem = sum(1 for ch, _, _, _ in batch_high if ch == 1)
    n_add = sum(1 for ch, _, _, _ in batch_high if ch == 0)
    assert n_rem > 0 and n_add > 0


def test_v42_trust_region_acceptance_and_configs():
    """Verify anti-throttle trust region, scaled P2 schedule, and backwards-compatible configs."""
    assert is_trust_region_accepted(True, True, False, 0.02, -0.01, 45000.0, 52000.0)
    # Catastrophic S1 neck throttling: 52k -> 637k J must be rejected!
    assert not is_trust_region_accepted(True, True, False, -0.05, -0.015, 637628.0, 52538.0)
    # Severe reward drop (> 0.08) rejected even below compliance multiplier
    assert not is_trust_region_accepted(True, True, False, -0.11, -0.01, 70000.0, 52000.0)
    # Grace step allows higher compliance as long as score doesn't drop worse than -0.02
    assert is_trust_region_accepted(True, True, False, -0.88, -0.92, 300000.0, 150000.0, in_grace_cur=True)
    # Disconnected rejected
    assert not is_trust_region_accepted(False, True, False, 0.10, -0.10, 100.0, 1000.0)

    assert build_scaled_p2_schedule((48, 32, 16, 8, 4), 64 * 32 * 8) == [48, 32, 16, 8, 4]
    sched_small = build_scaled_p2_schedule((48, 32, 16, 8, 4), 16 * 8 * 4)
    assert sched_small[0] < 48 and all(a > b for a, b in zip(sched_small, sched_small[1:]))

    assert PHASE2_V41_CONFIG.remove_min_vol_frac == 0.0
    assert PHASE1_V42_CONFIG.phase == "GROWTH"
    assert PHASE2_V42_CONFIG.action_batch_size == 48
    assert PHASE2_V42_CONFIG.remove_min_vol_frac == pytest.approx(0.080)
    assert PHASE2_V42_CONFIG.remove_floor_vol_frac == pytest.approx(0.105)

    # Verify full run_mcts_search with PHASE2_V42_CONFIG (batched & sequential paths) passes volume cleanly
    from alphabuilder.src.logic.mcts.engine import run_mcts_search

    shape_s = (16, 16, 4)
    dens_s = np.zeros(shape_s, dtype=np.float32)
    dens_s[:, 7:9, 1:3] = 1.0  # V = 0.0625 < 0.080
    mx_s, my_s, mz_s = generate_bc_masks(shape_s, "FULL_CLAMP")
    bc_s = (mx_s, my_s, mz_s)
    forces_s = (np.zeros(shape_s, dtype=np.float32), np.zeros(shape_s, dtype=np.float32), np.zeros(shape_s, dtype=np.float32))
    forces_s[1][15, 7:9, 1:3] = -1.0

    def _pred_single(st):
        return 0.1, np.ones(shape_s, dtype=np.float32), np.full(shape_s, 5.0, dtype=np.float32)

    def _pred_batch(sts):
        return [0.1] * len(sts), [np.ones(shape_s, dtype=np.float32)] * len(sts), [
            np.full(shape_s, 5.0, dtype=np.float32)
        ] * len(sts)

    for bsz in (4, 1):
        cfg_s = PHASE2_V42_CONFIG._replace(num_simulations=8, batch_size=bsz, action_batch_size=8, max_candidates=12 if hasattr(PHASE2_V42_CONFIG, "max_candidates") else 12) if False else PHASE2_V42_CONFIG._replace(num_simulations=8, batch_size=bsz, action_batch_size=8, macro_num_candidates=12)
        res_s = run_mcts_search(dens_s, bc_s, forces_s, _pred_single, config=cfg_s, predict_batch_fn=_pred_batch if bsz > 1 else None)
        assert res_s.root.children
        for k in res_s.root.children:
            if k != STOP_ACTION:
                adds_k = sum(1 for a in k if a[0] == 0)
                rems_k = sum(1 for a in k if a[0] == 1)
                assert rems_k == 0 or adds_k > rems_k
