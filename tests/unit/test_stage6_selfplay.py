"""Unit and numerical-parity tests for AlphaBuilder v4.0 Stage 6 Self-Play."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sqlite3
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphabuilder.src.logic.harvest.generators import generate_bc_masks
from alphabuilder.src.logic.mcts.engine import AlphaBuilderMCTS, MCTSConfig
from alphabuilder.src.logic.mcts.legal_moves import get_legal_moves, is_terminal_state
from alphabuilder.src.logic.mcts.node import MCTSNode
from alphabuilder.src.logic.selfplay import (
    GameInfo,
    backpropagate_value_targets,
    generate_game_id,
    initialize_selfplay_db,
    load_game,
    merge_selfplay_dbs,
    record_fem_probe,
    record_step,
    save_game,
    update_game_final,
)
from alphabuilder.src.logic.selfplay.evaluator import HAS_FENICS, evaluate_fem
from alphabuilder.src.logic.selfplay.fast_fem import evaluate_fem_fast
from alphabuilder.src.logic.selfplay.reward import calculate_pbrs_potential, check_structure_connectivity
from alphabuilder.src.logic.selfplay.runner import EpisodeConfig, run_episode_with_stats
from alphabuilder.src.logic.selfplay.scenarios_stage6 import (
    SelfPlayScenario,
    adjust_tier_weights,
    build_ood_benchmark_scenarios,
    is_near_canonical_scenario,
    sample_curriculum_scenario,
)
from alphabuilder.src.logic.storage import Phase
from alphabuilder.src.neural.dataset import make_symmetry_collate_fn
from alphabuilder.src.neural.inference import AlphaBuilderInference
from alphabuilder.src.neural.phys_uvit import PhysUViT3D
from alphabuilder.src.neural.selfplay_dataset import (
    BucketedResolutionBatchSampler,
    MixedReplayDataset,
    SelfPlayDataset,
)
from alphabuilder.src.neural.trainer import compute_total_loss, sed_loss


def _make_cantilever_bar(resolution=(24, 12, 4), hole: bool = False) -> np.ndarray:
    nx, ny, nz = resolution
    density = np.zeros(resolution, dtype=np.float32)
    density[:, ny // 4 : (3 * ny) // 4, :] = 1.0
    if hole and nx >= 16 and ny >= 10:
        density[nx // 3 : nx // 2, ny // 2 - 1 : ny // 2 + 1, :] = 0.0
    return density


@pytest.mark.skipif(not HAS_FENICS, reason="FEniCSx not available")
def test_fast_fem_vs_dolfinx_parity_and_sed():
    """Verify fast_amg matches dolfinx compliance, max displacement, and sum(SED) == C across general BCs."""
    res = (24, 12, 4)
    nx, ny, nz = res
    density = _make_cantilever_bar(res, hole=True)
    load_cfg = {"x": nx - 1, "y": ny // 2, "z_start": 1, "z_end": 3, "bc_type": "FULL_CLAMP"}

    # 1) FULL_CLAMP & 2) RAIL_XY
    for bc_type in ("FULL_CLAMP", "RAIL_XY"):
        cfg = dict(load_cfg, bc_type=bc_type)
        ref = evaluate_fem(density, cfg, backend="dolfinx", compute_sed=True)
        fast = evaluate_fem(density, cfg, backend="fast_amg", compute_sed=True)
        assert ref.valid and fast.valid
        assert abs(fast.compliance - ref.compliance) / ref.compliance < 1e-3
        assert abs(fast.max_displacement - ref.max_displacement) / ref.max_displacement < 1e-3
        assert ref.sed_map is not None and fast.sed_map is not None
        assert abs(float(fast.sed_map.astype(np.float64).sum()) - fast.compliance) / fast.compliance < 1e-5
        assert abs(float(ref.sed_map.astype(np.float64).sum()) - ref.compliance) / ref.compliance < 1e-5
        max_sed = float(np.max(ref.sed_map))
        assert float(np.max(np.abs(fast.sed_map - ref.sed_map))) / max_sed < 1e-3

    # 3) BRIDGE_MBB (pin at x=0 base + roller Uy at x=nx-1 base, midspan top load)
    bridge = np.zeros(res, dtype=np.float32)
    bridge[:, 0 : ny // 2 + 2, :] = 1.0
    bc_mbb = np.zeros((3, *res), dtype=np.float32)
    bc_mbb[:, 0, 0:3, :] = 1.0
    bc_mbb[1, nx - 1, 0:3, :] = 1.0
    f_mbb = np.zeros((3, *res), dtype=np.float32)
    f_mbb[1, nx // 2, ny // 2 + 1, 1:3] = -1.0
    cfg_mbb = {"x": nx // 2, "y": ny // 2 + 1, "z_start": 1, "z_end": 3, "bc_type": "BRIDGE_MBB"}

    ref_mbb = evaluate_fem(bridge, cfg_mbb, bc_masks=bc_mbb, forces=f_mbb, backend="dolfinx", compute_sed=True)
    fast_mbb = evaluate_fem(bridge, cfg_mbb, bc_masks=bc_mbb, forces=f_mbb, backend="fast_amg", compute_sed=True)
    assert ref_mbb.valid and fast_mbb.valid
    assert abs(fast_mbb.compliance - ref_mbb.compliance) / ref_mbb.compliance < 1e-3
    assert abs(fast_mbb.max_displacement - ref_mbb.max_displacement) / ref_mbb.max_displacement < 1e-3
    assert fast_mbb.sed_map is not None
    assert abs(float(fast_mbb.sed_map.sum()) - fast_mbb.compliance) / fast_mbb.compliance < 1e-5

    # 4) DUAL_PAD + angled 3D force
    bc_dual = np.zeros((3, *res), dtype=np.float32)
    bc_dual[:, 0, 0:4, :] = 1.0
    bc_dual[:, 0, ny - 4 : ny, :] = 1.0
    f_3d = np.zeros((3, *res), dtype=np.float32)
    f_3d[0, nx - 1, ny // 2 - 1 : ny // 2 + 1, 1:3] = 0.5
    f_3d[1, nx - 1, ny // 2 - 1 : ny // 2 + 1, 1:3] = -0.75
    f_3d[2, nx - 1, ny // 2 - 1 : ny // 2 + 1, 1:3] = 0.35
    solid_full = np.ones(res, dtype=np.float32)
    solid_full[8:12, 4:8, :] = 0.0
    ref_dual = evaluate_fem(solid_full, load_cfg, bc_masks=bc_dual, forces=f_3d, backend="dolfinx", compute_sed=True)
    fast_dual = evaluate_fem(solid_full, load_cfg, bc_masks=bc_dual, forces=f_3d, backend="fast_amg", compute_sed=True)
    assert ref_dual.valid and fast_dual.valid
    assert abs(fast_dual.compliance - ref_dual.compliance) / ref_dual.compliance < 1e-3
    assert abs(fast_dual.max_displacement - ref_dual.max_displacement) / ref_dual.max_displacement < 1e-3
    assert fast_dual.sed_map is not None
    assert abs(float(fast_dual.sed_map.sum()) - fast_dual.compliance) / fast_dual.compliance < 1e-5


def test_fast_fem_handles_disconnected_empty():
    """Verify fast_amg safely reports invalid on empty density grids."""
    res = (16, 8, 4)
    res_empty = evaluate_fem_fast(np.zeros(res, dtype=np.float32), {"x": 15, "y": 4, "z_start": 1, "z_end": 3})
    assert res_empty.valid is False
    assert np.isinf(res_empty.compliance)


def test_mcts_remove_quota_and_domain_mask():
    """Verify REMOVE quota/prior floor, preserved default behavior, and keep-out masking."""
    shape = (8, 6, 4)
    density = np.zeros(shape, dtype=np.float32)
    density[:, 2:4, 1:3] = 1.0
    domain_mask = np.ones(shape, dtype=np.float32)
    domain_mask[3:6, 0:6, 0:4] = 0.0

    valid_add, valid_remove = get_legal_moves(density, domain_mask=domain_mask)
    assert valid_add[3:6].sum() == 0.0

    # Disabled default: zero prior-dominated REMOVE candidates
    p_add = np.full(shape, 5.0, dtype=np.float32)
    p_rem = np.full(shape, -25.0, dtype=np.float32)

    node_default = MCTSNode()
    node_default.expand(p_add, p_rem, valid_add, valid_remove, top_k=12)
    rem_default = [a for a in node_default.children if a[0] == 1]
    assert len(rem_default) == 0

    node_quota = MCTSNode()
    node_quota.expand(
        p_add,
        p_rem,
        valid_add,
        valid_remove,
        top_k=12,
        min_remove_candidates=4,
        remove_prior_floor=0.25,
    )
    rem_quota = [c for a, c in node_quota.children.items() if a[0] == 1]
    add_quota = [c for a, c in node_quota.children.items() if a[0] == 0]
    assert len(rem_quota) == 4
    assert len(add_quota) == 8
    total_prior = sum(c.prior for c in node_quota.children.values())
    rem_prior_sum = sum(c.prior for c in rem_quota)
    assert total_prior == pytest.approx(1.0, abs=1e-5)
    assert rem_prior_sum == pytest.approx(0.25, abs=1e-4)

    # Check is_terminal_state blocks all keep-out empty region
    term, reason = is_terminal_state(
        np.zeros(shape, dtype=np.float32),
        min_volume_fraction=0.0,
        domain_mask=np.zeros(shape, dtype=np.float32),
    )
    assert term and reason == "no_legal_moves"

    # Verify MCTS search + PhysUViT3D respect domain_mask on keep-out voxels
    model = PhysUViT3D(in_channels=7, out_channels=2, feature_size=16, embed_dim=64, transformer_depth=1, num_heads=4)
    model.eval()
    state = torch.zeros(1, 7, 16, 12, 8)
    state[:, 1:4, 0, :, :] = 1.0
    state[:, 5, 15, 5:7, 3:5] = -1.0
    dm_t = torch.ones(1, 1, 16, 12, 8)
    dm_t[:, :, 2:6, :, :] = 0.0
    with torch.no_grad():
        pol, val = model(state, domain_mask=dm_t)
    assert pol.shape == (1, 2, 16, 12, 8)
    assert val.shape == (1, 1)

    mcts_cfg = MCTSConfig(num_simulations=4, batch_size=2, action_batch_size=4, phase="GROWTH", min_volume_fraction=0.0)
    d_small = np.zeros((16, 12, 8), dtype=np.float32)
    bc_small = np.zeros((3, 16, 12, 8), dtype=np.float32)
    bc_small[:, 0, :, :] = 1.0
    f_small = np.zeros((3, 16, 12, 8), dtype=np.float32)
    f_small[1, 15, 5:7, 3:5] = -1.0
    dm_np = np.ones((16, 12, 8), dtype=np.float32)
    dm_np[2:6, :, :] = 0.0

    def _eval_fn(states_np):
        arr = np.asarray(states_np, dtype=np.float32)
        t = torch.from_numpy(arr)
        is_single = t.ndim == 4
        if is_single:
            t = t.unsqueeze(0)
        dm_batch = torch.from_numpy(dm_np[None, None]).expand(t.shape[0], -1, -1, -1, -1)
        with torch.no_grad():
            p, v = model(t, domain_mask=dm_batch)
        if is_single:
            return float(v[0, 0].item()), p[0, 0].numpy(), p[0, 1].numpy()
        return v.squeeze(-1).numpy().tolist(), p[:, 0].numpy(), p[:, 1].numpy()

    mcts = AlphaBuilderMCTS(_eval_fn, num_simulations=4, batch_size=2, action_batch_size=4)
    mcts.config = mcts_cfg
    res_search = mcts.search(d_small, bc_small, f_small, domain_mask=dm_np)
    assert len(res_search.actions) > 0
    assert all(dm_np[a[1], a[2], a[3]] > 0.5 for a in res_search.actions if a[0] == 0)


def test_scenarios_stage6_tiers_and_pacing():
    """Verify S1-S6 exclusion, 4-tier scenario sampling, adaptive pacing gate, and OOD G1-G4."""
    assert is_near_canonical_scenario({"x": 63, "y": 16}, "FULL_CLAMP") is True
    assert is_near_canonical_scenario({"x": 62, "y": 17}, "FULL_CLAMP") is True
    assert is_near_canonical_scenario({"x": 58, "y": 16}, "RAIL_XY") is True
    assert is_near_canonical_scenario({"x": 58, "y": 16}, "FULL_CLAMP") is False
    assert is_near_canonical_scenario({"x": 40, "y": 8}, "FULL_CLAMP") is False

    rng = random.Random(123)
    seen_res = set()
    seen_keepout = False
    for tier, force_weights in (
        (0, (1.0, 0.0, 0.0, 0.0)),
        (1, (0.0, 1.0, 0.0, 0.0)),
        (2, (0.0, 0.0, 1.0, 0.0)),
        (3, (0.0, 0.0, 0.0, 1.0)),
    ):
        for _ in range(12):
            sc = sample_curriculum_scenario(generation=3, tier_weights=force_weights, rng=rng)
            assert sc.tier == tier
            assert sc.bc_masks.shape == (3, *sc.resolution)
            assert sc.forces.shape == (3, *sc.resolution)
            assert float(np.abs(sc.forces).sum()) > 0.0
            assert float(sc.bc_masks.sum()) > 0.0
            if tier == 0:
                assert not is_near_canonical_scenario(sc.load_config, sc.bc_type, tol_xy=1)
            if sc.domain_mask is not None:
                seen_keepout = True
                assert sc.domain_mask.shape == sc.resolution
            seen_res.add(sc.resolution)
    assert seen_keepout is True
    assert len(seen_res) >= 2

    throttled = adjust_tier_weights((0.25, 0.25, 0.25, 0.25), prev_tier_conn_rates={1: 0.40}, min_conn_rate=0.65)
    assert throttled[3] < 0.05
    assert throttled[0] > 0.25
    assert throttled[1] > 0.25

    ood = build_ood_benchmark_scenarios()
    assert [s["id"] for s in ood] == [
        "G1_dual_pad_angled",
        "G2_bridge_mbb",
        "G3_keepout_window",
        "G4_thick_3d_48x24x12",
    ]
    assert ood[2]["domain_mask"] is not None
    assert ood[3]["resolution"] == (48, 24, 12)


def test_reward_general_bc_connectivity_and_pbrs_distance():
    """Verify connectivity & PBRS distance shaping work cleanly for arbitrary custom BCs/3D forces."""
    res = (16, 8, 4)
    bc = np.zeros((3, *res), dtype=np.float32)
    bc[:, 0, 0:3, :] = 1.0
    bc[1, 15, 0:3, :] = 1.0
    forces = np.zeros((3, *res), dtype=np.float32)
    forces[1, 8, 6, 1:3] = -1.0
    cfg = {"x": 8, "y": 6, "z_start": 1, "z_end": 3, "bc_type": "BRIDGE_MBB"}

    near = np.zeros(res, dtype=np.float32)
    near[0:7, 1, 1:3] = 1.0

    far = np.zeros(res, dtype=np.float32)
    far[0:2, 1, 1:3] = 1.0

    sup_far, load_far = check_structure_connectivity(far, cfg, bc_masks=bc, forces=forces)
    sup_near, load_near = check_structure_connectivity(near, cfg, bc_masks=bc, forces=forces)
    assert sup_far is True and load_far is False
    assert sup_near is True and load_near is False

    p_far, _ = calculate_pbrs_potential(far, tuple(bc), tuple(forces), cfg)
    p_near, _ = calculate_pbrs_potential(near, tuple(bc), tuple(forces), cfg)
    assert p_near > p_far

    solid = np.ones(res, dtype=np.float32)
    sup_full, load_full = check_structure_connectivity(solid, cfg, bc_masks=bc, forces=forces)
    assert sup_full is True and load_full is True


def test_storage_merge_runner_and_training_dryrun(tmp_path: Path):
    """Verify bc_type propagation, fem_probes, merge_selfplay_dbs, Dataset, Sampler, loss, and 1-step train."""
    # 1) Create tiny PhysUViT3D checkpoint on CPU
    model = PhysUViT3D(in_channels=7, out_channels=2, feature_size=16, embed_dim=64, transformer_depth=1, num_heads=4)
    ckpt_path = tmp_path / "tiny_phys_uvit.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "architecture": "phys_uvit",
                "in_channels": 7,
                "out_channels": 2,
                "feature_size": 16,
                "embed_dim": 64,
                "transformer_depth": 1,
                "num_heads": 4,
            },
            "epoch": 0,
        },
        ckpt_path,
    )
    infer = AlphaBuilderInference(ckpt_path, device="cpu")

    # 2) Run a fast real episode on (16, 12, 8) RAIL_XY with keep-out domain_mask
    res = (16, 12, 8)
    mx, my, mz = generate_bc_masks(res, "RAIL_XY")
    bc_masks = np.stack([mx, my, mz], axis=0).astype(np.float32)
    forces = np.zeros((3, *res), dtype=np.float32)
    forces[1, 12, 5:7, 3:5] = -1.0
    dm = np.ones(res, dtype=np.float32)
    dm[6:9, 0:3, :] = 0.0
    sc = SelfPlayScenario(
        scenario_id="unit_rail_16",
        tier=2,
        family="rail_keepout",
        resolution=res,
        bc_type="RAIL_XY",
        bc_masks=bc_masks,
        forces=forces,
        domain_mask=dm,
        load_config={"x": 12, "y": 5, "z_start": 3, "z_end": 5, "bc_type": "RAIL_XY"},
    )
    db1 = tmp_path / "shard1.db"
    initialize_selfplay_db(db1)
    ep_cfg = EpisodeConfig(
        max_steps=8,
        max_phase1_steps=4,
        max_phase2_steps=2,
        resolution=res,
        bc_type="RAIL_XY",
        num_simulations_p1=4,
        num_simulations_p2=4,
        p1_explore_steps=2,
        p2_action_batch_size=4,
        p2_action_batch_size_late=2,
        p2_min_remove_candidates=4,
        p2_remove_probe_prob=0.5,
        fem_backend="fast_amg",
        record_sed_map=True,
        seed=7,
        verbose=False,
    )
    gid1, score1, steps1, stats1 = run_episode_with_stats(infer, db1, ep_cfg, scenario=sc)
    assert steps1 >= 1
    assert np.isfinite(score1)
    assert stats1["bc_type"] == "RAIL_XY"
    loaded_g1 = load_game(db1, gid1)
    assert loaded_g1 is not None
    assert loaded_g1.bc_type == "RAIL_XY"
    assert loaded_g1.domain_mask is not None
    assert loaded_g1.domain_mask[7, 1, 0] == pytest.approx(0.0)

    # 3) Write a second shard at different resolution (16, 12, 12) with Phase-2 SED + fem_probe, then merge
    res2 = (16, 12, 12)
    db2 = tmp_path / "shard2.db"
    initialize_selfplay_db(db2)
    gid2 = generate_game_id()
    mx2, my2, mz2 = generate_bc_masks(res2, "FULL_CLAMP")
    bc2 = np.stack([mx2, my2, mz2], axis=0).astype(np.float32)
    f2 = np.zeros((3, *res2), dtype=np.float32)
    f2[1, 15, 5:7, 5:7] = -1.0
    cfg2 = {"x": 15, "y": 5, "z_start": 5, "z_end": 7, "bc_type": "FULL_CLAMP"}
    save_game(
        db2,
        GameInfo(
            game_id=gid2,
            neural_engine="phys_uvit",
            checkpoint_version="unit-T3",
            bc_masks=bc2,
            forces=f2,
            load_config=cfg2,
            bc_type="FULL_CLAMP",
            resolution=res2,
            domain_mask=np.ones(res2, dtype=np.float32),
        ),
    )
    solid2 = _make_cantilever_bar(res2, hole=False)
    fem2 = evaluate_fem_fast(solid2, cfg2, bc_masks=bc2, forces=f2)
    assert fem2.valid and fem2.sed_map is not None

    # Minimal StateSnapshot / SearchResult objects compatible with record_step
    class _Snap:
        def __init__(self):
            self.game_id = gid2
            self.current_step = 0
            self.phase = Phase.REFINEMENT
            self.density = solid2

    class _SR:
        def __init__(self):
            self.visit_distribution = {(1, 8, 3, 2): 10, (0, 8, 2, 2): 4}
            self.root_value = 0.25
            self.num_simulations = 14
            self.root = None

    p_add = np.zeros(res2, dtype=np.float32)
    p_rem = np.zeros(res2, dtype=np.float32)
    p_add[8, 2, 2] = 0.2
    p_rem[8, 3, 2] = 0.8
    record_step(
        db2,
        _Snap(),
        _SR(),
        0.25,
        p_add,
        p_rem,
        island_analysis={"n_islands": 1, "loose_voxels": 0, "is_connected": True},
        compliance_fem=fem2.compliance,
        max_displacement=fem2.max_displacement,
        volume_fraction=fem2.vol_frac,
        executed_actions=[(1, 8, 3, 2)],
        fem_reward=0.42,
        sed_map=fem2.sed_map,
    )
    backpropagate_value_targets(db2, gid2)
    update_game_final(db2, gid2, 0.42, fem2.compliance, fem2.vol_frac, 1)
    record_fem_probe(
        db2,
        game_id=gid2,
        step=0,
        probe_type="sed_remove",
        density=solid2,
        is_connected=True,
        compliance_fem=fem2.compliance,
        max_displacement=fem2.max_displacement,
        volume_fraction=fem2.vol_frac,
        fem_reward=0.45,
        sed_map=fem2.sed_map,
    )

    merged_db = tmp_path / "merged.db"
    n_g, n_s, n_p = merge_selfplay_dbs([db1, db2], merged_db)
    assert n_g == 2 and n_s >= 2 and n_p == 1

    # 4) Verify SelfPlayDataset + MixedReplayDataset + BucketedResolutionBatchSampler + collate rot90 + 1 train step
    sp_ds = SelfPlayDataset([merged_db], augment=True, include_probes=True)
    assert len(sp_ds) >= 3
    mixed = MixedReplayDataset(sp_ds, warmup_ds=None, selfplay_fraction=1.0, min_epoch_samples=len(sp_ds), seed=42)
    mixed.set_epoch(0)
    sampler = BucketedResolutionBatchSampler(mixed, batch_size=2, shuffle=True, drop_last=False, seed=42)
    batches = list(sampler)
    assert len(batches) >= 2
    collate_rot = make_symmetry_collate_fn(enable_rot90=True)
    collate_plain = make_symmetry_collate_fn(enable_rot90=False)

    # Confirm all samples in a batch share identical spatial dims (no mixed resolutions)
    for idxs in batches:
        sample_list = [mixed[i] for i in idxs]
        shapes = {tuple(s["state"].shape[1:]) for s in sample_list}
        assert len(shapes) == 1
        batch = collate_plain(sample_list)
        assert batch["domain_mask"].shape[1:] == (1, *next(iter(shapes)))
        assert batch["sed_target"].shape[1:] == (1, *next(iter(shapes)))

    # Verify rot90 keeps domain_mask and sed_target aligned with rotated state
    sample_fixed = sp_ds[0]
    torch.manual_seed(0)
    batch_rot = collate_rot([sample_fixed])
    assert batch_rot["domain_mask"].shape[2:] == batch_rot["state"].shape[2:]
    assert batch_rot["sed_target"].shape[2:] == batch_rot["state"].shape[2:]

    # 5) 1-step PhysUViT3D training forward + backward with SED loss
    train_model = PhysUViT3D(in_channels=7, out_channels=2, feature_size=16, embed_dim=64, transformer_depth=1, num_heads=4)
    opt = torch.optim.AdamW(train_model.parameters(), lr=1e-3)
    b_res2 = collate_plain([mixed[i] for i in range(len(mixed)) if mixed.get_sample_resolution(i) == res2])
    assert b_res2["sed_mask"].sum().item() >= 1.0

    w_before = train_model.policy_head[-1].weight.detach().clone()
    pred_pol, pred_val, aux = train_model(
        b_res2["state"], domain_mask=b_res2["domain_mask"], return_aux=True, compute_sed=True
    )
    loss_tensor, metrics = compute_total_loss(
        pred_policy=pred_pol,
        pred_value=pred_val,
        target_policy=b_res2["policy"],
        target_value=b_res2["value"],
        legal_mask=b_res2["legal_mask"],
        policy_mask=b_res2["policy_mask"],
        pred_log_c=aux["log_compliance"],
        target_log_c=b_res2["log_compliance"],
        compliance_mask=b_res2["compliance_mask"],
        pred_conn_logit=aux["conn_logit"],
        target_conn=b_res2["conn_target"],
        conn_mask=b_res2["conn_mask"],
        pred_sed_u=aux["sed_u"],
        target_sed=b_res2["sed_target"],
        sed_mask=b_res2["sed_mask"],
        lambda_bp=0.0,
        lambda_log_c=0.10,
        lambda_conn=0.10,
        lambda_sed=0.25,
    )
    assert torch.isfinite(loss_tensor)
    assert metrics["sed_loss"] >= 0.0
    opt.zero_grad()
    loss_tensor.backward()
    opt.step()
    assert not torch.equal(w_before, train_model.policy_head[-1].weight.detach())


def test_sed_loss_masking():
    """Verify sed_loss returns finite scalar and respects zero mask safely."""
    pred = torch.randn(2, 4, 8, 8, 4, requires_grad=True)
    tgt = torch.rand(2, 1, 8, 8, 4)
    zero_mask = torch.zeros(2)
    assert float(sed_loss(pred, tgt, zero_mask).item()) == pytest.approx(0.0)
    full_mask = torch.ones(2)
    assert float(sed_loss(pred, tgt, full_mask).item()) > 0.0


def test_stage6_notebook_valid_json_and_cells():
    """Validate Kaggle Stage 6 notebook structure and zero hardcoded secrets."""
    nb_path = REPO_ROOT / "notebooks" / "alphabuilder-v4-stage6-selfplay.ipynb"
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 4
    all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert "run_selfplay_pipeline.py" in all_src
    assert "pyamg" in all_src
    assert "KGAT_" not in all_src
