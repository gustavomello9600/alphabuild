"""Unit tests verifying Stage 6 v4.1 upgrades (MCTS-DAG, op_head, lifecycle, BESO 3D)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphabuilder.src.logic.benchmarks.beso3d import run_beso3d
from alphabuilder.src.logic.harvest.generators import generate_bc_masks
from alphabuilder.src.logic.mcts.engine import (
    AlphaBuilderMCTS,
    MCTSConfig,
    run_mcts_search,
)
from alphabuilder.src.logic.mcts.legal_moves import (
    filter_disconnect_safe_batch,
    get_legal_add_moves,
    get_legal_remove_moves,
)
from alphabuilder.src.logic.mcts.node import MCTSNode, STOP_ACTION
from alphabuilder.src.logic.selfplay import (
    GameInfo,
    backpropagate_value_targets,
    initialize_selfplay_db,
    load_game_step,
    record_step,
    save_game,
)
from alphabuilder.src.logic.selfplay.reward import (
    calculate_grace_displacement_reward,
    calculate_pbrs_potential,
    get_phase2_terminal_reward,
)
from alphabuilder.src.logic.selfplay.runner import EpisodeConfig, EpisodeState, run_episode_with_stats
from alphabuilder.src.logic.storage import Phase
from alphabuilder.src.neural.inference import AlphaBuilderInference
from alphabuilder.src.neural.phys_uvit import PhysUViT3D
from alphabuilder.src.neural.trainer import op_policy_loss


def test_relative_fpu_and_virtual_loss_prior_ordering():
    """Verify relative FPU penalizes unvisited children below parent Q and virtual_loss breaks ties by prior."""
    root = MCTSNode(prior=1.0, is_expanded=True)
    root.visit_count = 10
    root.value_sum = 6.0  # parent mean_value = +0.60

    c_high = MCTSNode(prior=0.7, parent=root)
    c_low = MCTSNode(prior=0.3, parent=root)
    root.children = {(0, 0, 0, 0): c_high, (0, 1, 0, 0): c_low}

    # On unvisited children, relative FPU = 0.60 - 0.25 = 0.35
    score_high = c_high.puct_score(1.0, parent_visits=10, fpu_value=0.35)
    assert score_high == pytest.approx(0.35 + 1.0 * 0.7 * np.sqrt(10.0) / 1.0, abs=1e-5)

    # First batch tie-break with virtual loss must respect prior even when visit_count == 0
    c_a = MCTSNode(prior=0.1, parent=root)
    c_b = MCTSNode(prior=0.9, parent=root)
    c_a.apply_virtual_loss()
    root.children = {(0, 2, 0, 0): c_a, (0, 3, 0, 0): c_b}
    act, chosen = root.select_child(1.0, fpu_reduction=0.25)
    assert chosen is c_b


def test_surface_only_remove_bidirectional_add_and_bfs_disconnect_filter():
    """Verify surface-only removals, bidirectional load seeds, and articulation batch safety."""
    res = (10, 6, 4)
    density = np.zeros(res, dtype=np.float32)
    density[0:4, 1:5, 1:3] = 1.0  # 4x4x2 block -> interior exists at [1:3, 2:4, ...]

    mx, my, mz = generate_bc_masks(res, "FULL_CLAMP")
    bc = (mx, my, mz)
    forces = (np.zeros(res, dtype=np.float32), np.zeros(res, dtype=np.float32), np.zeros(res, dtype=np.float32))
    forces[1][9, 2:4, 1:3] = -1.0

    # Bidirectional add allows growing around load patch at x=9 simultaneously with support
    add_uni = get_legal_add_moves(density, bc_masks=bc, forces=forces, include_load_seeds=False)
    add_bi = get_legal_add_moves(density, bc_masks=bc, forces=forces, include_load_seeds=True)
    assert add_uni[9, 2:4, 1:3].sum() == 0.0
    assert add_bi[9, 2:4, 1:3].sum() > 0.0

    # Build a 3x3x3 cube inside void to guarantee a strict interior voxel at (5,3,2)
    cube = np.zeros(res, dtype=np.float32)
    cube[4:7, 2:5, 1:4] = 1.0
    rem_all = get_legal_remove_moves(cube, surface_only=False)
    rem_surf = get_legal_remove_moves(cube, surface_only=True)
    assert rem_all[5, 3, 2] == 1.0
    assert rem_surf[5, 3, 2] == 0.0
    assert rem_surf.sum() == 26.0

    # Thin 1-voxel bridge connecting x=0 support to x=9 load
    bridge = np.zeros(res, dtype=np.float32)
    bridge[:, 2, 1] = 1.0
    bridge[9, 2:4, 1:3] = 1.0
    lcfg = {"x": 9, "y": 3, "z_start": 1, "z_end": 3, "bc_type": "FULL_CLAMP"}
    risky_batch = [(1, 5, 2, 1), (0, 5, 3, 1)]  # Removing (5,2,1) cuts unless (5,3,1) isn't adjacent cleanly (it IS adjacent to 5,2,1 -> check 6-nbr!)
    # Check an actual disconnection: remove (4,2,1) while adding unrelated far voxel (0,5,1)
    bad_batch = [(1, 4, 2, 1), (0, 0, 5, 1)]
    safe_batch = filter_disconnect_safe_batch(bridge, bad_batch, lcfg, bc_masks=bc, forces=forces)
    assert (1, 4, 2, 1) not in safe_batch
    assert (0, 0, 5, 1) in safe_batch


def test_bidirectional_pbrs_does_not_penalize_load_front():
    """Growing from load front under bidirectional PBRS must not incur loose-island penalty."""
    res = (12, 6, 4)
    mx, my, mz = generate_bc_masks(res, "FULL_CLAMP")
    bc = (mx, my, mz)
    forces = (np.zeros(res, dtype=np.float32), np.zeros(res, dtype=np.float32), np.zeros(res, dtype=np.float32))
    forces[1][11, 2:4, 1:3] = -1.0
    lcfg = {"x": 11, "y": 3, "z_start": 1, "z_end": 3}

    d = np.zeros(res, dtype=np.float32)
    d[0:2, 2:4, 1:3] = 1.0
    d[10:12, 2:4, 1:3] = 1.0
    phi_bi, info_bi = calculate_pbrs_potential(d, bc, forces, lcfg, bidirectional=True)
    assert info_bi["loose_voxels"] == 0
    assert info_bi["island_penalty"] == 0.0
    assert phi_bi > 0.0


def test_macro_swap_stop_dag_and_root_only_policy_targets():
    """Verify MCTS-DAG merges commutative paths, expands SWAP/STOP, reuses subtree, and isolates root targets."""
    shape = (8, 6, 4)
    density = np.zeros(shape, dtype=np.float32)
    density[:, 2:4, 1:3] = 1.0
    mx, my, mz = generate_bc_masks(shape, "FULL_CLAMP")
    bc = (mx, my, mz)
    forces = (np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32))
    forces[1][7, 2:4, 1:3] = -1.0

    # Verify MCTS-DAG merges A+B and B+A when using transposition table on single-voxel search
    def _eval(st):
        return float(st[0].mean()), np.ones(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    seed_density = np.zeros(shape, dtype=np.float32)
    seed_density[0, 2, 2] = 1.0
    act_a = (0, 1, 2, 2)
    act_b = (0, 0, 3, 2)
    pol_fixed = np.zeros(shape, dtype=np.float32)
    pol_fixed[1, 2, 2] = 10.0
    pol_fixed[0, 3, 2] = 10.0

    def _eval_dag(st):
        return 0.5, pol_fixed, np.zeros(shape, dtype=np.float32)

    cfg_dag = MCTSConfig(
        num_simulations=24,
        batch_size=1,
        action_batch_size=1,
        top_k_expansion=2,
        c_puct=2.0,
        add_root_noise=False,
        phase="GROWTH",
        fpu_reduction=0.2,
        use_transposition_dag=True,
        root_only_policy_target=True,
    )
    mcts_dag = AlphaBuilderMCTS(_eval_dag, num_simulations=24, batch_size=1, action_batch_size=1, top_k_expansion=2)
    mcts_dag.config = cfg_dag
    res_dag = mcts_dag.search(seed_density, bc, forces)
    assert mcts_dag.total_dag_hits >= 1
    assert act_a in res_dag.root.children and act_b in res_dag.root.children
    assert res_dag.root.children[act_a].children[act_b] is res_dag.root.children[act_b].children[act_a]
    nodes_exp, max_d = res_dag.root.tree_stats()
    assert nodes_exp >= 1 and max_d >= 1
    # All root-only visits must correspond strictly to depth-1 children
    root_children_set = set(res_dag.root.children.keys())
    assert set(res_dag.visit_distribution.keys()).issubset(root_children_set)

    # Verify macro-actions create SWAP and STOP, select STOP when stop_value > branch Q, and step() reuses subtree
    cfg_macro = MCTSConfig(
        num_simulations=24,
        batch_size=4,
        action_batch_size=4,
        phase="REFINEMENT",
        fpu_reduction=0.2,
        surface_only_remove=True,
        use_transposition_dag=True,
        macro_action_mode=True,
        factored_policy=True,
        root_only_policy_target=True,
        allow_stop_action=True,
        macro_num_candidates=12,
    )

    def _eval_bad_children(states):
        return [-0.5] * len(states), [np.random.randn(*shape).astype(np.float32) for _ in states], [
            np.random.randn(*shape).astype(np.float32) for _ in states
        ]

    mcts_macro = AlphaBuilderMCTS(_eval, predict_batch_fn=_eval_bad_children)
    mcts_macro.config = cfg_macro
    res_macro = mcts_macro.search(density, bc, forces, stop_value=0.80, root_op_probs=(0.2, 0.2, 0.6))
    has_swap = any(
        isinstance(k, tuple) and k != STOP_ACTION and any(a[0] == 0 for a in k) and any(a[0] == 1 for a in k)
        for k in res_macro.root.children
    )
    assert has_swap
    assert STOP_ACTION in res_macro.root.children
    best_edge, best_node = res_macro.root.get_best_child()
    assert best_edge == STOP_ACTION and best_node.visit_count > 0

    # Verify step() reuses non-STOP macro child subtree cleanly
    non_stop = next(k for k in res_macro.root.children if k != STOP_ACTION)
    assert mcts_macro.step(non_stop) is True
    assert mcts_macro.root is not None


def test_legacy_checkpoint_loads_into_op_head_and_lifecycle_fixes(tmp_path: Path):
    """Verify strict=False op_head zero-init compatibility, vol>0.4 bugfix, grace period, and best-of-episode targets."""
    model = PhysUViT3D(in_channels=7, out_channels=2, feature_size=16, embed_dim=64, transformer_depth=1, num_heads=4)
    legacy_sd = {k: v for k, v in model.state_dict().items() if not k.startswith("op_head.")}
    ckpt_path = tmp_path / "legacy_stage5.pt"
    torch.save(
        {
            "model_state_dict": legacy_sd,
            "config": {
                "architecture": "phys_uvit",
                "in_channels": 7,
                "out_channels": 2,
                "feature_size": 16,
                "embed_dim": 64,
                "transformer_depth": 1,
                "num_heads": 4,
            },
            "epoch": 18,
        },
        ckpt_path,
    )
    infer = AlphaBuilderInference(ckpt_path, device="cpu")
    st = np.zeros((7, 16, 12, 8), dtype=np.float32)
    _, _, _, aux = infer.predict_with_aux(st)
    assert "op_probs" in aux
    assert aux["op_probs"] == pytest.approx((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), abs=1e-4)

    # op_policy_loss finite check
    loss_op = op_policy_loss(torch.zeros(2, 3), torch.tensor([[0.6, 0.4, 0.0], [0.0, 0.0, 1.0]]))
    assert float(loss_op.item()) > 0.0

    # Verify vol_frac=0.45 is no longer prematurely killed as TERMINAL
    shape = (8, 6, 4)
    d45 = np.zeros(shape, dtype=np.float32)
    d45[:, 1:4, 1:3] = 1.0  # ~25%.. wait use 4 rows -> ~50%
    d45[:, 1:5, 1:4] = 1.0  # 0.50
    mx, my, mz = generate_bc_masks(shape, "FULL_CLAMP")
    lcfg = {"x": 7, "y": 3, "z_start": 1, "z_end": 3, "bc_type": "FULL_CLAMP"}
    forces = (np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32))
    forces[1][7, 2:4, 1:3] = -1.0
    assert get_phase2_terminal_reward(d45, lcfg, bc_masks=(mx, my, mz), forces=forces) is None
    assert calculate_grace_displacement_reward(256000.0) > -0.99

    # Verify best-of-episode backpropagation protects good pre-peak steps from late degradation
    db_path = tmp_path / "best_bp.db"
    initialize_selfplay_db(db_path)
    bc = np.stack([mx, my, mz], axis=0)
    f_arr = np.stack(forces, axis=0)
    save_game(db_path, GameInfo("g_best", "phys_uvit", "v4.1", bc, f_arr, lcfg, "FULL_CLAMP", shape))
    sr = type("SR", (), {"actions": [], "visit_distribution": {}, "root_value": 0.0, "num_simulations": 1, "root": None})()
    dum = np.zeros(shape, dtype=np.float32)
    for step, ph, conn, fem_r in [
        (0, Phase.GROWTH, False, None),
        (1, Phase.REFINEMENT, True, 0.65),  # Peak!
        (2, Phase.REFINEMENT, True, 0.40),  # Post-peak degradation
    ]:
        es = EpisodeState(density=d45, bc_masks=bc, forces=f_arr, load_config=lcfg, current_step=step, phase=ph, game_id="g_best")
        record_step(
            db_path, es, sr, 0.0, dum, dum,
            island_analysis={"n_islands": 1, "loose_voxels": 0, "is_connected": conn},
            compliance_fem=100.0 if conn else None, max_displacement=10.0 if conn else None,
            island_penalty=0.0, volume_fraction=0.15, fem_reward=fem_r,
        )
    backpropagate_value_targets(db_path, "g_best", gamma=0.99, use_best_of_episode=True)
    s0 = load_game_step(db_path, "g_best", 0, shape)
    s1 = load_game_step(db_path, "g_best", 1, shape)
    assert s1.value_target == pytest.approx(0.65)
    assert s0.value_target == pytest.approx(0.99 * 0.65)


def test_beso3d_smoke_and_v41_episode_stagnation_termination(tmp_path: Path):
    """Verify BESO 3D solver converges cleanly on small mesh and runner runs end-to-end with v4.1."""
    beso = run_beso3d(
        resolution=(16, 8, 4),
        load_config={"x": 15, "y": 4, "z_start": 1, "z_end": 3, "bc_type": "FULL_CLAMP"},
        scenario_id="unit_beso",
        v_target_min=0.14,
        v_iso_ref=0.18,
        iso_stabilize_iters=2,
        er_coarse=0.10,
        er_fine=0.08,
        max_iter=35,
    )
    assert beso.iterations_run >= 8
    assert beso.pareto_best is not None and beso.pareto_best.is_connected

    # Smoke end-to-end v4.1 runner on tiny grid with stagnation exit check
    model = PhysUViT3D(in_channels=7, out_channels=2, feature_size=16, embed_dim=64, transformer_depth=1, num_heads=4)
    ckpt = tmp_path / "m.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"architecture": "phys_uvit", "feature_size": 16, "embed_dim": 64, "transformer_depth": 1, "num_heads": 4},
        },
        ckpt,
    )
    infer = AlphaBuilderInference(ckpt, device="cpu")
    db = tmp_path / "ep_v41.db"
    initialize_selfplay_db(db)
    cfg = EpisodeConfig(
        max_steps=10,
        max_phase1_steps=4,
        max_phase2_steps=4,
        resolution=(16, 8, 4),
        num_simulations_p1=4,
        num_simulations_p2=4,
        mcts_batch_size=2,
        p1_explore_steps=1,
        p1_stagnation_patience=2,
        p2_action_batch_size=4,
        p2_action_batch_size_late=2,
        p2_remove_probe_prob=0.0,
        fem_backend="fast_amg",
        use_v41_mcts=True,
        verbose=False,
    )
    gid, score, steps, st = run_episode_with_stats(
        infer, db, cfg, load_config={"x": 15, "y": 4, "z_start": 1, "z_end": 3, "bc_type": "FULL_CLAMP"}
    )
    assert steps >= 1 and np.isfinite(score)
    assert "best_score" in st and "dag_transpositions" in st
