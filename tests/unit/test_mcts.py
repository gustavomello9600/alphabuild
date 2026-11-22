"""
Unit tests for MCTS engine, node pruning normalization, protected remove moves,
batch inference forwarding, and faithful multi-branch PV recording.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from alphabuilder.src.logic.mcts.engine import (
    AlphaBuilderMCTS,
    MCTSConfig,
    PHASE1_CONFIG,
    PHASE2_CONFIG,
    SearchResult,
    apply_action,
    apply_actions,
    build_state_tensor,
    run_mcts_search,
    select_pv_branches,
)
from alphabuilder.src.logic.mcts.legal_moves import (
    apply_action_mask,
    count_legal_moves,
    get_legal_add_moves,
    get_legal_moves,
    get_legal_remove_moves,
    is_terminal_state,
    masked_softmax,
)
from alphabuilder.src.logic.mcts.node import MCTSNode
from alphabuilder.src.logic.selfplay.runner import EpisodeState
from alphabuilder.src.logic.selfplay.storage import (
    GameInfo,
    initialize_selfplay_db,
    load_game_step,
    record_step,
    save_game,
)
from alphabuilder.src.logic.storage import Phase


class TestMCTSNode:
    def test_initial_state(self):
        node = MCTSNode(prior=0.5)
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.prior == 0.5
        assert not node.is_expanded
        assert node.children == {}

    def test_mean_value_zero_visits(self):
        node = MCTSNode()
        assert node.mean_value == 0.0

    def test_mean_value_after_updates(self):
        node = MCTSNode()
        node.update(0.8)
        node.update(0.6)
        node.update(0.4)
        assert node.visit_count == 3
        assert node.value_sum == pytest.approx(1.8)
        assert node.mean_value == pytest.approx(0.6)

    def test_puct_score_exploration_and_exploitation(self):
        node = MCTSNode(prior=0.5)
        score = node.puct_score(c_puct=1.25, parent_visits=100)
        assert score == pytest.approx(1.25 * 0.5 * np.sqrt(100) / 1.0)

        node2 = MCTSNode(prior=0.1)
        node2.update(0.9)
        node2.update(0.9)
        node2.update(0.9)
        score2 = node2.puct_score(c_puct=1.25, parent_visits=10)
        assert score2 == pytest.approx(0.9 + 1.25 * 0.1 * np.sqrt(10) / 4.0)

    def test_select_child(self):
        root = MCTSNode()
        child1 = MCTSNode(prior=0.1, parent=root)
        child2 = MCTSNode(prior=0.8, parent=root)
        child3 = MCTSNode(prior=0.1, parent=root)
        child2.update(0.5)
        root.children = {(0, 0, 0, 0): child1, (0, 1, 0, 0): child2, (0, 2, 0, 0): child3}
        root.is_expanded = True
        action, selected = root.select_child(c_puct=1.25)
        assert action == (0, 1, 0, 0)
        assert selected is child2

    def test_expand_renormalizes_after_topk(self):
        node = MCTSNode()
        policy_add = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        policy_remove = np.zeros((2, 2, 2), dtype=np.float32)
        valid_add = np.ones((2, 2, 2), dtype=np.float32)
        valid_remove = np.zeros((2, 2, 2), dtype=np.float32)

        node.expand(policy_add, policy_remove, valid_add, valid_remove, top_k=3)
        assert node.is_expanded
        assert len(node.children) == 3
        total_prior = sum(c.prior for c in node.children.values())
        assert total_prior == pytest.approx(1.0, abs=1e-6)

    def test_get_top_k_actions(self):
        root = MCTSNode()
        for i in range(5):
            child = MCTSNode(parent=root)
            child.visit_count = 10 - i
            root.children[(0, i, 0, 0)] = child
        root.is_expanded = True
        assert root.get_top_k_actions(3) == [(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0)]


class TestLegalMoves:
    def test_legal_add_moves_empty_grid(self):
        density = np.zeros((4, 4, 4))
        valid_add = get_legal_add_moves(density)
        assert valid_add.sum() == 32
        assert valid_add[0:2, :, :].sum() == 32

        assert get_legal_add_moves(density, include_supports=False).sum() == 0

    def test_legal_add_moves_single_voxel(self):
        density = np.zeros((5, 5, 5))
        density[2, 2, 2] = 1.0
        valid_add = get_legal_add_moves(density)
        assert valid_add.sum() == 55
        assert valid_add[2, 2, 2] == 0
        assert get_legal_add_moves(density, include_supports=False).sum() == 6

    def test_legal_remove_moves_and_protection(self):
        density = np.zeros((4, 4, 4))
        density[0, 1, 1] = 1.0  # Support BC voxel
        density[1, 1, 1] = 1.0
        density[3, 2, 2] = 1.0  # Loaded voxel
        density[0, 0, 0] = 0.3

        assert get_legal_remove_moves(density, threshold=0.5).sum() == 3

        bc = (np.zeros_like(density), np.zeros_like(density), np.zeros_like(density))
        bc[0][0, :, :] = 1.0
        forces = (np.zeros_like(density), np.zeros_like(density), np.zeros_like(density))
        forces[1][3, 2, 2] = -1.0

        protected_rem = get_legal_remove_moves(density, threshold=0.5, bc_masks=bc, forces=forces)
        assert protected_rem.sum() == 1
        assert protected_rem[1, 1, 1] == 1.0
        assert protected_rem[0, 1, 1] == 0.0
        assert protected_rem[3, 2, 2] == 0.0

    def test_apply_action_mask_and_masked_softmax(self):
        policy = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        valid = np.array([[[1, 0], [1, 0]]])
        masked, _ = apply_action_mask(policy, np.zeros_like(policy), valid, np.zeros_like(valid))
        assert masked[0, 0, 1] == -np.inf

        prob_add, prob_rem = masked_softmax(
            np.ones((2, 2, 2)) * 0.5,
            np.ones((2, 2, 2)) * 0.5,
            np.ones((2, 2, 2)),
            np.zeros((2, 2, 2)),
        )
        assert prob_add.sum() + prob_rem.sum() == pytest.approx(1.0)
        assert prob_rem.sum() == 0.0

    def test_terminal_and_count(self):
        is_term, reason = is_terminal_state(np.ones((10, 10, 10)) * 0.9, max_volume_fraction=0.3)
        assert is_term and reason == "max_volume_exceeded"

        d_small = np.zeros((10, 10, 10))
        d_small[0, 0, 0] = 1.0
        is_term2, reason2 = is_terminal_state(d_small, min_volume_fraction=0.01)
        assert is_term2 and reason2 == "structure_too_small"

        d_cube = np.zeros((4, 4, 4))
        d_cube[1:3, 1:3, 1:3] = 1.0
        na, nr = count_legal_moves(d_cube)
        assert nr == 8 and na > 0


class TestEngineAndBlocoBFixes:
    def test_apply_action_add(self):
        density = np.zeros((4, 4, 4))
        result = apply_action(density, (0, 1, 2, 3))
        assert result[1, 2, 3] == 1.0
        assert density[1, 2, 3] == 0.0

    def test_apply_action_remove(self):
        density = np.ones((4, 4, 4))
        result = apply_action(density, (1, 1, 2, 3))
        assert result[1, 2, 3] == 0.0
        assert density[1, 2, 3] == 1.0

    def test_apply_actions_batch(self):
        density = np.zeros((4, 4, 4))
        result = apply_actions(density, [(0, 0, 0, 0), (0, 1, 1, 1), (0, 2, 2, 2)])
        assert result[0, 0, 0] == 1.0
        assert result[1, 1, 1] == 1.0
        assert result[2, 2, 2] == 1.0

    def test_build_state_tensor_shape(self):
        D, H, W = 8, 6, 4
        density = np.random.rand(D, H, W)
        bc = (np.zeros((D, H, W)), np.zeros((D, H, W)), np.zeros((D, H, W)))
        forces = (np.zeros((D, H, W)), np.zeros((D, H, W)), np.zeros((D, H, W)))
        state = build_state_tensor(density, bc, forces)
        assert state.shape == (7, D, H, W)
        assert state.dtype == np.float32

    def test_configs_aligned_with_spec(self):
        assert PHASE1_CONFIG.min_volume_fraction == 0.0
        assert PHASE2_CONFIG.action_batch_size == 8

    def test_phase1_runs_from_empty_grid(self):
        density = np.zeros((8, 6, 4), dtype=np.float32)
        bc_masks = (np.ones_like(density), np.zeros_like(density), np.zeros_like(density))
        bc_masks[0][1:, :, :] = 0.0
        forces = (np.zeros_like(density), np.zeros_like(density), np.zeros_like(density))

        def mock_predict(state):
            return 0.25, np.ones((8, 6, 4), dtype=np.float32), np.zeros((8, 6, 4), dtype=np.float32)

        cfg = PHASE1_CONFIG._replace(num_simulations=16, batch_size=1)
        res = run_mcts_search(density, bc_masks, forces, mock_predict, cfg)
        assert res.root_value > -0.9
        assert len(res.actions) > 0

    def test_wrapper_invokes_predict_batch_fn(self):
        density = np.zeros((8, 8, 8), dtype=np.float32)
        density[3:5, 3:5, 3:5] = 1.0
        bc = (np.zeros_like(density), np.zeros_like(density), np.zeros_like(density))
        forces = (np.zeros_like(density), np.zeros_like(density), np.zeros_like(density))

        calls = {"single": 0, "batch": 0}

        def single_fn(state):
            calls["single"] += 1
            return 0.5, np.ones_like(density), np.ones_like(density)

        def batch_fn(states):
            calls["batch"] += 1
            n = len(states)
            return [0.5] * n, [np.ones_like(density)] * n, [np.ones_like(density)] * n

        mcts = AlphaBuilderMCTS(
            predict_fn=single_fn,
            predict_batch_fn=batch_fn,
            num_simulations=12,
            batch_size=4,
            action_batch_size=4,
        )
        res = mcts.search(density, bc, forces)
        assert calls["batch"] >= 1
        assert len(res.actions) > 0

    def test_record_step_multi_branch_pv_stats(self):
        root = MCTSNode(prior=1.0, is_expanded=True)
        a1 = (0, 1, 1, 1)
        a1b = (0, 1, 2, 1)
        a2 = (0, 2, 1, 1)

        c1b = MCTSNode(prior=0.5, action_to_parent=a1b)
        c1b.visit_count = 15
        c1b.value_sum = 7.5
        c1 = MCTSNode(prior=0.6, action_to_parent=a1, parent=root, children={a1b: c1b}, is_expanded=True)
        c1.visit_count = 40
        c1.value_sum = 24.0
        c1b.parent = c1

        c2 = MCTSNode(prior=0.4, action_to_parent=a2, parent=root, is_expanded=True)
        c2.visit_count = 25
        c2.value_sum = 10.0

        root.children = {a1: c1, a2: c2}
        branches = select_pv_branches(root, max_actions=8, max_depth=4)
        assert len(branches) == 2
        flat_actions = [a for br in branches for a, _, _ in br]
        flat_stats = [e for br in branches for e in br]
        assert flat_actions == [a1, a1b, a2]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "selfplay.db"
            initialize_selfplay_db(db_path)
            res = (4, 4, 4)
            bc = np.zeros((3, *res), dtype=np.float32)
            forces = np.zeros((3, *res), dtype=np.float32)
            save_game(
                db_path,
                GameInfo("g1", "simple", "v3.3", bc, forces, {"x": 3, "y": 2, "z_start": 1, "z_end": 3}, "FULL_CLAMP", res),
            )
            state = EpisodeState(density=np.zeros(res, dtype=np.float32), bc_masks=bc, forces=forces, load_config={}, current_step=0, phase=Phase.GROWTH, game_id="g1")
            sr = SearchResult(actions=[a1, a2], visit_distribution=root.collect_subtree_visits(), root_value=0.5, num_simulations=65, root=root)
            record_step(
                db_path,
                state,
                sr,
                value=0.4,
                policy_add=np.ones(res, dtype=np.float32),
                policy_remove=np.zeros(res, dtype=np.float32),
                executed_actions=flat_actions,
                executed_action_stats=flat_stats,
            )
            loaded = load_game_step(db_path, "g1", 0, res)
            assert loaded is not None
            assert [sa.visits for sa in loaded.selected_actions] == [40, 15, 25]
            assert loaded.selected_actions[2].q_value == pytest.approx(0.4)
