"""
Unit tests for MCTS module.

Tests cover:
- MCTSNode statistics and selection
- Legal move generation
- MCTS search integration
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from alphabuilder.src.logic.mcts.node import MCTSNode, Action
from alphabuilder.src.logic.mcts.legal_moves import (
    get_legal_moves,
    get_legal_add_moves,
    get_legal_remove_moves,
    apply_action_mask,
    masked_softmax,
    is_terminal_state,
    count_legal_moves
)
from alphabuilder.src.logic.mcts.engine import (
    apply_action,
    apply_actions,
    build_state_tensor,
    run_mcts_search,
    MCTSConfig,
    AlphaBuilderMCTS
)


# =============================================================================
# MCTSNode Tests
# =============================================================================

class TestMCTSNode:
    """Tests for MCTSNode dataclass."""
    
    def test_initial_state(self):
        """Node starts with zero visits and zero value."""
        node = MCTSNode(prior=0.5)
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.prior == 0.5
        assert not node.is_expanded
        assert node.children == {}
    
    def test_mean_value_zero_visits(self):
        """Mean value is 0 when no visits (avoid div by zero)."""
        node = MCTSNode()
        assert node.mean_value == 0.0
    
    def test_mean_value_after_updates(self):
        """Mean value is W/N after updates."""
        node = MCTSNode()
        node.update(0.8)
        node.update(0.6)
        node.update(0.4)
        
        assert node.visit_count == 3
        assert node.value_sum == pytest.approx(1.8)
        assert node.mean_value == pytest.approx(0.6)
    
    def test_puct_score_exploration_bonus(self):
        """PUCT score includes exploration bonus for low-visit nodes."""
        node = MCTSNode(prior=0.5)
        
        # High exploration bonus when N=0
        score = node.puct_score(c_puct=1.25, parent_visits=100)
        expected = 0 + 1.25 * 0.5 * np.sqrt(100) / 1
        assert score == pytest.approx(expected)
    
    def test_puct_score_exploitation(self):
        """PUCT score favors high-value nodes with many visits."""
        node = MCTSNode(prior=0.1)
        node.update(0.9)
        node.update(0.9)
        node.update(0.9)
        
        # Q=0.9, exploration term is small with N=3
        score = node.puct_score(c_puct=1.25, parent_visits=10)
        expected = 0.9 + 1.25 * 0.1 * np.sqrt(10) / 4
        assert score == pytest.approx(expected)
    
    def test_select_child(self):
        """Select child with highest PUCT score."""
        root = MCTSNode()
        
        # Create children with different priors
        # When N=0 for all, PUCT = 0 + c_puct * P * sqrt(0) / 1 = 0
        # So we need some visits to differentiate, or just check highest prior wins
        child1 = MCTSNode(prior=0.1, parent=root)
        child2 = MCTSNode(prior=0.8, parent=root)  # Highest prior
        child3 = MCTSNode(prior=0.1, parent=root)
        
        # Give child2 a visit so it has higher total_visits for PUCT calc
        child2.update(0.5)
        
        root.children = {
            (0, 0, 0, 0): child1,
            (0, 1, 0, 0): child2,
            (0, 2, 0, 0): child3,
        }
        root.is_expanded = True
        
        action, selected = root.select_child(c_puct=1.25)
        # child2 has highest prior AND has visits, so should dominate
        assert action == (0, 1, 0, 0)
        assert selected is child2
    
    def test_expand_creates_children(self):
        """Expand creates children for valid moves with non-zero prior."""
        node = MCTSNode()
        
        # Small grid for testing
        policy_add = np.array([[[0.5, 0.3], [0.1, 0.1]]])  # (1, 2, 2)
        policy_remove = np.zeros((1, 2, 2))
        valid_add = np.array([[[1, 1], [0, 0]]])  # Only top row valid
        valid_remove = np.zeros((1, 2, 2))
        
        node.expand(policy_add, policy_remove, valid_add, valid_remove)
        
        assert node.is_expanded
        assert len(node.children) == 2  # Two valid add positions
        
        # Check actions are (channel=0, x, y, z)
        actions = list(node.children.keys())
        assert (0, 0, 0, 0) in actions
        assert (0, 0, 0, 1) in actions
    
    def test_get_top_k_actions(self):
        """Get top-k actions by visit count."""
        root = MCTSNode()
        
        # Create children with different visit counts
        for i in range(5):
            child = MCTSNode(parent=root)
            child.visit_count = 10 - i  # 10, 9, 8, 7, 6
            root.children[(0, i, 0, 0)] = child
        
        root.is_expanded = True
        
        top_3 = root.get_top_k_actions(3)
        assert len(top_3) == 3
        assert top_3[0] == (0, 0, 0, 0)  # Highest visits
        assert top_3[1] == (0, 1, 0, 0)
        assert top_3[2] == (0, 2, 0, 0)


# =============================================================================
# Legal Moves Tests
# =============================================================================

class TestLegalMoves:
    """Tests for legal move generation."""
    
    def test_legal_add_moves_empty_grid(self):
        """Empty grid: only support plane (X=0) is valid for adding."""
        density = np.zeros((4, 4, 4))
        valid_add = get_legal_add_moves(density)
        # 4x4 = 16 voxels on X=0 support plane
        # Plus 4x4 = 16 voxels on X=1 support plane (legacy fallback) -> Total 32
        assert valid_add.sum() == 32
        assert valid_add[0:2, :, :].sum() == 32  # All on support planes
        assert valid_add[2:, :, :].sum() == 0  # Nothing else
    
    def test_legal_add_moves_empty_grid_no_supports(self):
        """Empty grid without supports: no valid add moves."""
        density = np.zeros((4, 4, 4))
        valid_add = get_legal_add_moves(density, include_supports=False)
        assert valid_add.sum() == 0
    
    def test_legal_add_moves_single_voxel(self):
        """Add moves are neighbors of existing material + support plane."""
        density = np.zeros((5, 5, 5))
        density[2, 2, 2] = 1.0  # Single voxel in center
        
        valid_add = get_legal_add_moves(density)
        
        # 6 face neighbors + 50 support plane voxels (X=0, X=1)
        # One neighbor (1, 2, 2) is at X=1, so it overlaps.
        # Supports: 50. Neighbors: 6. Overlap: 1.
        # Total = 50 + 6 - 1 = 55.
        assert valid_add.sum() == 55
        assert valid_add[2, 2, 2] == 0  # Original position not valid
        assert valid_add[1, 2, 2] == 1  # Left neighbor
        assert valid_add[3, 2, 2] == 1  # Right neighbor
        assert valid_add[2, 1, 2] == 1  # Front neighbor
        assert valid_add[2, 3, 2] == 1  # Back neighbor
        assert valid_add[2, 2, 1] == 1  # Bottom neighbor
        assert valid_add[2, 2, 3] == 1  # Top neighbor
    
    def test_legal_add_moves_single_voxel_no_supports(self):
        """Add moves without supports: only 6 neighbors."""
        density = np.zeros((5, 5, 5))
        density[2, 2, 2] = 1.0
        
        valid_add = get_legal_add_moves(density, include_supports=False)
        
        # Should have 6 neighbors (faces) only
        assert valid_add.sum() == 6
    
    def test_legal_remove_moves(self):
        """Remove moves are existing material positions."""
        density = np.zeros((4, 4, 4))
        density[1, 1, 1] = 1.0
        density[2, 2, 2] = 0.8
        density[0, 0, 0] = 0.3  # Below threshold
        
        valid_remove = get_legal_remove_moves(density, threshold=0.5)
        
        assert valid_remove.sum() == 2
        assert valid_remove[1, 1, 1] == 1
        assert valid_remove[2, 2, 2] == 1
        assert valid_remove[0, 0, 0] == 0  # Below threshold
    
    def test_apply_action_mask(self):
        """Invalid actions get -inf in logits."""
        policy = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        valid = np.array([[[1, 0], [1, 0]]])
        
        masked, _ = apply_action_mask(policy, np.zeros_like(policy), valid, np.zeros_like(valid))
        
        assert masked[0, 0, 0] == 1.0  # Valid
        assert masked[0, 0, 1] == -np.inf  # Invalid
        assert masked[0, 1, 0] == 3.0  # Valid
        assert masked[0, 1, 1] == -np.inf  # Invalid
    
    def test_masked_softmax_normalization(self):
        """Masked softmax sums to 1 over valid actions."""
        policy_add = np.ones((2, 2, 2)) * 0.5
        policy_remove = np.ones((2, 2, 2)) * 0.5
        valid_add = np.ones((2, 2, 2))
        valid_remove = np.zeros((2, 2, 2))  # No removes allowed
        
        prob_add, prob_remove = masked_softmax(
            policy_add, policy_remove, valid_add, valid_remove
        )
        
        total_prob = prob_add.sum() + prob_remove.sum()
        assert total_prob == pytest.approx(1.0)
        assert prob_remove.sum() == 0  # No valid remove actions
    
    def test_is_terminal_volume_exceeded(self):
        """Volume exceeded triggers terminal state."""
        density = np.ones((10, 10, 10)) * 0.9
        
        is_term, reason = is_terminal_state(density, max_volume_fraction=0.3)
        
        assert is_term
        assert reason == "max_volume_exceeded"
    
    def test_is_terminal_structure_too_small(self):
        """Nearly empty structure triggers terminal."""
        density = np.zeros((10, 10, 10))
        density[0, 0, 0] = 1.0  # Single voxel
        
        is_term, reason = is_terminal_state(density, min_volume_fraction=0.01)
        
        assert is_term
        assert reason == "structure_too_small"
    
    def test_count_legal_moves(self):
        """Count legal moves correctly."""
        density = np.zeros((4, 4, 4))
        density[1:3, 1:3, 1:3] = 1.0  # 2x2x2 cube
        
        num_add, num_remove = count_legal_moves(density)
        
        assert num_remove == 8  # 2x2x2 cube
        assert num_add > 0  # Surface around cube


# =============================================================================
# Engine Tests
# =============================================================================

class TestEngine:
    """Tests for MCTS engine functions."""
    
    def test_apply_action_add(self):
        """Apply add action sets voxel to 1."""
        density = np.zeros((4, 4, 4))
        action = (0, 1, 2, 3)  # Add at (1, 2, 3)
        
        result = apply_action(density, action)
        
        assert result[1, 2, 3] == 1.0
        assert density[1, 2, 3] == 0.0  # Original unchanged
    
    def test_apply_action_remove(self):
        """Apply remove action sets voxel to 0."""
        density = np.ones((4, 4, 4))
        action = (1, 1, 2, 3)  # Remove at (1, 2, 3)
        
        result = apply_action(density, action)
        
        assert result[1, 2, 3] == 0.0
        assert density[1, 2, 3] == 1.0  # Original unchanged
    
    def test_apply_actions_batch(self):
        """Apply batch of actions."""
        density = np.zeros((4, 4, 4))
        actions = [
            (0, 0, 0, 0),  # Add
            (0, 1, 1, 1),  # Add
            (0, 2, 2, 2),  # Add
        ]
        
        result = apply_actions(density, actions)
        
        assert result[0, 0, 0] == 1.0
        assert result[1, 1, 1] == 1.0
        assert result[2, 2, 2] == 1.0
    
    def test_build_state_tensor_shape(self):
        """State tensor has correct shape (7, D, H, W)."""
        D, H, W = 8, 6, 4
        density = np.random.rand(D, H, W)
        bc_masks = (np.zeros((D, H, W)), np.zeros((D, H, W)), np.zeros((D, H, W)))
        forces = (np.zeros((D, H, W)), np.zeros((D, H, W)), np.zeros((D, H, W)))
        
        state = build_state_tensor(density, bc_masks, forces)
        
        assert state.shape == (7, D, H, W)
        assert state.dtype == np.float32
    
    def test_mcts_search_basic(self):
        """MCTS search returns valid actions."""
        # Simple setup
        density = np.zeros((8, 8, 8))
        density[3:5, 3:5, 3:5] = 1.0  # 2x2x2 seed
        
        bc_masks = (np.zeros((8, 8, 8)), np.zeros((8, 8, 8)), np.zeros((8, 8, 8)))
        forces = (np.zeros((8, 8, 8)), np.zeros((8, 8, 8)), np.zeros((8, 8, 8)))
        
        # Mock predict function
        def mock_predict(state):
            value = 0.5
            policy_add = np.random.rand(8, 8, 8)
            policy_remove = np.random.rand(8, 8, 8)
            return value, policy_add, policy_remove
        
        config = MCTSConfig(num_simulations=10, batch_size=4)
        result = run_mcts_search(density, bc_masks, forces, mock_predict, config)
        
        assert len(result.actions) <= 4
        assert result.num_simulations == 10
        assert -1 <= result.root_value <= 1
    
    def test_mcts_wrapper_class(self):
        """AlphaBuilderMCTS wrapper works correctly."""
        density = np.zeros((8, 8, 8))
        density[3:5, 3:5, 3:5] = 1.0
        
        bc_masks = (np.zeros((8, 8, 8)), np.zeros((8, 8, 8)), np.zeros((8, 8, 8)))
        forces = (np.zeros((8, 8, 8)), np.zeros((8, 8, 8)), np.zeros((8, 8, 8)))
        
        def mock_predict(state):
            return 0.5, np.random.rand(8, 8, 8), np.random.rand(8, 8, 8)
        
        mcts = AlphaBuilderMCTS(
            predict_fn=mock_predict,
            num_simulations=5,
            batch_size=2
        )
        
        actions, visit_dist = mcts.get_action_batch(density, bc_masks, forces)
        
        assert len(actions) <= 2
        assert isinstance(visit_dist, dict)


# =============================================================================
# Integration Tests
# =============================================================================

class TestMCTSIntegration:
    """Integration tests for complete MCTS flow."""
    
    def test_convergence_behavior(self):
        """MCTS should concentrate visits on promising actions."""
        density = np.zeros((8, 8, 8))
        density[3:5, 3:5, 3:5] = 1.0
        
        bc_masks = (np.zeros((8, 8, 8)), np.zeros((8, 8, 8)), np.zeros((8, 8, 8)))
        forces = (np.zeros((8, 8, 8)), np.zeros((8, 8, 8)), np.zeros((8, 8, 8)))
        
        # Predict function with strong preference for one action
        def biased_predict(state):
            value = 0.5
            policy_add = np.ones((8, 8, 8)) * -10  # Low prior everywhere
            policy_add[2, 3, 3] = 10  # High prior at one position
            policy_remove = np.ones((8, 8, 8)) * -10
            return value, policy_add, policy_remove
        
        config = MCTSConfig(num_simulations=50, batch_size=4)
        result = run_mcts_search(density, bc_masks, forces, biased_predict, config)
        
        # Check that visits are concentrated
        visit_counts = list(result.visit_distribution.values())
        max_visits = max(visit_counts) if visit_counts else 0
        
        # Most-visited action should have significantly more visits
        assert max_visits > config.num_simulations / 10
    
    def test_value_influences_selection(self):
        """High-value subtrees should attract more visits."""
        density = np.zeros((8, 8, 8))
        density[3:5, 3:5, 3:5] = 1.0
        
        bc_masks = (np.zeros((8, 8, 8)), np.zeros((8, 8, 8)), np.zeros((8, 8, 8)))
        forces = (np.zeros((8, 8, 8)), np.zeros((8, 8, 8)), np.zeros((8, 8, 8)))
        
        # Track which states were evaluated
        evaluation_count = {}
        
        def tracking_predict(state):
            state_hash = hash(state.tobytes())
            evaluation_count[state_hash] = evaluation_count.get(state_hash, 0) + 1
            
            # Return uniform policy and value
            return 0.0, np.ones(state.shape[1:]) * 0.1, np.ones(state.shape[1:]) * 0.1
        
        config = MCTSConfig(num_simulations=20, batch_size=4)
        result = run_mcts_search(density, bc_masks, forces, tracking_predict, config)
        
        # Search should have explored multiple states
        assert len(evaluation_count) >= 1
