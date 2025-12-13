"""
Unit tests for select_pv_sequences micro-batch action selection.

Verifies that the algorithm correctly extracts 32 actions from the MCTS tree
by iterating through root children and following depth-limited PV paths.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

# Import the function under test
from alphabuilder.src.logic.mcts.engine import select_pv_sequences, PHASE1_CONFIG
from alphabuilder.src.logic.mcts.node import MCTSNode


class TestSelectPVSequences:
    """Tests for the Principal Variation sequence selection algorithm."""

    def _create_mock_node(self, visit_count: int, mean_value: float = 0.0, children: dict = None):
        """Helper to create a mock MCTSNode."""
        node = MagicMock(spec=MCTSNode)
        node.visit_count = visit_count
        node.mean_value = mean_value
        node.children = children or {}
        return node

    def test_returns_empty_when_no_children(self):
        """Should return empty list when root has no children."""
        root = self._create_mock_node(100, children={})
        
        actions = select_pv_sequences(root, max_actions=32, max_depth=4)
        
        assert actions == []

    def test_extracts_single_action_from_single_child(self):
        """Should return single action when root has one leaf child."""
        child = self._create_mock_node(10, mean_value=0.5, children={})
        action = (0, 5, 5, 2)  # Add voxel at (5, 5, 2)
        root = self._create_mock_node(100, children={action: child})
        
        actions = select_pv_sequences(root, max_actions=32, max_depth=4)
        
        assert len(actions) == 1
        assert actions[0] == action

    def test_extracts_depth_path_from_deep_tree(self):
        """Should follow PV path up to max_depth."""
        # Build a depth-4 tree: root -> a1 -> a2 -> a3 -> a4
        a4 = (0, 8, 8, 3)
        child4 = self._create_mock_node(5, children={})
        
        a3 = (0, 7, 7, 3)
        child3 = self._create_mock_node(10, children={a4: child4})
        
        a2 = (0, 6, 6, 2)
        child2 = self._create_mock_node(20, children={a3: child3})
        
        a1 = (0, 5, 5, 1)
        child1 = self._create_mock_node(50, children={a2: child2})
        
        root = self._create_mock_node(100, children={a1: child1})
        
        actions = select_pv_sequences(root, max_actions=32, max_depth=4)
        
        # Should extract the full path: a1 -> a2 -> a3 -> a4
        assert len(actions) == 4
        assert actions == [a1, a2, a3, a4]

    def test_respects_max_depth_limit(self):
        """Should stop at max_depth even if tree is deeper."""
        # Build a depth-6 tree
        current = self._create_mock_node(1, children={})
        for i in range(5, -1, -1):  # 5, 4, 3, 2, 1, 0
            action = (0, i, i, i)
            current = self._create_mock_node(i * 10 + 10, children={action: current})
        root = current
        
        actions = select_pv_sequences(root, max_actions=32, max_depth=3)
        
        # Should only extract 3 actions (depth limited)
        assert len(actions) == 3

    def test_extracts_from_multiple_root_children(self):
        """Should iterate through root children to fill batch."""
        # Create 10 root children, each with depth-2 paths
        root_children = {}
        for i in range(10):
            a2 = (0, i, 10 + i, 2)
            child2 = self._create_mock_node(5, children={})
            
            a1 = (0, i, i, 1)
            child1 = self._create_mock_node(50 - i, children={a2: child2})
            
            root_children[a1] = child1
        
        root = self._create_mock_node(500, children=root_children)
        
        actions = select_pv_sequences(root, max_actions=20, max_depth=4)
        
        # Should extract 20 actions (10 children × 2 depth)
        assert len(actions) == 20

    def test_fills_to_max_actions(self):
        """Should continue until max_actions reached, using multiple children."""
        # Create 16 root children, each with depth-4 paths (potentially 64 actions)
        root_children = {}
        for i in range(16):
            # Build depth-4 path for each root child
            current = self._create_mock_node(1, children={})
            for d in range(3, -1, -1):
                action = (0, i, d, i % 4)
                current = self._create_mock_node((4 - d) * 10, children={action: current})
            
            root_action = (0, i, 100, i % 4)
            root_children[root_action] = current
        
        root = self._create_mock_node(1000, children=root_children)
        
        actions = select_pv_sequences(root, max_actions=32, max_depth=4)
        
        # Should extract exactly 32 actions
        assert len(actions) == 32

    def test_prevents_duplicate_voxels_in_batch(self):
        """Should skip actions that target already-locked voxels."""
        # Two children targeting same voxel
        same_voxel = (0, 5, 5, 2)
        child1 = self._create_mock_node(50, children={})
        child2 = self._create_mock_node(40, children={})  # Lower priority
        
        root = self._create_mock_node(100, children={
            same_voxel: child1,
            (0, 5, 5, 2): child2  # Same coords, should be skipped
        })
        
        actions = select_pv_sequences(root, max_actions=32, max_depth=4)
        
        # Should only have 1 action (duplicate skipped)
        assert len(actions) == 1

    def test_skips_locked_voxels_in_deep_paths(self):
        """Should skip nodes in PV path if voxel already used by earlier sequence."""
        # Two root children, second one's path crosses first's voxel
        collision_voxel = (0, 5, 5, 2)
        
        # First child: simple path
        child1 = self._create_mock_node(50, children={collision_voxel: self._create_mock_node(10)})
        
        # Second child: path that would use the collision voxel
        deep_child = self._create_mock_node(5, children={})
        child2_inner = self._create_mock_node(20, children={collision_voxel: deep_child})
        child2 = self._create_mock_node(40, children={(0, 6, 6, 3): child2_inner})
        
        root = self._create_mock_node(100, children={
            (0, 1, 1, 1): child1,  # Higher priority
            (0, 2, 2, 2): child2
        })
        
        actions = select_pv_sequences(root, max_actions=32, max_depth=4)
        
        # Should have 4 actions total: (0,1,1,1), (0,5,5,2), (0,2,2,2), (0,6,6,3)
        # But (0,5,5,2) is locked after first path, so second path stops early
        coords = [(a[1], a[2], a[3]) for a in actions]
        assert len(coords) == len(set(coords)), "All voxels should be unique"

    def test_verify_breadth_and_depth_distribution(self):
        """
        Verify that the selection strategy utilizes both breadth (multiple root children)
        and depth (sequences up to max_depth).
        
        Scenario:
        - Root has 10 children.
        - Each child has a deep path (depth 4).
        - We request 32 actions.
        
        Expected:
        - Breadth: Should use at least 8 root children (32 / 4 = 8).
        - Depth: Should extract sequences of length up to 4.
        """
        root_children = {}
        # Create 10 branches, each 4 deep
        for i in range(10):
            # Branch i: root -> (i,0) -> (i,1) -> (i,2) -> (i,3)
            curr = self._create_mock_node(1, children={})
            for d in range(3, -1, -1):
                action = (0, i, i, d) # Unique voxel per depth/branch
                if d == 0:
                    root_children[action] = curr
                else:
                    parent = self._create_mock_node(10, children={action: curr})
                    curr = parent
        
        root = self._create_mock_node(100, children=root_children)
        
        actions = select_pv_sequences(root, max_actions=32, max_depth=4)
        
        assert len(actions) == 32
        
        # Analyze distribution
        # Group by branch (x coordinate in our mock action)
        branches = {}
        for act in actions:
            branch_id = act[1]
            branches[branch_id] = branches.get(branch_id, 0) + 1
            
        # Verify Breadth: How many distinct branches (root children) were touched?
        # To get 32 actions with max_depth=4, we need at least 32/4 = 8 branches.
        # It could be more if some branches were shorter, but here they are all long.
        num_branches = len(branches)
        assert num_branches >= 8, f"Expected at least 8 branches, got {num_branches}"
        
        # Verify Depth: Check that we extracted deep sequences
        # In this perfect tree, we expect full depth 4 for the first 8 branches
        deep_branches = [b for b, count in branches.items() if count == 4]
        assert len(deep_branches) >= 8, f"Expected at least 8 branches to reach depth 4, got {len(deep_branches)}"

