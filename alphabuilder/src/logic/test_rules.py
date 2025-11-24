import numpy as np
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from alphabuilder.src.logic.game_types import DesignState, GameAction, Coord
from alphabuilder.src.logic.game_rules import get_legal_actions, apply_action, get_perimeter
from alphabuilder.src.logic.graph_ops import check_global_connectivity, get_articulation_points

class TestGameRules(unittest.TestCase):
    def test_articulation_point_removal(self):
        print("\nTesting Articulation Point Logic...")
        # Create a 3x3 grid with a bridge
        # 0 0 0
        # 1 1 1  <- Middle (1,1) is a bridge
        # 0 0 0
        grid = np.zeros((3, 3), dtype=np.int8)
        grid[1, :] = 1
        
        supports = ((1, 0),)
        loads = ((1, 2),)
        
        state = DesignState(
            grid=grid,
            supports=supports,
            loads=loads,
            phase='REFINEMENT', # Must be in refinement to remove
            is_connected=True,
            volume=3,
            perimeter=get_perimeter(grid)
        )
        
        legal_actions = get_legal_actions(state)
        legal_remove_coords = [a.coord for a in legal_actions if a.type == 'REMOVE']
        
        print(f"Legal REMOVE actions for bridge config: {legal_remove_coords}")
        
        # (1, 1) is the bridge. Removing it disconnects (1,0) from (1,2).
        # So (1, 1) should NOT be in legal_remove_coords.
        self.assertNotIn((1, 1), legal_remove_coords)
        
        # (1, 0) is support, (1, 2) is load. They are also protected by supports/loads constraints.
        
        # Let's try a T-shape to test non-support/load articulation
        # 0 1 0
        # 1 1 1
        # (0,1) is a leaf, removing it is fine.
        # (1,1) is still a bridge.
        
        grid2 = np.zeros((3, 3), dtype=np.int8)
        grid2[1, :] = 1
        grid2[0, 1] = 1
        
        state2 = DesignState(
            grid=grid2,
            supports=supports,
            loads=loads,
            phase='REFINEMENT',
            is_connected=True,
            volume=4,
            perimeter=get_perimeter(grid2)
        )
        
        legal_actions2 = get_legal_actions(state2)
        legal_remove_coords2 = [a.coord for a in legal_actions2 if a.type == 'REMOVE']
        
        print(f"Legal REMOVE actions for T-shape config: {legal_remove_coords2}")
        
        self.assertIn((0, 1), legal_remove_coords2) # Leaf, can remove
        self.assertNotIn((1, 1), legal_remove_coords2) # Bridge, cannot remove

    def test_immutability(self):
        print("\nTesting Immutability...")
        grid = np.zeros((3, 3), dtype=np.int8)
        grid[1, 1] = 1
        state = DesignState(
            grid=grid,
            supports=((1,1),),
            loads=((1,1),),
            phase='GROWTH',
            is_connected=True,
            volume=1,
            perimeter=get_perimeter(grid)
        )
        
        h1 = hash(state)
        
        # Apply action
        action = GameAction(type='ADD', coord=(0, 1))
        new_state = apply_action(state, action)
        
        h2 = hash(new_state)
        
        self.assertNotEqual(h1, h2)
        self.assertNotEqual(state.volume, new_state.volume)
        self.assertEqual(state.grid[0, 1], 0) # Original should be unchanged
        self.assertEqual(new_state.grid[0, 1], 1)
        print("Immutability verified.")

if __name__ == '__main__':
    unittest.main()
