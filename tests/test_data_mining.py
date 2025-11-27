"""
Unit Test for Data Mining Logic
"""

import numpy as np
import pytest
from alphabuilder.src.logic.data_mining import extract_discrete_actions

def test_extract_discrete_actions_removal():
    """Test that removal actions are correctly identified."""
    
    # Mock history
    # Step 0: Full 3x3x3 block
    state_0 = np.ones((3, 3, 3), dtype=np.int8)
    
    # Step 5: Center voxel removed
    state_5 = state_0.copy()
    state_5[1, 1, 1] = 0
    
    history = [
        {
            'step': 0,
            'binary_map': state_0,
            'compliance': 100.0,
            'max_displacement': 1.0
        },
        # Steps 1-4 skipped/irrelevant for jump_size=5
        {}, {}, {}, {},
        {
            'step': 5,
            'binary_map': state_5,
            'compliance': 90.0,
            'max_displacement': 0.9
        }
    ]
    
    # Extract
    samples = extract_discrete_actions(history, jump_size=5)
    
    assert len(samples) == 1
    sample = samples[0]
    
    assert sample['step'] == 0
    assert sample['target_value'] == 90.0
    
    # Verify Policy Target (Removal Mask)
    # Should be 0 everywhere except center (1,1,1) which is 1
    expected_policy = np.zeros((3, 3, 3), dtype=np.int8)
    expected_policy[1, 1, 1] = 1
    
    np.testing.assert_array_equal(sample['target_policy'], expected_policy)
    print("✅ Removal action correctly identified")

def test_extract_discrete_actions_no_change():
    """Test that no samples are generated if no change occurs."""
    state = np.ones((3, 3, 3), dtype=np.int8)
    
    history = [
        {'step': 0, 'binary_map': state, 'compliance': 100.0},
        {}, {}, {}, {},
        {'step': 5, 'binary_map': state, 'compliance': 100.0}
    ]
    
    samples = extract_discrete_actions(history, jump_size=5)
    assert len(samples) == 0
    print("✅ No samples generated for static state")

if __name__ == "__main__":
    test_extract_discrete_actions_removal()
    test_extract_discrete_actions_no_change()
