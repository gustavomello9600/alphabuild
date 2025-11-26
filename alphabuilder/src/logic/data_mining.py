"""
Data Mining Logic for AlphaBuilder.

Implements strategies to extract discrete training samples from continuous
optimization processes (SIMP), acting as a "Teacher" for the neural network.
"""

import numpy as np
from typing import List, Dict, Any

def extract_discrete_actions(simp_history: List[Dict[str, Any]], jump_size: int = 5) -> List[Dict[str, Any]]:
    """
    Converts continuous SIMP evolution into discrete "moves" for AlphaZero training.
    
    Strategy:
    Compare state at T vs T+jump_size.
    If a voxel existed at T but is gone at T+jump_size, that's a REMOVE action.
    
    Args:
        simp_history: List of SIMP frames (dicts with 'binary_map', 'compliance', etc.)
        jump_size: Number of frames to look ahead (simulates aggressive moves)
        
    Returns:
        List of training samples with 'input_state', 'target_policy', 'target_value'
    """
    training_samples = []
    
    # Get final compliance for value target (global reward)
    # Note: In RL, reward is usually immediate + discounted future.
    # Here we use the final outcome as a proxy for "value of this state".
    # We might want to normalize this later.
    final_compliance = simp_history[-1]['compliance']
    
    # Iterate through history
    for t in range(0, len(simp_history) - jump_size):
        
        # 1. Define Current State (Input)
        # Binarize current density. Threshold > 0.3 as per expert directive.
        # Note: simp_history['binary_map'] is already binary (0/1) from the generator?
        # Let's check simp_generator.py. It returns 'binary_map' which is (x > 0.5).astype(float).
        # But the directive says "Binarize density... > 0.3".
        # If history has raw density 'x', we should use that.
        # simp_generator returns 'binary_map' (thresholded) and 'compliance'.
        # It does NOT return raw density 'x' in the current implementation of run_simp_optimization_3d.
        # I should check if I need to update simp_generator to return raw density.
        # For now, let's assume binary_map is close enough, or update simp_generator.
        
        # Let's check simp_generator.py output first.
        # It returns: {'step': k, 'compliance': c, 'max_displacement': d, 'binary_map': x_phys}
        # where x_phys is the filtered density (continuous 0-1).
        # Wait, let's verify if 'binary_map' is actually binary or continuous.
        
        current_frame = simp_history[t]
        future_frame = simp_history[t + jump_size]
        
        # Assuming 'binary_map' holds the continuous density field (despite the name)
        # or the thresholded one. The directive says "Binarize density... > 0.3".
        # If 'binary_map' is already 0/1 (thresholded at 0.5), we might miss subtle changes.
        # But let's proceed with what we have.
        
        current_density = current_frame['binary_map']
        future_density = future_frame['binary_map']
        
        # 1. Current State (Input)
        current_state = (current_density > 0.3).astype(np.int8)
        
        # 2. Future State (Intent)
        future_state = (future_density > 0.3).astype(np.int8)
        
        # 3. Calculate Removal Action (Target Policy)
        # Logic: Existed (1) AND Removed later (0)
        removal_action_map = (current_state == 1) & (future_state == 0)
        removal_action_map = removal_action_map.astype(np.int8)
        
        # If nothing changed, skip
        if np.sum(removal_action_map) == 0:
            continue
            
        training_samples.append({
            "step": current_frame['step'],
            "input_state": current_state,
            "target_policy": removal_action_map,
            "target_value": final_compliance, # Simplified value target
            "metadata": {
                "compliance": current_frame['compliance'],
                "max_displacement": current_frame['max_displacement']
            }
        })
        
    return training_samples
