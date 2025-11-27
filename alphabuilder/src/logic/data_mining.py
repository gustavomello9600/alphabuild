"""
Data Mining Logic for AlphaBuilder.

Implements strategies to extract discrete training samples from continuous
optimization processes (SIMP), acting as a "Teacher" for the neural network.
"""

import numpy as np
from typing import List, Dict, Any

def extract_discrete_actions(
    simp_history: List[Dict[str, Any]], 
    jump_size: int = 5,
    resolution: tuple = (64, 32, 32)
) -> List[Dict[str, Any]]:
    """
    Converts continuous SIMP evolution into discrete "moves" for AlphaZero training.
    
    Args:
        simp_history: List of SIMP frames.
        jump_size: Lookahead.
        resolution: Grid resolution (D, H, W).
        
    Returns:
        List of training samples with 'input_state' (5-channel), 'target_policy' (2-channel), 'target_value'.
    """
    from alphabuilder.src.core.tensor_utils import build_input_tensor
    
    training_samples = []
    final_compliance = simp_history[-1]['compliance']
    
    for t in range(0, len(simp_history) - jump_size):
        current_frame = simp_history[t]
        future_frame = simp_history[t + jump_size]
        
        # 1. Input State (5-Channel Tensor)
        # Use raw density from history if available, or binary map
        # History has 'density_map' (continuous) and 'binary_map' (0/1)
        # We should use the continuous density for the network input?
        # Spec says: "Canal 0 (rho): Matriz de densidade binária atual"
        # So we use binary map.
        
        current_density = current_frame['binary_map']
        future_density = future_frame['binary_map']
        
        # Build 5-channel tensor using utility (adds Mask and Forces)
        input_tensor = build_input_tensor(current_density, resolution)
        
        # 2. Target Policy (2-Channel Tensor)
        # Channel 0: Add (Not used in SIMP usually)
        # Channel 1: Remove
        
        current_bin = (current_density > 0.5).astype(np.int8)
        future_bin = (future_density > 0.5).astype(np.int8)
        
        # Removal: Existed (1) AND Removed (0)
        removal_mask = (current_bin == 1) & (future_bin == 0)
        
        # Addition: Didn't exist (0) AND Added (1) (Rare in SIMP but possible with filtering)
        addition_mask = (current_bin == 0) & (future_bin == 1)
        
        D, H, W = resolution
        target_policy = np.zeros((2, D, H, W), dtype=np.float32)
        target_policy[0] = addition_mask
        target_policy[1] = removal_mask
        
        # If no change, skip
        if np.sum(removal_mask) == 0 and np.sum(addition_mask) == 0:
            continue
            
        training_samples.append({
            "step": current_frame['step'],
            "input_state": input_tensor,
            "target_policy": target_policy,
            "target_value": final_compliance,
            "metadata": {
                "compliance": current_frame['compliance'],
                "max_displacement": current_frame['max_displacement']
            }
        })
        
    return training_samples
