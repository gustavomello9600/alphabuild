"""
Tensor utilities for AlphaBuilder v3.1.

Builds 7-channel input tensors for the neural network.
"""
import numpy as np
from typing import Tuple, Optional, Dict, Any


def build_input_tensor_v31(
    density: np.ndarray,
    resolution: Tuple[int, int, int],
    load_config: Dict[str, Any],
    support_type: str = 'full_clamp'
) -> np.ndarray:
    """
    Construct the 7-channel input tensor for AlphaBuilder v3.1.
    
    Channels (spec v3.1):
    0: Density (ρ) - Material state (0.0 to 1.0)
    1: Mask X - 1.0 if displacement u_x is fixed
    2: Mask Y - 1.0 if displacement u_y is fixed
    3: Mask Z - 1.0 if displacement u_z is fixed
    4: Force X (Fx) - Normalized force component
    5: Force Y (Fy) - Normalized force component
    6: Force Z (Fz) - Normalized force component
    
    Args:
        density: 3D array (D, H, W) of density values (0-1).
        resolution: (D, H, W) tuple (same as density shape).
        load_config: Dict with 'x', 'y', 'z_start', 'z_end' for load position.
        support_type: 'full_clamp' (all DOFs fixed at X=0) or 'partial'.
        
    Returns:
        np.ndarray: (7, D, H, W) float32 tensor.
    """
    D, H, W = resolution
    
    # Ensure density is 3D
    if density.ndim == 1:
        density = density.reshape((D, H, W))
    
    tensor = np.zeros((7, D, H, W), dtype=np.float32)
    
    # Channel 0: Density
    tensor[0] = density.astype(np.float32)
    
    # Channels 1-3: Support Masks (X, Y, Z)
    # Default: Full clamp at X=0 (all DOFs fixed)
    if support_type == 'full_clamp':
        tensor[1, 0, :, :] = 1.0  # Mask X
        tensor[2, 0, :, :] = 1.0  # Mask Y
        tensor[3, 0, :, :] = 1.0  # Mask Z
    elif support_type == 'roller_y':
        # Roller: Only Y fixed at X=0
        tensor[2, 0, :, :] = 1.0  # Mask Y only
    elif support_type == 'roller_z':
        # Roller: Only Z fixed at X=0
        tensor[3, 0, :, :] = 1.0  # Mask Z only
        
    # Channels 4-6: Forces (Fx, Fy, Fz)
    # Load is surface traction at X=load_x, region defined by load_config
    lx = min(load_config['x'], D - 1)
    ly = load_config['y']
    lz_center = (load_config['z_start'] + load_config['z_end']) / 2.0
    
    # 2x2 load region
    load_half_width = 1.0
    y_min = max(0, int(ly - load_half_width))
    y_max = min(H, int(ly + load_half_width) + 1)
    z_min = max(0, int(lz_center - load_half_width))
    z_max = min(W, int(lz_center + load_half_width) + 1)
    
    # Force in -Y direction (normalized to -1.0)
    tensor[5, lx, y_min:y_max, z_min:z_max] = -1.0
        
    return tensor


# Legacy function for backwards compatibility
def build_input_tensor(
    density: np.ndarray,
    resolution: Tuple[int, int, int],
    load_position: Optional[Tuple[int, int, int]] = None,
    support_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Legacy 5-channel input tensor (deprecated, use build_input_tensor_v31).
    
    Channels:
    0: Density (0 or 1)
    1: Support Mask (1 where fixed)
    2: Force X (Normalized)
    3: Force Y (Normalized)
    4: Force Z (Normalized)
    """
    D, H, W = resolution
    
    if density.ndim == 1:
        density = density.reshape((D, H, W))
    
    tensor = np.zeros((5, D, H, W), dtype=np.float32)
    tensor[0] = density
    
    if support_mask is not None:
        tensor[1] = support_mask
    else:
        tensor[1, 0, :, :] = 1.0
        
    if load_position is not None:
        ld, lh, lw = load_position
        tensor[3, ld, lh, lw] = -1.0
    else:
        tensor[3, D-1, H//2, W//2] = -1.0
        
    return tensor
