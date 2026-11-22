"""
Tensor utilities for AlphaBuilder v3.1.

Builds 7-channel input tensors for the neural network.
"""
import numpy as np
from typing import Tuple, Dict, Any

from alphabuilder.src.logic.harvest.generators import get_load_region_bounds


def build_input_tensor_v31(
    density: np.ndarray,
    resolution: Tuple[int, int, int],
    load_config: Dict[str, Any],
    support_type: str = 'full_clamp'
) -> np.ndarray:
    """
    Construct the 7-channel input tensor for AlphaBuilder v3.1 / v3.3.
    
    Channels:
    0: Density (ρ) - Material state (0.0 to 1.0)
    1: Mask X - 1.0 if displacement u_x is fixed
    2: Mask Y - 1.0 if displacement u_y is fixed
    3: Mask Z - 1.0 if displacement u_z is fixed
    4: Force X (Fx) - Normalized force component
    5: Force Y (Fy) - Normalized force component
    6: Force Z (Fz) - Normalized force component
    """
    D, H, W = resolution
    
    # Ensure density is 3D
    if density.ndim == 1:
        density = density.reshape((D, H, W))
    
    tensor = np.zeros((7, D, H, W), dtype=np.float32)
    
    # Channel 0: Density
    tensor[0] = density.astype(np.float32)
    
    # Channels 1-3: Support Masks (X, Y, Z)
    st = str(support_type).upper()
    if st == 'FULL_CLAMP':
        tensor[1, 0, :, :] = 1.0  # Mask X
        tensor[2, 0, :, :] = 1.0  # Mask Y
        tensor[3, 0, :, :] = 1.0  # Mask Z
    elif st == 'RAIL_XY':
        tensor[1, 0, :, :] = 1.0  # Mask X
        tensor[2, 0, :, :] = 1.0  # Mask Y (Z free)
    elif st == 'ROLLER_Y':
        tensor[2, 0, :, :] = 1.0  # Mask Y only
    elif st == 'ROLLER_Z':
        tensor[3, 0, :, :] = 1.0  # Mask Z only
    else:
        tensor[1, 0, :, :] = 1.0
        tensor[2, 0, :, :] = 1.0
        tensor[3, 0, :, :] = 1.0
        
    # Channels 4-6: Forces (Fx, Fy, Fz) aligned with volumetric load region
    lx, y_min, y_max, z_min, z_max = get_load_region_bounds(resolution, load_config)
    tensor[5, lx, y_min:y_max, z_min:z_max] = -1.0
        
    return tensor
