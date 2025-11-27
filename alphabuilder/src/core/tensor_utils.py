import numpy as np
from typing import Tuple, Optional

def build_input_tensor(
    density: np.ndarray,
    resolution: Tuple[int, int, int],
    load_position: Optional[Tuple[int, int, int]] = None,
    support_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Construct the 5-channel input tensor for the neural network.
    
    Channels:
    0: Density (0 or 1)
    1: Support Mask (1 where fixed)
    2: Force X (Normalized)
    3: Force Y (Normalized)
    4: Force Z (Normalized)
    
    Args:
        density: 1D or 3D array of density values (0-1).
        resolution: (D, H, W) tuple.
        load_position: Optional (d, h, w) tuple for point load. If None, uses default Cantilever tip.
        support_mask: Optional 3D array for supports. If None, uses default Cantilever left wall.
        
    Returns:
        np.ndarray: (5, D, H, W) float32 tensor.
    """
    D, H, W = resolution
    
    # Ensure density is 3D
    if density.ndim == 1:
        density = density.reshape((D, H, W))
    
    tensor = np.zeros((5, D, H, W), dtype=np.float32)
    
    # Channel 0: Density
    tensor[0] = density
    
    # Channel 1: Support Mask
    if support_mask is not None:
        tensor[1] = support_mask
    else:
        # Default Cantilever: Fix Left Wall (x=0)
        # Note: In our resolution (L, H, W), usually L is x.
        # So index 0 is x.
        tensor[1, 0, :, :] = 1.0
        
    # Channels 2-4: Forces (Fx, Fy, Fz)
    # Default Cantilever: Point Load at Tip (x=max, y=mid, z=mid)
    # Force is -1.0 in Y direction.
    if load_position is not None:
        ld, lh, lw = load_position
        tensor[3, ld, lh, lw] = -1.0
    else:
        # Default: Tip Load
        # x=max (D-1), y=mid (H//2), z=mid (W//2)
        # Assuming D is Length (x), H is Height (y), W is Depth (z)
        tensor[3, D-1, H//2, W//2] = -1.0
        
    return tensor
