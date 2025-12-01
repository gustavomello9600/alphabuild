"""
Neural Network Model for AlphaBuilder v3.1.

Swin-UNETR based model with:
- 7 input channels
- Dynamic padding for arbitrary resolutions
- InstanceNorm3d
- Policy Head (2 channels) + Value Head (1 scalar)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Try to import MONAI's SwinUNETR, fallback to simple conv model
try:
    from monai.networks.nets import SwinUNETR
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


class AlphaBuilderV31(nn.Module):
    """
    AlphaBuilder v3.1 Neural Network.
    
    Input: (B, 7, D, H, W) - 7-channel state tensor
    Output: 
        - policy: (B, 2, D, H, W) - Add/Remove logits
        - value: (B, 1) - Quality score in [-1, 1]
    """
    
    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 2,
        feature_size: int = 24,
        use_swin: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.window_size = 2  # For dynamic padding
        
        if use_swin and HAS_MONAI:
            try:
                # Try newer MONAI API (without img_size)
                self.backbone = SwinUNETR(
                    in_channels=in_channels,
                    out_channels=feature_size,
                    feature_size=feature_size,
                    use_checkpoint=False,
                    spatial_dims=3,
                    norm_name="instance"
                )
                self.use_swin = True
            except TypeError:
                # Fallback to simple backbone if MONAI API incompatible
                self.backbone = SimpleBackbone(in_channels, feature_size)
                self.use_swin = False
        else:
            # Fallback: Simple 3D UNet-like architecture
            self.backbone = SimpleBackbone(in_channels, feature_size)
            self.use_swin = False
        
        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_size, out_channels, kernel_size=1)
        )
        
        # Value Head (Global pooling + MLP)
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(feature_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dynamic padding.
        
        Args:
            x: (B, 7, D, H, W) input tensor
            
        Returns:
            policy: (B, 2, D, H, W)
            value: (B, 1)
        """
        # Store original size
        original_size = x.shape[2:]  # (D, H, W)
        
        # Dynamic padding to make dimensions divisible by window_size
        x_padded, pad_sizes = self._dynamic_pad(x)
        
        # Backbone
        features = self.backbone(x_padded)
        
        # Crop features back to original size
        features = self._dynamic_crop(features, original_size, pad_sizes)
        
        # Heads
        policy = self.policy_head(features)
        value = self.value_head(features)
        
        return policy, value
    
    def _dynamic_pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """Pad input to make dimensions divisible by window_size."""
        B, C, D, H, W = x.shape
        
        # Calculate padding needed
        pad_d = (self.window_size - D % self.window_size) % self.window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        # F.pad format: (W_left, W_right, H_left, H_right, D_left, D_right)
        padding = (0, pad_w, 0, pad_h, 0, pad_d)
        
        x_padded = F.pad(x, padding, mode='constant', value=0)
        
        return x_padded, (pad_d, pad_h, pad_w)
    
    def _dynamic_crop(
        self, 
        x: torch.Tensor, 
        original_size: Tuple[int, int, int],
        pad_sizes: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Crop output back to original size."""
        D, H, W = original_size
        return x[:, :, :D, :H, :W]


class SimpleBackbone(nn.Module):
    """
    Simple 3D CNN backbone (fallback when MONAI not available).
    """
    
    def __init__(self, in_channels: int, feature_size: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv3d(in_channels, feature_size, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_size),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv3d(feature_size, feature_size * 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_size * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_size * 2, feature_size * 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_size * 2),
            nn.ReLU(inplace=True),
            
            # Back to feature_size
            nn.Conv3d(feature_size * 2, feature_size, kernel_size=1),
            nn.InstanceNorm3d(feature_size),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.encoder(x)

