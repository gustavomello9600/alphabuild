import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from dataclasses import dataclass
from typing import Tuple, Optional

# Constants
INPUT_SHAPE = (5, 64, 32, 32) # (Channels, Depth, Height, Width) - Adjusted for typical Colab memory
CHANNELS = 5

@dataclass
class ModelOutput:
    policy_logits: torch.Tensor # (B, 2, D, H, W)
    value_pred: torch.Tensor    # (B, 1)

class AlphaBuilderSwinUNETR(nn.Module):
    """
    Physics-Aware Swin-UNETR for Topology Optimization.
    
    Inputs:
        x: (Batch, 5, D, H, W)
           - Ch0: Density (0/1)
           - Ch1: Support Mask
           - Ch2-4: Force Vectors (Fx, Fy, Fz)
           
    Outputs:
        policy_logits: (Batch, 2, D, H, W) -> [Add_Score, Remove_Score]
        value_pred: (Batch, 1) -> Estimated Compliance/Success
    """
    def __init__(
        self, 
        img_size: Tuple[int, int, int] = (64, 32, 32),
        in_channels: int = 5,
        out_channels: int = 2,
        feature_size: int = 48,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        
        # 1. Backbone: Swin-UNETR (MONAI)
        # We use the encoder-decoder structure for the Policy Head
        self.swin_unetr = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims=3
        )
        
        # 2. Value Head (Attached to the Bottleneck)
        # SwinUNETR doesn't expose the bottleneck easily in its standard forward.
        # We might need to hook into it or use the encoder features if we want a true multi-head.
        # However, for simplicity and robustness, we can use a separate lightweight encoder 
        # OR (better) use the SwinUNETR's encoder output if accessible.
        # MONAI's SwinUNETR returns the segmentation map directly.
        
        # Strategy: We will use the SwinUNETR for the Policy.
        # For the Value, we will add a small parallel encoder or try to reuse features.
        # Given the complexity of modifying MONAI internals without inheritance, 
        # let's add a lightweight 3D CNN Encoder for the Value Head that shares nothing *yet* 
        # (to be optimized later) OR simply downsample the Policy output.
        
        # Better Strategy for V1: Use the Policy Logits to estimate Value? No, circular.
        # Let's add a separate simple encoder for Value to ensure it learns global structural stability.
        # This increases param count but guarantees the Value Head isn't confused by the Policy's pixel-wise task.
        
        # Calculate padded dimensions for Value Encoder
        # We pad to multiples of 32 in forward()
        pad_d = (32 - img_size[0] % 32) % 32
        pad_h = (32 - img_size[1] % 32) % 32
        pad_w = (32 - img_size[2] % 32) % 32
        
        d_padded = img_size[0] + pad_d
        h_padded = img_size[1] + pad_h
        w_padded = img_size[2] + pad_w
        
        self.value_encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=4, stride=4), # 64->16
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=4, stride=4), # 16->4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (d_padded//16) * (h_padded//16) * (w_padded//16), 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Note: The above is a placeholder. Ideally we want to tap into Swin's bottleneck.
        # But `monai.networks.nets.SwinUNETR` is a `nn.Module`.
        
    def forward(self, x: torch.Tensor) -> ModelOutput:
        # Pad input to be divisible by 32 (SwinUNETR requirement)
        # x shape: (B, C, D, H, W)
        b, c, d, h, w = x.shape
        
        # Calculate padding
        pd = (32 - d % 32) % 32
        ph = (32 - h % 32) % 32
        pw = (32 - w % 32) % 32
        
        # Pad if necessary
        if pd > 0 or ph > 0 or pw > 0:
            # F.pad expects (last_dim_left, last_dim_right, 2nd_last_left, ...)
            # So (w_left, w_right, h_left, h_right, d_left, d_right)
            x_padded = torch.nn.functional.pad(x, (0, pw, 0, ph, 0, pd))
        else:
            x_padded = x
            
        # Policy Forward
        # Output: (B, 2, D_pad, H_pad, W_pad)
        policy_logits_padded = self.swin_unetr(x_padded)
        
        # Value Forward (Simple Baseline)
        # We use the padded input for now
        value_pred = self.value_encoder(x_padded)
        
        # Unpad Policy Output
        if pd > 0 or ph > 0 or pw > 0:
            policy_logits = policy_logits_padded[..., :d, :h, :w]
        else:
            policy_logits = policy_logits_padded
        
        return ModelOutput(policy_logits=policy_logits, value_pred=value_pred)

def build_model(input_shape: Tuple[int, int, int] = (64, 32, 32)) -> AlphaBuilderSwinUNETR:
    return AlphaBuilderSwinUNETR(img_size=input_shape)

if __name__ == "__main__":
    # Smoke Test
    print("Running Swin-UNETR Smoke Test...")
    try:
        model = build_model()
        # Create dummy input (Batch=1, Ch=5, D=64, H=32, W=32)
        x = torch.randn(1, 5, 64, 32, 32)
        output = model(x)
        
        print(f"Input: {x.shape}")
        print(f"Policy Output: {output.policy_logits.shape}")
        print(f"Value Output: {output.value_pred.shape}")
        
        assert output.policy_logits.shape == (1, 2, 64, 32, 32)
        assert output.value_pred.shape == (1, 1)
        print("Test Passed!")
    except ImportError:
        print("MONAI or PyTorch not installed. Skipping execution.")
    except Exception as e:
        print(f"Test Failed: {e}")
