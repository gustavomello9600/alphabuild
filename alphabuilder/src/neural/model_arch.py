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
        
        # 2. Hook to capture Bottleneck Features (Block C)
        # The SwinViT module returns a list of hidden states. The last one is the bottleneck.
        self.bottleneck_features = None
        
        def hook_fn(module, input, output):
            # output is a list of hidden states from each stage
            # The last element corresponds to the bottleneck features
            if isinstance(output, (list, tuple)):
                self.bottleneck_features = output[-1]
            else:
                self.bottleneck_features = output
                
        # Register hook on the SwinViT backbone
        self.swin_unetr.swinViT.register_forward_hook(hook_fn)
        
        # 3. Value Head (Attached to Block C)
        # Bottleneck channels = feature_size * 2^4 = 48 * 16 = 768
        bottleneck_channels = feature_size * 16
        
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), # Global Average Pooling -> (B, 768, 1, 1, 1)
            nn.Flatten(),            # -> (B, 768)
            nn.Linear(bottleneck_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)        # -> (B, 1) Scalar Value
        )
        
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
            
        # 1. Policy Forward (Backbone + Decoder)
        # This triggers the hook, capturing self.bottleneck_features
        policy_logits_padded = self.swin_unetr(x_padded)
        
        # 2. Value Forward (Using captured Bottleneck features)
        # Ensure we have captured features
        if self.bottleneck_features is None:
            raise RuntimeError("Bottleneck features not captured. Hook failed.")
            
        value_pred = self.value_head(self.bottleneck_features)
        
        # Clear captured features to avoid memory leaks
        self.bottleneck_features = None
        
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
