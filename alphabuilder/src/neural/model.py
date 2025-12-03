"""
Neural Network Model for AlphaBuilder v3.1.

Architecture with separate paths from bottleneck:
- Value Head: bottleneck → MLP → scalar
- Policy Head: bottleneck → decoder → spatial output

Supports SwinUNETR (MONAI) or SimpleBackbone fallback.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

# Try to import MONAI components
try:
    from monai.networks.nets.swin_unetr import SwinTransformer
    from monai.networks.blocks import UnetrUpBlock, UnetOutBlock
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


class AlphaBuilderV31(nn.Module):
    """
    AlphaBuilder v3.1 Neural Network.
    
    Architecture:
        Input (B, 7, D, H, W)
              │
              ▼
        ┌─────────────┐
        │   ENCODER   │  (Swin Transformer or Simple CNN)
        └─────────────┘
              │
              ▼
        ┌─────────────┐
        │  BOTTLENECK │  (compressed representation)
        └─────────────┘
          │         │
          ▼         ▼
    ┌──────────┐  ┌──────────────┐
    │ VALUE    │  │ POLICY       │
    │ HEAD     │  │ DECODER      │
    │ (MLP)    │  │ (Upsample)   │
    └──────────┘  └──────────────┘
         │              │
         ▼              ▼
      (B, 1)      (B, 2, D, H, W)
    
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
        self.feature_size = feature_size
        self.use_swin = use_swin and HAS_MONAI
        
        # Bottleneck dimension (deepest representation)
        if self.use_swin:
            # SwinUNETR: bottleneck is 16x feature_size after 4 stages (768 for feature_size=48)
            # MONAI SwinTransformer returns hidden_states[4] with 16x embed_dim
            self.bottleneck_dim = feature_size * 16  # 384 for feature_size=24
        else:
            # SimpleBackbone: bottleneck is 4x feature_size
            self.bottleneck_dim = feature_size * 4  # 96 for feature_size=24
        
        # ============ ENCODER ============
        if self.use_swin:
            self.encoder = SwinEncoder(in_channels, feature_size)
        else:
            self.encoder = SimpleEncoder(in_channels, feature_size)
        
        # ============ VALUE HEAD (from bottleneck) ============
        # Bottleneck → Global Pool → MLP → Tanh
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(self.bottleneck_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # ============ POLICY DECODER (from bottleneck) ============
        if self.use_swin:
            self.policy_decoder = SwinPolicyDecoder(feature_size, out_channels)
        else:
            self.policy_decoder = SimplePolicyDecoder(feature_size, out_channels)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with separate paths from bottleneck.
        
        Args:
            x: (B, 7, D, H, W) input tensor
            
        Returns:
            policy: (B, 2, D, H, W)
            value: (B, 1)
        """
        original_size = x.shape[2:]  # (D, H, W)
        
        # Dynamic padding for Swin (needs divisibility by 32)
        if self.use_swin:
            x, pad_sizes = self._dynamic_pad(x, divisor=32)
        else:
            x, pad_sizes = self._dynamic_pad(x, divisor=2)
        
        padded_size = x.shape[2:]  # (D, H, W) after padding
        
        # Encoder: extract hierarchical features + bottleneck
        skip_features, bottleneck = self.encoder(x)
        
        # VALUE HEAD: from bottleneck directly
        value = self.value_head(bottleneck)
        
        # POLICY DECODER: from bottleneck with skip connections
        if self.use_swin:
            policy = self.policy_decoder(bottleneck, skip_features, padded_size)
        else:
            policy = self.policy_decoder(bottleneck, skip_features)
        
        # Crop back to original size
        policy = self._dynamic_crop(policy, original_size)
        
        return policy, value
    
    def _dynamic_pad(self, x: torch.Tensor, divisor: int) -> Tuple[torch.Tensor, Tuple]:
        """Pad input to make dimensions divisible by divisor."""
        B, C, D, H, W = x.shape
        
        pad_d = (divisor - D % divisor) % divisor
        pad_h = (divisor - H % divisor) % divisor
        pad_w = (divisor - W % divisor) % divisor
        
        padding = (0, pad_w, 0, pad_h, 0, pad_d)
        x_padded = F.pad(x, padding, mode='constant', value=0)
        
        return x_padded, (pad_d, pad_h, pad_w)
    
    def _dynamic_crop(self, x: torch.Tensor, original_size: Tuple[int, int, int]) -> torch.Tensor:
        """Crop output back to original size."""
        D, H, W = original_size
        return x[:, :, :D, :H, :W]


# ============================================================================
#                           SWIN TRANSFORMER ENCODER
# ============================================================================

class SwinEncoder(nn.Module):
    """
    Swin Transformer Encoder that returns skip features and bottleneck.
    
    Based on MONAI's SwinTransformer but exposes intermediate features.
    """
    
    def __init__(self, in_channels: int, feature_size: int):
        super().__init__()
        
        self.swin = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=3,
        )
        
        # Encoder blocks to process Swin outputs
        self.encoder1 = UnetrBasicBlock(feature_size, feature_size)
        self.encoder2 = UnetrBasicBlock(feature_size * 2, feature_size * 2)
        self.encoder3 = UnetrBasicBlock(feature_size * 4, feature_size * 4)
        self.encoder4 = UnetrBasicBlock(feature_size * 8, feature_size * 8)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Returns:
            skip_features: List of 4 feature maps at different resolutions
            bottleneck: Deepest feature map (compressed)
        """
        # Swin Transformer extracts hierarchical features
        hidden_states = self.swin(x)  # Returns list of 4 hidden states
        
        # Process each stage
        enc1 = self.encoder1(hidden_states[0])  # (B, C, D/2, H/2, W/2)
        enc2 = self.encoder2(hidden_states[1])  # (B, 2C, D/4, H/4, W/4)
        enc3 = self.encoder3(hidden_states[2])  # (B, 4C, D/8, H/8, W/8)
        enc4 = self.encoder4(hidden_states[3])  # (B, 8C, D/16, H/16, W/16)
        
        # Bottleneck is the deepest representation
        bottleneck = hidden_states[4] if len(hidden_states) > 4 else enc4
        
        skip_features = [enc1, enc2, enc3, enc4]
        
        return skip_features, bottleneck


class UnetrBasicBlock(nn.Module):
    """Basic encoder block with residual connection."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection if dimensions differ
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        x = x + residual
        x = self.relu(x)
        
        return x


# ============================================================================
#                           SWIN POLICY DECODER
# ============================================================================

class SwinPolicyDecoder(nn.Module):
    """
    Policy decoder that upsamples from bottleneck to full resolution.
    Uses skip connections from encoder.
    """
    
    def __init__(self, feature_size: int, out_channels: int):
        super().__init__()
        
        # Projection layer: bottleneck 16C → 8C (to match decoder expectation)
        self.bottleneck_proj = nn.Sequential(
            nn.Conv3d(feature_size * 16, feature_size * 8, kernel_size=1),
            nn.InstanceNorm3d(feature_size * 8),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks with upsampling
        # bottleneck (after proj): 8C → 4C
        self.decoder4 = DecoderBlock(feature_size * 8, feature_size * 4, feature_size * 8)
        # 4C → 2C
        self.decoder3 = DecoderBlock(feature_size * 4, feature_size * 2, feature_size * 4)
        # 2C → C
        self.decoder2 = DecoderBlock(feature_size * 2, feature_size, feature_size * 2)
        # C → C
        self.decoder1 = DecoderBlock(feature_size, feature_size, feature_size)
        
        # Final output projection
        self.out = nn.Sequential(
            nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_size, out_channels, kernel_size=1)
        )
    
    def forward(
        self, 
        bottleneck: torch.Tensor, 
        skip_features: List[torch.Tensor],
        original_padded_size: Tuple[int, int, int] = None
    ) -> torch.Tensor:
        """
        Args:
            bottleneck: (B, 16C, D/16, H/16, W/16) from SwinTransformer
            skip_features: [enc1, enc2, enc3, enc4] at different resolutions
            original_padded_size: Target output size (D, H, W) before cropping
        """
        enc1, enc2, enc3, enc4 = skip_features
        
        # Project bottleneck from 16C to 8C
        x = self.bottleneck_proj(bottleneck)  # → (B, 8C, D/16, H/16, W/16)
        
        # Upsample path with skip connections
        x = self.decoder4(x, enc4)            # → (B, 4C, D/8, H/8, W/8)
        x = self.decoder3(x, enc3)            # → (B, 2C, D/4, H/4, W/4)
        x = self.decoder2(x, enc2)            # → (B, C, D/2, H/2, W/2)
        x = self.decoder1(x, enc1)            # → (B, C, D/2, H/2, W/2) (enc1 is half res)
        
        # Final upsample to full resolution
        if original_padded_size is not None:
            x = F.interpolate(x, size=original_padded_size, mode='trilinear', align_corners=False)
        else:
            # Fallback: upsample 2x
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        
        # Final projection to policy logits
        policy = self.out(x)  # → (B, 2, D, H, W)
        
        return policy


class DecoderBlock(nn.Module):
    """Decoder block with transposed conv + skip connection fusion."""
    
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int):
        super().__init__()
        
        # Upsample
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels, 
            kernel_size=2, stride=2
        )
        
        # Fusion after concatenation with skip
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.fusion(x)
        
        return x


# ============================================================================
#                           SIMPLE BACKBONE (FALLBACK)
# ============================================================================

class SimpleEncoder(nn.Module):
    """
    Simple 3D CNN encoder for testing and fallback.
    """
    
    def __init__(self, in_channels: int, feature_size: int):
        super().__init__()
        
        # Stage 1: full res → half res
        self.stage1 = nn.Sequential(
            nn.Conv3d(in_channels, feature_size, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d(2)
            
        # Stage 2: half res → quarter res
        self.stage2 = nn.Sequential(
            nn.Conv3d(feature_size, feature_size * 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_size * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_size * 2, feature_size * 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_size * 2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool3d(2)
        
        # Stage 3: bottleneck
        self.stage3 = nn.Sequential(
            nn.Conv3d(feature_size * 2, feature_size * 4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_size * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_size * 4, feature_size * 4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_size * 4),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Returns:
            skip_features: [enc1, enc2] for decoder skip connections
            bottleneck: Deepest feature map
        """
        enc1 = self.stage1(x)           # (B, C, D, H, W)
        x = self.pool1(enc1)            # (B, C, D/2, H/2, W/2)
        
        enc2 = self.stage2(x)           # (B, 2C, D/2, H/2, W/2)
        x = self.pool2(enc2)            # (B, 2C, D/4, H/4, W/4)
        
        bottleneck = self.stage3(x)     # (B, 4C, D/4, H/4, W/4)
        
        skip_features = [enc1, enc2, None, None]  # Match Swin format
        
        return skip_features, bottleneck


class SimplePolicyDecoder(nn.Module):
    """
    Simple policy decoder for testing and fallback.
    """
    
    def __init__(self, feature_size: int, out_channels: int):
        super().__init__()
        
        # Upsample from bottleneck
        self.up1 = nn.ConvTranspose3d(feature_size * 4, feature_size * 2, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv3d(feature_size * 4, feature_size * 2, kernel_size=3, padding=1),  # concat with enc2
            nn.InstanceNorm3d(feature_size * 2),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.ConvTranspose3d(feature_size * 2, feature_size, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv3d(feature_size * 2, feature_size, kernel_size=3, padding=1),  # concat with enc1
            nn.InstanceNorm3d(feature_size),
            nn.ReLU(inplace=True),
        )
    
        # Final output
        self.out = nn.Sequential(
            nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_size, out_channels, kernel_size=1)
        )
    
    def forward(self, bottleneck: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        enc1, enc2, _, _ = skip_features
        
        x = self.up1(bottleneck)
        if x.shape[2:] != enc2.shape[2:]:
            x = F.interpolate(x, size=enc2.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, enc2], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        if x.shape[2:] != enc1.shape[2:]:
            x = F.interpolate(x, size=enc1.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, enc1], dim=1)
        x = self.conv2(x)
        
        policy = self.out(x)
        
        return policy
