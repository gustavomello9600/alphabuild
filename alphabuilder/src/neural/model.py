"""
Neural Network Model for AlphaBuilder v3.3.

Architecture with shared spatial encoder/decoder backbone and three heads:
- Value Head: scale-aware bottleneck + global pooling + physics stats -> MLP -> [-1, 1]
- Policy Head: full-resolution decoder features -> (B, 2, D, H, W) Add/Remove logits
- Blueprint Head (aux, Wu 2019): full-resolution decoder features -> (B, 1, D, H, W) final structure logits

Supports SwinUNETR (MONAI, with full-resolution enc0 skip) and upgraded SimpleBackbone (RF >= 64).
Uses GroupNorm (affine=True) everywhere instead of InstanceNorm3d so BC/force scales survive.
"""
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .phys_uvit import PhysUViT3D

try:
    from monai.networks.nets.swin_unetr import SwinTransformer
    HAS_MONAI = True
except ImportError:
    SwinTransformer = None  # type: ignore[assignment]
    HAS_MONAI = False


def _make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Create an affine GroupNorm layer whose group count cleanly divides num_channels."""
    target = min(max_groups, max(1, num_channels // 4))
    groups = 1
    for g in range(target, 0, -1):
        if num_channels % g == 0:
            groups = g
            break
    return nn.GroupNorm(groups, num_channels, affine=True)


class ResidualConvBlock3d(nn.Module):
    """3D residual block with GroupNorm and SiLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _make_group_norm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _make_group_norm(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


# Keep public name UnetrBasicBlock for compatibility while using GroupNorm
UnetrBasicBlock = ResidualConvBlock3d


class AlphaBuilderV31(nn.Module):
    """
    AlphaBuilder Neural Network (v3.3).

    Input:  (B, 7, D, H, W) - [density, mask_x, mask_y, mask_z, fx, fy, fz]
    Output:
        - Default `forward(x)`: (policy, value)
            - policy: (B, 2, D, H, W) Add/Remove logits (isolated from Blueprint Head)
            - value:  (B, 1) quality scalar in [-1, 1]
        - `forward(x, return_blueprint=True)`: (policy, value, blueprint)
            - blueprint: (B, 1, D, H, W) auxiliary final structure prediction logits
    """

    PHYS_STATS_DIM = 9

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 2,
        feature_size: int = 24,
        use_swin: bool = True,
    ):
        super().__init__()

        if use_swin and not HAS_MONAI:
            raise ImportError(
                "use_swin=True requires MONAI (`monai.networks.nets.swin_unetr.SwinTransformer`). "
                "Install `monai` or instantiate AlphaBuilderV31 with use_swin=False."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.use_swin = bool(use_swin)
        self.architecture = "swin" if self.use_swin else "simple"
        # Spatial channels: raw input + 3 normalized CoordConv channels + 1 valid domain mask
        self.aug_in_channels = in_channels + 4

        if self.use_swin:
            self.bottleneck_dim = feature_size * 16
            self.encoder = SwinEncoder(self.aug_in_channels, feature_size)
            self.policy_decoder = SwinPolicyDecoder(feature_size)
        else:
            self.bottleneck_dim = feature_size * 8
            self.encoder = SimpleEncoder(self.aug_in_channels, feature_size)
            self.policy_decoder = SimplePolicyDecoder(feature_size)

        # Scale-preserving physics conditioning (invariant to D4 x Z2 rigid symmetries)
        self.phys_film = nn.Sequential(
            nn.Linear(self.PHYS_STATS_DIM, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, self.bottleneck_dim * 2),
        )
        # Small initial modulation so force/BC scales influence features immediately without destabilizing
        with torch.no_grad():
            self.phys_film[-1].weight.mul_(0.1)
            self.phys_film[-1].bias.zero_()

        self.phys_value_proj = nn.Sequential(
            nn.Linear(self.PHYS_STATS_DIM, 64),
            nn.SiLU(inplace=True),
        )

        # Unnormalized full-resolution skip from raw BC/force channels directly to decoded features
        self.raw_phys_proj = nn.Conv3d(in_channels, feature_size, kernel_size=1)

        # Value Head: avg pool + max pool of bottleneck + invariant physics summary -> [-1, 1]
        self.value_head = nn.Sequential(
            nn.Linear(self.bottleneck_dim * 2 + 64, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        # Primary Policy Head (MCTS / inference logits)
        self.policy_head = nn.Sequential(
            nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1, bias=False),
            _make_group_norm(feature_size),
            nn.SiLU(inplace=True),
            nn.Conv3d(feature_size, out_channels, kernel_size=1),
        )

        # Auxiliary Blueprint Head (Wu 2019): predicts V_constructed without polluting MCTS logits
        self.blueprint_head = nn.Sequential(
            nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1, bias=False),
            _make_group_norm(feature_size),
            nn.SiLU(inplace=True),
            nn.Conv3d(feature_size, 1, kernel_size=1),
        )

    @staticmethod
    def _extract_physics_stats(x: torch.Tensor) -> torch.Tensor:
        """
        Extract 9 symmetry-invariant global physics descriptors from unpadded state.

        Preserves absolute load magnitude and FULL_CLAMP vs RAIL_XY BC distinctions
        even under normalized spatial backbones.
        """
        B, C = x.shape[:2]
        if C < 7:
            return x.new_zeros((B, AlphaBuilderV31.PHYS_STATS_DIM))

        vol_mean = x[:, 0].mean(dim=(1, 2, 3))
        bc_x = x[:, 1].mean(dim=(1, 2, 3))
        bc_y = x[:, 2].mean(dim=(1, 2, 3))
        bc_z = x[:, 3].mean(dim=(1, 2, 3))
        bc_xy_mean = 0.5 * (bc_x + bc_y)
        bc_xy_diff = (bc_x - bc_y).abs()

        f_norm = torch.sqrt(x[:, 4] ** 2 + x[:, 5] ** 2 + x[:, 6] ** 2 + 1e-12)
        f_max = f_norm.amax(dim=(1, 2, 3))
        f_log_max = torch.log1p(f_max)

        fx_l1 = x[:, 4].abs().mean(dim=(1, 2, 3))
        fy_l1 = x[:, 5].abs().mean(dim=(1, 2, 3))
        fz_l1 = x[:, 6].abs().mean(dim=(1, 2, 3))
        f_xy_mean = 0.5 * (fx_l1 + fy_l1)
        f_xy_diff = (fx_l1 - fy_l1).abs()

        return torch.stack(
            [
                vol_mean,
                bc_xy_mean,
                bc_xy_diff,
                bc_z,
                f_max,
                f_log_max,
                f_xy_mean,
                f_xy_diff,
                fz_l1,
            ],
            dim=1,
        )

    @staticmethod
    def _build_augmented_input(
        x: torch.Tensor,
        divisor: int,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int, int]]:
        """
        Append normalized CoordConv + valid domain mask and apply centered padding.
        """
        B, _, D, H, W = x.shape
        device, dtype = x.device, x.dtype

        cd = torch.linspace(-1.0, 1.0, steps=D, device=device, dtype=dtype).view(1, 1, D, 1, 1).expand(B, 1, D, H, W)
        ch = torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=dtype).view(1, 1, 1, H, 1).expand(B, 1, D, H, W)
        cw = torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=dtype).view(1, 1, 1, 1, W).expand(B, 1, D, H, W)
        domain_mask = torch.ones((B, 1, D, H, W), device=device, dtype=dtype)

        x_aug = torch.cat([x, cd, ch, cw, domain_mask], dim=1)

        pad_d = (divisor - D % divisor) % divisor
        pad_h = (divisor - H % divisor) % divisor
        pad_w = (divisor - W % divisor) % divisor

        d0 = pad_d // 2
        d1 = pad_d - d0
        h0 = pad_h // 2
        h1 = pad_h - h0
        w0 = pad_w // 2
        w1 = pad_w - w0

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x_aug = F.pad(x_aug, (w0, w1, h0, h1, d0, d1), mode="constant", value=0.0)

        return x_aug, (d0, D, h0, H, w0, W)

    @staticmethod
    def _crop_centered(
        tensor: torch.Tensor,
        crop_box: Tuple[int, int, int, int, int, int],
    ) -> torch.Tensor:
        """Crop centered-padded tensor back to original (D, H, W)."""
        d0, D, h0, H, w0, W = crop_box
        return tensor[:, :, d0 : d0 + D, h0 : h0 + H, w0 : w0 + W]

    def forward(
        self,
        x: torch.Tensor,
        return_blueprint: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, D, H, W)
            return_blueprint: If True, also return auxiliary blueprint logits (B, 1, D, H, W)

        Returns:
            (policy, value) or (policy, value, blueprint_logits)
        """
        phys_stats = self._extract_physics_stats(x)
        divisor = 32 if self.use_swin else 8
        x_aug, crop_box = self._build_augmented_input(x, divisor=divisor)

        skip_features, bottleneck = self.encoder(x_aug)

        # Apply invariant physics FiLM conditioning at the bottleneck
        film = self.phys_film(phys_stats)
        gamma, beta = film.chunk(2, dim=1)
        gamma = gamma.view(x.shape[0], self.bottleneck_dim, 1, 1, 1)
        beta = beta.view(x.shape[0], self.bottleneck_dim, 1, 1, 1)
        bottleneck = bottleneck * (1.0 + gamma) + beta

        # Value Head from pooled bottleneck + physics summary
        avg_pool = F.adaptive_avg_pool3d(bottleneck, 1).flatten(1)
        max_pool = F.adaptive_max_pool3d(bottleneck, 1).flatten(1)
        phys_emb = self.phys_value_proj(phys_stats)
        value = self.value_head(torch.cat([avg_pool, max_pool, phys_emb], dim=1))

        # Shared spatial decoder features cropped back to unpadded (D, H, W)
        dec_feat = self.policy_decoder(bottleneck, skip_features)
        dec_feat = self._crop_centered(dec_feat, crop_box)
        dec_feat = dec_feat + self.raw_phys_proj(x)

        policy = self.policy_head(dec_feat)

        if return_blueprint:
            blueprint = self.blueprint_head(dec_feat)
            return policy, value, blueprint
        return policy, value


# ============================================================================
#                           SWIN TRANSFORMER BACKBONE
# ============================================================================


class SwinEncoder(nn.Module):
    """
    Swin Transformer Encoder with full-resolution residual enc0 + 4 hierarchical stages.
    """

    def __init__(self, in_channels: int, feature_size: int):
        super().__init__()
        if SwinTransformer is None:
            raise ImportError("MONAI SwinTransformer is not available.")

        # Full-resolution skip stem (mirrors Hatamizadeh et al., Swin UNETR)
        self.encoder0 = ResidualConvBlock3d(in_channels, feature_size)

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

        self.encoder1 = ResidualConvBlock3d(feature_size, feature_size)
        self.encoder2 = ResidualConvBlock3d(feature_size * 2, feature_size * 2)
        self.encoder3 = ResidualConvBlock3d(feature_size * 4, feature_size * 4)
        self.encoder4 = ResidualConvBlock3d(feature_size * 8, feature_size * 8)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        enc0 = self.encoder0(x)  # (B, C, D, H, W)
        hidden_states = self.swin(x)
        # MONAI SwinTransformer returns [D/2 (C), D/4 (2C), D/8 (4C), D/16 (8C), D/32 (16C)]
        enc1 = self.encoder1(hidden_states[0])
        enc2 = self.encoder2(hidden_states[1])
        enc3 = self.encoder3(hidden_states[2])
        enc4 = self.encoder4(hidden_states[3])
        bottleneck = hidden_states[4]
        return [enc0, enc1, enc2, enc3, enc4], bottleneck


class DecoderBlock(nn.Module):
    """3D Decoder block with transposed conv + skip connection fusion (GroupNorm)."""

    def __init__(self, in_channels: int, out_channels: int, skip_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_group_norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_group_norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fusion(x)


class SwinPolicyDecoder(nn.Module):
    """
    Swin UNETR spatial decoder reaching full resolution via decoder0 + enc0 skip connection.
    """

    def __init__(self, feature_size: int):
        super().__init__()
        self.bottleneck_proj = nn.Sequential(
            nn.Conv3d(feature_size * 16, feature_size * 8, kernel_size=1, bias=False),
            _make_group_norm(feature_size * 8),
            nn.SiLU(inplace=True),
        )
        self.decoder4 = DecoderBlock(feature_size * 8, feature_size * 4, feature_size * 8)  # D/32 -> D/16
        self.decoder3 = DecoderBlock(feature_size * 4, feature_size * 2, feature_size * 4)  # D/16 -> D/8
        self.decoder2 = DecoderBlock(feature_size * 2, feature_size, feature_size * 2)      # D/8  -> D/4
        self.decoder1 = DecoderBlock(feature_size, feature_size, feature_size)              # D/4  -> D/2
        self.decoder0 = DecoderBlock(feature_size, feature_size, feature_size)              # D/2  -> D

    def forward(
        self,
        bottleneck: torch.Tensor,
        skip_features: List[torch.Tensor],
    ) -> torch.Tensor:
        enc0, enc1, enc2, enc3, enc4 = skip_features
        x = self.bottleneck_proj(bottleneck)
        x = self.decoder4(x, enc4)
        x = self.decoder3(x, enc3)
        x = self.decoder2(x, enc2)
        x = self.decoder1(x, enc1)
        x = self.decoder0(x, enc0)
        return x


# ============================================================================
#                           SIMPLE BACKBONE (RF >= 64)
# ============================================================================


class SimpleEncoder(nn.Module):
    """
    4-stage 3D Residual CNN encoder with dilated bottleneck (D-axis RF >= 64) and global context.
    """

    def __init__(self, in_channels: int, feature_size: int):
        super().__init__()
        C = feature_size

        self.stage1 = ResidualConvBlock3d(in_channels, C)   # 1x
        self.pool1 = nn.MaxPool3d(2)

        self.stage2 = ResidualConvBlock3d(C, C * 2)         # 1/2
        self.pool2 = nn.MaxPool3d(2)

        self.stage3 = ResidualConvBlock3d(C * 2, C * 4)     # 1/4
        self.pool3 = nn.MaxPool3d(2)

        # Stage 4 (1/8 resolution): dilated residual stack -> theoretical D-RF = 148 >= 64
        self.b1_conv = nn.Sequential(
            nn.Conv3d(C * 4, C * 8, kernel_size=3, padding=1, bias=False),
            _make_group_norm(C * 8),
            nn.SiLU(inplace=True),
        )
        self.b1_skip = nn.Conv3d(C * 4, C * 8, kernel_size=1, bias=False)

        self.b2_dilated = nn.Sequential(
            nn.Conv3d(C * 8, C * 8, kernel_size=3, dilation=(2, 2, 1), padding=(2, 2, 1), bias=False),
            _make_group_norm(C * 8),
            nn.SiLU(inplace=True),
        )
        self.b3_dilated = nn.Sequential(
            nn.Conv3d(C * 8, C * 8, kernel_size=3, dilation=(4, 2, 1), padding=(4, 2, 1), bias=False),
            _make_group_norm(C * 8),
            nn.SiLU(inplace=True),
        )
        self.global_ctx = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(C * 8, C * 4, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(C * 4, C * 8, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[List[Optional[torch.Tensor]], torch.Tensor]:
        enc1 = self.stage1(x)
        x = self.pool1(enc1)

        enc2 = self.stage2(x)
        x = self.pool2(enc2)

        enc3 = self.stage3(x)
        x = self.pool3(enc3)

        b = self.b1_conv(x) + self.b1_skip(x)
        b = b + self.b2_dilated(b)
        b = b + self.b3_dilated(b)
        b = b + self.global_ctx(b)

        return [enc1, enc2, enc3, None], b


class SimplePolicyDecoder(nn.Module):
    """
    4-stage 3D U-Net decoder with skip connections and global bottleneck conditioning.
    """

    def __init__(self, feature_size: int):
        super().__init__()
        C = feature_size

        self.up1 = nn.ConvTranspose3d(C * 8, C * 4, kernel_size=2, stride=2)
        self.conv1 = ResidualConvBlock3d(C * 8, C * 4)

        self.up2 = nn.ConvTranspose3d(C * 4, C * 2, kernel_size=2, stride=2)
        self.conv2 = ResidualConvBlock3d(C * 4, C * 2)

        self.up3 = nn.ConvTranspose3d(C * 2, C, kernel_size=2, stride=2)
        self.conv3 = ResidualConvBlock3d(C * 2, C)

        self.bottleneck_to_full = nn.Linear(C * 8, C)

    def forward(
        self,
        bottleneck: torch.Tensor,
        skip_features: List[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        enc1, enc2, enc3, _ = skip_features
        assert enc1 is not None and enc2 is not None and enc3 is not None

        x = self.up1(bottleneck)
        if x.shape[2:] != enc3.shape[2:]:
            x = F.interpolate(x, size=enc3.shape[2:], mode="trilinear", align_corners=False)
        x = self.conv1(torch.cat([x, enc3], dim=1))

        x = self.up2(x)
        if x.shape[2:] != enc2.shape[2:]:
            x = F.interpolate(x, size=enc2.shape[2:], mode="trilinear", align_corners=False)
        x = self.conv2(torch.cat([x, enc2], dim=1))

        x = self.up3(x)
        if x.shape[2:] != enc1.shape[2:]:
            x = F.interpolate(x, size=enc1.shape[2:], mode="trilinear", align_corners=False)
        x = self.conv3(torch.cat([x, enc1], dim=1))

        # Inject global bottleneck summary uniformly across full resolution
        global_emb = self.bottleneck_to_full(F.adaptive_avg_pool3d(bottleneck, 1).flatten(1))
        x = x + global_emb.view(x.shape[0], -1, 1, 1, 1)
        return x


VALID_ARCHITECTURES = ("simple", "swin", "phys_uvit")


def resolve_architecture(
    config: Optional[Dict[str, Any]] = None,
    use_swin: Optional[bool] = None,
) -> str:
    """Resolve canonical neural architecture name from checkpoint/run config."""
    cfg = config or {}
    arch = cfg.get("architecture")
    if isinstance(arch, str) and arch.strip():
        norm = arch.strip().lower().replace("-", "_").replace(" ", "_")
        if norm in ("swin_unetr", "swinunetr"):
            norm = "swin"
        elif norm in ("physuvit", "phys_uvit_3d", "physuvit3d"):
            norm = "phys_uvit"
        if norm not in VALID_ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {arch!r}. Supported: {VALID_ARCHITECTURES}")
        return norm
    swin_flag = bool(use_swin) if use_swin is not None else bool(cfg.get("use_swin", False))
    return "swin" if swin_flag else "simple"


def build_model(
    architecture: Optional[str] = None,
    in_channels: int = 7,
    out_channels: int = 2,
    feature_size: int = 24,
    use_swin: Optional[bool] = None,
    embed_dim: int = 192,
    transformer_depth: int = 5,
    num_heads: int = 6,
) -> nn.Module:
    """Instantiate AlphaBuilder neural architecture cleanly across v3.3 and v4.0."""
    arch = resolve_architecture(
        {"architecture": architecture, "use_swin": use_swin} if architecture is not None else None,
        use_swin=use_swin,
    )
    if arch == "phys_uvit":
        return PhysUViT3D(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            embed_dim=embed_dim,
            transformer_depth=transformer_depth,
            num_heads=num_heads,
        )
    return AlphaBuilderV31(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_swin=(arch == "swin"),
    )
