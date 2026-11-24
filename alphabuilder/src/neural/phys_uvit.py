"""
Phys-UViT 3D Neural Architecture for AlphaBuilder v4.0.

Implements a shallow 3D ConvNeXt/Residual U-Net paired with a Global Physics
Transformer bottleneck at 1/4x resolution (divisor=4 -> 0% zero-padding on 64x32x8):
- Arbitrary Spatial Input: (B, 7, D, H, W) + optional ternary domain_mask (keep-out)
- Multi-Load Encoder: set-equivariant encoding for K load cases -> Cross-Attention
- Global Physics Context Rail: D4 x Z2 invariant stats -> AdaLN-Zero / FiLM conditioning
- Shallow 3D Conv Encoder (1x -> 1/2x -> 1/4x) with GroupNorm + SiLU
- Global Physics Transformer Bottleneck: AdaLN + 3D RoPE Masked Global MHSA + Cross-Attn + SwiGLU
- Factored Value Head: [AvgPool, MaxPool (disconnection veto), AttnPool] + phys_emb
  -> log10(C), P(connected), V(s) in [-1, 1]
- Shallow 3D U-Net Decoder + Global Broadcast -> full-resolution Policy (AdaLN),
  optional Blueprint (warmup), and optional Physics SED + u heads.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Create an affine GroupNorm layer whose group count cleanly divides num_channels."""
    target = min(max_groups, max(1, num_channels // 4))
    groups = 1
    for g in range(target, 0, -1):
        if num_channels % g == 0:
            groups = g
            break
    return nn.GroupNorm(groups, num_channels, affine=True)


class ResidualConvBlock3d(nn.Module):
    """3D residual block with affine GroupNorm and SiLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = make_group_norm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = make_group_norm(out_channels)
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


class ConvNeXtBlock3D(nn.Module):
    """Lightweight 3D ConvNeXt block using GroupNorm and pointwise expansion."""

    def __init__(self, dim: int, mlp_ratio: float = 2.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.norm = make_group_norm(dim)
        self.pwconv1 = nn.Conv3d(dim, hidden_dim, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.pwconv2 = nn.Conv3d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(self.dwconv(x))
        h = self.pwconv2(self.act(self.pwconv1(h)))
        return residual + h


def _split_rope_dims(head_dim: int) -> Tuple[int, int, int]:
    """Split an even head_dim into even (d_x, d_y, d_z) counts with d_x == d_y."""
    if head_dim % 2 != 0:
        raise ValueError(f"3D RoPE requires even head_dim, got {head_dim}")
    pairs = head_dim // 2
    p_z = max(1, pairs // 4)
    if (pairs - p_z) % 2 != 0:
        p_z += 1
    p_xy = (pairs - p_z) // 2
    return p_xy * 2, p_xy * 2, p_z * 2


def build_3d_rope_cos_sin(
    coords_xyz: torch.Tensor,
    head_dim: int,
    theta: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build real-valued axial 3D RoPE cos/sin embeddings.

    Args:
        coords_xyz: Tensor of shape (..., N, 3) with (x, y, z) coordinates
        head_dim: Attention head dimension (must be even)
        theta: Base frequency

    Returns:
        cos, sin tensors of shape (..., 1, N, head_dim)
    """
    d_x, d_y, d_z = _split_rope_dims(head_dim)
    cos_parts = []
    sin_parts = []

    for axis_idx, d_axis in enumerate((d_x, d_y, d_z)):
        if d_axis == 0:
            continue
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_axis, 2, device=coords_xyz.device, dtype=torch.float32) / float(d_axis))
        )
        pos = coords_xyz[..., axis_idx].float()
        angles = pos.unsqueeze(-1) * inv_freq  # (..., N, d_axis // 2)
        emb = torch.cat([angles, angles], dim=-1)  # (..., N, d_axis)
        cos_parts.append(torch.cos(emb))
        sin_parts.append(torch.sin(emb))

    cos = torch.cat(cos_parts, dim=-1).unsqueeze(-3)
    sin = torch.cat(sin_parts, dim=-1).unsqueeze(-3)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_3d_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply axial 3D RoPE to q/k of shape (B, H, N, head_dim)."""
    head_dim = x.shape[-1]
    d_x, d_y, d_z = _split_rope_dims(head_dim)
    cos_t = cos.to(dtype=x.dtype, device=x.device)
    sin_t = sin.to(dtype=x.dtype, device=x.device)

    splits = [d for d in (d_x, d_y, d_z) if d > 0]
    x_chunks = torch.split(x, splits, dim=-1)
    cos_chunks = torch.split(cos_t, splits, dim=-1)
    sin_chunks = torch.split(sin_t, splits, dim=-1)

    out_chunks = [
        (xc * cc) + (_rotate_half(xc) * sc)
        for xc, cc, sc in zip(x_chunks, cos_chunks, sin_chunks)
    ]
    return torch.cat(out_chunks, dim=-1)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ShallowConvEncoder3D(nn.Module):
    """
    Shallow 3-stage 3D encoder (1x -> 1/2x -> 1/4x) preserving exact 3D depth Z=2 on 64x32x8.
    """

    def __init__(self, in_channels: int, feature_size: int = 32, embed_dim: int = 192):
        super().__init__()
        C = feature_size
        self.stage1 = ResidualConvBlock3d(in_channels, C)  # 1x
        self.down1 = nn.Sequential(
            nn.Conv3d(C, C * 2, kernel_size=2, stride=2, bias=False),
            make_group_norm(C * 2),
            nn.SiLU(inplace=True),
        )
        self.stage2 = ResidualConvBlock3d(C * 2, C * 2)  # 1/2x
        self.down2 = nn.Sequential(
            nn.Conv3d(C * 2, embed_dim, kernel_size=2, stride=2, bias=False),
            make_group_norm(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            ConvNeXtBlock3D(embed_dim, mlp_ratio=2.0),
            ConvNeXtBlock3D(embed_dim, mlp_ratio=2.0),
        )  # 1/4x

    def forward(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        enc1 = self.stage1(x)
        x2 = self.down1(enc1)
        enc2 = self.stage2(x2)
        x4 = self.down2(enc2)
        bottleneck = self.stage3(x4)
        return (enc1, enc2), bottleneck


class MultiLoadEncoder3D(nn.Module):
    """
    Set-equivariant Multi-Load Encoder converting K independent 3D load cases (B, K, 3, Dp, Hp, Wp)
    into compact load tokens with geometric 3D coordinates for Cross-Attention.
    """

    def __init__(self, embed_dim: int = 192):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=2, stride=2, bias=False),
            make_group_norm(32),
            nn.SiLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=2, stride=2, bias=False),
            make_group_norm(64),
            nn.SiLU(inplace=True),
            nn.Conv3d(64, embed_dim, kernel_size=1, bias=False),
            make_group_norm(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.pool_proj = nn.Sequential(
            nn.Conv3d(embed_dim * 2, embed_dim, kernel_size=1, bias=False),
            make_group_norm(embed_dim),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        loads: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            loads: Tensor of shape (B, K, 3, Dp, Hp, Wp)

        Returns:
            load_tokens: (B, K * N_L, embed_dim)
            load_coords: (B, K * N_L, 3) aligned with 1/4x lattice coordinates
        """
        B, K, C, Dp, Hp, Wp = loads.shape
        flat = loads.reshape(B * K, C, Dp, Hp, Wp)
        feat = self.stem(flat)  # (B*K, E, Dt, Ht, Wt)
        _, E, Dt, Ht, Wt = feat.shape

        sx = 2 if Dt >= 2 else 1
        sy = 2 if Ht >= 2 else 1
        sz = 2 if Wt >= 2 else 1
        k_pool = (sx, sy, sz)

        avg_f = F.avg_pool3d(feat, kernel_size=k_pool, stride=k_pool, ceil_mode=True)
        signed_peak = F.max_pool3d(feat, kernel_size=k_pool, stride=k_pool, ceil_mode=True) - F.max_pool3d(
            -feat, kernel_size=k_pool, stride=k_pool, ceil_mode=True
        )
        pooled = self.pool_proj(torch.cat([avg_f, signed_peak], dim=1))
        _, _, Dl, Hl, Wl = pooled.shape

        load_tokens = pooled.flatten(2).transpose(1, 2).reshape(B, K * Dl * Hl * Wl, E)

        # Compute exact patch center coordinates in the (Dt, Ht, Wt) frame
        gx = torch.arange(Dt, device=loads.device, dtype=torch.float32)
        gy = torch.arange(Ht, device=loads.device, dtype=torch.float32)
        gz = torch.arange(Wt, device=loads.device, dtype=torch.float32)
        mx, my, mz = torch.meshgrid(gx, gy, gz, indexing="ij")
        grid = torch.stack([mx, my, mz], dim=0).unsqueeze(0)  # (1, 3, Dt, Ht, Wt)
        pooled_coords = F.avg_pool3d(grid, kernel_size=k_pool, stride=k_pool, ceil_mode=True)
        coords_single = pooled_coords.squeeze(0).flatten(1).transpose(0, 1)  # (N_L, 3)
        load_coords = coords_single.unsqueeze(0).repeat(K, 1, 1).reshape(1, K * Dl * Hl * Wl, 3).expand(B, -1, -1)
        return load_tokens, load_coords


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate_val = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w_out = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.w_gate_val(x).chunk(2, dim=-1)
        return self.w_out(self.drop(F.silu(gate) * val))


class GlobalPhysicsTransformerBlock(nn.Module):
    """
    Unified Global Physics Transformer Block:
    1. AdaLN(phys) + 3D RoPE Masked Global MHSA
    2. AdaLN(phys) + Multi-Load Cross-Attention
    3. AdaLN(phys) + SwiGLU FFN
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 6,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Bottlenecked AdaLN + LayerScale-style residual gate generator
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 9 * embed_dim),
        )
        with torch.no_grad():
            self.ada_ln[-1].weight.mul_(0.05)
            bias_init = torch.zeros(9, embed_dim)
            bias_init[2] = 0.25  # gate1 (MHSA)
            bias_init[5] = 0.25  # gate2 (Multi-Load Cross-Attn)
            bias_init[8] = 0.25  # gate3 (SwiGLU)
            self.ada_ln[-1].bias.copy_(bias_init.reshape(-1))

        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj_self = nn.Linear(embed_dim, embed_dim, bias=False)

        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.q_cross = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_cross = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.proj_cross = nn.Linear(embed_dim, embed_dim, bias=False)

        self.norm3 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.ffn = SwiGLU(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        phys_emb: torch.Tensor,
        grid_cos: torch.Tensor,
        grid_sin: torch.Tensor,
        load_tokens: torch.Tensor,
        load_cos: torch.Tensor,
        load_sin: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, E)
            phys_emb: (B, E)
            grid_cos/sin: (1 or B, 1, N, head_dim)
            load_tokens: (B, M, E)
            load_cos/sin: (1 or B, 1, M, head_dim)
            token_valid: optional (B, N) bool tensor
            attn_bias: optional (B, 1, 1, N) additive float attention bias
        """
        B, N, E = tokens.shape
        H, Dh = self.num_heads, self.head_dim

        chunks = self.ada_ln(phys_emb).chunk(9, dim=-1)
        shift1, scale1, gate1, shift2, scale2, gate2, shift3, scale3, gate3 = chunks

        # 1. Masked Global 3D RoPE Self-Attention
        h1 = _modulate(self.norm1(tokens), shift1, scale1)
        qkv = self.qkv(h1).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q_rot = apply_3d_rope(q, grid_cos, grid_sin)
        k_rot = apply_3d_rope(k, grid_cos, grid_sin)
        bias_use = attn_bias.to(dtype=q_rot.dtype) if attn_bias is not None else None
        attn_out = F.scaled_dot_product_attention(q_rot, k_rot, v, attn_mask=bias_use)
        attn_out = self.proj_self(attn_out.transpose(1, 2).reshape(B, N, E))
        tokens = tokens + gate1.unsqueeze(1) * attn_out

        # 2. Multi-Load Cross-Attention (with aligned 3D RoPE on queries and load keys)
        M = load_tokens.shape[1]
        h2 = _modulate(self.norm2(tokens), shift2, scale2)
        qc = apply_3d_rope(self.q_cross(h2).reshape(B, N, H, Dh).transpose(1, 2), grid_cos, grid_sin)
        kv_c = self.kv_cross(load_tokens).reshape(B, M, 2, H, Dh).permute(2, 0, 3, 1, 4)
        kc = apply_3d_rope(kv_c[0], load_cos, load_sin)
        vc = kv_c[1]
        cross_out = F.scaled_dot_product_attention(qc, kc, vc)
        cross_out = self.proj_cross(cross_out.transpose(1, 2).reshape(B, N, E))
        tokens = tokens + gate2.unsqueeze(1) * cross_out

        # 3. AdaLN(phys) + SwiGLU FFN
        h3 = _modulate(self.norm3(tokens), shift3, scale3)
        tokens = tokens + gate3.unsqueeze(1) * self.ffn(h3)

        if token_valid is not None:
            tokens = tokens * token_valid.unsqueeze(-1).to(dtype=tokens.dtype)
        return tokens


class FactoredValueHead(nn.Module):
    """
    Factored Value Head using [AvgPool, MaxPool (disconnection veto), AttnPool] + phys_emb
    to predict decoupled physical observables (log10(C), conn_logit) and scalar V(s) in [-1, 1].
    """

    def __init__(self, embed_dim: int = 192):
        super().__init__()
        self.attn_score = nn.Linear(embed_dim, 1)
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim * 4, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
        )
        self.log_compliance_head = nn.Linear(128, 1)
        nn.init.constant_(self.log_compliance_head.bias, 3.0)

        self.conn_head = nn.Linear(128, 1)

        self.value_head = nn.Sequential(
            nn.Linear(128 + 2, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        phys_emb: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if token_valid is None:
            avg_pool = tokens.mean(dim=1)
            max_pool = tokens.amax(dim=1)
            attn_weights = F.softmax(self.attn_score(tokens), dim=1)
            attn_pool = (tokens * attn_weights).sum(dim=1)
        else:
            mask_f = token_valid.unsqueeze(-1).to(dtype=tokens.dtype)
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            avg_pool = (tokens * mask_f).sum(dim=1) / denom

            # Masked MaxPool veto
            neg_large = torch.finfo(tokens.dtype).min / 2.0
            masked_for_max = tokens.masked_fill(~token_valid.unsqueeze(-1), neg_large)
            max_pool = masked_for_max.amax(dim=1)

            logits = self.attn_score(tokens).masked_fill(~token_valid.unsqueeze(-1), -1e4)
            attn_weights = F.softmax(logits, dim=1) * mask_f
            attn_pool = (tokens * attn_weights).sum(dim=1)

        feat = torch.cat([avg_pool, max_pool, attn_pool, phys_emb], dim=1)
        h = self.trunk(feat)
        log_c = self.log_compliance_head(h)
        conn_logit = self.conn_head(h)
        conn_prob = torch.sigmoid(conn_logit)
        value = self.value_head(torch.cat([h, log_c, conn_prob], dim=1))
        return value, log_c, conn_logit


class ShallowConvDecoder3D(nn.Module):
    """Shallow 3D U-Net decoder (1/4x -> 1/2x -> 1x) with global context broadcast."""

    def __init__(self, feature_size: int = 32, embed_dim: int = 192):
        super().__init__()
        C = feature_size
        self.up1 = nn.ConvTranspose3d(embed_dim, C * 2, kernel_size=2, stride=2)
        self.conv1 = ResidualConvBlock3d(C * 4, C * 2)

        self.up2 = nn.ConvTranspose3d(C * 2, C, kernel_size=2, stride=2)
        self.conv2 = ResidualConvBlock3d(C * 2, C)

        self.global_bcast = nn.Linear(embed_dim, C)

    def forward(
        self,
        bottleneck: torch.Tensor,
        skip_features: Tuple[torch.Tensor, torch.Tensor],
        global_tokens_mean: torch.Tensor,
    ) -> torch.Tensor:
        enc1, enc2 = skip_features

        x = self.up1(bottleneck)
        if x.shape[2:] != enc2.shape[2:]:
            x = F.interpolate(x, size=enc2.shape[2:], mode="trilinear", align_corners=False)
        x = self.conv1(torch.cat([x, enc2], dim=1))

        x = self.up2(x)
        if x.shape[2:] != enc1.shape[2:]:
            x = F.interpolate(x, size=enc1.shape[2:], mode="trilinear", align_corners=False)
        x = self.conv2(torch.cat([x, enc1], dim=1))

        g = self.global_bcast(global_tokens_mean).view(x.shape[0], -1, 1, 1, 1)
        return x + g


class PhysUViT3D(nn.Module):
    """
    Phys-UViT 3D: Hybrid Shallow 3D U-Net + Global Physics Transformer Architecture.

    Preserves full compatibility with AlphaBuilderV31 callers:
    - `forward(x)` -> `(policy, value)`
    - `forward(x, return_blueprint=True)` -> `(policy, value, blueprint)`
    - `forward(x, return_aux=True, compute_sed=...)` -> `(policy, value, aux_dict)`
    """

    PHYS_STATS_DIM = 12

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 2,
        feature_size: int = 32,
        embed_dim: int = 192,
        transformer_depth: int = 5,
        num_heads: int = 6,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.architecture = "phys_uvit"
        self.use_swin = False
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.embed_dim = embed_dim
        self.transformer_depth = transformer_depth
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Input channels + valid_mask + keep_out_mask
        self.spatial_in_channels = in_channels + 2

        self.encoder = ShallowConvEncoder3D(
            in_channels=self.spatial_in_channels,
            feature_size=feature_size,
            embed_dim=embed_dim,
        )
        self.load_encoder = MultiLoadEncoder3D(embed_dim=embed_dim)

        self.phys_mlp = nn.Sequential(
            nn.Linear(self.PHYS_STATS_DIM, embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(inplace=True),
        )

        self.blocks = nn.ModuleList(
            [
                GlobalPhysicsTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(transformer_depth)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.factored_value_head = FactoredValueHead(embed_dim=embed_dim)
        self.decoder = ShallowConvDecoder3D(feature_size=feature_size, embed_dim=embed_dim)

        # Unnormalized full-resolution skip directly from raw BC/force channels
        self.raw_phys_proj = nn.Conv3d(in_channels, feature_size, kernel_size=1)

        # AdaLN/FiLM conditioning for phase/volume/physics on the Policy Head
        self.policy_film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 64),
            nn.SiLU(),
            nn.Linear(64, feature_size * 2),
        )
        with torch.no_grad():
            self.policy_film[-1].weight.mul_(0.1)
            self.policy_film[-1].bias.zero_()

        self.policy_head = nn.Sequential(
            nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1, bias=False),
            make_group_norm(feature_size),
            nn.SiLU(inplace=True),
            nn.Conv3d(feature_size, out_channels, kernel_size=1),
        )

        self.blueprint_head = nn.Sequential(
            nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1, bias=False),
            make_group_norm(feature_size),
            nn.SiLU(inplace=True),
            nn.Conv3d(feature_size, 1, kernel_size=1),
        )

        # Auxiliary dense physics prediction head: [SED, ux, uy, uz]
        self.sed_head = nn.Sequential(
            nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1, bias=False),
            make_group_norm(feature_size),
            nn.SiLU(inplace=True),
            nn.Conv3d(feature_size, 4, kernel_size=1),
        )

    @staticmethod
    def _extract_physics_stats(
        x: torch.Tensor,
        domain_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract 12 D4 x Z2 invariant global physics & phase context descriptors.
        """
        B, C = x.shape[:2]
        if C < 7:
            return x.new_zeros((B, PhysUViT3D.PHYS_STATS_DIM))

        rho = x[:, 0]
        if domain_mask is not None:
            dm = domain_mask[:, 0] if domain_mask.ndim == 5 else domain_mask
            valid = (dm > 0.5).to(dtype=x.dtype)
            valid_denom = valid.sum(dim=(1, 2, 3)).clamp(min=1.0)
            rho_v = rho * valid
            vol_mean = rho_v.sum(dim=(1, 2, 3)) / valid_denom
        else:
            rho_v = rho
            vol_mean = rho.mean(dim=(1, 2, 3))

        solid = (rho_v > 0.5).to(dtype=x.dtype)
        vol_solid_frac = solid.mean(dim=(1, 2, 3))

        bc_x = x[:, 1].mean(dim=(1, 2, 3))
        bc_y = x[:, 2].mean(dim=(1, 2, 3))
        bc_z = x[:, 3].mean(dim=(1, 2, 3))
        bc_xy_mean = 0.5 * (bc_x + bc_y)
        bc_xy_diff = (bc_x - bc_y).abs()

        f_norm = torch.sqrt(x[:, 4] ** 2 + x[:, 5] ** 2 + x[:, 6] ** 2 + 1e-12)
        f_max = f_norm.amax(dim=(1, 2, 3))
        f_log_max = torch.log1p(f_max)

        # Invariant coupling indicators (tell Growth vs Refinement phase smoothly)
        load_mask = (f_norm > 1e-3).to(dtype=x.dtype)
        bc_any = ((x[:, 1] > 0.5) | (x[:, 2] > 0.5) | (x[:, 3] > 0.5)).to(dtype=x.dtype)
        load_solid_overlap = (load_mask * solid).sum(dim=(1, 2, 3)) / load_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
        bc_solid_overlap = (bc_any * solid).sum(dim=(1, 2, 3)) / bc_any.sum(dim=(1, 2, 3)).clamp(min=1.0)

        fx_l1 = x[:, 4].abs().mean(dim=(1, 2, 3))
        fy_l1 = x[:, 5].abs().mean(dim=(1, 2, 3))
        fz_l1 = x[:, 6].abs().mean(dim=(1, 2, 3))
        f_xy_mean = 0.5 * (fx_l1 + fy_l1)
        f_xy_diff = (fx_l1 - fy_l1).abs()

        return torch.stack(
            [
                vol_mean,
                vol_solid_frac,
                load_solid_overlap,
                bc_solid_overlap,
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
    def _pad_centered_div4(
        x: torch.Tensor,
        domain_mask: Optional[torch.Tensor],
        loads: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Tuple[int, int, int, int, int, int],
    ]:
        """
        Pad inputs symmetrically to a multiple of 4 (0% padding on 64x32x8 and 32x64x8).
        """
        B, _, D, H, W = x.shape
        device, dtype = x.device, x.dtype

        if domain_mask is not None:
            dm = domain_mask.unsqueeze(1) if domain_mask.ndim == 4 else domain_mask
            dm = (dm > 0.5).to(device=device, dtype=dtype)
            keep_out = 1.0 - dm
            valid_mask = dm
        else:
            valid_mask = torch.ones((B, 1, D, H, W), device=device, dtype=dtype)
            keep_out = torch.zeros((B, 1, D, H, W), device=device, dtype=dtype)

        x_in = torch.cat([x, valid_mask, keep_out], dim=1)

        divisor = 4
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
            x_in = F.pad(x_in, (w0, w1, h0, h1, d0, d1), mode="constant", value=0.0)
            valid_padded = F.pad(valid_mask, (w0, w1, h0, h1, d0, d1), mode="constant", value=0.0)
            if loads is not None:
                K, C_l = loads.shape[1], loads.shape[2]
                loads_flat = loads.reshape(B * K, C_l, D, H, W)
                loads_padded = F.pad(loads_flat, (w0, w1, h0, h1, d0, d1), mode="constant", value=0.0)
                loads = loads_padded.reshape(B, K, C_l, D + pad_d, H + pad_h, W + pad_w)
        else:
            valid_padded = valid_mask

        return x_in, valid_padded, keep_out, loads, (d0, D, h0, H, w0, W)

    @staticmethod
    def _crop_centered(
        tensor: torch.Tensor,
        crop_box: Tuple[int, int, int, int, int, int],
    ) -> torch.Tensor:
        d0, D, h0, H, w0, W = crop_box
        return tensor[:, :, d0 : d0 + D, h0 : h0 + H, w0 : w0 + W]

    def forward(
        self,
        x: torch.Tensor,
        domain_mask: Optional[torch.Tensor] = None,
        loads: Optional[torch.Tensor] = None,
        return_blueprint: bool = False,
        return_aux: bool = False,
        compute_sed: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for PhysUViT3D.

        Args:
            x: Input tensor (B, 7, D, H, W)
            domain_mask: Optional binary valid design region mask (B, 1, D, H, W) or (B, D, H, W)
            loads: Optional multi-load tensor (B, K, 3, D, H, W); defaults to x[:, 4:7] as K=1
            return_blueprint: Return (policy, value, blueprint)
            return_aux: Return (policy, value, aux_dict)
            compute_sed: Compute SED + displacement head (only if return_aux=True)
        """
        if loads is None:
            loads_in = x[:, 4:7].unsqueeze(1) if x.shape[1] >= 7 else x.new_zeros((x.shape[0], 1, 3, *x.shape[2:]))
        else:
            loads_in = loads

        phys_stats = self._extract_physics_stats(x, domain_mask=domain_mask)
        phys_emb = self.phys_mlp(phys_stats)  # (B, E)

        x_in, valid_padded, keep_out, loads_padded, crop_box = self._pad_centered_div4(
            x, domain_mask=domain_mask, loads=loads_in
        )
        assert loads_padded is not None

        # 1. Shallow 3D Conv Encoder
        skip_features, bottleneck_map = self.encoder(x_in)
        B, E, Dt, Ht, Wt = bottleneck_map.shape
        N = Dt * Ht * Wt

        # 2. Compute 1/4x token validity mask (only trigger masking overhead if needed)
        token_valid_map = F.max_pool3d(valid_padded, kernel_size=4, stride=4) > 0.5
        token_valid_flat = token_valid_map.reshape(B, N)
        all_valid = bool(torch.all(token_valid_flat).item())
        if all_valid:
            token_valid: Optional[torch.Tensor] = None
            attn_bias: Optional[torch.Tensor] = None
        else:
            # Ensure at least token 0 is never masked if a pathological all-zero mask is passed
            none_valid = ~token_valid_flat.any(dim=1)
            if none_valid.any():
                token_valid_flat = token_valid_flat.clone()
                token_valid_flat[none_valid, 0] = True
            token_valid = token_valid_flat
            # Additive fp16-safe mask (B, 1, 1, N): 0.0 for valid keys, -1e4 for masked keys
            attn_bias = torch.zeros((B, 1, 1, N), device=x.device, dtype=torch.float32)
            attn_bias = attn_bias.masked_fill(~token_valid.unsqueeze(1).unsqueeze(1), -1e4)

        # 3. Prepare 3D grid coordinates & RoPE embeddings
        gx = torch.arange(Dt, device=x.device, dtype=torch.float32)
        gy = torch.arange(Ht, device=x.device, dtype=torch.float32)
        gz = torch.arange(Wt, device=x.device, dtype=torch.float32)
        mx, my, mz = torch.meshgrid(gx, gy, gz, indexing="ij")
        grid_coords = torch.stack([mx, my, mz], dim=-1).reshape(1, N, 3)  # (1, N, 3)
        grid_cos, grid_sin = build_3d_rope_cos_sin(grid_coords, self.head_dim)

        # 4. Multi-Load Encoder tokens & RoPE coordinates
        load_tokens, load_coords = self.load_encoder(loads_padded)
        load_cos, load_sin = build_3d_rope_cos_sin(load_coords, self.head_dim)

        # 5. Global Physics Transformer Bottleneck
        tokens = bottleneck_map.flatten(2).transpose(1, 2)  # (B, N, E)
        for blk in self.blocks:
            tokens = blk(
                tokens=tokens,
                phys_emb=phys_emb,
                grid_cos=grid_cos,
                grid_sin=grid_sin,
                load_tokens=load_tokens,
                load_cos=load_cos,
                load_sin=load_sin,
                token_valid=token_valid,
                attn_bias=attn_bias,
            )
        tokens = self.final_norm(tokens)
        if token_valid is not None:
            tokens = tokens * token_valid.unsqueeze(-1).to(dtype=tokens.dtype)

        # 6. Factored Value Head
        value, log_c, conn_logit = self.factored_value_head(tokens, phys_emb, token_valid=token_valid)

        # 7. Shallow 3D U-Net Decoder + Global Broadcast + Raw Physics Skip
        if token_valid is None:
            global_tokens_mean = tokens.mean(dim=1)
        else:
            denom = token_valid.unsqueeze(-1).to(dtype=tokens.dtype).sum(dim=1).clamp(min=1.0)
            global_tokens_mean = (tokens * token_valid.unsqueeze(-1).to(dtype=tokens.dtype)).sum(dim=1) / denom

        bottleneck_out = tokens.transpose(1, 2).reshape(B, E, Dt, Ht, Wt)
        dec_feat = self.decoder(bottleneck_out, skip_features, global_tokens_mean)
        dec_feat = self._crop_centered(dec_feat, crop_box)
        dec_feat = dec_feat + self.raw_phys_proj(x)

        # 8. AdaLN/FiLM Modulation on Policy Head
        gamma_p, beta_p = self.policy_film(phys_emb).chunk(2, dim=1)
        pol_feat = dec_feat * (1.0 + gamma_p.view(B, self.feature_size, 1, 1, 1)) + beta_p.view(
            B, self.feature_size, 1, 1, 1
        )
        policy = self.policy_head(pol_feat)
        if domain_mask is not None:
            policy = policy.masked_fill(keep_out > 0.5, -1e4)

        if return_aux:
            blueprint = self.blueprint_head(dec_feat)
            aux: Dict[str, Optional[torch.Tensor]] = {
                "blueprint": blueprint,
                "log_compliance": log_c,
                "conn_logit": conn_logit,
                "sed_u": self.sed_head(dec_feat) if compute_sed else None,
            }
            return policy, value, aux

        if return_blueprint:
            blueprint = self.blueprint_head(dec_feat)
            return policy, value, blueprint

        return policy, value
