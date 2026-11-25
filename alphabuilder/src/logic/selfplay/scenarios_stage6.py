"""
4-Tier Progressive Generalization Curriculum & OOD Benchmark Suite for AlphaBuilder v4.0 Stage 6.

Expands systematically in concentric physical rings around the Stage 5 cantilever warmup:
- Tier 0 (Base Cantilever Anchor, 64x32x8): FULL_CLAMP & RAIL_XY across eccentric X/Y/Z,
  strictly excluding S1-S6 held-out neighborhoods.
- Tier 1 (Near Domain Shift, 64x32x8): Partial/dual-pad clamps, 3D angled loads (Fx+Fy, Fy+Fz),
  and two-point simultaneous loads.
- Tier 2 (Topology Shift, 64x32x8): MBB bridges (pin + roller), L-shaped wall+floor supports,
  and cantilevers with interior keep-out obstacles (`domain_mask = 0`).
- Tier 3 (Aspect-Ratio & Thickness Shift): Multi-resolution `(48,32,8)`, `(64,24,8)`, `(48,24,12)`
  leveraging Phys-UViT 3D's `divisor=4` + `3D RoPE` scale invariance without zero-padding.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from alphabuilder.src.logic.harvest.generators import (
    generate_bc_masks,
    get_load_region_bounds,
)


# Canonical Stage 3/5 held-out scenarios S1-S6 (must never appear in training)
HELD_OUT_CANONICAL_SCENARIOS: Tuple[Tuple[int, int, str], ...] = (
    (63, 16, "FULL_CLAMP"),  # S1
    (63, 6, "FULL_CLAMP"),   # S2
    (44, 24, "FULL_CLAMP"),  # S3
    (58, 16, "RAIL_XY"),     # S4
    (52, 11, "RAIL_XY"),     # S5
    (36, 16, "FULL_CLAMP"),  # S6
)

# Normalized progressive tier weights across generations (T0, T1, T2, T3)
CURRICULUM_SCHEDULE: Dict[int, Tuple[float, float, float, float]] = {
    1: (0.60, 0.40, 0.00, 0.00),
    2: (0.35, 0.35, 0.30, 0.00),
    3: (0.25, 0.25, 0.25, 0.25),
    4: (0.20, 0.25, 0.30, 0.25),
    5: (0.15, 0.25, 0.30, 0.30),
}


@dataclass
class SelfPlayScenario:
    """Complete physical boundary/load/domain specification for a self-play or eval episode."""

    scenario_id: str
    tier: int
    family: str
    resolution: Tuple[int, int, int]
    bc_type: str
    bc_masks: np.ndarray                 # (3, D, H, W) float32
    forces: np.ndarray                   # (3, D, H, W) float32
    domain_mask: Optional[np.ndarray]    # (D, H, W) float32 or None when full domain valid
    load_config: Dict[str, Any]


def get_base_tier_weights(generation: int) -> Tuple[float, float, float, float]:
    """Return scheduled `(tier0, tier1, tier2, tier3)` sampling weights for generation `g >= 1`."""
    if generation <= 1:
        return CURRICULUM_SCHEDULE[1]
    if generation >= 5:
        return CURRICULUM_SCHEDULE[5]
    return CURRICULUM_SCHEDULE[int(generation)]


def adjust_tier_weights(
    base_weights: Tuple[float, float, float, float],
    prev_tier_conn_rates: Optional[Dict[int, float]] = None,
    min_conn_rate: float = 0.65,
) -> Tuple[float, float, float, float]:
    """
    Adaptive curriculum pacing gate: if any previously active tier had Phase-1 connection rate
    below `min_conn_rate`, throttle back new higher tiers so Phase 1 never collapses into junk.
    """
    if not prev_tier_conn_rates:
        return base_weights
    w = list(base_weights)
    for tier_idx in (0, 1, 2):
        rate = prev_tier_conn_rates.get(tier_idx)
        if rate is not None and rate < min_conn_rate:
            # Transfer weight from higher tiers down into the struggling tier and anchor
            redirect = 0.5 * w[tier_idx + 1]
            if tier_idx + 2 < 4:
                redirect += w[tier_idx + 2]
                w[tier_idx + 2] = 0.0
            w[tier_idx + 1] = max(0.0, w[tier_idx + 1] - 0.5 * w[tier_idx + 1])
            w[tier_idx] += 0.5 * redirect
            w[0] += 0.5 * redirect
    tot = sum(w)
    if tot <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(v / tot for v in w)  # type: ignore[return-value]


def is_near_canonical_scenario(
    load_config: Dict[str, Any],
    bc_type: str,
    tol_xy: int = 1,
) -> bool:
    """Return True if a candidate cantilever `(x, y, bc_type)` is near S1-S6 held-out tests."""
    lx = int(load_config["x"])
    ly = int(load_config["y"])
    bc = str(bc_type).upper()
    for sx, sy, s_bc in HELD_OUT_CANONICAL_SCENARIOS:
        if bc == s_bc and abs(lx - sx) <= tol_xy and abs(ly - sy) <= tol_xy:
            return True
    return False


def _paint_load_patch(
    forces: np.ndarray,
    resolution: Tuple[int, int, int],
    load_cfg: Dict[str, Any],
    vec: Tuple[float, float, float],
) -> None:
    lx, y_min, y_max, z_min, z_max = get_load_region_bounds(resolution, load_cfg)
    forces[0, lx, y_min:y_max, z_min:z_max] += float(vec[0])
    forces[1, lx, y_min:y_max, z_min:z_max] += float(vec[1])
    forces[2, lx, y_min:y_max, z_min:z_max] += float(vec[2])


def _sample_z_band(nz: int, rng: random.Random, eccentric_prob: float = 0.45) -> Tuple[int, int]:
    if nz >= 6 and rng.random() < eccentric_prob:
        z_start = int(rng.randint(0, nz - 2))
        z_end = min(nz, z_start + 2)
        return z_start, z_end
    zc = nz // 2
    return max(0, zc - 1), min(nz, zc + 1)


def _sample_tier0_scenario(
    resolution: Tuple[int, int, int],
    rng: random.Random,
    tier_override: int = 0,
    family_prefix: str = "cantilever",
) -> SelfPlayScenario:
    nx, ny, nz = resolution
    x_lo = min(nx - 1, max(nx // 2, 24))
    x_hi = nx - 1
    y_lo = min(ny - 4, 4)
    y_hi = max(y_lo, ny - 5)

    for _ in range(128):
        lx = int(rng.randint(x_lo, x_hi))
        ly = int(rng.randint(y_lo, y_hi))
        zs, ze = _sample_z_band(nz, rng, eccentric_prob=0.45)
        bc_type = "RAIL_XY" if rng.random() < 0.45 else "FULL_CLAMP"
        cfg = {"x": lx, "y": ly, "z_start": zs, "z_end": ze, "bc_type": bc_type}
        if resolution != (64, 32, 8) or not is_near_canonical_scenario(cfg, bc_type, tol_xy=1):
            mx, my, mz = generate_bc_masks(resolution, bc_type)
            bc_masks = np.stack([mx, my, mz], axis=0).astype(np.float32)
            forces = np.zeros((3, *resolution), dtype=np.float32)
            _paint_load_patch(forces, resolution, cfg, (0.0, -1.0, 0.0))
            return SelfPlayScenario(
                scenario_id=f"T{tier_override}_{family_prefix}_{bc_type}_{nx}x{ny}x{nz}_x{lx}y{ly}",
                tier=tier_override,
                family=f"{family_prefix}_{bc_type.lower()}",
                resolution=resolution,
                bc_type=bc_type,
                bc_masks=bc_masks,
                forces=forces,
                domain_mask=None,
                load_config=cfg,
            )

    cfg = {"x": min(nx - 1, 48), "y": ny // 2, "z_start": nz // 2 - 1, "z_end": nz // 2 + 1, "bc_type": "FULL_CLAMP"}
    mx, my, mz = generate_bc_masks(resolution, "FULL_CLAMP")
    bc_masks = np.stack([mx, my, mz], axis=0).astype(np.float32)
    forces = np.zeros((3, *resolution), dtype=np.float32)
    _paint_load_patch(forces, resolution, cfg, (0.0, -1.0, 0.0))
    return SelfPlayScenario(
        scenario_id=f"T{tier_override}_{family_prefix}_fallback_{nx}x{ny}x{nz}",
        tier=tier_override,
        family=f"{family_prefix}_full_clamp",
        resolution=resolution,
        bc_type="FULL_CLAMP",
        bc_masks=bc_masks,
        forces=forces,
        domain_mask=None,
        load_config=cfg,
    )


def _sample_tier1_scenario(
    resolution: Tuple[int, int, int],
    rng: random.Random,
    tier_override: int = 1,
) -> SelfPlayScenario:
    nx, ny, nz = resolution
    variant = rng.choice(["dual_pad_angled", "partial_center_3d", "two_point_load"])

    bc_masks = np.zeros((3, *resolution), dtype=np.float32)
    forces = np.zeros((3, *resolution), dtype=np.float32)

    if variant == "dual_pad_angled":
        pad_h = max(4, ny // 4)
        bc_masks[:, 0, 0:pad_h, :] = 1.0
        bc_masks[:, 0, ny - pad_h : ny, :] = 1.0
        lx = int(rng.randint(max(nx // 2 + 2, nx - 22), nx - 1))
        ly = int(rng.randint(max(4, ny // 4), min(ny - 5, 3 * ny // 4)))
        zs, ze = _sample_z_band(nz, rng, eccentric_prob=0.50)
        cfg = {"x": lx, "y": ly, "z_start": zs, "z_end": ze, "bc_type": "DUAL_PAD_CLAMP"}
        ang_deg = float(rng.choice([-35.0, -25.0, 25.0, 35.0]))
        rad = math.radians(ang_deg)
        vec = (math.sin(rad), -math.cos(rad), 0.0)
        _paint_load_patch(forces, resolution, cfg, vec)
        return SelfPlayScenario(
            scenario_id=f"T{tier_override}_dualpad_{nx}x{ny}x{nz}_x{lx}y{ly}",
            tier=tier_override,
            family="dual_pad_angled",
            resolution=resolution,
            bc_type="DUAL_PAD_CLAMP",
            bc_masks=bc_masks,
            forces=forces,
            domain_mask=None,
            load_config=cfg,
        )

    if variant == "partial_center_3d":
        y0 = max(2, ny // 4)
        y1 = min(ny - 2, 3 * ny // 4)
        bc_masks[:, 0, y0:y1, :] = 1.0
        lx = int(rng.randint(max(nx // 2, 24), nx - 1))
        ly = int(rng.randint(max(3, ny // 5), min(ny - 4, 4 * ny // 5)))
        zs, ze = _sample_z_band(nz, rng, eccentric_prob=0.60)
        cfg = {"x": lx, "y": ly, "z_start": zs, "z_end": ze, "bc_type": "PARTIAL_CLAMP"}
        fz_mag = float(rng.choice([-0.40, 0.40]))
        fy_mag = -math.sqrt(max(0.1, 1.0 - fz_mag * fz_mag))
        _paint_load_patch(forces, resolution, cfg, (0.0, fy_mag, fz_mag))
        return SelfPlayScenario(
            scenario_id=f"T{tier_override}_partial3d_{nx}x{ny}x{nz}_x{lx}y{ly}",
            tier=tier_override,
            family="partial_center_3d",
            resolution=resolution,
            bc_type="PARTIAL_CLAMP",
            bc_masks=bc_masks,
            forces=forces,
            domain_mask=None,
            load_config=cfg,
        )

    # Two-point simultaneous load with split resultant norm (~1.0 total magnitude)
    bc_type = "RAIL_XY" if rng.random() < 0.30 else "FULL_CLAMP"
    mx, my, mz = generate_bc_masks(resolution, bc_type)
    bc_masks = np.stack([mx, my, mz], axis=0).astype(np.float32)
    lx1 = int(rng.randint(max(nx // 3, 18), max(nx // 2 + 2, nx - 18)))
    lx2 = int(rng.randint(min(nx - 1, lx1 + 10), nx - 1))
    ly1 = int(rng.randint(4, max(5, ny // 2 - 2)))
    ly2 = int(rng.randint(min(ny - 5, ny // 2 + 2), ny - 5))
    zs1, ze1 = _sample_z_band(nz, rng, eccentric_prob=0.3)
    zs2, ze2 = _sample_z_band(nz, rng, eccentric_prob=0.3)
    cfg1 = {"x": lx1, "y": ly1, "z_start": zs1, "z_end": ze1, "bc_type": bc_type}
    cfg2 = {"x": lx2, "y": ly2, "z_start": zs2, "z_end": ze2, "bc_type": bc_type}
    _paint_load_patch(forces, resolution, cfg1, (0.0, -0.5, 0.0))
    _paint_load_patch(forces, resolution, cfg2, (0.0, -0.5, 0.0))
    return SelfPlayScenario(
        scenario_id=f"T{tier_override}_twoload_{nx}x{ny}x{nz}_x{lx1}_{lx2}",
        tier=tier_override,
        family="two_point_load",
        resolution=resolution,
        bc_type=bc_type,
        bc_masks=bc_masks,
        forces=forces,
        domain_mask=None,
        load_config=cfg2,
    )


def _sample_tier2_scenario(
    resolution: Tuple[int, int, int],
    rng: random.Random,
    tier_override: int = 2,
) -> SelfPlayScenario:
    nx, ny, nz = resolution
    variant = rng.choice(["bridge_mbb", "corner_l_support", "keepout_window"])

    bc_masks = np.zeros((3, *resolution), dtype=np.float32)
    forces = np.zeros((3, *resolution), dtype=np.float32)
    domain_mask: Optional[np.ndarray] = None

    if variant == "bridge_mbb":
        pad_y = max(4, ny // 6)
        bc_masks[:, 0, 0:pad_y, :] = 1.0         # Pin left base (Ux=Uy=Uz=0)
        bc_masks[1, nx - 1, 0:pad_y, :] = 1.0    # Roller right base (Uy=0)
        lx = int(rng.randint(max(nx // 4, nx // 2 - 6), min(nx - 4, nx // 2 + 6)))
        ly = int(rng.choice([max(2, ny - 2), 2]))  # Top-deck or bottom-deck bridge load
        zs, ze = _sample_z_band(nz, rng, eccentric_prob=0.35)
        cfg = {"x": lx, "y": ly, "z_start": zs, "z_end": ze, "bc_type": "BRIDGE_MBB"}
        _paint_load_patch(forces, resolution, cfg, (0.0, -1.0, 0.0))
        return SelfPlayScenario(
            scenario_id=f"T{tier_override}_bridge_{nx}x{ny}x{nz}_x{lx}y{ly}",
            tier=tier_override,
            family="bridge_mbb",
            resolution=resolution,
            bc_type="BRIDGE_MBB",
            bc_masks=bc_masks,
            forces=forces,
            domain_mask=None,
            load_config=cfg,
        )

    if variant == "corner_l_support":
        bc_masks[:, 0, 0 : max(8, ny // 2), :] = 1.0  # Left lower wall
        bc_masks[:, 0 : max(8, nx // 4), 0, :] = 1.0  # Floor anchor near wall
        lx = int(rng.randint(max(nx // 2, 24), nx - 1))
        ly = int(rng.randint(max(6, ny // 3), ny - 5))
        zs, ze = _sample_z_band(nz, rng, eccentric_prob=0.4)
        cfg = {"x": lx, "y": ly, "z_start": zs, "z_end": ze, "bc_type": "CORNER_L"}
        ang_deg = float(rng.choice([0.0, -25.0, 25.0]))
        rad = math.radians(ang_deg)
        _paint_load_patch(forces, resolution, cfg, (math.sin(rad), -math.cos(rad), 0.0))
        return SelfPlayScenario(
            scenario_id=f"T{tier_override}_cornerL_{nx}x{ny}x{nz}_x{lx}y{ly}",
            tier=tier_override,
            family="corner_l_support",
            resolution=resolution,
            bc_type="CORNER_L",
            bc_masks=bc_masks,
            forces=forces,
            domain_mask=None,
            load_config=cfg,
        )

    # Keep-out window cantilever (guarantees clear passage above & below window)
    mx, my, mz = generate_bc_masks(resolution, "FULL_CLAMP")
    bc_masks = np.stack([mx, my, mz], axis=0).astype(np.float32)
    lx = int(rng.randint(max(nx - 12, nx // 2 + 8), nx - 1))
    ly = int(rng.randint(max(6, ny // 3), min(ny - 7, 2 * ny // 3)))
    zs, ze = _sample_z_band(nz, rng, eccentric_prob=0.25)
    cfg = {"x": lx, "y": ly, "z_start": zs, "z_end": ze, "bc_type": "KEEPOUT_WINDOW"}
    _paint_load_patch(forces, resolution, cfg, (0.0, -1.0, 0.0))

    domain_mask = np.ones(resolution, dtype=np.float32)
    wx0 = max(10, nx // 4)
    wx1 = min(lx - 6, max(wx0 + 8, nx // 2 + 4))
    wy0 = max(6, ny // 2 - 5)
    wy1 = min(ny - 6, ny // 2 + 5)
    if wx1 > wx0 + 2 and wy1 > wy0 + 2:
        domain_mask[wx0:wx1, wy0:wy1, :] = 0.0
    # Guarantee BC and load regions stay strictly inside valid domain_mask
    domain_mask[0:2, :, :] = 1.0
    lx_b, y0_b, y1_b, z0_b, z1_b = get_load_region_bounds(resolution, cfg)
    domain_mask[max(0, lx_b - 1) : min(nx, lx_b + 2), max(0, y0_b - 1) : min(ny, y1_b + 1), :] = 1.0
    return SelfPlayScenario(
        scenario_id=f"T{tier_override}_keepout_{nx}x{ny}x{nz}_x{lx}y{ly}",
        tier=tier_override,
        family="keepout_window",
        resolution=resolution,
        bc_type="KEEPOUT_WINDOW",
        bc_masks=bc_masks,
        forces=forces,
        domain_mask=domain_mask,
        load_config=cfg,
    )


def sample_curriculum_scenario(
    generation: int = 1,
    tier_weights: Optional[Sequence[float]] = None,
    rng: Optional[random.Random] = None,
) -> SelfPlayScenario:
    """
    Sample a self-play scenario according to the 4-tier curriculum weights.
    """
    r = rng if rng is not None else random.Random()
    weights = list(tier_weights) if tier_weights is not None else list(get_base_tier_weights(generation))
    tier = r.choices([0, 1, 2, 3], weights=weights, k=1)[0]

    if tier == 0:
        return _sample_tier0_scenario((64, 32, 8), r, tier_override=0)
    if tier == 1:
        return _sample_tier1_scenario((64, 32, 8), r, tier_override=1)
    if tier == 2:
        return _sample_tier2_scenario((64, 32, 8), r, tier_override=2)

    # Tier 3: multi-resolution domain variation (all dims divisible by 4 -> 0% padding)
    res_choice = r.choice([(48, 32, 8), (64, 24, 8), (48, 24, 12)])
    sub_roll = r.random()
    if sub_roll < 0.45:
        return _sample_tier0_scenario(res_choice, r, tier_override=3, family_prefix="multires_cant")
    if sub_roll < 0.75:
        return _sample_tier1_scenario(res_choice, r, tier_override=3)
    return _sample_tier2_scenario(res_choice, r, tier_override=3)


def build_ood_benchmark_scenarios() -> List[Dict[str, Any]]:
    """
    Construct the deterministic 4-scenario OOD generalization suite `G1-G4`
    compatible with `evaluate_inference_scenarios.py`.
    """
    specs: List[Dict[str, Any]] = []

    # G1: Dual-pad clamp + 30 deg angled load (64x32x8)
    res1 = (64, 32, 8)
    bc1 = np.zeros((3, *res1), dtype=np.float32)
    bc1[:, 0, 0:8, :] = 1.0
    bc1[:, 0, 24:32, :] = 1.0
    f1 = np.zeros((3, *res1), dtype=np.float32)
    cfg1 = {"x": 58, "y": 16, "z_start": 3, "z_end": 5, "bc_type": "DUAL_PAD_CLAMP"}
    rad1 = math.radians(30.0)
    _paint_load_patch(f1, res1, cfg1, (math.sin(rad1), -math.cos(rad1), 0.0))
    specs.append(
        {
            "id": "G1_dual_pad_angled",
            "title": "OOD G1: Dupla aba x=0 + carga inclinada 30 deg",
            "resolution": res1,
            "bc_type": "DUAL_PAD_CLAMP",
            "load_config": cfg1,
            "bc_masks": bc1,
            "forces": f1,
            "domain_mask": None,
        }
    )

    # G2: MBB Bridge (pin at x=0 base + roller Uy at x=63 base, midspan top load)
    res2 = (64, 32, 8)
    bc2 = np.zeros((3, *res2), dtype=np.float32)
    bc2[:, 0, 0:6, :] = 1.0
    bc2[1, 63, 0:6, :] = 1.0
    f2 = np.zeros((3, *res2), dtype=np.float32)
    cfg2 = {"x": 32, "y": 30, "z_start": 3, "z_end": 5, "bc_type": "BRIDGE_MBB"}
    _paint_load_patch(f2, res2, cfg2, (0.0, -1.0, 0.0))
    specs.append(
        {
            "id": "G2_bridge_mbb",
            "title": "OOD G2: Ponte bi-apoiada MBB (pino + rolete)",
            "resolution": res2,
            "bc_type": "BRIDGE_MBB",
            "load_config": cfg2,
            "bc_masks": bc2,
            "forces": f2,
            "domain_mask": None,
        }
    )

    # G3: Keep-out window cantilever (64x32x8)
    res3 = (64, 32, 8)
    mx3, my3, mz3 = generate_bc_masks(res3, "FULL_CLAMP")
    bc3 = np.stack([mx3, my3, mz3], axis=0).astype(np.float32)
    f3 = np.zeros((3, *res3), dtype=np.float32)
    cfg3 = {"x": 60, "y": 16, "z_start": 3, "z_end": 5, "bc_type": "KEEPOUT_WINDOW"}
    _paint_load_patch(f3, res3, cfg3, (0.0, -1.0, 0.0))
    dm3 = np.ones(res3, dtype=np.float32)
    dm3[18:38, 10:22, :] = 0.0
    specs.append(
        {
            "id": "G3_keepout_window",
            "title": "OOD G3: Cantilever com janela proibida central",
            "resolution": res3,
            "bc_type": "KEEPOUT_WINDOW",
            "load_config": cfg3,
            "bc_masks": bc3,
            "forces": f3,
            "domain_mask": dm3,
        }
    )

    # G4: Multi-resolution thick 3D beam (48x24x12, Z=3 tokens in bottleneck)
    res4 = (48, 24, 12)
    mx4, my4, mz4 = generate_bc_masks(res4, "FULL_CLAMP")
    bc4 = np.stack([mx4, my4, mz4], axis=0).astype(np.float32)
    f4 = np.zeros((3, *res4), dtype=np.float32)
    cfg4 = {"x": 47, "y": 14, "z_start": 1, "z_end": 3, "bc_type": "FULL_CLAMP"}
    _paint_load_patch(f4, res4, cfg4, (0.0, -1.0, 0.0))
    specs.append(
        {
            "id": "G4_thick_3d_48x24x12",
            "title": "OOD G4: Bloco 3D multi-res (48x24x12, Z excentrico)",
            "resolution": res4,
            "bc_type": "FULL_CLAMP",
            "load_config": cfg4,
            "bc_masks": bc4,
            "forces": f4,
            "domain_mask": None,
        }
    )

    return specs
