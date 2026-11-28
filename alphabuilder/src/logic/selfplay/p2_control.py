"""Pure-NumPy Phase-2 expansion, SED probe mode, and trust-region helpers for Stage 6 v4.2."""

from __future__ import annotations

from typing import List, Optional, Tuple


def select_sed_probe_mode(
    vol_frac: float,
    remove_min_vol: float = 0.080,
    expand_target_vol: float = 0.105,
    carve_high_vol: float = 0.145,
) -> Tuple[str, float]:
    """
    Select counterfactual SED probe tag and target removal fraction for `vol_frac`:
    - `vol_frac < remove_min_vol`   -> `sed_expand_topk`    (0.00 remove, pure outer-SED ADD)
    - `vol_frac < expand_target_vol` -> `sed_grow_swap_topk` (0.25 remove / 0.75 ADD)
    - `vol_frac > carve_high_vol`    -> `sed_carve_topk`     (0.67 remove / 0.33 ADD)
    - intermediate volumes           -> `sed_hybrid_topk`    (0.50 remove / 0.50 ADD)
    """
    v = float(vol_frac)
    if v < float(remove_min_vol):
        return "sed_expand_topk", 0.0
    if v < float(expand_target_vol):
        return "sed_grow_swap_topk", 0.25
    if v > float(carve_high_vol):
        return "sed_carve_topk", 2.0 / 3.0
    return "sed_hybrid_topk", 0.50


def is_trust_region_accepted(
    cand_connected: bool,
    cand_valid: bool,
    cand_collapsed: bool,
    cand_score: float,
    cur_score: float,
    cand_compliance: Optional[float],
    cur_compliance: Optional[float],
    in_grace_cur: bool = False,
    drop_tol: float = 0.08,
    comp_mult: float = 1.80,
) -> bool:
    """
    Validate a Phase-2 candidate move against physical trust-region guards so thin-beam
    neck throttling (`52k -> 637k J`) or catastrophic drops are rejected before state mutation.
    """
    if not cand_connected or not cand_valid or cand_collapsed:
        return False
    if in_grace_cur:
        return bool(float(cand_score) >= float(cur_score) - 0.02)
    if float(cand_score) < float(cur_score) - float(drop_tol):
        return False
    if (
        cur_compliance is not None
        and cand_compliance is not None
        and float(cur_compliance) > 0.0
        and float(cand_compliance) > float(comp_mult) * float(cur_compliance)
    ):
        return False
    return True


def build_scaled_p2_schedule(
    base_schedule: Tuple[int, ...],
    domain_size: int,
) -> List[int]:
    """
    Scale a Phase-2 coarse-to-fine batch schedule monotonically by domain size
    (`(domain_size / 16384)**(1/3)`), preserving exact values at canonical `64x32x8`.
    """
    scale = (float(max(1, int(domain_size))) / 16384.0) ** (1.0 / 3.0)
    out: List[int] = []
    for v in base_schedule:
        scaled = max(2, int(round(float(v) * scale)))
        if not out or scaled < out[-1]:
            out.append(scaled)
    return out or [4]
