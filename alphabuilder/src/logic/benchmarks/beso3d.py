"""
Soft-Kill 3D Bi-directional Evolutionary Structural Optimization (BESO 3D)
aligned 1-to-1 with AlphaBuilder's StructuredHex8FEM physics and v3.3 reward.

Reference: Huang & Xie (2007, 2010), FEA via PyAMG (`StructuredHex8FEM`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from alphabuilder.src.logic.harvest.generators import (
    generate_bc_masks,
    get_load_region_bounds,
)
from alphabuilder.src.logic.selfplay.evaluator import evaluate_main_island
from alphabuilder.src.logic.selfplay.fast_fem import _get_solver
from alphabuilder.src.logic.selfplay.reward import (
    analyze_structure_islands,
    calculate_island_penalty,
    calculate_reward,
)


@dataclass
class BESOSnapshot:
    """Single audited BESO 3D design snapshot (e.g., iso-V or Pareto-Best)."""

    tag: str
    iteration: int
    volume_fraction: float
    compliance: float
    max_displacement: float
    fem_reward: float
    island_penalty: float
    total_score: float
    is_connected: bool
    n_islands: int
    loose_voxels: int
    geometry: Dict[str, Any]
    density: np.ndarray = field(repr=False)


@dataclass
class BESOResult:
    """Full trajectory and key benchmarks from a BESO 3D run."""

    scenario_id: str
    resolution: Tuple[int, int, int]
    bc_type: str
    iterations_run: int
    elapsed_sec: float
    iso_v_target: float
    iso_v: Optional[BESOSnapshot]
    pareto_best: Optional[BESOSnapshot]
    final_snapshot: Optional[BESOSnapshot]
    history: List[Dict[str, Any]]


def _build_conical_filter_kernel(r_min: float) -> np.ndarray:
    """Build normalized 3D conical spatial filter kernel w(d) = max(0, r_min - d)."""
    rad = int(np.ceil(float(r_min)))
    grid = np.arange(-rad, rad + 1, dtype=np.float64)
    dx, dy, dz = np.meshgrid(grid, grid, grid, indexing="ij")
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    kernel = np.maximum(0.0, float(r_min) - dist)
    s = float(kernel.sum())
    return kernel / s if s > 0.0 else kernel


def _resolve_bc_and_forces(
    resolution: Tuple[int, int, int],
    load_config: Dict[str, Any],
    bc_type: str,
    bc_masks: Optional[np.ndarray],
    forces: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    if bc_masks is None:
        mx, my, mz = generate_bc_masks(resolution, bc_type)
        bc_arr = np.stack([mx, my, mz], axis=0).astype(np.float32)
    else:
        bc_arr = np.asarray(bc_masks, dtype=np.float32)

    if forces is None:
        fx = np.zeros(resolution, dtype=np.float32)
        fy = np.zeros(resolution, dtype=np.float32)
        fz = np.zeros(resolution, dtype=np.float32)
        lx, y0, y1, z0, z1 = get_load_region_bounds(resolution, load_config)
        fy[lx, y0:y1, z0:z1] = -1.0
        f_arr = np.stack([fx, fy, fz], axis=0).astype(np.float32)
    else:
        f_arr = np.asarray(forces, dtype=np.float32)

    return bc_arr, f_arr


def _compute_geom_summary(
    density: np.ndarray,
    load_config: Dict[str, Any],
) -> Dict[str, Any]:
    nx, ny, _ = density.shape
    solid = density > 0.5
    solid_voxels = int(solid.sum())
    if solid_voxels == 0:
        return {
            "solid_voxels": 0,
            "volume_fraction": 0.0,
            "inertia_y_norm": 0.0,
            "hollow_ratio_xy": 0.0,
            "overshoot_frac": 0.0,
            "z_centroid": None,
        }
    xs, ys, zs = np.nonzero(solid)
    lx = int(load_config.get("x", nx - 1))
    y_mean = float(ys.mean())
    inertia_y_norm = float(np.mean(((ys - y_mean) / (ny / 2.0)) ** 2))

    proj_xy = np.any(solid, axis=2)
    occ = 0
    env = 0
    for x in range(nx):
        col = np.flatnonzero(proj_xy[x])
        if col.size > 0:
            occ += int(col.size)
            env += int(col.max() - col.min() + 1)
    hollow_ratio_xy = float(1.0 - occ / env) if env > 0 else 0.0
    overshoot_frac = float(np.sum(xs > lx + 2) / solid_voxels)
    return {
        "solid_voxels": solid_voxels,
        "volume_fraction": float(solid.mean()),
        "inertia_y_norm": inertia_y_norm,
        "hollow_ratio_xy": hollow_ratio_xy,
        "overshoot_frac": overshoot_frac,
        "z_centroid": float(zs.mean()),
    }


def _beso_threshold_update(
    density: np.ndarray,
    sens: np.ndarray,
    target_voxels: int,
    valid_domain: np.ndarray,
    protected_solid: np.ndarray,
    ar_max: float,
) -> np.ndarray:
    """
    Bisection threshold update enforcing valid_domain, protected load patches,
    and Huang & Xie admission ratio limit `ar_max`.
    """
    free_mask = valid_domain & (~protected_solid)
    n_prot = int((valid_domain & protected_solid).sum())
    total_voxels = density.size
    cur_solid = density > 0.5

    max_add = max(1, int(round(ar_max * total_voxels)))
    k_target = max(0, int(target_voxels) - n_prot)
    free_coords = np.flatnonzero(free_mask)
    if free_coords.size == 0 or k_target <= 0:
        out = np.zeros_like(density, dtype=np.float32)
        out[valid_domain & protected_solid] = 1.0
        return out

    if k_target >= free_coords.size:
        out = np.zeros_like(density, dtype=np.float32)
        out[valid_domain] = 1.0
        return out

    sens_free = sens.ravel()[free_coords]
    cur_free = cur_solid.ravel()[free_coords]

    def _apply_th(th: float) -> Tuple[np.ndarray, int]:
        cand = sens_free > th
        add_mask = cand & (~cur_free)
        n_add = int(add_mask.sum())
        if n_add > max_add:
            add_idx = np.flatnonzero(add_mask)
            keep_top = add_idx[np.argsort(sens_free[add_idx])[-max_add:]]
            cand = cur_free & (sens_free > th)
            cand[keep_top] = True
        return cand, int(cand.sum())

    lo = float(sens_free.min()) - 1e-12
    hi = float(sens_free.max()) + 1e-12
    best_cand = None
    best_diff = 1 << 30
    for _ in range(44):
        mid = 0.5 * (lo + hi)
        cand, count = _apply_th(mid)
        diff = abs(count - k_target)
        if diff < best_diff:
            best_diff = diff
            best_cand = cand
        if count > k_target:
            lo = mid
        elif count < k_target:
            hi = mid
        else:
            best_cand = cand
            break

    out_flat = np.zeros(total_voxels, dtype=np.float32)
    out_flat[np.flatnonzero(valid_domain & protected_solid)] = 1.0
    if best_cand is not None:
        out_flat[free_coords[best_cand]] = 1.0
    return out_flat.reshape(density.shape)


def _prune_detached_floaters(
    density: np.ndarray,
    bc_arr: np.ndarray,
    protected_solid: np.ndarray,
) -> np.ndarray:
    """Remove isolated zero-stress floating islands while keeping support-anchored structure & load pad."""
    solid = density > 0.5
    labeled, n_comp = ndimage.label(solid)
    if n_comp <= 1:
        return density
    sup_mask = (bc_arr[0] > 0.5) | (bc_arr[1] > 0.5) | (bc_arr[2] > 0.5)
    sup_lbls = np.unique(labeled[sup_mask])
    keep = sup_lbls[sup_lbls > 0]
    if keep.size == 0:
        return density
    keep_mask = np.isin(labeled, keep) | protected_solid
    return np.where(keep_mask, density, 0.0).astype(np.float32)


def run_beso3d(
    resolution: Tuple[int, int, int],
    load_config: Dict[str, Any],
    bc_type: str = "FULL_CLAMP",
    bc_masks: Optional[np.ndarray] = None,
    forces: Optional[np.ndarray] = None,
    domain_mask: Optional[np.ndarray] = None,
    scenario_id: str = "beso_custom",
    v_target_min: float = 0.060,
    v_iso_ref: float = 0.135,
    iso_stabilize_iters: int = 8,
    er_coarse: float = 0.03,
    er_fine: float = 0.02,
    ar_max: float = 0.02,
    r_min: float = 2.0,
    x_min: float = 1e-2,
    max_iter: int = 120,
    conv_tol: float = 1e-3,
    protect_cg1_halo: bool = True,
    clean_floaters: bool = True,
    progress_cb: Optional[Callable[[int, float, float, float], None]] = None,
) -> BESOResult:
    """
    Execute Soft-Kill BESO 3D on the exact Hex8 PyAMG elasticity solver.

    Sweeps down from full valid domain through a stabilized fixed-volume `iso-V`
    benchmark around `v_iso_ref` and down to `v_target_min` to capture the exact
    Pareto-optimal binary structure under AlphaBuilder v3.3 reward.
    """
    t0 = time.perf_counter()
    bc_arr, f_arr = _resolve_bc_and_forces(resolution, load_config, bc_type, bc_masks, forces)
    bc_tuple = (bc_arr[0], bc_arr[1], bc_arr[2])
    f_tuple = (f_arr[0], f_arr[1], f_arr[2])

    valid_domain = (np.asarray(domain_mask) > 0.5) if domain_mask is not None else np.ones(resolution, dtype=bool)
    force_mag = np.abs(f_arr[0]) + np.abs(f_arr[1]) + np.abs(f_arr[2])
    load_raw = force_mag > 1e-6
    if protect_cg1_halo:
        protected_solid = ndimage.binary_dilation(load_raw, iterations=1) & valid_domain
    else:
        protected_solid = load_raw & valid_domain

    kernel = _build_conical_filter_kernel(r_min=r_min)
    denom_weights = ndimage.convolve(valid_domain.astype(np.float64), kernel, mode="constant", cval=0.0)
    denom_weights = np.where(valid_domain & (denom_weights > 1e-12), denom_weights, 1.0)

    solver = _get_solver(resolution, nu=0.25)
    density = valid_domain.astype(np.float32)

    eps_fem = 1e-6
    void_scale = float((x_min ** 2) / eps_fem)  # (x_min^(p-1)) / eps for p=3

    prev_sens: Optional[np.ndarray] = None
    history: List[Dict[str, Any]] = []
    iso_snap: Optional[BESOSnapshot] = None
    pareto_snap: Optional[BESOSnapshot] = None
    final_snap: Optional[BESOSnapshot] = None

    in_iso_hold = False
    iso_hold_done = False
    iso_hold_left = int(iso_stabilize_iters)
    iso_comp_hist: List[float] = []
    disconn_streak = 0
    total_voxels = density.size

    for it in range(max_iter):
        vol_frac = float(np.mean(density > 0.5))
        comp, max_disp, _, sed_map, valid = solver.solve(
            density=density,
            load_config=load_config,
            bc_type=bc_type,
            bc_masks=bc_arr,
            forces=f_arr,
        )
        if not valid:
            break

        isl = analyze_structure_islands(
            density, load_config, threshold=0.5, bc_masks=bc_tuple, forces=f_tuple
        )
        solid_vox = int((density > 0.5).sum())
        isl_pen = float(calculate_island_penalty(isl["n_islands"], isl["loose_voxels"], solid_vox))

        eval_comp = float(comp)
        eval_disp = float(max_disp)
        if isl["is_connected"] and isl["n_islands"] > 1:
            main_res = evaluate_main_island(
                density,
                load_config,
                isl["main_island_mask"],
                bc_type=bc_type,
                backend="fast_amg",
                bc_masks=bc_arr,
                forces=f_arr,
                compute_sed=False,
            )
            if main_res.valid:
                eval_comp = float(main_res.compliance)
                eval_disp = float(main_res.max_displacement)

        fem_r = float(
            calculate_reward(
                compliance=eval_comp,
                vol_frac=vol_frac,
                is_valid=bool(isl["is_connected"]),
                max_displacement=eval_disp,
            )
        )
        total_score = float(np.clip(fem_r - isl_pen, -1.0, 1.0)) if isl["is_connected"] else -1.0

        geom = _compute_geom_summary(density, load_config)
        snap = BESOSnapshot(
            tag="iter",
            iteration=it,
            volume_fraction=vol_frac,
            compliance=eval_comp,
            max_displacement=eval_disp,
            fem_reward=fem_r,
            island_penalty=isl_pen,
            total_score=total_score,
            is_connected=bool(isl["is_connected"]),
            n_islands=int(isl["n_islands"]),
            loose_voxels=int(isl["loose_voxels"]),
            geometry=geom,
            density=density.copy(),
        )
        final_snap = snap

        history.append(
            {
                "iter": it,
                "vol_frac": round(vol_frac, 6),
                "compliance": float(eval_comp),
                "max_displacement": float(eval_disp),
                "fem_reward": float(fem_r),
                "island_penalty": float(isl_pen),
                "score": float(total_score),
                "is_connected": bool(isl["is_connected"]),
                "n_islands": int(isl["n_islands"]),
                "inertia_y_norm": float(geom["inertia_y_norm"]),
                "hollow_ratio_xy": float(geom["hollow_ratio_xy"]),
                "in_iso_hold": bool(in_iso_hold),
            }
        )

        if isl["is_connected"] and fem_r > -0.999:
            disconn_streak = 0
            if (pareto_snap is None) or (total_score > pareto_snap.total_score):
                pareto_snap = BESOSnapshot(
                    tag="pareto_best",
                    iteration=snap.iteration,
                    volume_fraction=snap.volume_fraction,
                    compliance=snap.compliance,
                    max_displacement=snap.max_displacement,
                    fem_reward=snap.fem_reward,
                    island_penalty=snap.island_penalty,
                    total_score=snap.total_score,
                    is_connected=snap.is_connected,
                    n_islands=snap.n_islands,
                    loose_voxels=snap.loose_voxels,
                    geometry=snap.geometry,
                    density=snap.density,
                )
            if in_iso_hold or abs(vol_frac - v_iso_ref) <= 0.012:
                if (
                    iso_snap is None
                    or abs(vol_frac - v_iso_ref) < abs(iso_snap.volume_fraction - v_iso_ref) - 0.004
                    or (
                        abs(vol_frac - v_iso_ref) <= abs(iso_snap.volume_fraction - v_iso_ref) + 0.004
                        and total_score > iso_snap.total_score
                    )
                ):
                    iso_snap = BESOSnapshot(
                        tag="iso_v",
                        iteration=snap.iteration,
                        volume_fraction=snap.volume_fraction,
                        compliance=snap.compliance,
                        max_displacement=snap.max_displacement,
                        fem_reward=snap.fem_reward,
                        island_penalty=snap.island_penalty,
                        total_score=snap.total_score,
                        is_connected=snap.is_connected,
                        n_islands=snap.n_islands,
                        loose_voxels=snap.loose_voxels,
                        geometry=snap.geometry,
                        density=snap.density,
                    )
        else:
            disconn_streak += 1
            if disconn_streak >= 3:
                break

        if progress_cb is not None:
            progress_cb(it, vol_frac, eval_comp, total_score)

        solid_bool = density > 0.5
        sed64 = sed_map.astype(np.float64)
        alpha_raw = np.where(solid_bool, sed64, void_scale * sed64)
        alpha_raw = np.where(valid_domain, alpha_raw, 0.0)
        alpha_filt = ndimage.convolve(alpha_raw, kernel, mode="constant", cval=0.0) / denom_weights
        if prev_sens is not None:
            alpha = 0.5 * (alpha_filt + prev_sens)
        else:
            alpha = alpha_filt
        prev_sens = alpha_filt

        if (not iso_hold_done) and (not in_iso_hold) and (vol_frac <= v_iso_ref * (1.0 + er_fine)):
            in_iso_hold = True
            iso_comp_hist = [eval_comp]
            next_vol = float(v_iso_ref)
        elif in_iso_hold:
            iso_comp_hist.append(eval_comp)
            iso_hold_left -= 1
            converged_iso = (
                len(iso_comp_hist) >= 5
                and max(iso_comp_hist[-4:]) > 1e-9
                and abs(iso_comp_hist[-1] - iso_comp_hist[-4]) / max(iso_comp_hist[-4:]) < conv_tol
            )
            if iso_hold_left <= 0 or converged_iso:
                in_iso_hold = False
                iso_hold_done = True
                next_vol = max(v_target_min, vol_frac * (1.0 - er_fine))
            else:
                next_vol = float(v_iso_ref)
        else:
            if vol_frac <= v_target_min + 1e-4:
                break
            er = er_coarse if vol_frac > 0.25 else er_fine
            next_vol = max(v_target_min, vol_frac * (1.0 - er))

        target_vox = int(round(next_vol * total_voxels))
        density = _beso_threshold_update(
            density=density,
            sens=alpha,
            target_voxels=target_vox,
            valid_domain=valid_domain,
            protected_solid=protected_solid,
            ar_max=ar_max,
        )
        if clean_floaters:
            density = _prune_detached_floaters(density, bc_arr, protected_solid)

    elapsed = time.perf_counter() - t0
    return BESOResult(
        scenario_id=scenario_id,
        resolution=resolution,
        bc_type=bc_type,
        iterations_run=len(history),
        elapsed_sec=round(elapsed, 2),
        iso_v_target=float(v_iso_ref),
        iso_v=iso_snap,
        pareto_best=pareto_snap,
        final_snapshot=final_snap,
        history=history,
    )
