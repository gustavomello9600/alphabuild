#!/usr/bin/env python3
"""
Audit script for AlphaBuilder v3.3 harvest SQLite databases.

Verifies:
1. SQLite integrity (`PRAGMA integrity_check`) and zero orphan episodes.
2. Step sequence continuity (0..N-1) with zero collisions or gaps.
3. Exact spatial alignment of `forces`, `bc_masks`, `bc_type`, and `load_config`.
4. Phase 1 & Phase 2 policy target legality (100% inside legal/unprotected masks, sum checks).
5. Statistical distributions by phase and BC type (`FULL_CLAMP` vs `RAIL_XY`).
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.logic.storage import (
    deserialize_array,
    deserialize_sparse,
    sparse_decode,
)
from alphabuilder.src.logic.harvest.generators import get_load_region_bounds
from alphabuilder.src.logic.mcts.legal_moves import (
    get_legal_add_moves,
    get_legal_remove_moves,
    get_protected_regions,
)


def _quantiles(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def audit_database(db_path: Path, sample_steps: int = 250) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    integrity = cursor.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        errors.append(f"SQLite integrity_check failed: {integrity}")

    ep_rows = cursor.execute("""
        SELECT episode_id, seed, bc_masks_blob, forces_blob, v_constructed_blob,
               load_config, bc_type, strategy, resolution,
               final_compliance, final_volume, final_max_displacement,
               final_is_connected, simp_iters, duration_sec
        FROM episodes
        ORDER BY rowid
    """).fetchall()

    total_records = cursor.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    phase_counts = dict(
        cursor.execute("SELECT phase, COUNT(*) FROM records GROUP BY phase").fetchall()
    )

    episodes_by_id: Dict[str, Dict[str, Any]] = {}
    bc_compliance: Dict[str, List[float]] = {"FULL_CLAMP": [], "RAIL_XY": []}
    bc_disp: Dict[str, List[float]] = {"FULL_CLAMP": [], "RAIL_XY": []}
    connected_count = 0

    for row in ep_rows:
        (
            ep_id, seed, bc_blob, force_blob, v_const_blob,
            load_json, bc_type, strategy, res_json,
            fin_c, fin_v, fin_d, fin_conn, simp_iters, dur
        ) = row

        res = tuple(json.loads(res_json))
        load_cfg = json.loads(load_json)
        bc_masks = deserialize_array(bc_blob)
        forces = deserialize_array(force_blob)
        v_const = deserialize_array(v_const_blob) if v_const_blob else None

        episodes_by_id[ep_id] = {
            "seed": seed,
            "res": res,
            "load_cfg": load_cfg,
            "bc_type": bc_type,
            "strategy": strategy,
            "bc_masks": bc_masks,
            "forces": forces,
        }

        if fin_conn:
            connected_count += 1
        for field_name, field_val in (
            ("final_compliance", fin_c),
            ("final_volume", fin_v),
            ("final_max_displacement", fin_d),
        ):
            if field_val is None or not np.isfinite(float(field_val)):
                errors.append(f"Episode {ep_id} (seed={seed}): non-finite or null {field_name}={field_val}")
        if bc_type in bc_compliance and fin_c is not None and np.isfinite(float(fin_c)):
            bc_compliance[bc_type].append(float(fin_c))
        if bc_type in bc_disp and fin_d is not None and np.isfinite(float(fin_d)):
            bc_disp[bc_type].append(float(fin_d))
        if fin_conn and fin_d is not None and float(fin_d) > 1e5:
            warnings.append(f"Episode {ep_id} (seed={seed}): connected final_max_displacement={float(fin_d):.2e} > 1e5")

        # Check BC mask consistency
        mz_sum = float(np.sum(bc_masks[2, 0, :, :]))
        mx_sum = float(np.sum(bc_masks[0, 0, :, :]))
        my_sum = float(np.sum(bc_masks[1, 0, :, :]))
        expected_plane = float(res[1] * res[2])
        if bc_type == "FULL_CLAMP":
            if not (mx_sum == expected_plane and my_sum == expected_plane and mz_sum == expected_plane):
                errors.append(f"Episode {ep_id}: FULL_CLAMP mask sums invalid ({mx_sum}, {my_sum}, {mz_sum})")
        elif bc_type == "RAIL_XY":
            if not (mx_sum == expected_plane and my_sum == expected_plane and mz_sum == 0.0):
                errors.append(f"Episode {ep_id}: RAIL_XY mask sums invalid ({mx_sum}, {my_sum}, {mz_sum})")

        # Check force alignment with load_config['x']
        lx, y_min, y_max, z_min, z_max = get_load_region_bounds(res, load_cfg)
        fy = forces[1]
        expected_patch = fy[lx, y_min:y_max, z_min:z_max]
        total_nonzero = int(np.sum(np.abs(forces) > 1e-6))
        patch_nonzero = int(np.sum(expected_patch < -0.5))
        if total_nonzero == 0 or total_nonzero != patch_nonzero:
            errors.append(
                f"Episode {ep_id}: force tensor misaligned with load_config x={lx} "
                f"(patch_nz={patch_nonzero}, total_nz={total_nonzero})"
            )

        if v_const is not None and strategy == "BEZIER":
            anchor_occ = float(np.mean(v_const[lx, y_min:y_max, z_min:z_max] > 0.5))
            if anchor_occ < 1.0:
                errors.append(
                    f"Episode {ep_id}: v_constructed load patch occupancy {anchor_occ:.2f} < 1.0 at x={lx}"
                )

        # Check step continuity & final step uniqueness
        steps = cursor.execute(
            "SELECT step, phase, is_final_step FROM records WHERE episode_id = ? ORDER BY step ASC",
            (ep_id,),
        ).fetchall()
        if not steps:
            errors.append(f"Episode {ep_id} (seed={seed}): ORPHAN episode with 0 records")
            continue

        step_nums = [s[0] for s in steps]
        if step_nums != list(range(len(steps))):
            errors.append(f"Episode {ep_id}: non-contiguous steps (min={min(step_nums)}, max={max(step_nums)}, len={len(steps)})")

        final_flags = [s[0] for s in steps if s[2] == 1]
        if len(final_flags) != 1 or final_flags[0] != step_nums[-1]:
            errors.append(f"Episode {ep_id}: invalid is_final_step flags={final_flags}, expected [{step_nums[-1]}]")

    # Global NaN/Inf and anomalous connected-displacement scan across all records
    non_finite_records = cursor.execute("""
        SELECT COUNT(*) FROM records
        WHERE fitness_score IS NULL
           OR fitness_score != fitness_score
           OR ABS(fitness_score) > 1e300
           OR (compliance IS NOT NULL AND (compliance != compliance OR ABS(compliance) > 1e300))
           OR (vol_frac IS NOT NULL AND (vol_frac != vol_frac OR ABS(vol_frac) > 1e300))
           OR (max_displacement IS NOT NULL AND (max_displacement != max_displacement OR ABS(max_displacement) > 1e300))
    """).fetchone()[0]
    if non_finite_records:
        errors.append(f"{non_finite_records} records contain NaN/Inf/NULL scalar values")

    high_disp_conn_steps = cursor.execute("""
        SELECT COUNT(*) FROM records
        WHERE is_connected = 1 AND max_displacement IS NOT NULL AND max_displacement > 1e5
    """).fetchone()[0]
    if high_disp_conn_steps:
        warnings.append(f"{high_disp_conn_steps} connected step(s) have max_displacement > 1e5")

    # Sample steps for full tensor & policy legality checks
    if total_records <= sample_steps:
        sampled_rows = cursor.execute("""
            SELECT episode_id, step, phase, density_blob, policy_add_blob, policy_remove_blob,
                   fitness_score, compliance, vol_frac, max_displacement, is_final_step, is_connected
            FROM records
        """).fetchall()
    else:
        sampled_rows = cursor.execute("""
            SELECT episode_id, step, phase, density_blob, policy_add_blob, policy_remove_blob,
                   fitness_score, compliance, vol_frac, max_displacement, is_final_step, is_connected
            FROM records
            WHERE is_final_step = 1
            UNION ALL
            SELECT episode_id, step, phase, density_blob, policy_add_blob, policy_remove_blob,
                   fitness_score, compliance, vol_frac, max_displacement, is_final_step, is_connected
            FROM (
                SELECT episode_id, step, phase, density_blob, policy_add_blob, policy_remove_blob,
                       fitness_score, compliance, vol_frac, max_displacement, is_final_step, is_connected
                FROM records
                ORDER BY RANDOM()
                LIMIT ?
            )
        """, (sample_steps,)).fetchall()

    p1_values: List[float] = []
    p2_values: List[float] = []
    p2_log10_c: List[float] = []
    p2_vols: List[float] = []
    p2_disps: List[float] = []
    illegal_add_violations = 0
    illegal_rem_violations = 0
    sum_violations = 0
    non_finite_sampled = 0

    for row in sampled_rows:
        ep_id, step, phase, dens_b, add_b, rem_b, val, comp, vol, disp, is_final, is_conn = row
        ep = episodes_by_id.get(ep_id)
        if ep is None:
            errors.append(f"Record ({ep_id}, {step}) references missing episode")
            continue

        for s_name, s_val in (("fitness_score", val), ("compliance", comp), ("vol_frac", vol), ("max_displacement", disp)):
            if s_val is not None and not np.isfinite(float(s_val)):
                non_finite_sampled += 1
        if val is None or not np.isfinite(float(val)):
            continue

        res = ep["res"]
        bc_masks_tuple = (ep["bc_masks"][0], ep["bc_masks"][1], ep["bc_masks"][2])
        force_mag = np.linalg.norm(ep["forces"], axis=0)

        density = deserialize_array(dens_b).astype(np.float32)
        add_idx, add_val = deserialize_sparse(add_b)
        rem_idx, rem_val = deserialize_sparse(rem_b)
        pol_add = sparse_decode(add_idx, add_val, res)
        pol_rem = sparse_decode(rem_idx, rem_val, res)

        valid_add = get_legal_add_moves(density, threshold=0.5, include_supports=True, bc_masks=bc_masks_tuple)
        protected = get_protected_regions(bc_masks_tuple, force_mag)
        valid_rem = get_legal_remove_moves(density, threshold=0.5) * (1.0 - protected)

        bad_add = float(np.sum(pol_add * (1.0 - valid_add)))
        bad_rem = float(np.sum(pol_rem * (1.0 - valid_rem)))
        if bad_add > 1e-5:
            illegal_add_violations += 1
        if bad_rem > 1e-5:
            illegal_rem_violations += 1

        add_sum = float(np.sum(pol_add))
        rem_sum = float(np.sum(pol_rem))
        tot_sum = add_sum + rem_sum

        if phase == "GROWTH":
            p1_values.append(float(val))
            if rem_sum > 1e-6 or abs(add_sum - 1.0) > 1e-3:
                sum_violations += 1
        else:
            p2_values.append(float(val))
            if comp is not None and comp > 0:
                p2_log10_c.append(float(np.log10(max(comp, 1e-9))))
            if vol is not None:
                p2_vols.append(float(vol))
            if disp is not None:
                p2_disps.append(float(disp))
            if is_final:
                if tot_sum > 1e-6:
                    sum_violations += 1
            else:
                if not (tot_sum < 1e-6 or abs(tot_sum - 1.0) < 1e-3):
                    sum_violations += 1

    if non_finite_sampled:
        errors.append(f"{non_finite_sampled} non-finite scalar field(s) found in sampled steps")
    if illegal_add_violations:
        errors.append(f"{illegal_add_violations}/{len(sampled_rows)} sampled steps have policy_add outside legal moves")
    if illegal_rem_violations:
        errors.append(f"{illegal_rem_violations}/{len(sampled_rows)} sampled steps have policy_remove outside legal/unprotected moves")
    if sum_violations:
        errors.append(f"{sum_violations}/{len(sampled_rows)} sampled steps failed policy normalization/phase sum invariants")

    if len(ep_rows) == 0:
        errors.append("Database contains 0 episodes")
    elif connected_count < len(ep_rows):
        warnings.append(f"{len(ep_rows) - connected_count}/{len(ep_rows)} episodes finished disconnected at threshold=0.5")

    conn.close()

    report = {
        "db_path": str(db_path),
        "sqlite_integrity": integrity,
        "episodes": len(ep_rows),
        "connected_episodes": connected_count,
        "total_records": total_records,
        "phase_counts": phase_counts,
        "sampled_steps_checked": len(sampled_rows),
        "bc_stats": {
            bc: {
                "compliance": _quantiles(bc_compliance[bc]),
                "max_displacement": _quantiles(bc_disp[bc]),
            }
            for bc in ("FULL_CLAMP", "RAIL_XY")
        },
        "value_stats": {
            "GROWTH": _quantiles(p1_values),
            "REFINEMENT": _quantiles(p2_values),
        },
        "phase2_physics_stats": {
            "log10_compliance": _quantiles(p2_log10_c),
            "vol_frac": _quantiles(p2_vols),
            "max_displacement": _quantiles(p2_disps),
        },
        "errors": errors,
        "warnings": warnings,
        "passed": len(errors) == 0,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit AlphaBuilder v3.3 Harvest DB")
    parser.add_argument("db_path", type=str, help="Path to SQLite database")
    parser.add_argument("--sample-steps", type=int, default=250, help="Max steps to tensor-audit")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to save JSON report")
    args = parser.parse_args()

    report = audit_database(Path(args.db_path), sample_steps=args.sample_steps)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
