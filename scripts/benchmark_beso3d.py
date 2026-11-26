#!/usr/bin/env python3
"""
Parallel Empirical Benchmark of Soft-Kill BESO 3D on S1..S6 (Canonical) + G1..G4 (OOD).

Runs all 10 scenarios directly on StructuredHex8FEM (`fast_amg`), captures both
converged `iso-V` (matching SIMP_cont volume) and `Pareto-Best` (max Score v3.3),
and optionally cross-checks parity against DOLFINx if available.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphabuilder.src.logic.benchmarks.beso3d import BESOSnapshot, run_beso3d
from alphabuilder.src.logic.harvest.generators import generate_bc_masks, get_load_region_bounds
from alphabuilder.src.logic.selfplay.evaluator import HAS_FENICS, evaluate_main_island
from alphabuilder.src.logic.selfplay.reward import analyze_structure_islands
from alphabuilder.src.logic.selfplay.scenarios_stage6 import build_ood_benchmark_scenarios
from scripts.evaluate_inference_scenarios import RESOLUTION, SCENARIOS

EVAL_STAGE5_JSON = REPO_ROOT / "data" / "analysis" / "eval_stage5_phys_uvit_results.json"
OUT_JSON = REPO_ROOT / "data" / "analysis" / "beso3d_benchmark_results.json"
OUT_NPZ = REPO_ROOT / "data" / "analysis" / "beso3d_geometries.npz"


def _snap_to_dict(s: Optional[BESOSnapshot]) -> Optional[Dict[str, Any]]:
    if s is None:
        return None
    return {
        "tag": s.tag,
        "iteration": int(s.iteration),
        "volume_fraction": float(s.volume_fraction),
        "compliance": float(s.compliance),
        "max_displacement": float(s.max_displacement),
        "fem_reward": float(s.fem_reward),
        "island_penalty": float(s.island_penalty),
        "total_score": float(s.total_score),
        "is_connected": bool(s.is_connected),
        "n_islands": int(s.n_islands),
        "loose_voxels": int(s.loose_voxels),
        "geometry": dict(s.geometry),
    }


def _run_one_task(task: Tuple[Dict[str, Any], float, str]) -> Tuple[str, Dict[str, Any], Dict[str, np.ndarray]]:
    sc, v_iso_ref, suite = task
    res = tuple(int(v) for v in sc.get("resolution", RESOLUTION))
    bc_type = str(sc["bc_type"])
    lcfg = dict(sc["load_config"])
    lcfg.setdefault("bc_type", bc_type)

    bc_masks = sc.get("bc_masks")
    forces = sc.get("forces")
    domain_mask = sc.get("domain_mask")
    if bc_masks is None:
        mx, my, mz = generate_bc_masks(res, bc_type)
        bc_masks = np.stack([mx, my, mz], axis=0).astype(np.float32)
    if forces is None:
        fx = np.zeros(res, dtype=np.float32)
        fy = np.zeros(res, dtype=np.float32)
        fz = np.zeros(res, dtype=np.float32)
        lx, y0, y1, z0, z1 = get_load_region_bounds(res, lcfg)
        fy[lx, y0:y1, z0:z1] = -1.0
        forces = np.stack([fx, fy, fz], axis=0).astype(np.float32)

    beso_res = run_beso3d(
        resolution=res,  # type: ignore[arg-type]
        load_config=lcfg,
        bc_type=bc_type,
        bc_masks=np.asarray(bc_masks, dtype=np.float32),
        forces=np.asarray(forces, dtype=np.float32),
        domain_mask=np.asarray(domain_mask, dtype=np.float32) if domain_mask is not None else None,
        scenario_id=str(sc["id"]),
        v_target_min=0.060,
        v_iso_ref=float(v_iso_ref),
        iso_stabilize_iters=8,
        er_coarse=0.03,
        er_fine=0.02,
        ar_max=0.02,
        r_min=2.0,
        x_min=1e-2,
        max_iter=125,
        protect_cg1_halo=True,
        clean_floaters=True,
    )

    rec = {
        "scenario_id": str(sc["id"]),
        "suite": suite,
        "title": str(sc["title"]),
        "resolution": list(res),
        "bc_type": bc_type,
        "iterations_run": int(beso_res.iterations_run),
        "elapsed_sec": float(beso_res.elapsed_sec),
        "iso_v_target": float(beso_res.iso_v_target),
        "iso_v": _snap_to_dict(beso_res.iso_v),
        "pareto_best": _snap_to_dict(beso_res.pareto_best),
        "history": beso_res.history,
    }
    densities: Dict[str, np.ndarray] = {}
    if beso_res.iso_v is not None:
        densities[f"{sc['id']}__iso_v"] = (beso_res.iso_v.density > 0.5).astype(np.uint8)
    if beso_res.pareto_best is not None:
        densities[f"{sc['id']}__pareto"] = (beso_res.pareto_best.density > 0.5).astype(np.uint8)
    return str(sc["id"]), rec, densities


def main() -> None:
    t_start = time.perf_counter()
    simp_vols: Dict[str, float] = {}
    simp_summary: Dict[str, Dict[str, Any]] = {}
    if EVAL_STAGE5_JSON.exists():
        stage5 = json.loads(EVAL_STAGE5_JSON.read_text(encoding="utf-8"))
        simp_summary = stage5.get("simp_baselines", {})
        for sid, sdata in simp_summary.items():
            if sdata.get("vol_cont"):
                simp_vols[sid] = float(sdata["vol_cont"])

    tasks: List[Tuple[Dict[str, Any], float, str]] = []
    for sc in SCENARIOS:
        sc_copy = dict(sc)
        sc_copy["resolution"] = RESOLUTION
        tasks.append((sc_copy, simp_vols.get(sc["id"], 0.135), "canonical"))
    for sc in build_ood_benchmark_scenarios():
        tasks.append((dict(sc), 0.135, "ood"))

    records: Dict[str, Dict[str, Any]] = {}
    all_densities: Dict[str, np.ndarray] = {}

    n_workers = min(4, len(tasks))
    print(f"Running BESO 3D benchmark on {len(tasks)} scenarios using {n_workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_run_one_task, t): t[0]["id"] for t in tasks}
        for fut in as_completed(futs):
            sid, rec, dens = fut.result()
            records[sid] = rec
            all_densities.update(dens)
            iso = rec["iso_v"] or {}
            par = rec["pareto_best"] or {}
            print(
                f"  [{sid}] iters={rec['iterations_run']:3d} time={rec['elapsed_sec']:5.1f}s | "
                f"isoV({iso.get('volume_fraction', 0)*100:4.1f}%): C={iso.get('compliance', 0):6.1f} Score={iso.get('total_score', -1):+.3f} | "
                f"Pareto({par.get('volume_fraction', 0)*100:4.1f}%): C={par.get('compliance', 0):6.1f} Score={par.get('total_score', -1):+.3f}",
                flush=True,
            )

    ordered_ids = [t[0]["id"] for t in tasks]
    ordered_recs = [records[sid] for sid in ordered_ids]

    def _agg(suite_name: str, key: str) -> Dict[str, Any]:
        sub = [r[key] for r in ordered_recs if r["suite"] == suite_name and r.get(key)]
        return {
            "n": len(sub),
            "mean_volume": float(np.mean([s["volume_fraction"] for s in sub])) if sub else None,
            "mean_compliance": float(np.mean([s["compliance"] for s in sub])) if sub else None,
            "mean_score": float(np.mean([s["total_score"] for s in sub])) if sub else None,
            "mean_inertia_y": float(np.mean([s["geometry"]["inertia_y_norm"] for s in sub])) if sub else None,
            "mean_hollow_xy": float(np.mean([s["geometry"]["hollow_ratio_xy"] for s in sub])) if sub else None,
            "connected_count": int(sum(1 for s in sub if s["is_connected"])),
        }

    dolfinx_parity = []
    if HAS_FENICS:
        for sid in ("S1_cantilever_long_center", "G2_bridge_mbb"):
            key_p = f"{sid}__pareto"
            if key_p in all_densities:
                sc_obj = next(t[0] for t in tasks if t[0]["id"] == sid)
                d_arr = all_densities[key_p].astype(np.float32)
                res = tuple(int(v) for v in sc_obj.get("resolution", RESOLUTION))
                lcfg = dict(sc_obj["load_config"])
                lcfg.setdefault("bc_type", sc_obj["bc_type"])
                bc_m = sc_obj.get("bc_masks")
                f_m = sc_obj.get("forces")
                if bc_m is None:
                    mx, my, mz = generate_bc_masks(res, sc_obj["bc_type"])
                    bc_m = np.stack([mx, my, mz], axis=0).astype(np.float32)
                isl = analyze_structure_islands(
                    d_arr, lcfg, bc_masks=(bc_m[0], bc_m[1], bc_m[2]),
                    forces=(f_m[0], f_m[1], f_m[2]) if f_m is not None else None,
                )
                dolf = evaluate_main_island(
                    d_arr, lcfg, isl["main_island_mask"], bc_type=sc_obj["bc_type"],
                    backend="dolfinx", bc_masks=sc_obj.get("bc_masks"), forces=sc_obj.get("forces"),
                )
                fast_c = records[sid]["pareto_best"]["compliance"]
                if dolf.valid:
                    rel_err = abs(fast_c - float(dolf.compliance)) / float(dolf.compliance)
                    dolfinx_parity.append({
                        "scenario_id": sid,
                        "fast_amg_compliance": float(fast_c),
                        "dolfinx_compliance": float(dolf.compliance),
                        "rel_err": float(rel_err),
                    })

    summary = {
        "canonical_iso_v": _agg("canonical", "iso_v"),
        "canonical_pareto_best": _agg("canonical", "pareto_best"),
        "ood_iso_v": _agg("ood", "iso_v"),
        "ood_pareto_best": _agg("ood", "pareto_best"),
        "mean_iter_time_sec": float(np.mean([r["elapsed_sec"] / max(1, r["iterations_run"]) for r in ordered_recs])),
        "total_wall_sec": round(time.perf_counter() - t_start, 2),
        "dolfinx_parity": dolfinx_parity,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps({"summary": summary, "scenarios": ordered_recs}, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(OUT_NPZ, **all_densities)
    print(
        f"\nBESO 3D COMPLETE in {summary['total_wall_sec']:.1f}s | "
        f"S1..S6 iso-V Score={summary['canonical_iso_v']['mean_score']:+.4f} (C={summary['canonical_iso_v']['mean_compliance']:.1f}) | "
        f"Pareto Score={summary['canonical_pareto_best']['mean_score']:+.4f} (C={summary['canonical_pareto_best']['mean_compliance']:.1f}) | "
        f"G1..G4 Pareto Score={summary['ood_pareto_best']['mean_score']:+.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
