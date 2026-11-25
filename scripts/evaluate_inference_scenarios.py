#!/usr/bin/env python3
"""Real-Inference + MCTS + FEniTop SIMP Multi-Scenario Evaluation Harness (v3.3).

Evaluates a trained AlphaBuilder v3.3 neural checkpoint across 6 canonical 3D structural
scenarios using the full v3.3 game loop (`AlphaBuilderMCTS`, `select_pv_branches`,
PBRS shaping in Phase 1, and real FEniCSx FEM evaluation in Phase 2), alongside
FEniTop SIMP baselines.

All trajectories are persisted in standard Studio v4 Self-Play SQLite format
(`games` + `game_steps` with sparse `COO1` policy/MCTS visit/Q tensors and
displacement maps), and full quantitative diagnostics are written to JSON.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphabuilder.src.core.physics_model import PhysicalProperties
from alphabuilder.src.logic.harvest.config import SIMPConfig
from alphabuilder.src.logic.harvest.generators import (
    generate_bc_masks,
    get_load_region_bounds,
)
from alphabuilder.src.logic.harvest.optimization import run_fenitop_optimization
from alphabuilder.src.logic.mcts.engine import (
    AlphaBuilderMCTS,
    PHASE1_CONFIG,
    PHASE2_CONFIG,
    SearchResult,
    build_state_tensor,
    select_pv_branches,
)
from alphabuilder.src.logic.selfplay import (
    GameInfo,
    backpropagate_value_targets,
    initialize_selfplay_db,
    record_step,
    save_game,
    update_game_final,
)
from alphabuilder.src.logic.selfplay.evaluator import evaluate_main_island
from alphabuilder.src.logic.selfplay.reward import (
    analyze_structure_islands,
    calculate_connectivity_reward,
    calculate_island_penalty,
    calculate_pbrs_potential,
    calculate_reward,
    get_guidance_alpha,
    get_phase2_terminal_reward,
)
from alphabuilder.src.logic.storage import (
    Phase,
    deserialize_array,
    serialize_array,
    serialize_sparse,
    sparse_encode,
)
from alphabuilder.src.neural.inference import AlphaBuilderInference


RESOLUTION: Tuple[int, int, int] = (64, 32, 8)

SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "S1_cantilever_long_center",
        "title": "Cantilever longa (x=63, carga centrada)",
        "bc_type": "FULL_CLAMP",
        "load_config": {"x": 63, "y": 16, "z_start": 3, "z_end": 5, "bc_type": "FULL_CLAMP"},
    },
    {
        "id": "S2_cantilever_tip_bottom",
        "title": "Cantilever assimétrica inferior (x=63, y=6)",
        "bc_type": "FULL_CLAMP",
        "load_config": {"x": 63, "y": 6, "z_start": 3, "z_end": 5, "bc_type": "FULL_CLAMP"},
    },
    {
        "id": "S3_cantilever_midspan_top",
        "title": "Cantilever intermediária superior (x=44, y=24)",
        "bc_type": "FULL_CLAMP",
        "load_config": {"x": 44, "y": 24, "z_start": 3, "z_end": 5, "bc_type": "FULL_CLAMP"},
    },
    {
        "id": "S4_rail_xy_center",
        "title": "Suporte RAIL_XY centrado (x=58, y=16)",
        "bc_type": "RAIL_XY",
        "load_config": {"x": 58, "y": 16, "z_start": 3, "z_end": 5, "bc_type": "RAIL_XY"},
    },
    {
        "id": "S5_rail_xy_3d_eccentric",
        "title": "Suporte RAIL_XY excêntrico 3D (x=52, y=11, z=[1,3])",
        "bc_type": "RAIL_XY",
        "load_config": {"x": 52, "y": 11, "z_start": 1, "z_end": 3, "bc_type": "RAIL_XY"},
    },
    {
        "id": "S6_short_deep_clamp",
        "title": "Cantilever curta profunda (x=36, y=16)",
        "bc_type": "FULL_CLAMP",
        "load_config": {"x": 36, "y": 16, "z_start": 3, "z_end": 5, "bc_type": "FULL_CLAMP"},
    },
]


@dataclass
class EvalEpisodeState:
    """Mutable episode container compatible with `selfplay.storage.record_step`."""

    density: np.ndarray
    bc_masks: np.ndarray
    forces: np.ndarray
    load_config: Dict[str, Any]
    current_step: int = 0
    phase: Phase = Phase.GROWTH
    is_connected: bool = False
    game_id: str = ""


def _build_masks_and_forces(
    resolution: Tuple[int, int, int],
    load_config: Dict[str, Any],
    bc_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    mask_x, mask_y, mask_z = generate_bc_masks(resolution, bc_type)
    bc_masks = np.stack([mask_x, mask_y, mask_z], axis=0).astype(np.float32)

    fx = np.zeros(resolution, dtype=np.float32)
    fy = np.zeros(resolution, dtype=np.float32)
    fz = np.zeros(resolution, dtype=np.float32)
    lx, y_min, y_max, z_min, z_max = get_load_region_bounds(resolution, load_config)
    fy[lx, y_min:y_max, z_min:z_max] = -1.0
    forces = np.stack([fx, fy, fz], axis=0).astype(np.float32)
    return bc_masks, forces


def _dice_and_iou(pred_mask: np.ndarray, target_mask: np.ndarray) -> Tuple[float, float]:
    pred = pred_mask.astype(bool)
    tgt = target_mask.astype(bool)
    inter = float(np.logical_and(pred, tgt).sum())
    pred_sum = float(pred.sum())
    tgt_sum = float(tgt.sum())
    union = float(np.logical_or(pred, tgt).sum())
    dice = (2.0 * inter / (pred_sum + tgt_sum)) if (pred_sum + tgt_sum) > 0.0 else 0.0
    iou = (inter / union) if union > 0.0 else 0.0
    return float(dice), float(iou)


def _span_y_slice(coords_x: np.ndarray, coords_y: np.ndarray, x_lo: float, x_hi: float) -> Optional[float]:
    mask = (coords_x >= x_lo) & (coords_x <= x_hi)
    if not np.any(mask):
        return None
    ys = coords_y[mask]
    return float(int(ys.max()) - int(ys.min()) + 1)


def compute_geometry_metrics(
    density: np.ndarray,
    load_config: Dict[str, Any],
    bc_masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Dict[str, Any]:
    """Compute quantitative structural shape & truss/beam diagnostics on a 3D density grid."""
    nx, ny, nz = density.shape
    solid = density > 0.5
    isl = analyze_structure_islands(density, load_config, threshold=0.5, bc_masks=bc_masks)
    solid_voxels = int(solid.sum())
    vol_frac = float(solid.mean())

    if solid_voxels == 0:
        return {
            "solid_voxels": 0,
            "volume_fraction": 0.0,
            "is_connected": False,
            "n_islands": 0,
            "loose_voxels": 0,
            "x_min": None,
            "x_max": None,
            "y_span_root": None,
            "y_span_mid": None,
            "y_span_tip": None,
            "inertia_y_norm": 0.0,
            "hollow_ratio_xy": 0.0,
            "overshoot_frac": 0.0,
            "z_centroid": None,
        }

    xs, ys, zs = np.nonzero(solid)
    lx = int(load_config["x"])
    y_span_root = _span_y_slice(xs, ys, 0, max(1.0, lx / 3.0))
    y_span_mid = _span_y_slice(xs, ys, lx / 3.0, 2.0 * lx / 3.0)
    y_span_tip = _span_y_slice(xs, ys, 2.0 * lx / 3.0, float(lx))

    y_mean = float(ys.mean())
    inertia_y_norm = float(np.mean(((ys - y_mean) / (ny / 2.0)) ** 2))

    # 2D side projection (X, Y): measure hollow web/truss envelope vs solid beam
    proj_xy = np.any(solid, axis=2)
    occupied_xy = 0
    envelope_xy = 0
    for x in range(nx):
        col = np.flatnonzero(proj_xy[x])
        if col.size > 0:
            occupied_xy += int(col.size)
            envelope_xy += int(col.max() - col.min() + 1)
    hollow_ratio_xy = float(1.0 - occupied_xy / envelope_xy) if envelope_xy > 0 else 0.0

    overshoot_frac = float(np.sum(xs > lx + 2) / solid_voxels)

    return {
        "solid_voxels": solid_voxels,
        "volume_fraction": vol_frac,
        "is_connected": bool(isl["is_connected"]),
        "n_islands": int(isl["n_islands"]),
        "loose_voxels": int(isl["loose_voxels"]),
        "x_min": int(xs.min()),
        "x_max": int(xs.max()),
        "y_span_root": y_span_root,
        "y_span_mid": y_span_mid,
        "y_span_tip": y_span_tip,
        "inertia_y_norm": inertia_y_norm,
        "hollow_ratio_xy": hollow_ratio_xy,
        "overshoot_frac": overshoot_frac,
        "z_centroid": float(zs.mean()),
    }


def _delete_game(db_path: Path, game_id: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM game_steps WHERE game_id = ?", (game_id,))
        conn.execute("DELETE FROM games WHERE game_id = ?", (game_id,))
        conn.commit()
    finally:
        conn.close()


def _game_exists(db_path: Path, game_id: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()
        if not row or row[0] == 0:
            return False
        srow = conn.execute(
            "SELECT COUNT(*) FROM game_steps WHERE game_id = ?", (game_id,)
        ).fetchone()
        return bool(srow and srow[0] > 0)
    finally:
        conn.close()


def _load_existing_simp_summary(
    db_path: Path,
    scenario: Dict[str, Any],
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    game_id = f"simp__{scenario['id']}"
    if not _game_exists(db_path, game_id):
        return None
    conn = sqlite3.connect(db_path)
    try:
        g = conn.execute(
            "SELECT final_score, final_compliance, final_volume, total_steps FROM games WHERE game_id = ?",
            (game_id,),
        ).fetchone()
        last = conn.execute(
            """
            SELECT density_blob, compliance_fem, max_displacement, volume_fraction,
                   is_connected, fem_reward, reward_components_json
            FROM game_steps
            WHERE game_id = ?
            ORDER BY step DESC LIMIT 1
            """,
            (game_id,),
        ).fetchone()
    finally:
        conn.close()

    if not last or last[0] is None:
        return None
    density = deserialize_array(last[0]).astype(np.float32)
    load_cfg = dict(scenario["load_config"])
    bc_type = scenario["bc_type"]
    load_cfg["bc_type"] = bc_type
    bc_masks, _ = _build_masks_and_forces(RESOLUTION, load_cfg, bc_type)
    bc_tuple = (bc_masks[0], bc_masks[1], bc_masks[2])

    extra: Dict[str, Any] = {}
    if last[6]:
        try:
            extra = json.loads(last[6])
        except json.JSONDecodeError:
            extra = {}

    if "comp_bin" not in extra:
        binary = (density > 0.5).astype(np.float32)
        isl = analyze_structure_islands(binary, load_cfg, bc_masks=bc_tuple)
        fem_bin = evaluate_main_island(binary, load_cfg, isl["main_island_mask"], bc_type=bc_type)
        comp_bin = float(fem_bin.compliance) if fem_bin.valid else None
        max_disp_bin = float(fem_bin.max_displacement) if fem_bin.valid else None
        vol_bin = float(np.mean(binary > 0.5))
        reward_bin = (
            float(calculate_reward(comp_bin, vol_bin, bool(isl["is_connected"]), max_disp_bin))
            if comp_bin is not None
            else None
        )
    else:
        comp_bin = extra.get("comp_bin")
        max_disp_bin = extra.get("max_disp_bin")
        vol_bin = extra.get("vol_bin", float(np.mean(density > 0.5)))
        reward_bin = extra.get("reward_bin")

    geom = compute_geometry_metrics((density > 0.5).astype(np.float32), load_cfg, bc_tuple)
    summary = {
        "game_id": game_id,
        "steps": int(g[3] or 0) if g else 0,
        "comp_cont": float(last[1]) if last[1] is not None else (float(g[1]) if g and g[1] is not None else None),
        "vol_cont": float(last[3]) if last[3] is not None else (float(g[2]) if g and g[2] is not None else None),
        "max_disp_cont": float(last[2]) if last[2] is not None else None,
        "reward_cont": float(last[5]) if last[5] is not None else None,
        "comp_bin": comp_bin,
        "vol_bin": vol_bin,
        "max_disp_bin": max_disp_bin,
        "reward_bin": reward_bin,
        "is_connected_bin": bool(geom["is_connected"]),
        "geometry": geom,
    }
    return density, summary


def run_simp_baseline(
    db_path: Path,
    scenario: Dict[str, Any],
    max_iter: int = 60,
    target_vol_frac: float = 0.15,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run FEniTop SIMP on one scenario and persist all steps to `db_path`."""
    game_id = f"simp__{scenario['id']}"
    load_cfg = dict(scenario["load_config"])
    bc_type = scenario["bc_type"]
    load_cfg["bc_type"] = bc_type

    print(f"\n[SIMP] {scenario['id']} ({bc_type}, load={load_cfg}) ...", flush=True)
    t0 = time.perf_counter()
    bc_masks, forces = _build_masks_and_forces(RESOLUTION, load_cfg, bc_type)
    bc_tuple = (bc_masks[0], bc_masks[1], bc_masks[2])
    force_tuple = (forces[0], forces[1], forces[2])

    props = PhysicalProperties(E=100.0, nu=0.25)
    simp_cfg = SIMPConfig(
        vol_frac=target_vol_frac,
        max_iter=max_iter,
        r_min=1.5,
        load_config=load_cfg,
    )
    history = run_fenitop_optimization(
        RESOLUTION,
        props,
        simp_cfg,
        initial_density=None,
        strategy="FULL_DOMAIN",
        record_displacement_map=True,
    )
    if not history:
        raise RuntimeError(f"FEniTop SIMP returned empty history for {scenario['id']}")

    final_rho = np.asarray(history[-1]["density_map"], dtype=np.float32)
    binary_final = (final_rho > 0.5).astype(np.float32)
    isl_bin = analyze_structure_islands(binary_final, load_cfg, bc_masks=bc_tuple)
    fem_bin = evaluate_main_island(binary_final, load_cfg, isl_bin["main_island_mask"], bc_type=bc_type)
    comp_bin = float(fem_bin.compliance) if fem_bin.valid else None
    max_disp_bin = float(fem_bin.max_displacement) if fem_bin.valid else None
    vol_bin = float(np.mean(binary_final > 0.5))
    reward_bin = (
        float(calculate_reward(comp_bin, vol_bin, bool(isl_bin["is_connected"]), max_disp_bin))
        if comp_bin is not None
        else None
    )

    _delete_game(db_path, game_id)
    save_game(
        db_path,
        GameInfo(
            game_id=game_id,
            neural_engine="fenitop-simp",
            checkpoint_version="simp-ref-v3.3",
            bc_masks=bc_masks,
            forces=forces,
            load_config=load_cfg,
            bc_type=bc_type,
            resolution=RESOLUTION,
            total_steps=len(history),
        ),
    )

    conn = sqlite3.connect(db_path)
    try:
        rows = []
        last_cont_reward = 0.0
        for step_idx, entry in enumerate(history):
            rho_3d = np.asarray(entry["density_map"], dtype=np.float32)
            disp_3d = entry.get("displacement_map")
            comp = float(entry["compliance"])
            vol = float(entry["vol_frac"])
            max_disp = float(entry["max_displacement"])
            fem_score = float(entry["fem_score"])
            isl = analyze_structure_islands(rho_3d, load_cfg, threshold=0.5, bc_masks=bc_tuple)
            _, conn_frac = calculate_connectivity_reward(rho_3d, bc_tuple, force_tuple, threshold=0.5)
            vox = int((rho_3d > 0.5).sum())
            isl_pen = calculate_island_penalty(isl["n_islands"], isl["loose_voxels"], vox)
            cont_reward = float(calculate_reward(comp, vol, True, max_disp))
            last_cont_reward = cont_reward

            comp_payload = {
                "base_reward": 0.0,
                "connectivity_bonus": 0.0,
                "connected_load_fraction": float(conn_frac),
                "fem_reward": cont_reward,
                "island_penalty": float(isl_pen),
                "total": cont_reward,
                "beta": float(entry.get("beta", 1.0)),
            }
            if step_idx == len(history) - 1:
                comp_payload.update(
                    {
                        "comp_bin": comp_bin,
                        "max_disp_bin": max_disp_bin,
                        "vol_bin": vol_bin,
                        "reward_bin": reward_bin,
                    }
                )

            rows.append(
                (
                    game_id,
                    step_idx,
                    "REFINEMENT",
                    serialize_array(rho_3d),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "[]",
                    fem_score,
                    None,
                    int(isl["n_islands"]),
                    int(isl["loose_voxels"]),
                    1 if isl["is_connected"] else 0,
                    comp,
                    max_disp,
                    float(isl_pen),
                    vol,
                    json.dumps(comp_payload),
                    float(conn_frac),
                    entry.get("final_fem_score"),
                    reward_bin if reward_bin is not None else cont_reward,
                    cont_reward,
                    serialize_array(np.asarray(disp_3d, dtype=np.float32))
                    if disp_3d is not None
                    else None,
                )
            )
        conn.executemany(
            """
            INSERT INTO game_steps (
                game_id, step, phase, density_blob,
                policy_add_blob, policy_remove_blob,
                mcts_visit_add_blob, mcts_visit_remove_blob,
                mcts_q_add_blob, mcts_q_remove_blob,
                selected_actions_json, value, mcts_stats_json,
                n_islands, loose_voxels, is_connected,
                compliance_fem, max_displacement, island_penalty, volume_fraction,
                reward_components_json, connected_load_fraction, value_target,
                final_reward, fem_reward, displacement_map_blob
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    final_entry = history[-1]
    comp_cont = float(final_entry["compliance"])
    vol_cont = float(final_entry["vol_frac"])
    max_disp_cont = float(final_entry["max_displacement"])
    final_score = reward_bin if reward_bin is not None else last_cont_reward
    update_game_final(
        db_path=db_path,
        game_id=game_id,
        final_score=final_score,
        final_compliance=comp_cont,
        final_volume=vol_cont,
        total_steps=len(history),
    )
    backpropagate_value_targets(db_path, game_id)

    geom = compute_geometry_metrics(binary_final, load_cfg, bc_tuple)
    elapsed = time.perf_counter() - t0
    print(
        f"[SIMP] {scenario['id']} done in {elapsed:.1f}s | steps={len(history)} | "
        f"C_cont={comp_cont:.1f} (V={vol_cont*100:.2f}%) | "
        f"C_bin={f'{comp_bin:.1f}' if comp_bin is not None else 'INF'} "
        f"(V_bin={vol_bin*100:.2f}%, conn={geom['is_connected']}, hollow={geom['hollow_ratio_xy']:.2f})",
        flush=True,
    )
    summary = {
        "game_id": game_id,
        "steps": len(history),
        "elapsed_sec": round(elapsed, 2),
        "comp_cont": comp_cont,
        "vol_cont": vol_cont,
        "max_disp_cont": max_disp_cont,
        "reward_cont": last_cont_reward,
        "comp_bin": comp_bin,
        "vol_bin": vol_bin,
        "max_disp_bin": max_disp_bin,
        "reward_bin": reward_bin,
        "is_connected_bin": bool(geom["is_connected"]),
        "geometry": geom,
    }
    return final_rho, summary


def _update_mcts_q_blobs(
    db_path: Path,
    game_id: str,
    step_idx: int,
    root: Any,
    resolution: Tuple[int, int, int],
) -> None:
    if root is None or not getattr(root, "children", None):
        return
    q_add = np.zeros(resolution, dtype=np.float32)
    q_rem = np.zeros(resolution, dtype=np.float32)
    for (ch, x, y, z), child in root.children.items():
        if ch == 0:
            q_add[x, y, z] = float(child.mean_value)
        else:
            q_rem[x, y, z] = float(child.mean_value)
    qa_idx, qa_val = sparse_encode(q_add)
    qr_idx, qr_val = sparse_encode(q_rem)
    qa_blob = serialize_sparse(qa_idx, qa_val) if len(qa_idx) > 0 else None
    qr_blob = serialize_sparse(qr_idx, qr_val) if len(qr_idx) > 0 else None
    if qa_blob is None and qr_blob is None:
        return
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            UPDATE game_steps
            SET mcts_q_add_blob = ?, mcts_q_remove_blob = ?
            WHERE game_id = ? AND step = ?
            """,
            (qa_blob, qr_blob, game_id, step_idx),
        )
        conn.commit()
    finally:
        conn.close()


def run_neural_mcts_episode(
    model: AlphaBuilderInference,
    db_path: Path,
    scenario: Dict[str, Any],
    prefix: str,
    checkpoint_version: str,
    num_simulations: int,
    max_phase1_steps: int,
    max_phase2_steps: int,
    simp_binary_mask: Optional[np.ndarray],
    simp_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run one deterministic MCTS + FEniCSx episode on `scenario` and persist to `db_path`."""
    game_id = f"{prefix}__{scenario['id']}"
    load_cfg = dict(scenario["load_config"])
    bc_type = scenario["bc_type"]
    load_cfg["bc_type"] = bc_type

    print(
        f"\n[{prefix.upper()}] Starting {scenario['id']} ({bc_type}, sims={num_simulations}) ...",
        flush=True,
    )
    ep_start = time.perf_counter()

    bc_masks, forces = _build_masks_and_forces(RESOLUTION, load_cfg, bc_type)
    bc_tuple = (bc_masks[0], bc_masks[1], bc_masks[2])
    force_tuple = (forces[0], forces[1], forces[2])

    state = EvalEpisodeState(
        density=np.zeros(RESOLUTION, dtype=np.float32),
        bc_masks=bc_masks,
        forces=forces,
        load_config=load_cfg,
        current_step=0,
        phase=Phase.GROWTH,
        is_connected=False,
        game_id=game_id,
    )

    engine_name = getattr(
        model,
        "architecture",
        "swin-unetr" if getattr(model, "use_swin", False) else "simple",
    )
    _delete_game(db_path, game_id)
    save_game(
        db_path,
        GameInfo(
            game_id=game_id,
            neural_engine=engine_name,
            checkpoint_version=checkpoint_version,
            bc_masks=bc_masks,
            forces=forces,
            load_config=load_cfg,
            bc_type=bc_type,
            resolution=RESOLUTION,
            total_steps=0,
        ),
    )

    alpha_guide = get_guidance_alpha(generation=0, decay_generations=5)
    phi_root = [0.0]

    def _shape_value(
        net_val: float,
        density_grid: np.ndarray,
        masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
        force_vecs: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> float:
        isl = analyze_structure_islands(density_grid, load_cfg, bc_masks=masks)
        total_vox = int((density_grid > 0.5).sum())
        isl_pen = calculate_island_penalty(isl["n_islands"], isl["loose_voxels"], total_vox)
        if state.phase == Phase.REFINEMENT:
            if not isl["is_connected"]:
                return -1.0
            return float(np.clip(float(net_val) - isl_pen, -1.0, 1.0))
        if alpha_guide > 0.0:
            phi_leaf, _ = calculate_pbrs_potential(density_grid, masks, force_vecs, load_cfg)
            return float(np.clip(float(net_val) + alpha_guide * (phi_leaf - phi_root[0]), -1.0, 1.0))
        return float(np.clip(float(net_val), -1.0, 1.0))

    def hybrid_predict(state_tensor: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        nv, pa, pr = model.predict(state_tensor)
        masks = (state_tensor[1], state_tensor[2], state_tensor[3])
        fvecs = (state_tensor[4], state_tensor[5], state_tensor[6])
        return _shape_value(nv, state_tensor[0], masks, fvecs), pa, pr

    def hybrid_predict_batch(
        state_tensors: List[np.ndarray],
    ) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
        nvs, pas, prs = model.predict_batch(state_tensors)
        shaped = [
            _shape_value(nv, st[0], (st[1], st[2], st[3]), (st[4], st[5], st[6]))
            for nv, st in zip(nvs, state_tensors)
        ]
        return shaped, pas, prs

    p1_cfg = PHASE1_CONFIG._replace(num_simulations=num_simulations, add_root_noise=False)
    p2_cfg = PHASE2_CONFIG._replace(num_simulations=num_simulations, add_root_noise=False)

    mcts = AlphaBuilderMCTS(
        predict_fn=hybrid_predict,
        num_simulations=p1_cfg.num_simulations,
        batch_size=p1_cfg.batch_size,
        action_batch_size=p1_cfg.action_batch_size,
        c_puct=p1_cfg.c_puct,
        top_k_expansion=p1_cfg.top_k_expansion,
        predict_batch_fn=hybrid_predict_batch,
    )

    # Diagnostics accumulators
    bp_step0: Optional[np.ndarray] = None
    bp_conn: Optional[np.ndarray] = None
    step0_val_raw: Optional[float] = None
    phase1_steps_to_connect: Optional[int] = None
    phase2_steps_run = 0
    end_reason = "UNKNOWN"
    compliance_history: List[float] = []
    p2_value_preds: List[float] = []
    p2_fem_rewards: List[float] = []
    refutations_p1: List[bool] = []
    refutations_p2: List[bool] = []
    top8_p1: List[float] = []
    top8_p2: List[float] = []
    depths_all: List[int] = []
    step_secs_p1: List[float] = []
    step_secs_p2: List[float] = []
    p2_add_actions = 0
    p2_rem_actions = 0
    last_recorded_step = -1
    last_valid_compliance: Optional[float] = None
    last_final_score = 0.0
    is_phys_uvit = getattr(model, "architecture", "") == "phys_uvit"
    p2_log_c_preds: List[float] = []
    p2_log_c_true: List[float] = []
    p1_conn_probs: List[float] = []
    p2_conn_probs: List[float] = []

    def _compute_mcts_diagnostics(result: SearchResult, pa: np.ndarray, pr: np.ndarray) -> Tuple[float, bool, int]:
        vdist = result.visit_distribution or {}
        total_v = sum(vdist.values())
        conc = 0.0
        refut = False
        if total_v > 0:
            sorted_v = sorted(vdist.values(), reverse=True)
            conc = float(sum(sorted_v[:8]) / total_v)
            most_visited = max(vdist.items(), key=lambda it: it[1])[0]
            all_pol = np.concatenate([pa.flatten(), pr.flatten()])
            best_idx = int(np.argmax(all_pol))
            sp = pa.size
            if best_idx < sp:
                best_pol = (0,) + np.unravel_index(best_idx, pa.shape)
            else:
                best_pol = (1,) + np.unravel_index(best_idx - sp, pr.shape)
            refut = bool(most_visited != best_pol)
        max_d = 0
        if result.root is not None and hasattr(result.root, "tree_stats"):
            _, max_d = result.root.tree_stats()
        return conc, refut, int(max_d)

    while True:
        step_t0 = time.perf_counter()
        cur_phase = state.phase
        cfg = p1_cfg if cur_phase == Phase.GROWTH else p2_cfg
        mcts.config = cfg

        phi_root[0], _ = calculate_pbrs_potential(state.density, bc_tuple, force_tuple, load_cfg)
        state_tensor = build_state_tensor(state.density, bc_tuple, force_tuple)

        step_log_c: Optional[float] = None
        if is_phys_uvit:
            raw_net_val, policy_add, policy_remove, step_aux = model.predict_with_aux(state_tensor)
            step_bp = step_aux.get("blueprint_probs")
            step_log_c = step_aux.get("log_compliance")  # type: ignore[assignment]
            step_conn_prob = step_aux.get("conn_prob")
            if state.current_step == 0:
                bp_step0 = step_bp  # type: ignore[assignment]
                step0_val_raw = float(raw_net_val)
            elif cur_phase == Phase.REFINEMENT and bp_conn is None:
                bp_conn = step_bp  # type: ignore[assignment]
            if step_conn_prob is not None:
                if cur_phase == Phase.GROWTH:
                    p1_conn_probs.append(float(step_conn_prob))  # type: ignore[arg-type]
                else:
                    p2_conn_probs.append(float(step_conn_prob))  # type: ignore[arg-type]
        elif state.current_step == 0:
            raw_net_val, policy_add, policy_remove, bp_step0 = model.predict_with_blueprint(state_tensor)
            step0_val_raw = float(raw_net_val)
        elif cur_phase == Phase.REFINEMENT and bp_conn is None:
            raw_net_val, policy_add, policy_remove, bp_conn = model.predict_with_blueprint(state_tensor)
        else:
            raw_net_val, policy_add, policy_remove = model.predict(state_tensor)

        result = mcts.search(state.density, bc_tuple, force_tuple)

        vol_frac = float(np.mean(state.density > 0.5))
        isl = analyze_structure_islands(state.density, load_cfg, bc_masks=bc_tuple)
        total_vox = int((state.density > 0.5).sum())
        isl_pen = calculate_island_penalty(isl["n_islands"], isl["loose_voxels"], total_vox)
        conn_bonus_raw, conn_frac = calculate_connectivity_reward(state.density, bc_tuple, force_tuple)
        conn_bonus = conn_bonus_raw if cur_phase == Phase.GROWTH else 0.0

        compliance: Optional[float] = None
        max_disp: Optional[float] = None
        fem_reward: Optional[float] = None
        displacement_map: Optional[np.ndarray] = None
        fem_collapsed = False

        if cur_phase == Phase.REFINEMENT:
            if not isl["is_connected"]:
                fem_reward = -1.0
                fem_collapsed = True
            else:
                fem_res = evaluate_main_island(
                    state.density,
                    load_cfg,
                    isl["main_island_mask"],
                    bc_type=bc_type,
                )
                if fem_res.valid:
                    compliance = float(fem_res.compliance)
                    max_disp = float(fem_res.max_displacement)
                    displacement_map = fem_res.displacement_map
                    fem_reward = float(
                        calculate_reward(
                            compliance=compliance,
                            vol_frac=vol_frac,
                            is_valid=True,
                            max_displacement=max_disp,
                        )
                    )
                    last_valid_compliance = compliance
                    if fem_reward == -1.0:
                        fem_collapsed = True
                    else:
                        compliance_history.append(compliance)
                        p2_value_preds.append(float(raw_net_val))
                        p2_fem_rewards.append(float(fem_reward))
                        if is_phys_uvit and step_log_c is not None and compliance is not None:
                            p2_log_c_preds.append(float(step_log_c))
                            p2_log_c_true.append(float(np.log10(max(compliance, 1e-6))))
                else:
                    fem_reward = -1.0
                    fem_collapsed = True

        pv_branches = select_pv_branches(result.root, max_actions=cfg.action_batch_size, max_depth=4)
        if pv_branches:
            executed_actions = [act for branch in pv_branches for act, _, _ in branch]
            executed_action_stats = [entry for branch in pv_branches for entry in branch]
        elif result.actions:
            executed_actions = list(result.actions)
            executed_action_stats = None
        else:
            executed_actions = []
            executed_action_stats = None

        conc, refut, max_d = _compute_mcts_diagnostics(result, policy_add, policy_remove)
        depths_all.append(max_d)
        if cur_phase == Phase.GROWTH:
            refutations_p1.append(refut)
            top8_p1.append(conc)
        else:
            refutations_p2.append(refut)
            top8_p2.append(conc)

        reward_components = {
            "base_reward": 0.0,
            "connectivity_bonus": float(conn_bonus),
            "connected_load_fraction": float(conn_frac),
            "fem_reward": float(fem_reward) if fem_reward is not None else 0.0,
            "island_penalty": float(isl_pen),
            "loose_penalty": calculate_island_penalty(1, isl["loose_voxels"], total_vox, penalty_per_island=0.0)
            if isl["loose_voxels"] > 0
            else 0.0,
            "volume_penalty": 0.0,
            "validity_penalty": -1.0 if fem_collapsed else 0.0,
            "n_islands": int(isl["n_islands"]),
            "loose_voxels": int(isl["loose_voxels"]),
            "disconnected_volume_fraction": isl["loose_voxels"] / total_vox if total_vox > 0 else 0.0,
            "total": (float(fem_reward) if fem_reward is not None else float(conn_bonus)) - float(isl_pen),
        }

        record_step(
            db_path,
            state,
            result,
            float(raw_net_val),
            policy_add,
            policy_remove,
            island_analysis=isl,
            compliance_fem=compliance,
            max_displacement=max_disp,
            island_penalty=isl_pen,
            volume_fraction=vol_frac,
            reward_components=reward_components,
            executed_actions=executed_actions,
            connected_load_fraction=conn_frac,
            value_target=fem_reward if (cur_phase == Phase.REFINEMENT and fem_reward is not None) else None,
            fem_reward=fem_reward,
            displacement_map=displacement_map,
            executed_action_stats=executed_action_stats,
        )
        _update_mcts_q_blobs(db_path, game_id, state.current_step, result.root, RESOLUTION)
        last_recorded_step = state.current_step

        if cur_phase == Phase.REFINEMENT and fem_reward is not None:
            last_final_score = float(np.clip(0.5 * fem_reward + 0.5 * raw_net_val - isl_pen, -1.0, 1.0))
        else:
            last_final_score = float(np.clip(raw_net_val, -1.0, 1.0))

        step_elapsed = time.perf_counter() - step_t0
        if cur_phase == Phase.GROWTH:
            step_secs_p1.append(step_elapsed)
        else:
            step_secs_p2.append(step_elapsed)

        if state.current_step % 10 == 0 or cur_phase == Phase.REFINEMENT and phase2_steps_run in (0, max_phase2_steps - 1):
            c_str = f"{compliance:.1f}" if compliance is not None else "---"
            print(
                f"  [{scenario['id']}] step={state.current_step:3d} {cur_phase.value:10s} | "
                f"V={vol_frac*100:5.2f}% conn={str(isl['is_connected']):5s} isl={isl['n_islands']} | "
                f"C={c_str} v_net={raw_net_val:+.3f} fem_r={fem_reward if fem_reward is not None else '---'} | "
                f"{step_elapsed:.1f}s",
                flush=True,
            )

        if fem_collapsed:
            end_reason = "COLLAPSE"
            last_final_score = -1.0
            break

        # Apply actions
        if cur_phase == Phase.REFINEMENT:
            phase2_steps_run += 1
            p2_add_actions += sum(1 for c, _, _, _ in executed_actions if c == 0)
            p2_rem_actions += sum(1 for c, _, _, _ in executed_actions if c == 1)

        for channel, x, y, z in executed_actions:
            state.density[x, y, z] = 1.0 if channel == 0 else 0.0

        # Convergence / Phase-2 budget checks (before or after transition)
        if cur_phase == Phase.REFINEMENT:
            term_r = get_phase2_terminal_reward(state.density, load_cfg)
            if term_r is not None:
                state.current_step += 1
                end_reason = "TERMINAL"
                last_final_score = float(term_r)
                break
            if len(compliance_history) >= 15:
                w = compliance_history[-15:]
                if max(w) > 1e-9 and (max(w) - min(w)) / max(w) < 1e-3:
                    state.current_step += 1
                    end_reason = "CONVERGENCE"
                    break
            if phase2_steps_run >= max_phase2_steps:
                state.current_step += 1
                end_reason = "MAX_PHASE2_STEPS"
                break

        # Update tree cleanly
        if len(pv_branches) == 1 and pv_branches[0]:
            for act, _, _ in pv_branches[0]:
                mcts.step(act)
            if cur_phase == Phase.GROWTH:
                mcts.eval_cache = {}
        elif not pv_branches and len(executed_actions) == 1:
            mcts.step(executed_actions[0])
            if cur_phase == Phase.GROWTH:
                mcts.eval_cache = {}
        else:
            mcts.reset()

        if cur_phase == Phase.GROWTH:
            post_isl = analyze_structure_islands(state.density, load_cfg, bc_masks=bc_tuple)
            if post_isl["is_connected"]:
                phase1_steps_to_connect = state.current_step + 1
                state.phase = Phase.REFINEMENT
                state.is_connected = True
                mcts.reset()
                print(
                    f"  >>> [{scenario['id']}] Connected at post-step {phase1_steps_to_connect} "
                    f"(V={state.density.mean()*100:.2f}%) -> REFINEMENT",
                    flush=True,
                )
            elif state.current_step + 1 >= max_phase1_steps:
                state.current_step += 1
                end_reason = "PHASE1_TIMEOUT"
                last_final_score = -1.0
                break

        state.current_step += 1

    # Record final post-action terminal snapshot if actions mutated density after `last_recorded_step`
    state_tensor_final = build_state_tensor(state.density, bc_tuple, force_tuple)
    final_log_c_pred: Optional[float] = None
    final_conn_prob: Optional[float] = None
    if is_phys_uvit:
        raw_val_final, pa_final, pr_final, final_aux = model.predict_with_aux(state_tensor_final)
        bp_final = final_aux["blueprint_probs"]  # type: ignore[assignment]
        final_log_c_pred = float(final_aux["log_compliance"])  # type: ignore[arg-type]
        final_conn_prob = float(final_aux["conn_prob"])  # type: ignore[arg-type]
    else:
        raw_val_final, pa_final, pr_final, bp_final = model.predict_with_blueprint(state_tensor_final)
    isl_final = analyze_structure_islands(state.density, load_cfg, bc_masks=bc_tuple)
    vol_final = float(np.mean(state.density > 0.5))
    vox_final = int((state.density > 0.5).sum())
    isl_pen_final = calculate_island_penalty(isl_final["n_islands"], isl_final["loose_voxels"], vox_final)
    _, conn_frac_final = calculate_connectivity_reward(state.density, bc_tuple, force_tuple)

    comp_final: Optional[float] = None
    max_disp_final: Optional[float] = None
    fem_reward_final: Optional[float] = None
    disp_map_final: Optional[np.ndarray] = None
    if isl_final["is_connected"]:
        fem_res_f = evaluate_main_island(state.density, load_cfg, isl_final["main_island_mask"], bc_type=bc_type)
        if fem_res_f.valid:
            comp_final = float(fem_res_f.compliance)
            max_disp_final = float(fem_res_f.max_displacement)
            disp_map_final = fem_res_f.displacement_map
            fem_reward_final = float(
                calculate_reward(comp_final, vol_final, True, max_disp_final)
            )
            last_valid_compliance = comp_final
            last_final_score = float(np.clip(fem_reward_final - isl_pen_final, -1.0, 1.0))

    if state.current_step != last_recorded_step:
        dummy_res = SearchResult(
            actions=[],
            visit_distribution={},
            root_value=float(raw_val_final),
            num_simulations=0,
            root=None,
        )
        rc_final = {
            "base_reward": 0.0,
            "connectivity_bonus": 0.0,
            "connected_load_fraction": float(conn_frac_final),
            "fem_reward": float(fem_reward_final) if fem_reward_final is not None else -1.0,
            "island_penalty": float(isl_pen_final),
            "loose_penalty": 0.0,
            "volume_penalty": 0.0,
            "validity_penalty": 0.0 if isl_final["is_connected"] else -1.0,
            "n_islands": int(isl_final["n_islands"]),
            "loose_voxels": int(isl_final["loose_voxels"]),
            "disconnected_volume_fraction": isl_final["loose_voxels"] / vox_final if vox_final > 0 else 0.0,
            "total": float(last_final_score),
        }
        record_step(
            db_path,
            state,
            dummy_res,
            float(raw_val_final),
            pa_final,
            pr_final,
            island_analysis=isl_final,
            compliance_fem=comp_final,
            max_displacement=max_disp_final,
            island_penalty=isl_pen_final,
            volume_fraction=vol_final,
            reward_components=rc_final,
            executed_actions=[],
            connected_load_fraction=conn_frac_final,
            value_target=fem_reward_final if fem_reward_final is not None else -1.0,
            fem_reward=fem_reward_final,
            displacement_map=disp_map_final,
            executed_action_stats=None,
        )

    total_recorded_steps = state.current_step + 1
    update_game_final(
        db_path=db_path,
        game_id=game_id,
        final_score=float(last_final_score),
        final_compliance=last_valid_compliance,
        final_volume=vol_final,
        total_steps=total_recorded_steps,
    )
    backpropagate_value_targets(db_path, game_id)

    # Blueprint & SIMP comparison metrics
    bp_step0_dice = bp_step0_iou = None
    bp_conn_dice = bp_conn_iou = None
    bp_final_dice = bp_final_iou = None
    mcts_final_dice = mcts_final_iou = None
    if simp_binary_mask is not None:
        if bp_step0 is not None:
            bp_step0_dice, bp_step0_iou = _dice_and_iou(bp_step0 > 0.5, simp_binary_mask)
        if bp_conn is not None:
            bp_conn_dice, bp_conn_iou = _dice_and_iou(bp_conn > 0.5, simp_binary_mask)
        bp_final_dice, bp_final_iou = _dice_and_iou(bp_final > 0.5, simp_binary_mask)
        mcts_final_dice, mcts_final_iou = _dice_and_iou(state.density > 0.5, simp_binary_mask)

    geom = compute_geometry_metrics(state.density, load_cfg, bc_tuple)
    comp_start = compliance_history[0] if compliance_history else None
    comp_best = min(compliance_history + ([comp_final] if comp_final is not None else [])) if (compliance_history or comp_final is not None) else None
    comp_reduction_pct = (
        float((comp_start - comp_final) / comp_start * 100.0)
        if (comp_start and comp_final and comp_start > 0)
        else None
    )

    val_mae = val_bias = val_corr = None
    if p2_value_preds and p2_fem_rewards:
        vp = np.asarray(p2_value_preds, dtype=np.float64)
        vr = np.asarray(p2_fem_rewards, dtype=np.float64)
        val_mae = float(np.mean(np.abs(vp - vr)))
        val_bias = float(np.mean(vp - vr))
        if vp.size >= 3 and np.std(vp) > 1e-9 and np.std(vr) > 1e-9:
            val_corr = float(np.corrcoef(vp, vr)[0, 1])

    physics_aux_diagnostics: Optional[Dict[str, Any]] = None
    if is_phys_uvit:
        log_c_mae = log_c_bias = log_c_corr = None
        if p2_log_c_preds and p2_log_c_true:
            lp = np.asarray(p2_log_c_preds, dtype=np.float64)
            lt = np.asarray(p2_log_c_true, dtype=np.float64)
            log_c_mae = float(np.mean(np.abs(lp - lt)))
            log_c_bias = float(np.mean(lp - lt))
            if lp.size >= 3 and np.std(lp) > 1e-9 and np.std(lt) > 1e-9:
                log_c_corr = float(np.corrcoef(lp, lt)[0, 1])
        physics_aux_diagnostics = {
            "n_log_c_samples": len(p2_log_c_preds),
            "log_c_mae": log_c_mae,
            "log_c_bias_pred_minus_true": log_c_bias,
            "log_c_pearson_r": log_c_corr,
            "final_log_c_pred": final_log_c_pred,
            "final_log_c_true": float(np.log10(max(comp_final, 1e-6))) if comp_final is not None else None,
            "p1_conn_prob_mean": float(np.mean(p1_conn_probs)) if p1_conn_probs else None,
            "p2_conn_prob_mean": float(np.mean(p2_conn_probs)) if p2_conn_probs else None,
            "final_conn_prob": final_conn_prob,
        }

    simp_comp_bin = simp_summary.get("comp_bin") if simp_summary else None
    simp_comp_cont = simp_summary.get("comp_cont") if simp_summary else None
    comp_ratio_vs_simp_bin = (
        float(comp_final / simp_comp_bin) if (comp_final is not None and simp_comp_bin) else None
    )
    comp_ratio_vs_simp_cont = (
        float(comp_final / simp_comp_cont) if (comp_final is not None and simp_comp_cont) else None
    )

    ep_elapsed = time.perf_counter() - ep_start
    print(
        f"[{prefix.upper()}] {scenario['id']} DONE in {ep_elapsed:.1f}s ({end_reason}) | "
        f"P1={phase1_steps_to_connect} P2={phase2_steps_run} | V={vol_final*100:.2f}% | "
        f"C_start={f'{comp_start:.1f}' if comp_start else '---'} -> C_final={f'{comp_final:.1f}' if comp_final else '---'} | "
        f"Score={last_final_score:+.3f} | hollow={geom['hollow_ratio_xy']:.2f} | "
        f"BP_Dice={f'{bp_step0_dice:.3f}' if bp_step0_dice is not None else '---'} MCTS_Dice={f'{mcts_final_dice:.3f}' if mcts_final_dice is not None else '---'}",
        flush=True,
    )

    return {
        "scenario_id": scenario["id"],
        "title": scenario["title"],
        "bc_type": bc_type,
        "load_config": load_cfg,
        "game_id": game_id,
        "checkpoint": str(model.checkpoint_path),
        "checkpoint_version": checkpoint_version,
        "neural_engine": engine_name,
        "elapsed_sec": round(ep_elapsed, 2),
        "end_reason": end_reason,
        "phase1_steps_to_connect": phase1_steps_to_connect,
        "phase2_steps_run": phase2_steps_run,
        "total_recorded_steps": total_recorded_steps,
        "connected_final": bool(isl_final["is_connected"]),
        "step0_raw_value": step0_val_raw,
        "final_raw_value": float(raw_val_final),
        "final_volume": vol_final,
        "final_compliance": comp_final,
        "best_compliance": comp_best,
        "phase2_start_compliance": comp_start,
        "compliance_reduction_pct": comp_reduction_pct,
        "final_max_displacement": max_disp_final,
        "final_fem_reward": fem_reward_final,
        "final_score": float(last_final_score),
        "comp_ratio_vs_simp_bin": comp_ratio_vs_simp_bin,
        "comp_ratio_vs_simp_cont": comp_ratio_vs_simp_cont,
        "blueprint_step0_dice": bp_step0_dice,
        "blueprint_step0_iou": bp_step0_iou,
        "blueprint_conn_dice": bp_conn_dice,
        "blueprint_conn_iou": bp_conn_iou,
        "blueprint_final_dice": bp_final_dice,
        "blueprint_final_iou": bp_final_iou,
        "mcts_final_dice": mcts_final_dice,
        "mcts_final_iou": mcts_final_iou,
        "geometry": geom,
        "mcts_diagnostics": {
            "num_simulations": num_simulations,
            "refutation_rate_p1": float(np.mean(refutations_p1)) if refutations_p1 else None,
            "refutation_rate_p2": float(np.mean(refutations_p2)) if refutations_p2 else None,
            "mean_top8_conc_p1": float(np.mean(top8_p1)) if top8_p1 else None,
            "mean_top8_conc_p2": float(np.mean(top8_p2)) if top8_p2 else None,
            "mean_max_depth": float(np.mean(depths_all)) if depths_all else None,
            "mean_step_sec_p1": float(np.mean(step_secs_p1)) if step_secs_p1 else None,
            "mean_step_sec_p2": float(np.mean(step_secs_p2)) if step_secs_p2 else None,
            "p2_add_actions_total": p2_add_actions,
            "p2_rem_actions_total": p2_rem_actions,
        },
        "value_calibration_p2": {
            "n_samples": len(p2_value_preds),
            "mae": val_mae,
            "bias_net_minus_fem": val_bias,
            "pearson_r": val_corr,
        },
        "physics_aux_diagnostics": physics_aux_diagnostics,
    }


def _finalize_sqlite(db_path: Path) -> Tuple[str, int]:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        integrity = conn.execute("PRAGMA integrity_check;").fetchone()[0]
        count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        return str(integrity), int(count)
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AlphaBuilder v3.3 inference + MCTS across 6 scenarios.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to neural checkpoint (.pt)")
    parser.add_argument("--prefix", type=str, required=True, help="Game ID prefix (e.g., ep2, ep30)")
    parser.add_argument("--checkpoint-version", type=str, required=True, help="Label stored in games.checkpoint_version")
    parser.add_argument("--db-path", type=Path, default=Path("data/eval_inference_v3_3.db"))
    parser.add_argument("--json-out", type=Path, default=Path("data/analysis/eval_ep2_results.json"))
    parser.add_argument("--device", type=str, default="xpu")
    parser.add_argument("--num-simulations", type=int, default=160)
    parser.add_argument("--max-phase1-steps", type=int, default=60)
    parser.add_argument("--max-phase2-steps", type=int, default=30)
    parser.add_argument("--run-simp", dest="run_simp", action="store_true", help="Run FEniTop SIMP baselines if absent")
    parser.add_argument("--skip-simp", dest="run_simp", action="store_false", help="Reuse SIMP baselines from DB if present")
    parser.set_defaults(run_simp=False)
    parser.add_argument("--scenarios", type=str, default="", help="Optional comma-separated subset of scenario IDs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initialize_selfplay_db(args.db_path)

    selected = SCENARIOS
    if args.scenarios.strip():
        wanted = {s.strip() for s in args.scenarios.split(",") if s.strip()}
        selected = [s for s in SCENARIOS if s["id"] in wanted]
        if not selected:
            raise ValueError(f"No matching scenarios for filter: {wanted}")

    simp_summaries: Dict[str, Dict[str, Any]] = {}
    simp_masks: Dict[str, np.ndarray] = {}

    for sc in selected:
        existing = _load_existing_simp_summary(args.db_path, sc)
        if existing is not None:
            rho, summ = existing
            simp_masks[sc["id"]] = (rho > 0.5)
            simp_summaries[sc["id"]] = summ
            print(f"[SIMP] Reusing existing baseline for {sc['id']} (C_bin={summ['comp_bin']})", flush=True)
        elif args.run_simp:
            rho, summ = run_simp_baseline(args.db_path, sc, max_iter=60, target_vol_frac=0.15)
            simp_masks[sc["id"]] = (rho > 0.5)
            simp_summaries[sc["id"]] = summ
            gc.collect()

    print(f"\nLoading checkpoint {args.checkpoint} on device={args.device} ...", flush=True)
    model = AlphaBuilderInference(args.checkpoint, device=args.device)
    print(
        f"Loaded model: arch={getattr(model, 'architecture', 'swin-unetr' if model.use_swin else 'simple')} | "
        f"device={model.device_str} | epoch={model.epoch} | val_loss={model.val_loss}",
        flush=True,
    )

    neural_results: List[Dict[str, Any]] = []
    for sc in selected:
        res = run_neural_mcts_episode(
            model=model,
            db_path=args.db_path,
            scenario=sc,
            prefix=args.prefix,
            checkpoint_version=args.checkpoint_version,
            num_simulations=args.num_simulations,
            max_phase1_steps=args.max_phase1_steps,
            max_phase2_steps=args.max_phase2_steps,
            simp_binary_mask=simp_masks.get(sc["id"]),
            simp_summary=simp_summaries.get(sc["id"]),
        )
        neural_results.append(res)
        gc.collect()

    integrity, game_count = _finalize_sqlite(args.db_path)
    print(
        f"\nDB finalized: {args.db_path} | integrity_check={integrity} | total_games={game_count}",
        flush=True,
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": model.epoch,
        "checkpoint_val_loss": float(model.val_loss) if np.isfinite(model.val_loss) else None,
        "prefix": args.prefix,
        "checkpoint_version": args.checkpoint_version,
        "device": model.device_str,
        "num_simulations": args.num_simulations,
        "max_phase1_steps": args.max_phase1_steps,
        "max_phase2_steps": args.max_phase2_steps,
        "db_path": str(args.db_path),
        "db_integrity": integrity,
        "db_total_games": game_count,
        "simp_baselines": simp_summaries,
        "episodes": neural_results,
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Summary JSON saved to {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
