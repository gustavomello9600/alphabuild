"""Remote dataset and self-play endpoints (read-only SQLite, v3.3 blobs)."""

from __future__ import annotations

import io
import json
import pickle
import sqlite3
import time
import zlib
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import numpy as np
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, Response

from .. import wire
from ..schemas import ApiError
from ..worker import load_region_bounds

router = APIRouter(tags=["datasets"])


def _data_dir(request: Request) -> Path:
    return request.app.state.ctx.config.data_dir


def _connect(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise ApiError(404, "NOT_FOUND", f"Database not found: {path.name}")
    uri = path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA query_only = ON")
    return conn


def _resolve_db(request: Request, db_id: str) -> Path:
    path = request.app.state.ctx.config.safe_path(_data_dir(request), db_id)
    if not path.is_file():
        raise ApiError(404, "NOT_FOUND", f"Database not found: {db_id}")
    return path


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _kind(conn: sqlite3.Connection) -> str:
    tables = _tables(conn)
    if {"games", "game_steps"} <= tables:
        return "selfplay"
    if {"episodes", "records"} <= tables:
        try:
            count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        except sqlite3.Error:
            count = 0
        if count > 0:
            columns = _columns(conn, "episodes")
            return "harvest_v33" if "v_constructed_blob" in columns else "harvest_v2"
    if "training_data" in tables:
        return "legacy_v1"
    if {"episodes", "records"} <= tables:
        return "harvest_v33" if "v_constructed_blob" in _columns(conn, "episodes") else "harvest_v2"
    raise ApiError(400, "INVALID_INPUT", "unrecognized AlphaBuilder database schema")


def _decode_array(blob: bytes) -> np.ndarray:
    raw = zlib.decompress(blob)
    if raw[:6] == b"\x93NUMPY":
        return np.load(io.BytesIO(raw), allow_pickle=False)
    if raw[:1] == b"\x80":
        raise ApiError(
            422,
            "LEGACY_PICKLE_BLOB",
            "Legacy pickle blob detected; migrate the database to v3.3 first.",
        )
    raise ApiError(422, "INVALID_INPUT", "unrecognized blob format")


def _decode_sparse(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    raw = zlib.decompress(blob)
    if raw[:4] == b"COO1":
        n = int(np.frombuffer(raw, dtype="<u4", count=1, offset=4)[0])
        idx = np.frombuffer(raw, dtype="<i4", count=n, offset=8).astype(np.int32)
        vals = np.frombuffer(raw, dtype="<f4", count=n, offset=8 + 4 * n).astype(np.float32)
        return idx, vals
    if raw[:1] == b"\x80":
        raise ApiError(
            422,
            "LEGACY_PICKLE_BLOB",
            "Legacy pickle blob detected; migrate the database to v3.3 first.",
        )
    raise ApiError(422, "INVALID_INPUT", "unrecognized sparse blob format")


def _dense(blob: Optional[bytes], shape: tuple[int, ...], default: float = 0.0) -> np.ndarray:
    if not blob:
        return np.full(shape, default, dtype=np.float32)
    arr = np.asarray(_decode_array(blob), dtype=np.float32)
    return arr.reshape(shape)


def _sparse_dense(blob: Optional[bytes], shape: tuple[int, ...]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    if blob:
        idx, vals = _decode_sparse(blob)
        out.reshape(-1)[idx] = vals
    return out


def _has_legacy(conn: sqlite3.Connection, table: str, column: str) -> bool:
    if column not in _columns(conn, table):
        return False
    try:
        rows = conn.execute(
            f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 32"
        ).fetchall()
    except sqlite3.Error:
        return False
    for row in rows:
        try:
            if zlib.decompress(row[0])[:1] == b"\x80":
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _describe(path: Path) -> dict[str, Any]:
    conn = _connect(path)
    try:
        kind = _kind(conn)
        if kind == "selfplay":
            item_count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            total_steps = conn.execute("SELECT COUNT(*) FROM game_steps").fetchone()[0]
            table, column = "game_steps", "density_blob"
        elif kind in {"harvest_v33", "harvest_v2"}:
            item_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            total_steps = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            table, column = "records", "density_blob"
        else:
            item_count = conn.execute(
                "SELECT COUNT(DISTINCT episode_id) FROM training_data"
            ).fetchone()[0]
            total_steps = conn.execute("SELECT COUNT(*) FROM training_data").fetchone()[0]
            table, column = "training_data", "state_blob"
        has_legacy = _has_legacy(conn, table, column)
    finally:
        conn.close()
    return {
        "db_id": str(path),
        "name": path.name,
        "kind": kind,
        "item_count": int(item_count),
        "total_steps": int(total_steps),
        "size_mb": round(path.stat().st_size / (1024.0 * 1024.0), 2),
        "has_legacy_pickle": has_legacy,
    }


def _row_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    return row[key] if key in row.keys() else default


def _parse_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _supported_vectors(
    conn: sqlite3.Connection, table: str, row: sqlite3.Row, resolution: tuple[int, int, int]
) -> tuple[list[dict], list[dict]]:
    supports: list[dict] = []
    forces: list[dict] = []
    if "bc_masks_blob" in row.keys() and row["bc_masks_blob"]:
        masks = _dense(row["bc_masks_blob"], (3, *resolution))
        for x, y, z in np.argwhere(np.max(masks, axis=0) > 0.5):
            supports.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "z": int(z),
                    "mask_x": bool(masks[0, x, y, z] > 0.5),
                    "mask_y": bool(masks[1, x, y, z] > 0.5),
                    "mask_z": bool(masks[2, x, y, z] > 0.5),
                }
            )
    if "forces_blob" in row.keys() and row["forces_blob"]:
        f = _dense(row["forces_blob"], (3, *resolution))
        mag = np.abs(f[0]) + np.abs(f[1]) + np.abs(f[2])
        for x, y, z in np.argwhere(mag > 1e-6):
            forces.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "z": int(z),
                    "fx": float(f[0, x, y, z]),
                    "fy": float(f[1, x, y, z]),
                    "fz": float(f[2, x, y, z]),
                }
            )
    return supports, forces


def _reward_from_values(
    compliance: Optional[float],
    volume: Optional[float],
    max_disp: Optional[float],
    connected: Optional[bool],
    island_penalty: float = 0.0,
) -> dict[str, Any]:
    try:
        from alphabuilder.src.logic.selfplay.reward import (
            calculate_compliance_score,
            calculate_displacement_penalty,
            calculate_reward,
            calculate_volume_bonus,
        )

        if compliance is None or volume is None:
            return {
                "compliance_score": None,
                "volume_bonus": None,
                "fem_score": None,
                "displacement_penalty": 0.0,
                "island_penalty": float(island_penalty),
                "connectivity_bonus": 0.0,
                "total_reward": float(1.0 if connected else -1.0),
                "is_collapsed": not bool(connected),
            }
        score = calculate_reward(compliance, volume, bool(connected), max_disp)
        return {
            "compliance_score": float(calculate_compliance_score(compliance)),
            "volume_bonus": float(calculate_volume_bonus(volume)),
            "fem_score": float(
                np.clip(
                    calculate_compliance_score(compliance) + calculate_volume_bonus(volume),
                    -1.0,
                    1.0,
                )
            ),
            "displacement_penalty": float(calculate_displacement_penalty(max_disp)),
            "island_penalty": float(island_penalty),
            "connectivity_bonus": 0.0,
            "total_reward": float(np.clip(score - island_penalty, -1.0, 1.0)),
            "is_collapsed": score == -1.0,
        }
    except Exception:  # noqa: BLE001
        return {
            "compliance_score": None,
            "volume_bonus": None,
            "fem_score": None,
            "displacement_penalty": 0.0,
            "island_penalty": float(island_penalty),
            "connectivity_bonus": 0.0,
            "total_reward": float(1.0 if connected else -1.0),
            "is_collapsed": not bool(connected),
        }


@router.get("/datasets")
async def list_datasets(request: Request) -> dict:
    data_dir = _data_dir(request)
    items = []
    if data_dir.is_dir():
        for path in sorted(data_dir.rglob("*.db")):
            if path.name.endswith("-wal") or path.name.endswith("-shm"):
                continue
            try:
                descriptor = _describe(path)
            except Exception as exc:  # noqa: BLE001
                descriptor = {
                    "db_id": str(path),
                    "name": path.name,
                    "kind": "legacy_v1",
                    "item_count": 0,
                    "total_steps": 0,
                    "size_mb": round(path.stat().st_size / (1024.0 * 1024.0), 2),
                    "has_legacy_pickle": False,
                    "error": str(exc),
                }
            descriptor["db_id"] = path.relative_to(data_dir).as_posix()
            items.append(descriptor)
    return {"databases": items}


@router.get("/datasets/{db_id:path}/episodes")
async def list_episodes(
    request: Request,
    db_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    strategy: Optional[str] = None,
    bc_type: Optional[str] = None,
    connected_only: bool = False,
    sort: str = "recent",
    search: Optional[str] = None,
) -> dict:
    path = _resolve_db(request, db_id)
    conn = _connect(path)
    try:
        kind = _kind(conn)
        if kind == "selfplay":
            scalar_cols = [c for c in _columns(conn, "games") if not c.endswith("_blob")]
            rows = conn.execute(f"SELECT {', '.join(scalar_cols)} FROM games").fetchall()
        elif kind in {"harvest_v33", "harvest_v2"}:
            scalar_cols = [c for c in _columns(conn, "episodes") if not c.endswith("_blob")]
            rows = conn.execute(f"SELECT {', '.join(scalar_cols)} FROM episodes").fetchall()
        else:
            rows = conn.execute(
                "SELECT DISTINCT episode_id FROM training_data"
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            if kind == "selfplay":
                entry_id = row["game_id"]
                item = {
                    "entry_id": entry_id,
                    "kind": kind,
                    "total_steps": int(_row_get(row, "total_steps") or 0),
                    "steps_phase1": 0,
                    "steps_phase2": 0,
                    "final_reward": _row_get(row, "final_score"),
                    "final_compliance": _row_get(row, "final_compliance"),
                    "final_volume": _row_get(row, "final_volume"),
                    "bc_type": _row_get(row, "bc_type"),
                    "strategy": None,
                    "neural_engine": _row_get(row, "neural_engine"),
                    "checkpoint_version": _row_get(row, "checkpoint_version"),
                    "seed": None,
                    "is_connected": None,
                    "created_at": str(_row_get(row, "created_at", "")),
                }
                if strategy and item["strategy"] != strategy:
                    continue
                if bc_type and item["bc_type"] != bc_type:
                    continue
                if search and search.lower() not in entry_id.lower():
                    continue
            elif kind in {"harvest_v33", "harvest_v2"}:
                entry_id = row["episode_id"]
                fc = _row_get(row, "final_compliance")
                fv = _row_get(row, "final_volume")
                fdisp = _row_get(row, "final_max_displacement")
                fconn_raw = _row_get(row, "final_is_connected")
                is_conn = bool(fconn_raw) if fconn_raw is not None else None
                est_reward = None
                if fc is not None and fv is not None:
                    est_reward = _reward_from_values(
                        float(fc),
                        float(fv),
                        float(fdisp) if fdisp is not None else None,
                        True if is_conn is None else is_conn,
                    ).get("total_reward")
                item = {
                    "entry_id": entry_id,
                    "kind": kind,
                    "total_steps": 0,
                    "steps_phase1": 0,
                    "steps_phase2": 0,
                    "final_reward": est_reward,
                    "final_compliance": fc,
                    "final_volume": fv,
                    "bc_type": _row_get(row, "bc_type"),
                    "strategy": _row_get(row, "strategy"),
                    "neural_engine": None,
                    "checkpoint_version": _row_get(row, "code_version"),
                    "seed": _row_get(row, "seed"),
                    "is_connected": is_conn,
                    "created_at": str(_row_get(row, "created_at", "")),
                }
                if strategy and item["strategy"] != strategy:
                    continue
                if bc_type and item["bc_type"] != bc_type:
                    continue
                if connected_only and not item["is_connected"]:
                    continue
                if search and search.lower() not in entry_id.lower():
                    continue
            else:
                entry_id = row["episode_id"]
                item = {
                    "entry_id": entry_id,
                    "kind": kind,
                    "total_steps": 0,
                    "steps_phase1": 0,
                    "steps_phase2": 0,
                    "final_reward": None,
                    "final_compliance": None,
                    "final_volume": None,
                    "bc_type": None,
                    "strategy": None,
                    "neural_engine": None,
                    "checkpoint_version": None,
                    "seed": None,
                    "is_connected": None,
                    "created_at": "",
                }
            items.append(item)
        if sort == "seed_asc":
            items.sort(key=lambda i: (i.get("seed") is None, i.get("seed") or 0))
        elif sort == "reward_desc":
            items.sort(key=lambda i: i.get("final_reward") if i.get("final_reward") is not None else -99, reverse=True)
        elif sort == "compliance_asc":
            items.sort(key=lambda i: i.get("final_compliance") if i.get("final_compliance") is not None else 1e99)
        else:
            items.sort(key=lambda i: i.get("created_at", ""), reverse=True)
        total_count = len(items)
        page_items = items[offset : offset + limit]
        for item in page_items:
            entry_id = item["entry_id"]
            if kind == "selfplay":
                total = conn.execute(
                    "SELECT COUNT(*) FROM game_steps WHERE game_id = ?", (entry_id,)
                ).fetchone()[0]
                p1 = conn.execute(
                    "SELECT COUNT(*) FROM game_steps WHERE game_id = ? AND phase='GROWTH'",
                    (entry_id,),
                ).fetchone()[0]
                item["total_steps"] = int(total)
                item["steps_phase1"] = int(p1)
                item["steps_phase2"] = int(total - p1)
            elif kind in {"harvest_v33", "harvest_v2"}:
                total = conn.execute(
                    "SELECT COUNT(*) FROM records WHERE episode_id = ?", (entry_id,)
                ).fetchone()[0]
                p1 = conn.execute(
                    "SELECT COUNT(*) FROM records WHERE episode_id = ? AND phase='GROWTH'",
                    (entry_id,),
                ).fetchone()[0]
                last = conn.execute(
                    "SELECT fitness_score, reward_components_json "
                    "FROM records WHERE episode_id = ? ORDER BY step DESC LIMIT 1",
                    (entry_id,),
                ).fetchone()
                if last:
                    components = _parse_json(last[1], {})
                    if isinstance(components, dict) and components.get("total") is not None:
                        item["final_reward"] = components.get("total")
                    elif last[0] is not None:
                        item["final_reward"] = last[0]
                item["total_steps"] = int(total)
                item["steps_phase1"] = int(p1)
                item["steps_phase2"] = int(total - p1)
            else:
                total = conn.execute(
                    "SELECT COUNT(*) FROM training_data WHERE episode_id = ?", (entry_id,)
                ).fetchone()[0]
                item["total_steps"] = int(total)
        return {"total_count": total_count, "items": page_items}
    finally:
        conn.close()


def _episode_metadata(request: Request, db_id: str, episode_id: str) -> dict[str, Any]:
    path = _resolve_db(request, db_id)
    conn = _connect(path)
    try:
        kind = _kind(conn)
        if kind == "selfplay":
            row = conn.execute("SELECT * FROM games WHERE game_id = ?", (episode_id,)).fetchone()
            if not row:
                raise ApiError(404, "NOT_FOUND", f"Game not found: {episode_id}")
            resolution = tuple(int(v) for v in _parse_json(row["resolution"], [64, 32, 8]))
            rows = conn.execute(
                "SELECT step, phase, value, volume_fraction, compliance_fem, max_displacement, is_connected, mcts_stats_json, displacement_map_blob "
                "FROM game_steps WHERE game_id = ? ORDER BY step",
                (episode_id,),
            ).fetchall()
            value_history = [float(r[2]) for r in rows]
            volume_history = [float(r[3]) if r[3] is not None else None for r in rows]
            compliance_history = [float(r[4]) if r[4] is not None else None for r in rows]
            phase_history = [r[1] for r in rows]
            refutation_steps = []
            for r in rows:
                stats = _parse_json(r[7], {})
                if stats.get("refutation"):
                    refutation_steps.append(int(r[0]))
            supports, forces = _supported_vectors(conn, "games", row, resolution)
            return {
                "entry_id": episode_id,
                "kind": kind,
                "resolution": list(resolution),
                "bc_type": row["bc_type"],
                "strategy": None,
                "neural_engine": row["neural_engine"],
                "checkpoint_version": row["checkpoint_version"],
                "seed": None,
                "total_steps": len(rows),
                "steps_phase1": sum(1 for p in phase_history if p == "GROWTH"),
                "steps_phase2": sum(1 for p in phase_history if p == "REFINEMENT"),
                "final_score": row["final_score"],
                "final_compliance": row["final_compliance"],
                "final_volume": row["final_volume"],
                "final_max_displacement": next(
                    (r[5] for r in reversed(rows) if r[5] is not None), None
                ),
                "real_scale_factor": float(_row_get(row, "initial_cantilever_problem_real_scale_factor", 500000.0) or 500000.0),
                "stiffness_e": float(_row_get(row, "stiffness_E", 200e9) or 200e9),
                "load_config": _parse_json(row["load_config"], {}),
                "has_blueprint": False,
                "has_rho_cont": False,
                "has_displacement": any(r[8] is not None for r in rows),
                "supports": supports,
                "forces": forces,
                "phase_history": phase_history,
                "value_history": value_history,
                "volume_history": volume_history,
                "compliance_history": compliance_history,
                "refutation_steps": refutation_steps,
            }
        if kind not in {"harvest_v33", "harvest_v2"}:
            raise ApiError(400, "INVALID_INPUT", "metadata endpoint supports harvest/selfplay DBs")
        row = conn.execute("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)).fetchone()
        if not row:
            raise ApiError(404, "NOT_FOUND", f"Episode not found: {episode_id}")
        resolution = tuple(int(v) for v in _parse_json(row["resolution"], [64, 32, 8]))
        rows = conn.execute(
            "SELECT step, phase, fitness_score, vol_frac, compliance, max_displacement, is_connected, rho_cont_blob, displacement_blob "
            "FROM records WHERE episode_id = ? ORDER BY step",
            (episode_id,),
        ).fetchall()
        supports, forces = _supported_vectors(conn, "episodes", row, resolution)
        record_columns = _columns(conn, "records")
        has_rho_cont = (
            "rho_cont_blob" in record_columns
            and conn.execute(
                "SELECT COUNT(*) FROM records WHERE episode_id = ? AND rho_cont_blob IS NOT NULL",
                (episode_id,),
            ).fetchone()[0]
            > 0
        )
        has_displacement = (
            "displacement_blob" in record_columns
            and conn.execute(
                "SELECT COUNT(*) FROM records WHERE episode_id = ? AND displacement_blob IS NOT NULL",
                (episode_id,),
            ).fetchone()[0]
            > 0
        )
        return {
            "entry_id": episode_id,
            "kind": kind,
            "resolution": list(resolution),
            "bc_type": row["bc_type"],
            "strategy": _row_get(row, "strategy"),
            "neural_engine": None,
            "checkpoint_version": _row_get(row, "code_version"),
            "seed": _row_get(row, "seed"),
            "total_steps": len(rows),
            "steps_phase1": sum(1 for r in rows if r[1] == "GROWTH"),
            "steps_phase2": sum(1 for r in rows if r[1] == "REFINEMENT"),
            "final_score": None,
            "final_compliance": _row_get(row, "final_compliance"),
            "final_volume": _row_get(row, "final_volume"),
            "final_max_displacement": _row_get(row, "final_max_displacement"),
            "real_scale_factor": 500000.0,
            "stiffness_e": 200e9,
            "load_config": _parse_json(row["load_config"], {}),
            "has_blueprint": _row_get(row, "v_constructed_blob") is not None,
            "has_rho_cont": has_rho_cont,
            "has_displacement": has_displacement,
            "supports": supports,
            "forces": forces,
            "phase_history": [r[1] for r in rows],
            "value_history": [float(r[2]) for r in rows],
            "volume_history": [float(r[3]) if r[3] is not None else None for r in rows],
            "compliance_history": [float(r[4]) if r[4] is not None else None for r in rows],
            "refutation_steps": [],
        }
    finally:
        conn.close()


@router.get("/datasets/{db_id:path}/episodes/{episode_id}/metadata")
async def episode_metadata(request: Request, db_id: str, episode_id: str) -> dict:
    return _episode_metadata(request, db_id, episode_id)


def _frame_metadata(
    step: int,
    phase: str,
    value: float,
    volume: float,
    compliance: Optional[float],
    max_disp: Optional[float],
    connected: bool,
    n_islands: int,
    loose: int,
    island_penalty: float,
    connected_fraction: float,
    reward_v33: dict,
    mcts_stats: Optional[dict],
    selected_actions: list[dict],
) -> dict[str, Any]:
    return {
        "step": int(step),
        "phase": phase,
        "value": float(value),
        "value_target": None,
        "fem_reward": None,
        "compliance_fem": compliance,
        "max_displacement": max_disp,
        "volume_fraction": float(volume),
        "is_connected": bool(connected),
        "n_islands": int(n_islands),
        "loose_voxels": int(loose),
        "connected_load_fraction": float(connected_fraction),
        "island_penalty": float(island_penalty),
        "reward_v33": reward_v33,
        "mcts_stats": mcts_stats,
        "selected_actions": selected_actions,
    }


def _harvest_frame(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    resolution: tuple[int, int, int],
    blueprint: Optional[np.ndarray],
) -> bytes:
    density = _dense(row["density_blob"], resolution)
    prob_add = _sparse_dense(row["policy_add_blob"], resolution)
    prob_rem = _sparse_dense(row["policy_remove_blob"], resolution)
    rho = _dense(row["rho_cont_blob"], resolution) if row["rho_cont_blob"] else None
    disp = _dense(row["displacement_blob"], resolution) if row["displacement_blob"] else None
    components = _parse_json(row["reward_components_json"], {})
    island_penalty = float(components.get("island_penalty", 0.0) or 0.0)
    conn_frac = float(components.get("connected_load_fraction", 0.0) or 0.0)
    n_islands = int(components.get("n_islands", 1) or 1)
    loose = int(components.get("loose_voxels", 0) or 0)
    metadata = _frame_metadata(
        row["step"],
        row["phase"],
        row["fitness_score"],
        row["vol_frac"] if row["vol_frac"] is not None else float(np.mean(density > 0.5)),
        row["compliance"],
        row["max_displacement"],
        bool(row["is_connected"]),
        n_islands,
        loose,
        island_penalty,
        conn_frac,
        _reward_from_values(
            row["compliance"],
            row["vol_frac"],
            row["max_displacement"],
            bool(row["is_connected"]),
            island_penalty,
        ),
        None,
        [],
    )
    metadata["step"] = int(row["step"])
    tensors = [
        wire.dense_u8(wire.TAG_DENSITY, resolution, (density > 0.5).astype(np.uint8).reshape(-1)),
        wire.coo1_f32(
            wire.TAG_POLICY_ADD,
            resolution,
            *wire.sparse_from_dense(prob_add, 1e-7),
        ),
        wire.coo1_f32(
            wire.TAG_POLICY_REMOVE,
            resolution,
            *wire.sparse_from_dense(prob_rem, 1e-7),
        ),
    ]
    if rho is not None:
        tensors.append(wire.dense_f16(wire.TAG_RHO_CONT, resolution, rho.reshape(-1)))
    if disp is not None:
        tensors.append(wire.dense_f16(wire.TAG_DISPLACEMENT_MAG, resolution, disp.reshape(-1)))
    if blueprint is not None:
        tensors.append(wire.dense_f16(wire.TAG_BLUEPRINT_PROBS, resolution, blueprint.reshape(-1)))
    return wire.encode_packet(wire.PAYLOAD_STEP_FRAME, metadata, tensors, compress=True, flags=0x01)


@router.get("/datasets/{db_id:path}/episodes/{episode_id}/frames")
async def episode_frames(
    request: Request,
    db_id: str,
    episode_id: str,
    start: int = Query(default=0, ge=0),
    end: int = Query(default=50, ge=0),
    include_rho_cont: bool = False,
) -> Response:
    if end <= start or end - start > 100:
        raise ApiError(400, "INVALID_INPUT", "frame window must satisfy 0 < end-start <= 100")
    path = _resolve_db(request, db_id)
    conn = _connect(path)
    try:
        kind = _kind(conn)
        if kind not in {"harvest_v33", "harvest_v2"}:
            raise ApiError(400, "INVALID_INPUT", "frames endpoint supports harvest DBs")
        episode = conn.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if not episode:
            raise ApiError(404, "NOT_FOUND", f"Episode not found: {episode_id}")
        resolution = tuple(int(v) for v in _parse_json(episode["resolution"], [64, 32, 8]))
        blueprint = None
        if episode["v_constructed_blob"] if "v_constructed_blob" in episode.keys() else None:
            blueprint = _dense(episode["v_constructed_blob"], resolution)
        rows = conn.execute(
            "SELECT * FROM records WHERE episode_id = ? AND step >= ? AND step < ? ORDER BY step",
            (episode_id, start, end),
        ).fetchall()
        frames = [_harvest_frame(conn, row, resolution, blueprint) for row in rows]
    finally:
        conn.close()
    return Response(content=wire.encode_step_batch(frames), media_type="application/octet-stream")


@router.get("/selfplay/games")
async def list_games(
    request: Request,
    db_id: str = Query(default="selfplay_games.db"),
    engine: Optional[str] = None,
    version: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    search: Optional[str] = None,
) -> dict:
    path = _resolve_db(request, db_id)
    conn = _connect(path)
    try:
        if _kind(conn) != "selfplay":
            raise ApiError(400, "INVALID_INPUT", "not a self-play database")
        where = []
        params: list[Any] = []
        if engine:
            where.append("neural_engine = ?")
            params.append(engine)
        if version:
            where.append("checkpoint_version = ?")
            params.append(version)
        if search:
            where.append("game_id LIKE ?")
            params.append(f"%{search}%")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        rows = conn.execute(f"SELECT * FROM games{clause} ORDER BY created_at DESC", params).fetchall()
        total = len(rows)
        items = []
        for row in rows[offset : offset + limit]:
            items.append(
                {
                    "game_id": row["game_id"],
                    "neural_engine": row["neural_engine"],
                    "checkpoint_version": row["checkpoint_version"],
                    "final_score": row["final_score"],
                    "final_compliance": row["final_compliance"],
                    "final_volume": row["final_volume"],
                    "total_steps": int(row["total_steps"] or 0),
                    "created_at": str(row["created_at"]),
                }
            )
        return {"total_count": total, "items": items}
    finally:
        conn.close()


@router.get("/selfplay/games/{game_id}/metadata")
async def game_metadata(request: Request, game_id: str, db_id: str = Query(default="selfplay_games.db")) -> dict:
    return _episode_metadata(request, db_id, game_id)


@router.get("/selfplay/games/{game_id}/steps")
async def game_steps(
    request: Request,
    game_id: str,
    db_id: str = Query(default="selfplay_games.db"),
    start: int = Query(default=0, ge=0),
    end: int = Query(default=50, ge=0),
) -> Response:
    if end <= start or end - start > 100:
        raise ApiError(400, "INVALID_INPUT", "frame window must satisfy 0 < end-start <= 100")
    path = _resolve_db(request, db_id)
    conn = _connect(path)
    try:
        if _kind(conn) != "selfplay":
            raise ApiError(400, "INVALID_INPUT", "not a self-play database")
        game = conn.execute("SELECT * FROM games WHERE game_id = ?", (game_id,)).fetchone()
        if not game:
            raise ApiError(404, "NOT_FOUND", f"Game not found: {game_id}")
        resolution = tuple(int(v) for v in _parse_json(game["resolution"], [64, 32, 8]))
        rows = conn.execute(
            "SELECT * FROM game_steps WHERE game_id = ? AND step >= ? AND step < ? ORDER BY step",
            (game_id, start, end),
        ).fetchall()
        frames = []
        for row in rows:
            density = _dense(row["density_blob"], resolution)
            prob_add = _sparse_dense(row["policy_add_blob"], resolution)
            prob_rem = _sparse_dense(row["policy_remove_blob"], resolution)
            visit_add = _sparse_dense(row["mcts_visit_add_blob"], resolution)
            visit_rem = _sparse_dense(row["mcts_visit_remove_blob"], resolution)
            q_add = _sparse_dense(row["mcts_q_add_blob"], resolution)
            q_rem = _sparse_dense(row["mcts_q_remove_blob"], resolution)
            stats = _parse_json(row["mcts_stats_json"], {})
            actions = _parse_json(row["selected_actions_json"], [])
            components = _parse_json(row["reward_components_json"], {})
            island_penalty = float(row["island_penalty"] or 0.0)
            metadata = _frame_metadata(
                row["step"],
                row["phase"],
                row["value"],
                float(row["volume_fraction"] or float(np.mean(density > 0.5))),
                row["compliance_fem"],
                row["max_displacement"],
                bool(row["is_connected"]),
                int(row["n_islands"] or 1),
                int(row["loose_voxels"] or 0),
                island_penalty,
                float(row["connected_load_fraction"] or 0.0),
                _reward_from_values(
                    row["compliance_fem"],
                    row["volume_fraction"],
                    row["max_displacement"],
                    bool(row["is_connected"]),
                    island_penalty,
                ),
                stats,
                actions,
            )
            metadata["value_target"] = row["value_target"] if "value_target" in row.keys() else None
            metadata["fem_reward"] = row["fem_reward"] if "fem_reward" in row.keys() else None
            metadata["reward_components"] = components
            tensors = [
                wire.dense_u8(wire.TAG_DENSITY, resolution, (density > 0.5).astype(np.uint8).reshape(-1)),
                wire.coo1_f32(wire.TAG_POLICY_ADD, resolution, *wire.sparse_from_dense(prob_add, 1e-7)),
                wire.coo1_f32(wire.TAG_POLICY_REMOVE, resolution, *wire.sparse_from_dense(prob_rem, 1e-7)),
                wire.coo1_f32(wire.TAG_MCTS_VISIT_ADD, resolution, *wire.sparse_from_dense(visit_add, 0.0)),
                wire.coo1_f32(wire.TAG_MCTS_VISIT_REMOVE, resolution, *wire.sparse_from_dense(visit_rem, 0.0)),
                wire.coo1_f32(wire.TAG_MCTS_Q_ADD, resolution, *wire.sparse_from_dense(q_add, 0.0)),
                wire.coo1_f32(wire.TAG_MCTS_Q_REMOVE, resolution, *wire.sparse_from_dense(q_rem, 0.0)),
            ]
            if row["displacement_map_blob"] if "displacement_map_blob" in row.keys() else None:
                disp = _dense(row["displacement_map_blob"], resolution)
                tensors.append(wire.dense_f16(wire.TAG_DISPLACEMENT_MAG, resolution, disp.reshape(-1)))
            frames.append(wire.encode_packet(wire.PAYLOAD_STEP_FRAME, metadata, tensors, compress=True, flags=0x01))
    finally:
        conn.close()
    return Response(content=wire.encode_step_batch(frames), media_type="application/octet-stream")


def _reencode_array(blob: Optional[bytes]) -> Optional[bytes]:
    if not blob:
        return None
    raw = zlib.decompress(blob)
    if raw[:6] == b"\x93NUMPY":
        return blob
    if raw[:1] == b"\x80":
        arr = np.asarray(pickle.loads(raw))
    else:
        arr = np.asarray(np.load(io.BytesIO(raw), allow_pickle=False))
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(arr), allow_pickle=False)
    return zlib.compress(buffer.getvalue(), level=6)


def _reencode_sparse(blob: Optional[bytes]) -> Optional[bytes]:
    if not blob:
        return None
    raw = zlib.decompress(blob)
    if raw[:4] == b"COO1":
        return blob
    if raw[:1] == b"\x80":
        data = pickle.loads(raw)
    else:
        data = np.load(io.BytesIO(raw), allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
    if not isinstance(data, dict):
        raise ValueError("unrecognized legacy sparse payload")
    n = int(data.get("n", 0))
    indices = np.frombuffer(data["indices"], dtype=np.int32).copy()[:n]
    values = np.frombuffer(data["values"], dtype=np.float32).copy()[:n]
    payload = (
        b"COO1"
        + int(n).to_bytes(4, "little")
        + np.ascontiguousarray(indices, dtype="<i4").tobytes()
        + np.ascontiguousarray(values, dtype="<f4").tobytes()
    )
    return zlib.compress(payload, level=6)


def _migrate_harvest(source: Path, output: Path) -> dict:
    from alphabuilder.src.logic.storage import initialize_database

    if output.exists():
        raise ApiError(409, "ENGINE_BUSY", f"Migration output already exists: {output.name}")
    initialize_database(output, wal_mode=True)
    src = _connect(source)
    dst = sqlite3.connect(output)
    try:
        episode_columns = _columns(src, "episodes")
        rows = src.execute("SELECT * FROM episodes").fetchall()
        for row in rows:
            record = dict(row)
            if record.get("bc_masks_blob"):
                record["bc_masks_blob"] = _reencode_array(record["bc_masks_blob"])
            if record.get("forces_blob"):
                record["forces_blob"] = _reencode_array(record["forces_blob"])
            if record.get("v_constructed_blob"):
                record["v_constructed_blob"] = _reencode_array(record["v_constructed_blob"])
            keys = [key for key in record.keys() if key in episode_columns]
            placeholders = ",".join("?" for _ in keys)
            dst.execute(
                f"INSERT OR REPLACE INTO episodes ({','.join(keys)}) VALUES ({placeholders})",
                [record[key] for key in keys],
            )
        record_columns = _columns(src, "records")
        for row in src.execute("SELECT * FROM records").fetchall():
            record = dict(row)
            for column in ("density_blob", "rho_cont_blob", "displacement_blob"):
                if record.get(column):
                    record[column] = _reencode_array(record[column])
            for column in ("policy_add_blob", "policy_remove_blob"):
                if record.get(column):
                    record[column] = _reencode_sparse(record[column])
            keys = [key for key in record.keys() if key in record_columns]
            placeholders = ",".join("?" for _ in keys)
            dst.execute(
                f"INSERT OR REPLACE INTO records ({','.join(keys)}) VALUES ({placeholders})",
                [record[key] for key in keys],
            )
        dst.commit()
        episodes = dst.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        records = dst.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    finally:
        src.close()
        dst.close()
    return {"migrated": True, "output_db_id": output.name, "episodes": int(episodes), "records": int(records)}


@router.post("/datasets/{db_id:path}/migrate-v33")
async def migrate_v33(request: Request, db_id: str) -> dict:
    source = _resolve_db(request, db_id)
    out_path = source.with_name(source.stem + "_v33.db")
    conn = _connect(source)
    try:
        kind = _kind(conn)
    finally:
        conn.close()
    if kind not in {"harvest_v2", "harvest_v33"}:
        raise ApiError(
            400,
            "INVALID_INPUT",
            "migrate-v33 currently supports harvest databases with episodes/records",
        )
    return _migrate_harvest(source, out_path)
