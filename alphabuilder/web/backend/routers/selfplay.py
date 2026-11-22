from fastapi import APIRouter, HTTPException
from pathlib import Path
import sqlite3
import json
import struct
import numpy as np
import base64
from fastapi import Response
from typing import Optional, List
from pydantic import BaseModel

try:
    from ..fastapi_utils import get_selfplay_db_path, deserialize_array, deserialize_sparse, sparse_decode
except (ImportError, ValueError):
    from fastapi_utils import get_selfplay_db_path, deserialize_array, deserialize_sparse, sparse_decode

router = APIRouter()

# --- Pydantic Models ---

class SelfPlayGameSummary(BaseModel):
    game_id: str
    neural_engine: str
    checkpoint_version: str
    final_score: Optional[float]
    final_compliance: Optional[float]
    final_volume: Optional[float]
    total_steps: int
    created_at: str
    initial_cantilever_problem_real_scale_factor: float = 500000.0
    stiffness_E: float = 200e9

class SelectedActionData(BaseModel):
    channel: int
    x: int
    y: int
    z: int
    visits: int
    q_value: float

class MCTSStatsData(BaseModel):
    num_simulations: int
    nodes_expanded: int
    max_depth: int
    cache_hits: int
    top8_concentration: float
    refutation: bool

class SelfPlayStepData(BaseModel):
    step: int
    phase: str
    tensor_shape: list[int]
    tensor_data: list[float]
    policy_add: list[float]
    policy_remove: list[float]
    mcts_visit_add: list[float]
    mcts_visit_remove: list[float]
    mcts_q_add: list[float]
    mcts_q_remove: list[float]
    selected_actions: list[SelectedActionData]
    value: float
    mcts_stats: MCTSStatsData
    n_islands: int = 1
    is_connected: bool = False
    volume_fraction: float = 0.0
    # Detailed reward fields
    compliance_fem: Optional[float] = None
    max_displacement: Optional[float] = None
    island_penalty: float = 0.0
    loose_voxels: int = 0
    reward_components: Optional[dict] = None
    displacement_map: Optional[str] = None  # Base64 encoded float32 array


class SelfPlayGameData(BaseModel):
    game_id: str
    neural_engine: str
    checkpoint_version: str
    bc_type: str
    resolution: list[int]
    final_score: Optional[float]
    final_compliance: Optional[float]
    final_volume: Optional[float]
    final_volume: Optional[float]
    total_steps: int
    initial_cantilever_problem_real_scale_factor: float = 500000.0
    stiffness_E: float = 200e9
    steps: list[SelfPlayStepData]

class SelfPlayGameMetadata(BaseModel):
    """Metadata for a self-play game, for efficient loading."""
    game_id: str
    neural_engine: str
    checkpoint_version: str
    bc_type: str
    resolution: list[int]
    final_score: Optional[float]
    final_compliance: Optional[float]
    final_volume: Optional[float]
    total_steps: int
    initial_cantilever_problem_real_scale_factor: float = 500000.0
    stiffness_E: float = 200e9
    value_history: Optional[list[float]] = None

# --- Internal helpers ---

def _decode_sparse_or_zeros(blob, resolution):
    """Decode sparse blob, return zeros if None."""
    if blob is None:
        return np.zeros(resolution, dtype=np.float32)
    indices, values = deserialize_sparse(blob)
    return sparse_decode(indices, values, resolution)

def _load_game_info(db_path, game_id):
    """Load game metadata from DB."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT neural_engine, checkpoint_version, bc_masks_blob, forces_blob,
               load_config, bc_type, resolution, final_score, final_compliance,
               final_volume, total_steps
        FROM games WHERE game_id = ?
    """, (game_id,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "game_id": game_id,
        "neural_engine": row[0],
        "checkpoint_version": row[1],
        "bc_masks": deserialize_array(row[2]),
        "forces": deserialize_array(row[3]),
        "load_config": json.loads(row[4]),
        "bc_type": row[5],
        "resolution": tuple(json.loads(row[6])),
        "final_score": row[7],
        "final_compliance": row[8],
        "final_volume": row[9],
        "total_steps": row[10] or 0,
    }

# --- Endpoints ---

@router.get("/games", response_model=list[SelfPlayGameSummary])
def list_selfplay_games(
    engine: Optional[str] = None,
    version: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    deprecated: bool = False
):
    """List self-play games with optional filtering."""
    db_path = get_selfplay_db_path(deprecated)
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = """
        SELECT game_id, neural_engine, checkpoint_version,
               final_score, final_compliance, final_volume,
               total_steps, created_at
        FROM games WHERE 1=1
    """
    params = []
    if engine:
        query += " AND neural_engine = ?"
        params.append(engine)
    if version:
        query += " AND checkpoint_version = ?"
        params.append(version)
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [
        SelfPlayGameSummary(
            game_id=row[0],
            neural_engine=row[1],
            checkpoint_version=row[2],
            final_score=row[3],
            final_compliance=row[4],
            final_volume=row[5],
            total_steps=row[6] or 0,
            created_at=row[7] or "",
        )
        for row in rows
    ]

@router.get("/games/{game_id}/metadata", response_model=SelfPlayGameMetadata)
def get_selfplay_game_metadata(game_id: str, deprecated: bool = False):
    """Get game metadata without steps - fast initial load."""
    db_path = get_selfplay_db_path(deprecated)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Self-play database not found")

    game = _load_game_info(db_path, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM game_steps WHERE game_id = ? ORDER BY step ASC", (game_id,))
    value_history = [row[0] for row in cursor.fetchall()]
    conn.close()

    return SelfPlayGameMetadata(
        game_id=game["game_id"],
        neural_engine=game["neural_engine"],
        checkpoint_version=game["checkpoint_version"],
        bc_type=game["bc_type"],
        resolution=list(game["resolution"]),
        final_score=game["final_score"],
        final_compliance=game["final_compliance"],
        final_volume=game["final_volume"],
        total_steps=game["total_steps"],
        value_history=value_history,
    )

@router.get("/games/{game_id}/steps", response_model=list[SelfPlayStepData])
def get_selfplay_game_steps(
    game_id: str,
    start: int = 0,
    end: int = 50,
    deprecated: bool = False
):
    """Get range of steps for a self-play game."""
    db_path = get_selfplay_db_path(deprecated)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Self-play database not found")

    game = _load_game_info(db_path, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    resolution = game["resolution"]
    bc_masks = game["bc_masks"]
    forces = game["forces"]

    limit = end - start
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT phase, density_blob, policy_add_blob, policy_remove_blob,
               mcts_visit_add_blob, mcts_visit_remove_blob,
               mcts_q_add_blob, mcts_q_remove_blob,
               selected_actions_json, value, mcts_stats_json, step,
               n_islands, loose_voxels, is_connected, volume_fraction,
               reward_components_json, compliance_fem, island_penalty,
               connected_load_fraction, max_displacement
        FROM game_steps
        WHERE game_id = ?
        ORDER BY step
        LIMIT ? OFFSET ?
    """, (game_id, limit, start))
    rows = cursor.fetchall()
    conn.close()

    step_data = []
    for row in rows:
        density = deserialize_array(row[1])
        policy_add = _decode_sparse_or_zeros(row[2], resolution)
        policy_remove = _decode_sparse_or_zeros(row[3], resolution)
        mcts_visit_add = _decode_sparse_or_zeros(row[4], resolution)
        mcts_visit_remove = _decode_sparse_or_zeros(row[5], resolution)
        mcts_q_add = _decode_sparse_or_zeros(row[6], resolution)
        mcts_q_remove = _decode_sparse_or_zeros(row[7], resolution)

        actions_data = json.loads(row[8]) if row[8] else []
        selected_actions = [
            SelectedActionData(
                channel=a["channel"],
                x=a["x"], y=a["y"], z=a["z"],
                visits=a["visits"],
                q_value=a["q_value"],
            )
            for a in actions_data
        ]

        stats_data = json.loads(row[10]) if row[10] else {}
        mcts_stats = MCTSStatsData(
            num_simulations=stats_data.get("num_simulations", 0),
            nodes_expanded=stats_data.get("nodes_expanded", 0),
            max_depth=stats_data.get("max_depth", 0),
            cache_hits=stats_data.get("cache_hits", 0),
            top8_concentration=stats_data.get("top8_concentration", 0.0),
            refutation=stats_data.get("refutation", False),
        )

        reward_components = json.loads(row[16]) if row[16] else {}

        state_tensor = np.concatenate([
            density[None],
            bc_masks,
            forces,
        ], axis=0)

        step_data.append(SelfPlayStepData(
            step=row[11],
            phase=row[0],
            tensor_shape=list(state_tensor.shape),
            tensor_data=state_tensor.flatten().tolist(),
            policy_add=policy_add.flatten().tolist(),
            policy_remove=policy_remove.flatten().tolist(),
            mcts_visit_add=mcts_visit_add.flatten().tolist(),
            mcts_visit_remove=mcts_visit_remove.flatten().tolist(),
            mcts_q_add=mcts_q_add.flatten().tolist(),
            mcts_q_remove=mcts_q_remove.flatten().tolist(),
            selected_actions=selected_actions,
            value=row[9],
            mcts_stats=mcts_stats,
            n_islands=row[12] if row[12] is not None else 1,
            loose_voxels=row[13] if row[13] is not None else 0,
            is_connected=bool(row[14]) if row[14] is not None else False,
            volume_fraction=row[15] if row[15] is not None else 0.0,
            compliance_fem=row[17],
            max_displacement=row[20],
            island_penalty=row[18] if row[18] is not None else 0.0,
            reward_components=reward_components if reward_components else None,
            displacement_map=None,
        ))

    return step_data

@router.get("/games/{game_id}/steps/binary")
def get_selfplay_game_steps_binary(
    game_id: str,
    start: int = 0,
    end: int = 50,
    deprecated: bool = False
):
    """Get range of steps in binary format for efficient transfer."""
    db_path = get_selfplay_db_path(deprecated)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Self-play database not found")

    game = _load_game_info(db_path, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    resolution = game["resolution"]
    bc_masks = game["bc_masks"]
    forces = game["forces"]
    D, H, W = resolution

    limit = end - start
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT phase, density_blob, policy_add_blob, policy_remove_blob,
               mcts_visit_add_blob, mcts_visit_remove_blob,
               mcts_q_add_blob, mcts_q_remove_blob,
               selected_actions_json, value, mcts_stats_json, step,
               n_islands, loose_voxels, is_connected, volume_fraction,
               reward_components_json, compliance_fem, island_penalty,
               connected_load_fraction, max_displacement
        FROM game_steps
        WHERE game_id = ?
        ORDER BY step
        LIMIT ? OFFSET ?
    """, (game_id, limit, start))
    rows = cursor.fetchall()
    conn.close()

    output_buffer = bytearray()

    for row in rows:
        density = deserialize_array(row[1])
        policy_add = _decode_sparse_or_zeros(row[2], resolution)
        policy_remove = _decode_sparse_or_zeros(row[3], resolution)
        mcts_visit_add = _decode_sparse_or_zeros(row[4], resolution)
        mcts_visit_remove = _decode_sparse_or_zeros(row[5], resolution)

        state_tensor = np.concatenate([density[None], bc_masks, forces], axis=0).astype(np.float32)

        tensor_bytes = state_tensor.tobytes()
        policy_add_bytes = policy_add.astype(np.float32).tobytes()
        policy_rem_bytes = policy_remove.astype(np.float32).tobytes()
        mcts_add_bytes = mcts_visit_add.astype(np.float32).tobytes()
        mcts_rem_bytes = mcts_visit_remove.astype(np.float32).tobytes()

        phase_code = 0 if row[0] == "GROWTH" else 1

        actions_data = json.loads(row[8]) if row[8] else []
        selected_actions_bytes = struct.pack("<i", len(actions_data))
        for a in actions_data:
            selected_actions_bytes += struct.pack(
                "<5if", a["channel"], a["x"], a["y"], a["z"], a["visits"], a["q_value"]
            )

        stats_data = json.loads(row[10]) if row[10] else {}

        extension_bytes = struct.pack(
            "<3i3f",
            row[12] if row[12] is not None else 1,
            row[13] if row[13] is not None else 0,
            1 if (row[14] is not None and row[14]) else 0,
            float(row[17] or 0.0),
            float(row[18] or 0.0),
            float(row[20] or 0.0),
        )

        rc_json = json.dumps(json.loads(row[16]) if row[16] else {}).encode("utf-8")
        rc_header = struct.pack("<i", len(rc_json))

        disp_header = struct.pack("<i", 0)

        total_data_size = (
            len(tensor_bytes) + len(policy_add_bytes) + len(policy_rem_bytes) +
            len(mcts_add_bytes) + len(mcts_rem_bytes) +
            len(selected_actions_bytes) + len(extension_bytes) +
            len(rc_header) + len(rc_json) + len(disp_header)
        )

        header = struct.pack(
            "<10i",
            row[11],
            phase_code,
            D, H, W,
            total_data_size,
            stats_data.get("num_simulations", 0),
            int(row[9] * 10000),
            int((row[19] or 0.0) * 10000),
            int((row[15] or 0.0) * 10000),
        )

        output_buffer.extend(header)
        output_buffer.extend(tensor_bytes)
        output_buffer.extend(policy_add_bytes)
        output_buffer.extend(policy_rem_bytes)
        output_buffer.extend(mcts_add_bytes)
        output_buffer.extend(mcts_rem_bytes)
        output_buffer.extend(selected_actions_bytes)
        output_buffer.extend(extension_bytes)
        output_buffer.extend(rc_header)
        output_buffer.extend(rc_json)
        output_buffer.extend(disp_header)

    return Response(content=bytes(output_buffer), media_type="application/octet-stream")
