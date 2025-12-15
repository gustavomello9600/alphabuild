from fastapi import APIRouter, HTTPException
from pathlib import Path
import sqlite3
import json
import zlib
import pickle
import numpy as np
import base64
from typing import Optional

try:
    from ..fastapi_utils import (
        DatabaseInfo, EpisodeSummary, EpisodeData, Frame, EpisodeMetadata,
        find_all_databases, resolve_db_path, detect_schema_version,
        deserialize_array, deserialize_sparse, sparse_decode
    )
except (ImportError, ValueError):
    from fastapi_utils import (
        DatabaseInfo, EpisodeSummary, EpisodeData, Frame, EpisodeMetadata,
        find_all_databases, resolve_db_path, detect_schema_version,
        deserialize_array, deserialize_sparse, sparse_decode
    )

router = APIRouter()

# --- Helper Functions (specific to training data) ---

def get_db_info(db_path: Path) -> DatabaseInfo:
    """Get summarized info for a database file."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    schema = detect_schema_version(db_path)
    ep_count = 0
    total_steps = 0
    
    try:
        if schema == "v2":
            cursor.execute("SELECT COUNT(*) FROM episodes")
            ep_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM records")
            total_steps = cursor.fetchone()[0]
        else:
            cursor.execute("SELECT COUNT(DISTINCT episode_id) FROM training_data")
            ep_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM training_data")
            total_steps = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        pass
        
    conn.close()
    
    return DatabaseInfo(
        id=db_path.name,
        name=db_path.stem,
        path=str(db_path),
        episode_count=ep_count,
        total_steps=total_steps,
        size_mb=db_path.stat().st_size / (1024 * 1024),
        schema_version=schema
    )

def get_episode_metadata_v2(db_path: Path, episode_id: str) -> EpisodeMetadata:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT final_compliance, final_volume, bc_type, strategy, resolution,
               bc_masks_blob, forces_blob, episode_id
        FROM episodes WHERE episode_id = ?
    """, (episode_id,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Episode metadata not found")
        
    # Get stats from records
    cursor.execute("SELECT COUNT(*) FROM records WHERE episode_id = ?", (episode_id,))
    total_steps = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM records WHERE episode_id = ? AND phase='GROWTH'", (episode_id,))
    steps_phase1 = cursor.fetchone()[0]
    steps_phase2 = total_steps - steps_phase1
    
    # Get final reward
    cursor.execute("SELECT fitness_score FROM records WHERE episode_id = ? ORDER BY step DESC LIMIT 1", (episode_id,))
    last_row = cursor.fetchone()
    final_reward = last_row[0] if last_row else None
        
    # Get fitness history for graph
    cursor.execute("SELECT fitness_score FROM records WHERE episode_id = ? ORDER BY step ASC", (episode_id,))
    fitness_rows = cursor.fetchall()
    fitness_history = [r[0] for r in fitness_rows]
    
    conn.close()
    
    # helper for resolving none
    def get_float(val): return float(val) if val is not None else None
    
    resolution = json.loads(row[4]) if row[4] else None
    bc_masks = deserialize_array(row[5]).flatten().tolist()
    forces = deserialize_array(row[6]).flatten().tolist()
    
    return EpisodeMetadata(
        episode_id=episode_id,
        steps_phase1=steps_phase1,
        steps_phase2=steps_phase2,
        total_steps=total_steps,
        final_reward=final_reward,
        final_compliance=get_float(row[0]),
        final_volume_fraction=get_float(row[1]),
        bc_type=row[2],
        strategy=row[3],
        resolution=resolution,
        bc_masks=bc_masks,
        forces=forces,
        fitness_history=fitness_history
    )

def get_episode_metadata_v1(db_path: Path, episode_id: str) -> EpisodeMetadata:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Calculate stats
    cursor.execute("SELECT COUNT(*) FROM training_data WHERE episode_id = ?", (episode_id,))
    total_steps = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM training_data WHERE episode_id = ? AND phase='GROWTH'", (episode_id,))
    steps_p1 = cursor.fetchone()[0]
    
    steps_p2 = total_steps - steps_p1
    
    # Get final metrics
    cursor.execute("""
        SELECT fitness_score, metadata 
        FROM training_data 
        WHERE episode_id = ? 
        ORDER BY step DESC LIMIT 1
    """, (episode_id,))
    last_row = cursor.fetchone()
    
    final_reward = last_row[0] if last_row else None
    final_meta = json.loads(last_row[1]) if last_row and last_row[1] else {}
    
    # Get fitness history
    cursor.execute("SELECT fitness_score FROM training_data WHERE episode_id = ? ORDER BY step ASC", (episode_id,))
    fitness_rows = cursor.fetchall()
    fitness_history = [r[0] for r in fitness_rows]
    
    # Infer resolution from first frame state blob if possible
    resolution = None
    cursor.execute("SELECT state_blob FROM training_data WHERE episode_id = ? LIMIT 1", (episode_id,))
    blob_row = cursor.fetchone()
    if blob_row:
        try:
            state = deserialize_array(blob_row[0])
            resolution = list(state.shape[1:]) # (7, D, H, W) -> [D, H, W]
        except:
            pass

    conn.close()
    
    return EpisodeMetadata(
        episode_id=episode_id,
        steps_phase1=steps_p1,
        steps_phase2=steps_p2,
        total_steps=total_steps,
        final_reward=final_reward,
        final_compliance=final_meta.get('compliance'),
        final_volume_fraction=final_meta.get('volume_fraction'),
        resolution=resolution,
        fitness_history=fitness_history
    )

def load_episode_frames_v2(db_path: Path, episode_id: str, limit: int, offset: int) -> list[Frame]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Need resolution to reconstruct dense arrays
    cursor.execute("SELECT resolution FROM episodes WHERE episode_id = ?", (episode_id,))
    res_row = cursor.fetchone()
    if not res_row: 
        conn.close()
        return []
        
    resolution = tuple(json.loads(res_row[0]))
    
    cursor.execute("""
        SELECT step, phase, density_blob, policy_add_blob, policy_remove_blob, 
               fitness_score, is_connected, reward_components_json, displacement_blob
        FROM records 
        WHERE episode_id = ? 
        ORDER BY step ASC
        LIMIT ? OFFSET ?
    """, (episode_id, limit, offset))
    
    rows = cursor.fetchall()
    conn.close()
    
    frames = []
    for row in rows:
        density = deserialize_array(row[2])
        
        # Decode sparse policies
        p_add_idx, p_add_val = deserialize_sparse(row[3]) if row[3] else ([], [])
        p_rem_idx, p_rem_val = deserialize_sparse(row[4]) if row[4] else ([], [])
        
        policy_add = sparse_decode(p_add_idx, p_add_val, resolution)
        policy_remove = sparse_decode(p_rem_idx, p_rem_val, resolution)
        
        # Deserialize JSON fields
        action_sequence = None # Not available in training data v2 schema
        reward_components = json.loads(row[7]) if row[7] else None
        
        if reward_components:
            # Backfill missing fields for Pydantic validation
            if 'base_reward' not in reward_components:
                reward_components['base_reward'] = reward_components.get('fem_score', 0.0)
            if 'total' not in reward_components:
                reward_components['total'] = row[5]

        # Helper to encode displacement map
        displacement_b64 = None
        if row[8] is not None:
             disp_map = deserialize_array(row[8])
             displacement_b64 = base64.b64encode(disp_map.astype(np.float32).tobytes()).decode('utf-8')

        frames.append(Frame(
            step=row[0],
            phase=row[1],
            tensor_shape=[1, *resolution], # Send only density
            tensor_data=density.flatten().tolist(),
            fitness_score=row[5],
            compliance=None, # In v2 derived from generic metadata or added later
            volume_fraction=None, # Calculated on frontend or added to query if explicit
            policy_add=policy_add.flatten().tolist(),
            policy_remove=policy_remove.flatten().tolist(),
            action_sequence=action_sequence,
            reward_components=reward_components,
            displacement_map=displacement_b64
        ))
        
    return frames

def load_episode_frames_v1(db_path: Path, episode_id: str, limit: int, offset: int) -> list[Frame]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT step, phase, state_blob, fitness_score, metadata, policy_blob
        FROM training_data
        WHERE episode_id = ?
        ORDER BY step ASC
        LIMIT ? OFFSET ?
    """, (episode_id, limit, offset))
    
    rows = cursor.fetchall()
    conn.close()
    
    frames = []
    for row in rows:
        state_tensor = deserialize_array(row[2])
        meta = json.loads(row[4]) if row[4] else {}
        
        policy = deserialize_array(row[5]) if row[5] else None
        p_add = np.maximum(policy, 0) if policy is not None else None
        p_rem = np.abs(np.minimum(policy, 0)) if policy is not None else None
        
        frames.append(Frame(
            step=row[0],
            phase=row[1],
            tensor_shape=list(state_tensor.shape),
            tensor_data=state_tensor.flatten().tolist(),
            fitness_score=row[3],
            compliance=meta.get('compliance'),
            volume_fraction=meta.get('volume_fraction'),
            policy_add=p_add.flatten().tolist() if p_add is not None else None,
            policy_remove=p_rem.flatten().tolist() if p_rem is not None else None
        ))
        
    return frames


# --- Endpoints ---

@router.get("/databases", response_model=list[DatabaseInfo])
def list_databases():
    """List all available training databases."""
    db_files = find_all_databases()
    return [get_db_info(db) for db in db_files]

@router.get("/databases/{db_id}/episodes", response_model=list[EpisodeSummary])
def list_episodes(
    db_id: str, 
    limit: int = 50, 
    offset: int = 0,
    min_reward: Optional[float] = None
):
    """List episodes in a database with pagination and filtering."""
    db_path = resolve_db_path(db_id)
    if not db_path:
        raise HTTPException(status_code=404, detail="Database not found")
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    schema = detect_schema_version(db_path)
    
    episodes = []
    
    if schema == "v2":
        query = """
            SELECT episode_id, final_compliance, final_volume, strategy, bc_type, created_at
            FROM episodes
            WHERE 1=1
        """
        params = []
        if min_reward is not None:
             # Can't filter by min_reward efficiently without column in episodes
             pass 
            
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        for row in rows:
            ep_id = row[0]
            # Get steps counts
            cursor.execute("SELECT COUNT(*) FROM records WHERE episode_id = ?", (ep_id,))
            total_steps = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM records WHERE episode_id = ? AND phase='GROWTH'", (ep_id,))
            steps_p1 = cursor.fetchone()[0]
            steps_p2 = total_steps - steps_p1
            
            # Get final reward
            cursor.execute("SELECT fitness_score FROM records WHERE episode_id = ? ORDER BY step DESC LIMIT 1", (ep_id,))
            last_rec = cursor.fetchone()
            reward = last_rec[0] if last_rec else None

            # Check filter manually if needed
            if min_reward is not None and (reward is None or reward < min_reward):
                continue

            episodes.append(EpisodeSummary(
                episode_id=ep_id,
                steps_phase1=steps_p1,
                steps_phase2=steps_p2,
                total_steps=total_steps,
                final_reward=reward,
                final_compliance=row[1],
                final_volume_fraction=row[2],
                strategy=row[3],
                bc_type=row[4]
            ))
    else:
        # Legacy schema query
        cursor.execute("SELECT DISTINCT episode_id FROM training_data LIMIT ? OFFSET ?", (limit, offset))
        ids = [r[0] for r in cursor.fetchall()]
        
        for ep_id in ids:
            # Need to aggregate manual stats for legacy
            cursor.execute("SELECT COUNT(*) FROM training_data WHERE episode_id = ?", (ep_id,))
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT fitness_score FROM training_data WHERE episode_id = ? ORDER BY step DESC LIMIT 1", (ep_id,))
            last = cursor.fetchone()
            reward = last[0] if last else None
            
            episodes.append(EpisodeSummary(
                episode_id=ep_id,
                steps_phase1=0,
                steps_phase2=0,
                total_steps=total,
                final_reward=reward,
                final_compliance=None,
                final_volume_fraction=None
            ))
            
    conn.close()
    return episodes

@router.get("/databases/{db_id}/episodes/{episode_id}/metadata", response_model=EpisodeMetadata)
def get_episode_metadata(db_id: str, episode_id: str):
    """Get lightweight metadata for an episode."""
    db_path = resolve_db_path(db_id)
    if not db_path:
        raise HTTPException(status_code=404, detail="Database not found")
        
    if detect_schema_version(db_path) == "v2":
        return get_episode_metadata_v2(db_path, episode_id)
    return get_episode_metadata_v1(db_path, episode_id)

@router.get("/databases/{db_id}/episodes/{episode_id}/frames", response_model=list[Frame])
def get_episode_frames(db_id: str, episode_id: str, start: int = 0, end: int = 100):
    """Get range of frames for an episode."""
    db_path = resolve_db_path(db_id)
    if not db_path:
        raise HTTPException(status_code=404, detail="Database not found")
        
    # Cap limit
    limit = min(end - start, 100)
    
    if detect_schema_version(db_path) == "v2":
        return load_episode_frames_v2(db_path, episode_id, limit, start)
    return load_episode_frames_v1(db_path, episode_id, limit, start)
