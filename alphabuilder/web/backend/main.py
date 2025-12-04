"""
FastAPI Backend for AlphaBuilder Training Data Viewer.

Serves data from SQLite databases for visualization in the frontend.
Supports both legacy schema (v1) and optimized schema (v2).
"""

import json
import pickle
import zlib
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="AlphaBuilder Training Data API")

# CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_SEARCH_PATHS = [
    PROJECT_ROOT,
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "tests" / "data",
]


# --- Pydantic Models ---
class DatabaseInfo(BaseModel):
    id: str
    name: str
    path: str
    episode_count: int
    total_steps: int
    size_mb: float
    schema_version: str  # 'v1' or 'v2'


class EpisodeSummary(BaseModel):
    episode_id: str
    steps_phase1: int
    steps_phase2: int
    total_steps: int
    final_reward: Optional[float]
    final_compliance: Optional[float]
    final_volume_fraction: Optional[float]
    strategy: Optional[str] = None
    bc_type: Optional[str] = None


class Frame(BaseModel):
    step: int
    phase: str
    tensor_shape: list[int]
    tensor_data: list[float]
    fitness_score: float
    compliance: Optional[float]
    volume_fraction: Optional[float]
    policy_add: Optional[list[float]] = None
    policy_remove: Optional[list[float]] = None


class EpisodeData(BaseModel):
    episode_id: str
    frames: list[Frame]


# --- Schema Detection ---
def detect_schema_version(db_path: Path) -> str:
    """Detect which schema version the database uses."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check for v2 schema (episodes table)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='episodes'")
        has_episodes = cursor.fetchone() is not None
        
        # Check for v1 schema (training_data table)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_data'")
        has_training_data = cursor.fetchone() is not None
        
        conn.close()
        
        if has_episodes:
            return "v2"
        elif has_training_data:
            return "v1"
        else:
            # Unknown schema, default to v2 but will be handled gracefully
            return "unknown"
    except sqlite3.Error:
        conn.close()
        return "unknown"


# --- Sparse Decode Helper ---
def sparse_decode(indices: np.ndarray, values: np.ndarray, shape: tuple) -> np.ndarray:
    """Reconstruct dense array from sparse COO format."""
    flat = np.zeros(np.prod(shape), dtype=np.float32)
    if len(indices) > 0:
        flat[indices] = values
    return flat.reshape(shape)


def deserialize_sparse(blob: bytes) -> tuple:
    """Deserialize sparse arrays from compressed bytes."""
    data = pickle.loads(zlib.decompress(blob))
    n = data['n']
    indices = np.frombuffer(data['indices'], dtype=np.int32)[:n]
    values = np.frombuffer(data['values'], dtype=np.float32)[:n]
    return indices, values


def deserialize_array(blob: bytes) -> np.ndarray:
    """Deserialize numpy array from compressed bytes."""
    return pickle.loads(zlib.decompress(blob))


# --- Helper Functions ---
def find_databases() -> list[Path]:
    """Find all .db files in search paths."""
    databases = []
    for search_path in DATA_SEARCH_PATHS:
        if search_path.exists():
            for db_file in search_path.glob("*.db"):
                if db_file not in databases:
                    databases.append(db_file)
    return databases


def get_database_info(db_path: Path) -> DatabaseInfo:
    """Get information about a database."""
    schema_version = detect_schema_version(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    episode_count = 0
    total_steps = 0
    
    try:
        if schema_version == "v2":
            cursor.execute("SELECT COUNT(*) FROM episodes")
            episode_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM records")
            total_steps = cursor.fetchone()[0]
        elif schema_version == "v1":
            cursor.execute("SELECT COUNT(DISTINCT episode_id) FROM training_data")
            episode_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM training_data")
            total_steps = cursor.fetchone()[0]
        else:
            # Unknown schema - try to detect tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            # If no known tables, leave counts at 0
            pass
    except sqlite3.OperationalError as e:
        # Table doesn't exist or other SQL error
        print(f"Warning: Error reading database {db_path.name}: {e}")
        episode_count = 0
        total_steps = 0
    finally:
        conn.close()
    
    size_mb = db_path.stat().st_size / (1024 * 1024)
    
    # Calculate relative path, fallback to absolute if outside PROJECT_ROOT
    try:
        path_str = str(db_path.relative_to(PROJECT_ROOT))
    except ValueError:
        # Database is outside PROJECT_ROOT, use absolute path
        path_str = str(db_path)
    
    return DatabaseInfo(
        id=db_path.stem,
        name=db_path.name,
        path=path_str,
        episode_count=episode_count,
        total_steps=total_steps,
        size_mb=round(size_mb, 2),
        schema_version=schema_version,
    )


def get_episodes_from_db_v2(db_path: Path) -> list[EpisodeSummary]:
    """Get episodes from v2 schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT episode_id, bc_type, strategy, final_compliance, final_volume
        FROM episodes ORDER BY created_at
    """)
    episode_rows = cursor.fetchall()
    
    episodes = []
    for ep_id, bc_type, strategy, final_compliance, final_volume in episode_rows:
        # Get step counts by phase
        cursor.execute("""
            SELECT phase, COUNT(*) FROM records 
            WHERE episode_id = ? GROUP BY phase
        """, (ep_id,))
        phase_counts = dict(cursor.fetchall())
        
        steps_phase1 = phase_counts.get("GROWTH", 0)
        steps_phase2 = phase_counts.get("REFINEMENT", 0)
        
        # Get final fitness score
        cursor.execute("""
            SELECT fitness_score FROM records 
            WHERE episode_id = ? ORDER BY step DESC LIMIT 1
        """, (ep_id,))
        final_row = cursor.fetchone()
        final_reward = final_row[0] if final_row else None
        
        episodes.append(EpisodeSummary(
            episode_id=ep_id,
            steps_phase1=steps_phase1,
            steps_phase2=steps_phase2,
            total_steps=steps_phase1 + steps_phase2,
            final_reward=final_reward,
            final_compliance=final_compliance,
            final_volume_fraction=final_volume,
            strategy=strategy,
            bc_type=bc_type,
        ))
    
    conn.close()
    return episodes


def get_episodes_from_db_v1(db_path: Path) -> list[EpisodeSummary]:
    """Get episodes from v1 (legacy) schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT DISTINCT episode_id FROM training_data ORDER BY episode_id")
    episode_ids = [row[0] for row in cursor.fetchall()]
    
    episodes = []
    for ep_id in episode_ids:
        cursor.execute("""
            SELECT phase, COUNT(*) FROM training_data 
            WHERE episode_id = ? GROUP BY phase
        """, (ep_id,))
        phase_counts = dict(cursor.fetchall())
        
        steps_phase1 = phase_counts.get("GROWTH", 0)
        steps_phase2 = phase_counts.get("REFINEMENT", 0)
        
        cursor.execute("""
            SELECT fitness_score, metadata FROM training_data 
            WHERE episode_id = ? ORDER BY step DESC LIMIT 1
        """, (ep_id,))
        final_row = cursor.fetchone()
        
        final_reward = None
        final_compliance = None
        final_volume_fraction = None
        
        if final_row:
            final_reward = final_row[0]
            if final_row[1]:
                try:
                    metadata = json.loads(final_row[1])
                    final_compliance = metadata.get("compliance")
                    final_volume_fraction = metadata.get("vol_frac")
                except json.JSONDecodeError:
                    pass
        
        episodes.append(EpisodeSummary(
            episode_id=ep_id,
            steps_phase1=steps_phase1,
            steps_phase2=steps_phase2,
            total_steps=steps_phase1 + steps_phase2,
            final_reward=final_reward,
            final_compliance=final_compliance,
            final_volume_fraction=final_volume_fraction,
        ))
    
    conn.close()
    return episodes


def get_episodes_from_db(db_path: Path) -> list[EpisodeSummary]:
    """Get episodes, auto-detecting schema version."""
    if detect_schema_version(db_path) == "v2":
        return get_episodes_from_db_v2(db_path)
    return get_episodes_from_db_v1(db_path)


def load_episode_data_v2(db_path: Path, episode_id: str) -> EpisodeData:
    """Load episode from v2 schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Load episode info (BC masks + forces)
    cursor.execute("""
        SELECT bc_masks_blob, forces_blob, resolution, final_compliance, final_volume
        FROM episodes WHERE episode_id = ?
    """, (episode_id,))
    ep_row = cursor.fetchone()
    
    if not ep_row:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    
    bc_masks = deserialize_array(ep_row[0])  # (3, D, H, W)
    forces = deserialize_array(ep_row[1])     # (3, D, H, W)
    resolution = tuple(json.loads(ep_row[2]))
    
    # Load all steps
    cursor.execute("""
        SELECT step, phase, density_blob, policy_add_blob, policy_remove_blob, fitness_score
        FROM records WHERE episode_id = ? ORDER BY step ASC
    """, (episode_id,))
    rows = cursor.fetchall()
    conn.close()
    
    frames = []
    for row in rows:
        step, phase, density_blob, policy_add_blob, policy_remove_blob, fitness_score = row
        
        # Deserialize density
        density = deserialize_array(density_blob)  # (D, H, W)
        
        # Reconstruct full state tensor (7, D, H, W)
        state_tensor = np.concatenate([
            density[None],  # (1, D, H, W)
            bc_masks,       # (3, D, H, W)
            forces          # (3, D, H, W)
        ], axis=0)
        
        # Decode sparse policy
        policy_add = None
        policy_remove = None
        
        if policy_add_blob:
            try:
                indices, values = deserialize_sparse(policy_add_blob)
                policy_add_arr = sparse_decode(indices, values, resolution)
                policy_add = policy_add_arr.flatten().tolist()
            except Exception:
                pass
        
        if policy_remove_blob:
            try:
                indices, values = deserialize_sparse(policy_remove_blob)
                policy_remove_arr = sparse_decode(indices, values, resolution)
                policy_remove = policy_remove_arr.flatten().tolist()
            except Exception:
                pass
        
        # Compute volume fraction from density
        volume_fraction = float(np.mean(density > 0.5))
        
        frames.append(Frame(
            step=step,
            phase=phase,
            tensor_shape=list(state_tensor.shape),
            tensor_data=state_tensor.flatten().tolist(),
            fitness_score=fitness_score,
            compliance=None,  # Not stored per-step in v2
            volume_fraction=volume_fraction,
            policy_add=policy_add,
            policy_remove=policy_remove,
        ))
    
    return EpisodeData(episode_id=episode_id, frames=frames)


def load_episode_data_v1(db_path: Path, episode_id: str) -> EpisodeData:
    """Load episode from v1 (legacy) schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT step, phase, state_blob, fitness_score, metadata, policy_blob
        FROM training_data WHERE episode_id = ? ORDER BY step ASC
    """, (episode_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    
    frames = []
    for row in rows:
        step, phase, state_blob, fitness_score, metadata_json, policy_blob = row
        
        # Try to deserialize state_blob (may be compressed or not)
        try:
            # First try decompressing (for zlib-compressed data)
            state_tensor = pickle.loads(zlib.decompress(state_blob))
        except (zlib.error, TypeError):
            # If decompression fails, try direct unpickling
            try:
                state_tensor = pickle.loads(state_blob)
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to deserialize state data: {str(e)}"
                )
        
        compliance = None
        volume_fraction = None
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
                compliance = metadata.get("compliance")
                volume_fraction = metadata.get("vol_frac")
            except json.JSONDecodeError:
                pass
        
        policy_add = None
        policy_remove = None
        if policy_blob:
            try:
                # Try decompressing first (for zlib-compressed data)
                try:
                    policy = pickle.loads(zlib.decompress(policy_blob))
                except (zlib.error, TypeError):
                    # If decompression fails, try direct unpickling
                    policy = pickle.loads(policy_blob)
                
                if isinstance(policy, np.ndarray) and policy.ndim == 4 and policy.shape[0] == 2:
                    policy_add = policy[0].flatten().tolist()
                    policy_remove = policy[1].flatten().tolist()
            except Exception as e:
                # Log but don't fail the entire request
                print(f"Warning: Failed to deserialize policy: {e}")
                pass
        
        frames.append(Frame(
            step=step,
            phase=phase,
            tensor_shape=list(state_tensor.shape),
            tensor_data=state_tensor.flatten().tolist(),
            fitness_score=fitness_score,
            compliance=compliance,
            volume_fraction=volume_fraction,
            policy_add=policy_add,
            policy_remove=policy_remove,
        ))
    
    return EpisodeData(episode_id=episode_id, frames=frames)


def load_episode_data(db_path: Path, episode_id: str) -> EpisodeData:
    """Load episode, auto-detecting schema version."""
    if detect_schema_version(db_path) == "v2":
        return load_episode_data_v2(db_path, episode_id)
    return load_episode_data_v1(db_path, episode_id)


# --- API Endpoints ---
@app.get("/")
def root():
    return {"status": "ok", "message": "AlphaBuilder Training Data API"}


@app.get("/databases", response_model=list[DatabaseInfo])
def list_databases():
    """List all available databases."""
    try:
        databases = find_databases()
        result = []
        for db in databases:
            try:
                result.append(get_database_info(db))
            except Exception as e:
                # Log error but continue processing other databases
                print(f"Error processing database {db.name}: {e}")
                # Still include the database with minimal info
                try:
                    size_mb = db.stat().st_size / (1024 * 1024)
                    try:
                        path_str = str(db.relative_to(PROJECT_ROOT))
                    except ValueError:
                        path_str = str(db)
                    result.append(DatabaseInfo(
                        id=db.stem,
                        name=db.name,
                        path=path_str,
                        episode_count=0,
                        total_steps=0,
                        size_mb=round(size_mb, 2),
                        schema_version="unknown",
                    ))
                except Exception:
                    # Skip this database if we can't even get basic info
                    pass
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao listar databases: {str(e)}"
        )


@app.get("/databases/{db_id}/episodes", response_model=list[EpisodeSummary])
def list_episodes(db_id: str):
    """List all episodes in a database."""
    databases = find_databases()
    db_path = next((db for db in databases if db.stem == db_id), None)
    
    if not db_path:
        raise HTTPException(status_code=404, detail=f"Database {db_id} not found")
    
    return get_episodes_from_db(db_path)


@app.get("/databases/{db_id}/episodes/{episode_id}", response_model=EpisodeData)
def get_episode(db_id: str, episode_id: str):
    """Get full episode data for replay."""
    databases = find_databases()
    db_path = next((db for db in databases if db.stem == db_id), None)
    
    if not db_path:
        raise HTTPException(status_code=404, detail=f"Database {db_id} not found")
    
    return load_episode_data(db_path, episode_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
