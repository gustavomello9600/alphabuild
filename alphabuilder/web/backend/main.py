"""
FastAPI Backend for AlphaBuilder Training Data Viewer.

Serves data from SQLite databases for visualization in the frontend.
"""

import json
import pickle
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
# Look for databases in project root and data folder
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


class EpisodeSummary(BaseModel):
    episode_id: str
    steps_phase1: int
    steps_phase2: int
    total_steps: int
    final_reward: Optional[float]
    final_compliance: Optional[float]
    final_volume_fraction: Optional[float]


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
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get episode count
    cursor.execute("SELECT COUNT(DISTINCT episode_id) FROM training_data")
    episode_count = cursor.fetchone()[0]
    
    # Get total steps
    cursor.execute("SELECT COUNT(*) FROM training_data")
    total_steps = cursor.fetchone()[0]
    
    conn.close()
    
    # Get file size
    size_mb = db_path.stat().st_size / (1024 * 1024)
    
    return DatabaseInfo(
        id=db_path.stem,
        name=db_path.name,
        path=str(db_path.relative_to(PROJECT_ROOT)),
        episode_count=episode_count,
        total_steps=total_steps,
        size_mb=round(size_mb, 2),
    )


def get_episodes_from_db(db_path: Path) -> list[EpisodeSummary]:
    """Get summary of all episodes in a database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all unique episode IDs
    cursor.execute("SELECT DISTINCT episode_id FROM training_data ORDER BY episode_id")
    episode_ids = [row[0] for row in cursor.fetchall()]
    
    episodes = []
    for ep_id in episode_ids:
        # Get steps by phase
        cursor.execute("""
            SELECT phase, COUNT(*) as count 
            FROM training_data 
            WHERE episode_id = ? 
            GROUP BY phase
        """, (ep_id,))
        phase_counts = dict(cursor.fetchall())
        
        steps_phase1 = phase_counts.get("GROWTH", 0)
        steps_phase2 = phase_counts.get("REFINEMENT", 0)
        total_steps = steps_phase1 + steps_phase2
        
        # Get final step data
        cursor.execute("""
            SELECT fitness_score, metadata 
            FROM training_data 
            WHERE episode_id = ? 
            ORDER BY step DESC 
            LIMIT 1
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
                    final_volume_fraction = metadata.get("volume_fraction") or metadata.get("vol_frac")
                except json.JSONDecodeError:
                    pass
        
        episodes.append(EpisodeSummary(
            episode_id=ep_id,
            steps_phase1=steps_phase1,
            steps_phase2=steps_phase2,
            total_steps=total_steps,
            final_reward=final_reward,
            final_compliance=final_compliance,
            final_volume_fraction=final_volume_fraction,
        ))
    
    conn.close()
    return episodes


def load_episode_data(db_path: Path, episode_id: str) -> EpisodeData:
    """Load full episode data for replay."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT step, phase, state_blob, fitness_score, metadata, policy_blob
        FROM training_data
        WHERE episode_id = ?
        ORDER BY step ASC
    """, (episode_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    
    frames = []
    for row in rows:
        step, phase, state_blob, fitness_score, metadata_json, policy_blob = row
        
        # Deserialize state tensor
        state_tensor = pickle.loads(state_blob)
        
        # Parse metadata
        compliance = None
        volume_fraction = None
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
                compliance = metadata.get("compliance")
                volume_fraction = metadata.get("volume_fraction") or metadata.get("vol_frac")
            except json.JSONDecodeError:
                pass
        
        # Parse policy if available
        # Policy format: (2, D, H, W) where channel 0 = ADD, channel 1 = REMOVE
        policy_add = None
        policy_remove = None
        if policy_blob:
            try:
                policy = pickle.loads(policy_blob)
                if isinstance(policy, dict):
                    # Legacy dict format
                    if "add" in policy:
                        policy_add = np.array(policy["add"]).flatten().tolist()
                    if "remove" in policy:
                        policy_remove = np.array(policy["remove"]).flatten().tolist()
                elif isinstance(policy, np.ndarray):
                    if policy.ndim == 4 and policy.shape[0] == 2:
                        # v3.1 format: (2, D, H, W) - channel 0 = ADD, channel 1 = REMOVE
                        policy_add = policy[0].flatten().tolist()
                        policy_remove = policy[1].flatten().tolist()
                    elif policy.ndim == 3:
                        # Legacy 3D format - assume it's ADD only
                        policy_add = policy.flatten().tolist()
                    else:
                        # Fallback
                        policy_add = policy.flatten().tolist()
            except Exception as e:
                print(f"Error parsing policy: {e}")
        
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


# --- API Endpoints ---
@app.get("/")
def root():
    return {"status": "ok", "message": "AlphaBuilder Training Data API"}


@app.get("/databases", response_model=list[DatabaseInfo])
def list_databases():
    """List all available databases."""
    databases = find_databases()
    return [get_database_info(db) for db in databases]


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


