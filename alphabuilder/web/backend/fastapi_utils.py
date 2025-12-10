"""
Common utilities and models for FastAPI backend.
"""

from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import numpy as np
import zlib
import pickle

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_SEARCH_PATHS = [
    PROJECT_ROOT,
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "tests" / "data",
]
SELFPLAY_DB_PATH = PROJECT_ROOT / "data" / "selfplay_games.db"

# --- Pydantic Models (Shared) ---
class DatabaseInfo(BaseModel):
    id: str
    name: str
    path: str
    episode_count: int
    total_steps: int
    size_mb: float
    schema_version: str  # 'v1' or 'v2'

class Action(BaseModel):
    channel: int
    x: int
    y: int
    z: int
    visits: Optional[int] = 0
    q_value: Optional[float] = 0.0

class RewardComponents(BaseModel):
    base_reward: float
    connectivity_bonus: float = 0.0
    fem_reward: float = 0.0
    island_penalty: float = 0.0
    loose_penalty: float = 0.0
    volume_penalty: float = 0.0
    validity_penalty: float = 0.0
    total: float

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
    # New fields for replay
    action_sequence: Optional[List[Action]] = None
    reward_components: Optional[RewardComponents] = None

class EpisodeMetadata(BaseModel):
    """Metadata for an episode, sufficient for initialization without loading all frames."""
    episode_id: str
    steps_phase1: int
    steps_phase2: int
    total_steps: int
    final_reward: Optional[float]
    final_compliance: Optional[float]
    final_volume_fraction: Optional[float]
    bc_type: Optional[str] = None
    strategy: Optional[str] = None
    resolution: Optional[list[int]] = None
    # For v2 reconstruction (flattened arrays)
    bc_masks: Optional[list[float]] = None
    forces: Optional[list[float]] = None
    # For full graph context
    fitness_history: Optional[list[float]] = None

class EpisodeData(BaseModel):
    episode_id: str
    frames: list[Frame]

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

# --- Helper Functions ---
def find_all_databases() -> list[Path]:
    """Find all .db files in search paths."""
    db_paths = set()
    for search_dir in DATA_SEARCH_PATHS:
        if search_dir.exists():
            for db_file in search_dir.glob("*.db"):
                if "selfplay" not in db_file.name: # Exclude selfplay DBs from training data search
                     db_paths.add(db_file.resolve())
    return sorted(list(db_paths))

def resolve_db_path(db_id: str) -> Optional[Path]:
    """Resolve a short DB ID (filename) to full path."""
    for db_path in find_all_databases():
        if db_path.name == db_id:
            return db_path
    return None

def detect_schema_version(db_path: Path) -> str:
    """Detect if database uses v1 or v2 schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='episodes'")
        has_episodes = cursor.fetchone() is not None
        return "v2" if has_episodes else "v1"
    except sqlite3.Error:
        return "v1"
    finally:
        conn.close()

def deserialize_array(blob: bytes) -> np.ndarray:
    """Deserialize numpy array from compressed bytes."""
    return pickle.loads(zlib.decompress(blob))

def deserialize_sparse(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Deserialize sparse arrays from compressed bytes."""
    data = pickle.loads(zlib.decompress(blob))
    n = data['n']
    indices = np.frombuffer(data['indices'], dtype=np.int32)[:n]
    values = np.frombuffer(data['values'], dtype=np.float32)[:n]
    return indices, values

def sparse_decode(indices: np.ndarray, values: np.ndarray, shape: tuple) -> np.ndarray:
    """Reconstruct dense array from sparse COO format."""
    flat = np.zeros(np.prod(shape), dtype=np.float32)
    if len(indices) > 0:
        flat[indices] = values
    return flat.reshape(shape)

def get_selfplay_db_path(deprecated: bool = False) -> Path:
    """Get path to self-play database."""
    if deprecated:
        return PROJECT_ROOT / "data" / "selfplay_games_deprecated.db"
    return SELFPLAY_DB_PATH
