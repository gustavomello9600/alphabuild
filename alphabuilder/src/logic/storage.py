"""
SQLite Storage Management for AlphaBuilder Training Data.

Optimized schema v2:
- Episodes table: BC masks and forces stored ONCE per episode
- Records table: Only density and sparse policy per step
- ~88% storage reduction compared to v1

Provides functions to create and manage the training database following
the immutable data lake pattern specified in the blueprint.
"""

import sqlite3
import pickle
import zlib
import uuid
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum
import numpy as np


class Phase(Enum):
    """Episode phase enumeration."""
    GROWTH = "GROWTH"
    REFINEMENT = "REFINEMENT"


# =============================================================================
# Sparse Encoding/Decoding for Policy
# =============================================================================

def sparse_encode(dense_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert dense array to sparse COO format.
    
    Args:
        dense_array: Dense numpy array of any shape
        
    Returns:
        Tuple of (indices, values) where indices are flat int32 and values are float32
    """
    flat = dense_array.flatten().astype(np.float32)
    nonzero_mask = np.abs(flat) > 1e-7
    indices = np.where(nonzero_mask)[0].astype(np.int32)
    values = flat[nonzero_mask]
    return indices, values


def sparse_decode(indices: np.ndarray, values: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Reconstruct dense array from sparse COO format.
    
    Args:
        indices: Flat indices (int32)
        values: Values at those indices (float32)
        shape: Target shape
        
    Returns:
        Dense numpy array
    """
    flat = np.zeros(np.prod(shape), dtype=np.float32)
    if len(indices) > 0:
        flat[indices] = values
    return flat.reshape(shape)


def serialize_sparse(indices: np.ndarray, values: np.ndarray) -> bytes:
    """Serialize sparse arrays to compressed bytes."""
    data = {
        'indices': indices.tobytes(),
        'values': values.tobytes(),
        'n': len(indices)
    }
    return zlib.compress(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))


def deserialize_sparse(blob: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Deserialize sparse arrays from compressed bytes."""
    data = pickle.loads(zlib.decompress(blob))
    n = data['n']
    indices = np.frombuffer(data['indices'], dtype=np.int32)[:n]
    values = np.frombuffer(data['values'], dtype=np.float32)[:n]
    return indices, values


# =============================================================================
# Standard Serialization
# =============================================================================

def serialize_array(arr: np.ndarray) -> bytes:
    """Serialize numpy array to compressed bytes."""
    return zlib.compress(pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL))


def deserialize_array(blob: bytes) -> np.ndarray:
    """Deserialize numpy array from compressed bytes."""
    return pickle.loads(zlib.decompress(blob))


# Legacy aliases for backward compatibility
def serialize_state(state_tensor: np.ndarray) -> bytes:
    """Legacy: Serialize a state tensor to bytes for storage."""
    return serialize_array(state_tensor)


def deserialize_state(state_blob: bytes) -> np.ndarray:
    """Legacy: Deserialize a state tensor from bytes."""
    return deserialize_array(state_blob)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EpisodeInfo:
    """Episode-level information (stored once per episode)."""
    episode_id: str
    bc_masks: np.ndarray      # (3, D, H, W) - mask_x, mask_y, mask_z
    forces: np.ndarray        # (3, D, H, W) - fx, fy, fz
    load_config: Dict[str, Any]
    bc_type: str
    strategy: str             # 'BEZIER' or 'FULL_DOMAIN'
    resolution: Tuple[int, int, int]
    final_compliance: Optional[float] = None
    final_volume: Optional[float] = None


@dataclass
class StepRecord:
    """Single step record (density + sparse policy)."""
    episode_id: str
    step: int
    phase: Phase
    density: np.ndarray           # (D, H, W) - continuous density
    policy_add: np.ndarray        # (D, H, W) - add targets (will be sparse encoded)
    policy_remove: np.ndarray     # (D, H, W) - remove targets (will be sparse encoded)
    fitness_score: float
    is_final_step: bool = False
    is_connected: bool = False


# Legacy dataclass for backward compatibility
@dataclass
class TrainingRecord:
    """Legacy: Single training record to be persisted."""
    episode_id: str
    step: int
    phase: Phase
    state_blob: bytes
    fitness_score: float
    valid_fem: bool
    metadata: Optional[Dict[str, Any]] = None
    policy_blob: Optional[bytes] = None


# =============================================================================
# Database Initialization
# =============================================================================

def initialize_database(db_path: Path) -> None:
    """
    Create the training database with the optimized v2 schema.
    
    Args:
        db_path: Path to the SQLite database file.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    
    # Episodes table: BC and forces stored ONCE per episode
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY,
            bc_masks_blob BLOB NOT NULL,
            forces_blob BLOB NOT NULL,
            load_config TEXT NOT NULL,
            bc_type TEXT NOT NULL,
            strategy TEXT NOT NULL,
            resolution TEXT NOT NULL,
            final_compliance REAL,
            final_volume REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Records table: only density and sparse policy
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            phase TEXT NOT NULL,
            density_blob BLOB NOT NULL,
            policy_add_blob BLOB,
            policy_remove_blob BLOB,
            fitness_score REAL NOT NULL,
            is_final_step INTEGER DEFAULT 0,
            is_connected INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (episode_id) REFERENCES episodes(episode_id),
            UNIQUE(episode_id, step)
        )
    """)
    
    # Indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_records_episode 
        ON records(episode_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_records_phase 
        ON records(phase)
    """)
    
    # Legacy table for backward compatibility (if needed)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            phase TEXT NOT NULL,
            state_blob BLOB NOT NULL,
            fitness_score REAL NOT NULL,
            valid_fem INTEGER NOT NULL,
            metadata TEXT,
            policy_blob BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(episode_id, step)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_episode_id 
        ON training_data(episode_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_phase 
        ON training_data(phase)
    """)
    
    conn.commit()
    conn.close()


# =============================================================================
# Save Functions (New Schema)
# =============================================================================

def save_episode(db_path: Path, episode: EpisodeInfo) -> None:
    """
    Save episode-level information (BC masks, forces, etc.).
    Should be called ONCE at the start of each episode.
    
    Args:
        db_path: Path to the SQLite database file.
        episode: Episode information to save.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO episodes 
            (episode_id, bc_masks_blob, forces_blob, load_config, bc_type, 
             strategy, resolution, final_compliance, final_volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            episode.episode_id,
            serialize_array(episode.bc_masks),
            serialize_array(episode.forces),
            json.dumps(episode.load_config),
            episode.bc_type,
            episode.strategy,
            json.dumps(episode.resolution),
            episode.final_compliance,
            episode.final_volume
        ))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error saving episode: {e}")
    finally:
        conn.close()


def save_step(db_path: Path, record: StepRecord) -> None:
    """
    Save a single step record with sparse policy encoding.
    
    Args:
        db_path: Path to the SQLite database file.
        record: Step record to save.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Encode policy as sparse
    add_indices, add_values = sparse_encode(record.policy_add)
    rem_indices, rem_values = sparse_encode(record.policy_remove)
    
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO records 
            (episode_id, step, phase, density_blob, policy_add_blob, 
             policy_remove_blob, fitness_score, is_final_step, is_connected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.episode_id,
            record.step,
            record.phase.value,
            serialize_array(record.density),
            serialize_sparse(add_indices, add_values),
            serialize_sparse(rem_indices, rem_values),
            record.fitness_score,
            1 if record.is_final_step else 0,
            1 if record.is_connected else 0
        ))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error saving step: {e}")
    finally:
        conn.close()


def update_episode_final(db_path: Path, episode_id: str, 
                         final_compliance: float, final_volume: float) -> None:
    """Update episode with final compliance and volume after optimization."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            UPDATE episodes 
            SET final_compliance = ?, final_volume = ?
            WHERE episode_id = ?
        """, (final_compliance, final_volume, episode_id))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error updating episode: {e}")
    finally:
        conn.close()


# =============================================================================
# Load Functions (New Schema)
# =============================================================================

def load_episode(db_path: Path, episode_id: str) -> Optional[EpisodeInfo]:
    """Load episode information by ID."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT bc_masks_blob, forces_blob, load_config, bc_type, 
               strategy, resolution, final_compliance, final_volume
        FROM episodes WHERE episode_id = ?
    """, (episode_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return None
    
    return EpisodeInfo(
        episode_id=episode_id,
        bc_masks=deserialize_array(row[0]),
        forces=deserialize_array(row[1]),
        load_config=json.loads(row[2]),
        bc_type=row[3],
        strategy=row[4],
        resolution=tuple(json.loads(row[5])),
        final_compliance=row[6],
        final_volume=row[7]
    )


def load_step(db_path: Path, episode_id: str, step: int, 
              resolution: Tuple[int, int, int]) -> Optional[StepRecord]:
    """Load a single step record and decode sparse policy."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT phase, density_blob, policy_add_blob, policy_remove_blob,
               fitness_score, is_final_step, is_connected
        FROM records WHERE episode_id = ? AND step = ?
    """, (episode_id, step))
    
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return None
    
    # Decode sparse policy
    add_indices, add_values = deserialize_sparse(row[2]) if row[2] else (np.array([], dtype=np.int32), np.array([], dtype=np.float32))
    rem_indices, rem_values = deserialize_sparse(row[3]) if row[3] else (np.array([], dtype=np.int32), np.array([], dtype=np.float32))
    
    return StepRecord(
        episode_id=episode_id,
        step=step,
        phase=Phase(row[0]),
        density=deserialize_array(row[1]),
        policy_add=sparse_decode(add_indices, add_values, resolution),
        policy_remove=sparse_decode(rem_indices, rem_values, resolution),
        fitness_score=row[4],
        is_final_step=bool(row[5]),
        is_connected=bool(row[6])
    )


# =============================================================================
# Query Functions
# =============================================================================

def get_episode_count(db_path: Path) -> int:
    """Get the total number of unique episodes in the database."""
    db_path = Path(db_path)
    if not db_path.exists():
        return 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Try new schema first
    try:
        cursor.execute("SELECT COUNT(*) FROM episodes")
        count = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        count = 0
        
    # If no episodes found in new schema (or table empty), check legacy
    if count == 0:
        try:
            cursor.execute("SELECT COUNT(DISTINCT episode_id) FROM training_data")
            legacy_count = cursor.fetchone()[0]
            count += legacy_count
        except sqlite3.OperationalError:
            pass
    
    conn.close()
    return count


def get_record_count(db_path: Path) -> int:
    """Get total number of step records."""
    db_path = Path(db_path)
    if not db_path.exists():
        return 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT COUNT(*) FROM records")
        count = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        cursor.execute("SELECT COUNT(*) FROM training_data")
        count = cursor.fetchone()[0]
    
    conn.close()
    return count


def list_episodes(db_path: Path) -> List[str]:
    """List all episode IDs in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT episode_id FROM episodes ORDER BY created_at")
        episodes = [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        cursor.execute("SELECT DISTINCT episode_id FROM training_data")
        episodes = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return episodes


def get_episode_steps(db_path: Path, episode_id: str) -> List[int]:
    """Get list of step numbers for an episode."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT step FROM records WHERE episode_id = ? ORDER BY step
        """, (episode_id,))
        steps = [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        cursor.execute("""
            SELECT step FROM training_data WHERE episode_id = ? ORDER BY step
        """, (episode_id,))
        steps = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return steps


def generate_episode_id() -> str:
    """Generate a unique episode ID."""
    return str(uuid.uuid4())


# =============================================================================
# Legacy Functions (Backward Compatibility)
# =============================================================================

def save_record(db_path: Path, record: TrainingRecord) -> None:
    """
    Legacy: Save a single training record to the old schema.
    Kept for backward compatibility with existing code.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    metadata_json = json.dumps(record.metadata) if record.metadata else None
    
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO training_data 
            (episode_id, step, phase, state_blob, fitness_score, valid_fem, metadata, policy_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.episode_id,
            record.step,
            record.phase.value,
            record.state_blob,
            record.fitness_score,
            1 if record.valid_fem else 0,
            metadata_json,
            record.policy_blob
        ))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error saving record: {e}")
    finally:
        conn.close()


def has_new_schema(db_path: Path) -> bool:
    """Check if database uses the new optimized schema."""
    if not Path(db_path).exists():
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT COUNT(*) FROM episodes")
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    except sqlite3.OperationalError:
        conn.close()
        return False
