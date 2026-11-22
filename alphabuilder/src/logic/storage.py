"""
SQLite Storage Management for AlphaBuilder Training Data (v2 / v3.3).

Schema v3.3:
- Version-independent serialization (`\x93NUMPY` + zlib for dense arrays,
  compact `COO1` binary + zlib for sparse policy tensors)
- Atomic per-episode persistence via `save_complete_episode`
- Episode metadata for distributed sharding (`seed`, `shard_id`, `worker_id`,
  `code_version`, `v_constructed_blob`, `final_max_displacement`,
  `final_is_connected`, `simp_iters`, `duration_sec`)
- Raw physics per step (`rho_cont_blob`, `compliance`, `vol_frac`,
  `max_displacement`, `beta`) for recalibration without re-running FEM
"""

import io
import sqlite3
import struct
import pickle
import zlib
import uuid
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np


_COO1_MAGIC = b"COO1"
_ALPHABUILD_UUID_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "alphabuilder.project.ufs")


class Phase(Enum):
    """Episode phase enumeration."""
    GROWTH = "GROWTH"
    REFINEMENT = "REFINEMENT"


def deterministic_episode_id(
    seed: int,
    resolution: Tuple[int, int, int],
    schema_tag: str = "v3.3",
) -> str:
    """Generate a deterministic UUIDv5 episode ID from seed, resolution and schema."""
    nx, ny, nz = resolution
    return str(uuid.uuid5(_ALPHABUILD_UUID_NS, f"{schema_tag}:{nx}x{ny}x{nz}:{int(seed)}"))


# =============================================================================
# Sparse Encoding/Decoding for Policy
# =============================================================================

def sparse_encode(dense_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert dense array to sparse COO format.

    Returns:
        Tuple of (indices, values) where indices are flat int32 and values are float32
    """
    flat = dense_array.flatten().astype(np.float32)
    nonzero_mask = np.abs(flat) > 1e-7
    indices = np.where(nonzero_mask)[0].astype(np.int32)
    values = flat[nonzero_mask]
    return indices, values


def sparse_decode(indices: np.ndarray, values: np.ndarray, shape: tuple) -> np.ndarray:
    """Reconstruct dense array from sparse COO format."""
    flat = np.zeros(np.prod(shape), dtype=np.float32)
    if len(indices) > 0:
        flat[indices] = values
    return flat.reshape(shape)


def serialize_sparse(indices: np.ndarray, values: np.ndarray) -> bytes:
    """Serialize sparse COO arrays using compact version-independent COO1 format."""
    idx_bytes = np.ascontiguousarray(indices, dtype='<i4').tobytes()
    val_bytes = np.ascontiguousarray(values, dtype='<f4').tobytes()
    n = len(indices)
    payload = _COO1_MAGIC + struct.pack('<I', n) + idx_bytes + val_bytes
    return zlib.compress(payload, level=6)


def deserialize_sparse(blob: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Deserialize sparse arrays (supports COO1 binary and legacy pickle)."""
    raw = zlib.decompress(blob)
    if raw[:4] == _COO1_MAGIC:
        n = struct.unpack('<I', raw[4:8])[0]
        idx_end = 8 + n * 4
        val_end = idx_end + n * 4
        indices = np.frombuffer(raw[8:idx_end], dtype='<i4').astype(np.int32, copy=True)
        values = np.frombuffer(raw[idx_end:val_end], dtype='<f4').astype(np.float32, copy=True)
        return indices, values

    data = pickle.loads(raw)
    n = data['n']
    indices = np.frombuffer(data['indices'], dtype=np.int32).copy()[:n]
    values = np.frombuffer(data['values'], dtype=np.float32).copy()[:n]
    return indices, values


# =============================================================================
# Standard Serialization (NumPy version-independent)
# =============================================================================

def serialize_array(arr: np.ndarray) -> bytes:
    """Serialize numpy array to zlib-compressed `.npy` bytes (no pickle)."""
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return zlib.compress(buf.getvalue(), level=6)


def deserialize_array(blob: bytes) -> np.ndarray:
    """Deserialize numpy array from `.npy` or legacy pickle compressed bytes."""
    raw = zlib.decompress(blob)
    if raw[:6] == b'\x93NUMPY':
        return np.load(io.BytesIO(raw), allow_pickle=False)
    return pickle.loads(raw)


# Legacy aliases for backward compatibility
def serialize_state(state_tensor: np.ndarray) -> bytes:
    """Legacy: Serialize a state tensor to bytes for storage."""
    return serialize_array(state_tensor)


def deserialize_state(state_blob: bytes) -> np.ndarray:
    """Legacy: Deserialize a state tensor from bytes."""
    return deserialize_array(state_blob)


def _encode_density_compact(density: np.ndarray) -> np.ndarray:
    """Store binary 0/1 density as uint8 for ~4x raw reduction; keep non-binary as float32."""
    arr = np.asarray(density)
    if arr.dtype == np.uint8 or arr.dtype == bool:
        return arr.astype(np.uint8)
    is_binary = np.all((arr == 0.0) | (arr == 1.0))
    if is_binary:
        return arr.astype(np.uint8)
    return arr.astype(np.float32)


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
    seed: Optional[int] = None
    shard_id: Optional[str] = None
    worker_id: Optional[str] = None
    code_version: Optional[str] = None
    v_constructed: Optional[np.ndarray] = None
    final_max_displacement: Optional[float] = None
    final_is_connected: Optional[bool] = None
    simp_iters: Optional[int] = None
    duration_sec: Optional[float] = None


@dataclass
class StepRecord:
    """Single step record (density + sparse policy + optional v3.3 physics)."""
    episode_id: str
    step: int
    phase: Phase
    density: np.ndarray           # (D, H, W)
    policy_add: np.ndarray        # (D, H, W) - add targets (sparse encoded)
    policy_remove: np.ndarray     # (D, H, W) - remove targets (sparse encoded)
    fitness_score: float
    is_final_step: bool = False
    is_connected: bool = False
    displacement_map: Optional[np.ndarray] = None
    reward_components: Optional[Dict[str, float]] = None
    rho_cont: Optional[np.ndarray] = None
    compliance: Optional[float] = None
    vol_frac: Optional[float] = None
    max_displacement: Optional[float] = None
    beta: Optional[int] = None


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
# Database Initialization & Schema Migration
# =============================================================================

_EPISODE_V33_COLUMNS = [
    ("seed", "INTEGER"),
    ("shard_id", "TEXT"),
    ("worker_id", "TEXT"),
    ("code_version", "TEXT"),
    ("v_constructed_blob", "BLOB"),
    ("final_max_displacement", "REAL"),
    ("final_is_connected", "INTEGER"),
    ("simp_iters", "INTEGER"),
    ("duration_sec", "REAL"),
]

_RECORD_V33_COLUMNS = [
    ("rho_cont_blob", "BLOB"),
    ("compliance", "REAL"),
    ("vol_frac", "REAL"),
    ("max_displacement", "REAL"),
    ("beta", "INTEGER"),
]


def _ensure_columns(cursor: sqlite3.Cursor, table: str, columns: List[Tuple[str, str]]) -> None:
    cursor.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cursor.fetchall()}
    for col_name, col_type in columns:
        if col_name not in existing:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")


def initialize_database(db_path: Path, wal_mode: bool = True) -> None:
    """
    Create or upgrade the training database to schema v3.3.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    if wal_mode:
        conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    
    # Episodes table: BC, forces, blueprint and provenance stored ONCE per episode
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY,
            seed INTEGER,
            shard_id TEXT,
            worker_id TEXT,
            code_version TEXT,
            bc_masks_blob BLOB NOT NULL,
            forces_blob BLOB NOT NULL,
            v_constructed_blob BLOB,
            load_config TEXT NOT NULL,
            bc_type TEXT NOT NULL,
            strategy TEXT NOT NULL,
            resolution TEXT NOT NULL,
            final_compliance REAL,
            final_volume REAL,
            final_max_displacement REAL,
            final_is_connected INTEGER,
            simp_iters INTEGER,
            duration_sec REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    _ensure_columns(cursor, "episodes", _EPISODE_V33_COLUMNS)
    
    # Records table: compact density, sparse policy, value target and raw scalar physics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            phase TEXT NOT NULL,
            density_blob BLOB NOT NULL,
            rho_cont_blob BLOB,
            policy_add_blob BLOB,
            policy_remove_blob BLOB,
            fitness_score REAL NOT NULL,
            compliance REAL,
            vol_frac REAL,
            max_displacement REAL,
            beta INTEGER,
            is_final_step INTEGER DEFAULT 0,
            is_connected INTEGER DEFAULT 0,
            displacement_blob BLOB,
            reward_components_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (episode_id) REFERENCES episodes(episode_id),
            UNIQUE(episode_id, step)
        )
    """)
    _ensure_columns(cursor, "records", _RECORD_V33_COLUMNS)
    
    # Indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_records_episode 
        ON records(episode_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_records_phase 
        ON records(phase)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_episodes_seed
        ON episodes(seed)
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
# Save Functions (v2 / v3.3 Schema)
# =============================================================================

def _episode_insert_params(episode: EpisodeInfo) -> tuple:
    v_const_blob = None
    if episode.v_constructed is not None:
        v_mask = (episode.v_constructed > 0.5).astype(np.uint8)
        v_const_blob = serialize_array(v_mask)

    final_conn_int = (
        None if episode.final_is_connected is None
        else (1 if episode.final_is_connected else 0)
    )
    return (
        episode.episode_id,
        episode.seed,
        episode.shard_id,
        episode.worker_id,
        episode.code_version,
        serialize_array(episode.bc_masks.astype(np.float32)),
        serialize_array(episode.forces.astype(np.float32)),
        v_const_blob,
        json.dumps(episode.load_config),
        episode.bc_type,
        episode.strategy,
        json.dumps(episode.resolution),
        episode.final_compliance,
        episode.final_volume,
        episode.final_max_displacement,
        final_conn_int,
        episode.simp_iters,
        episode.duration_sec,
    )


def _step_insert_params(record: StepRecord) -> tuple:
    add_indices, add_values = sparse_encode(record.policy_add)
    rem_indices, rem_values = sparse_encode(record.policy_remove)
    rho_cont_blob = (
        serialize_array(record.rho_cont.astype(np.float16))
        if record.rho_cont is not None else None
    )
    disp_blob = (
        serialize_array(record.displacement_map.astype(np.float16))
        if record.displacement_map is not None else None
    )
    return (
        record.episode_id,
        record.step,
        record.phase.value,
        serialize_array(_encode_density_compact(record.density)),
        rho_cont_blob,
        serialize_sparse(add_indices, add_values),
        serialize_sparse(rem_indices, rem_values),
        float(record.fitness_score),
        record.compliance,
        record.vol_frac,
        record.max_displacement,
        record.beta,
        1 if record.is_final_step else 0,
        1 if record.is_connected else 0,
        disp_blob,
        json.dumps(record.reward_components) if record.reward_components else None,
    )


_EPISODE_UPSERT_SQL = """
    INSERT OR REPLACE INTO episodes (
        episode_id, seed, shard_id, worker_id, code_version,
        bc_masks_blob, forces_blob, v_constructed_blob,
        load_config, bc_type, strategy, resolution,
        final_compliance, final_volume, final_max_displacement,
        final_is_connected, simp_iters, duration_sec
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_RECORD_UPSERT_SQL = """
    INSERT OR REPLACE INTO records (
        episode_id, step, phase, density_blob, rho_cont_blob,
        policy_add_blob, policy_remove_blob, fitness_score,
        compliance, vol_frac, max_displacement, beta,
        is_final_step, is_connected, displacement_blob, reward_components_json
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def save_episode(db_path: Path, episode: EpisodeInfo) -> None:
    """Save episode-level information."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(_EPISODE_UPSERT_SQL, _episode_insert_params(episode))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error saving episode: {e}")
    finally:
        conn.close()


def save_step(db_path: Path, record: StepRecord) -> None:
    """Save a single step record with sparse policy encoding."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(_RECORD_UPSERT_SQL, _step_insert_params(record))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error saving step: {e}")
    finally:
        conn.close()


def save_complete_episode(
    db_path: Path,
    episode: EpisodeInfo,
    steps: List[StepRecord],
) -> None:
    """
    Atomically persist an episode and all its step records in a single SQLite transaction.

    Prevents orphan episodes if a worker crashes mid-episode.
    """
    if not steps:
        raise ValueError(f"Refusing to save episode {episode.episode_id} with 0 steps")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN")
        cursor.execute("DELETE FROM records WHERE episode_id = ?", (episode.episode_id,))
        cursor.execute(_EPISODE_UPSERT_SQL, _episode_insert_params(episode))
        step_params = [_step_insert_params(s) for s in steps]
        cursor.executemany(_RECORD_UPSERT_SQL, step_params)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
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
# Load Functions (v2 / v3.3 Schema)
# =============================================================================

def load_episode(db_path: Path, episode_id: str) -> Optional[EpisodeInfo]:
    """Load episode information by ID."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT bc_masks_blob, forces_blob, load_config, bc_type, 
               strategy, resolution, final_compliance, final_volume,
               seed, shard_id, worker_id, code_version, v_constructed_blob,
               final_max_displacement, final_is_connected, simp_iters, duration_sec
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
        final_volume=row[7],
        seed=row[8],
        shard_id=row[9],
        worker_id=row[10],
        code_version=row[11],
        v_constructed=deserialize_array(row[12]) if row[12] else None,
        final_max_displacement=row[13],
        final_is_connected=bool(row[14]) if row[14] is not None else None,
        simp_iters=row[15],
        duration_sec=row[16],
    )


def load_step(db_path: Path, episode_id: str, step: int, 
              resolution: Tuple[int, int, int]) -> Optional[StepRecord]:
    """Load a single step record and decode sparse policy."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT phase, density_blob, policy_add_blob, policy_remove_blob,
               fitness_score, is_final_step, is_connected, displacement_blob,
               reward_components_json, rho_cont_blob, compliance, vol_frac,
               max_displacement, beta
        FROM records WHERE episode_id = ? AND step = ?
    """, (episode_id, step))
    
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return None
    
    add_indices, add_values = deserialize_sparse(row[2]) if row[2] else (np.array([], dtype=np.int32), np.array([], dtype=np.float32))
    rem_indices, rem_values = deserialize_sparse(row[3]) if row[3] else (np.array([], dtype=np.int32), np.array([], dtype=np.float32))
    
    return StepRecord(
        episode_id=episode_id,
        step=step,
        phase=Phase(row[0]),
        density=deserialize_array(row[1]).astype(np.float32),
        policy_add=sparse_decode(add_indices, add_values, resolution),
        policy_remove=sparse_decode(rem_indices, rem_values, resolution),
        fitness_score=row[4],
        is_final_step=bool(row[5]),
        is_connected=bool(row[6]),
        displacement_map=deserialize_array(row[7]).astype(np.float32) if row[7] else None,
        reward_components=json.loads(row[8]) if row[8] else None,
        rho_cont=deserialize_array(row[9]).astype(np.float32) if row[9] else None,
        compliance=row[10],
        vol_frac=row[11],
        max_displacement=row[12],
        beta=row[13],
    )


def get_completed_seeds(db_path: Path) -> Set[int]:
    """Return the set of seeds that have at least one persisted step record."""
    db_path = Path(db_path)
    if not db_path.exists():
        return set()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT DISTINCT e.seed
            FROM episodes e
            INNER JOIN records r ON e.episode_id = r.episode_id
            WHERE e.seed IS NOT NULL
        """)
        seeds = {int(row[0]) for row in cursor.fetchall()}
    except sqlite3.OperationalError:
        seeds = set()
    finally:
        conn.close()
    return seeds


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
