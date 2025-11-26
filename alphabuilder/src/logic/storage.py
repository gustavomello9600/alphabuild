"""
SQLite Storage Management for AlphaBuilder Training Data.

Provides functions to create and manage the training database following
the immutable data lake pattern specified in the blueprint.
"""

import sqlite3
import pickle
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np


class Phase(Enum):
    """Episode phase enumeration."""
    GROWTH = "GROWTH"
    REFINEMENT = "REFINEMENT"


@dataclass
class TrainingRecord:
    """Single training record to be persisted."""
    episode_id: str
    step: int
    phase: Phase
    state_blob: bytes  # Pickled NumPy tensor
    fitness_score: float
    valid_fem: bool
    metadata: Optional[Dict[str, Any]] = None
    policy_blob: Optional[bytes] = None  # NEW: Target action mask


def initialize_database(db_path: Path) -> None:
    """
    Create the training database with the required schema.
    
    Args:
        db_path: Path to the SQLite database file.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    
    # Enable Write-Ahead Logging (WAL) for concurrency
    conn.execute("PRAGMA journal_mode=WAL;")
    
    cursor = conn.cursor()
    
    # Create table following blueprint schema
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
    
    # Create index for efficient querying
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


def save_record(db_path: Path, record: TrainingRecord) -> None:
    """
    Save a single training record to the database.
    
    Args:
        db_path: Path to the SQLite database file.
        record: Training record to save.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Serialize metadata if present
    import json
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


def serialize_state(state_tensor: np.ndarray) -> bytes:
    """
    Serialize a state tensor to bytes for storage.
    
    Args:
        state_tensor: NumPy array representing the state.
        
    Returns:
        Pickled bytes.
    """
    return pickle.dumps(state_tensor, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_state(state_blob: bytes) -> np.ndarray:
    """
    Deserialize a state tensor from bytes.
    
    Args:
        state_blob: Pickled bytes.
        
    Returns:
        NumPy array.
    """
    return pickle.loads(state_blob)


def get_episode_count(db_path: Path) -> int:
    """
    Get the total number of unique episodes in the database.
    
    Args:
        db_path: Path to the SQLite database file.
        
    Returns:
        Number of unique episodes.
    """
    if not db_path.exists():
        return 0
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(DISTINCT episode_id) FROM training_data")
    count = cursor.fetchone()[0]
    
    conn.close()
    return count


def generate_episode_id() -> str:
    """Generate a unique episode ID."""
    return str(uuid.uuid4())
