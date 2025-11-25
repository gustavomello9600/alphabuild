import pytest
import numpy as np
import sqlite3
from alphabuilder.src.logic.storage import (
    initialize_database,
    save_record,
    get_episode_count,
    TrainingRecord,
    Phase,
    serialize_state,
    deserialize_state
)

def test_initialize_database(temp_db):
    """Test database initialization."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_data'")
    assert cursor.fetchone() is not None
    
    conn.close()

def test_save_and_count_records(temp_db):
    """Test saving records and counting episodes."""
    assert get_episode_count(temp_db) == 0
    
    state = np.zeros((10, 10), dtype=np.int32)
    blob = serialize_state(state)
    
    record = TrainingRecord(
        episode_id="ep1",
        step=0,
        phase=Phase.GROWTH,
        state_blob=blob,
        fitness_score=0.5,
        valid_fem=True,
        metadata={}
    )
    
    save_record(temp_db, record)
    
    # Count should be 1 (distinct episode_id)
    assert get_episode_count(temp_db) == 1
    
    # Add another step for same episode
    record2 = TrainingRecord(
        episode_id="ep1",
        step=1,
        phase=Phase.REFINEMENT,
        state_blob=blob,
        fitness_score=0.6,
        valid_fem=True,
        metadata={}
    )
    save_record(temp_db, record2)
    
    assert get_episode_count(temp_db) == 1
    
    # Add new episode
    record3 = TrainingRecord(
        episode_id="ep2",
        step=0,
        phase=Phase.GROWTH,
        state_blob=blob,
        fitness_score=0.1,
        valid_fem=True,
        metadata={}
    )
    save_record(temp_db, record3)
    
    assert get_episode_count(temp_db) == 2

def test_serialization():
    """Test state serialization and deserialization."""
    original = np.random.randint(0, 2, (10, 10)).astype(np.int32)
    blob = serialize_state(original)
    restored = deserialize_state(blob)
    
    np.testing.assert_array_equal(original, restored)
