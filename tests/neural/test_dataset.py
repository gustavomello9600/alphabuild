import pytest
import tensorflow as tf
import numpy as np
import json
import sqlite3
import pickle
from alphabuilder.src.logic.storage import (
    save_record, TrainingRecord, Phase, serialize_state
)
from alphabuilder.src.neural.dataset import create_dataset, data_generator, get_dataset_shape

def test_data_generator(temp_db):
    """Test data generator yields correct data."""
    # Create dummy data
    state = np.zeros((32, 64), dtype=np.int32)
    blob = serialize_state(state)
    
    metadata = json.dumps({"max_displacement": 1.5})
    
    record = TrainingRecord(
        episode_id="ep1",
        step=0,
        phase=Phase.REFINEMENT,
        state_blob=blob,
        fitness_score=0.5,
        valid_fem=True,
        metadata=metadata # Pass raw json string or dict? Storage expects dict but saves as string?
        # Storage.py handles dict -> json conversion.
        # Wait, storage.py expects dict in dataclass, but saves as JSON.
        # Let's check storage.py... it takes dict.
    )
    # Re-create record with dict metadata
    record = TrainingRecord(
        episode_id="ep1",
        step=0,
        phase=Phase.REFINEMENT,
        state_blob=blob,
        fitness_score=0.5,
        valid_fem=True,
        metadata={"max_displacement": 1.5}
    )
    
    save_record(temp_db, record)
    
    # Test generator
    gen = data_generator(temp_db)
    x, y = next(gen)
    
    assert x.shape == (32, 64, 3)
    assert y == 1.5

def test_get_dataset_shape(temp_db):
    """Test auto-detection of dataset shape."""
    # Insert a record with known shape (16, 32)
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    
    state = np.zeros((16, 32), dtype=np.float32)
    state_blob = pickle.dumps(state)
    
    cursor.execute("""
        INSERT INTO training_data (episode_id, step, phase, state_blob, fitness_score, valid_fem, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, ("test_ep", 1, "REFINEMENT", state_blob, 1.0, 1, "{}"))
    conn.commit()
    conn.close()
    
    # Test detection
    shape = get_dataset_shape(temp_db)
    assert shape == (16, 32, 3)
    
    # Test empty DB
    assert get_dataset_shape("non_existent.db") is None

def test_create_dataset(temp_db):
    """Test tf.data.Dataset creation."""
    # Create dummy data
    state = np.zeros((32, 64), dtype=np.int32)
    blob = serialize_state(state)
    record = TrainingRecord(
        episode_id="ep1",
        step=0,
        phase=Phase.REFINEMENT,
        state_blob=blob,
        fitness_score=0.5,
        valid_fem=True,
        metadata={"max_displacement": 1.5}
    )
    save_record(temp_db, record)
    
    dataset = create_dataset(temp_db, batch_size=2)
    
    for x, y in dataset.take(1):
        # Batch size is 2, but we only have 1 record, so it yields 1
        assert x.shape[1:] == (32, 64, 3)
        assert y.shape[0] == 1
