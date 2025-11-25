import pytest
import tensorflow as tf
import numpy as np
import json
from alphabuilder.src.logic.storage import (
    save_record, TrainingRecord, Phase, serialize_state
)
from alphabuilder.src.neural.dataset import create_dataset, data_generator

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
