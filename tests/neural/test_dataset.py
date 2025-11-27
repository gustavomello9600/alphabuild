import pytest
import torch
import numpy as np
import json
import sqlite3
import pickle
from alphabuilder.src.logic.storage import (
    save_record, TrainingRecord, Phase, serialize_state
)
from alphabuilder.src.neural.dataset import CantileverDataset

def test_cantilever_dataset(temp_db):
    """Test CantileverDataset iteration."""
    # Create dummy data
    state = np.zeros((32, 64, 64), dtype=np.int32) # 3D
    blob = serialize_state(state)
    
    metadata = {"compliance": 100.0}
    
    record = TrainingRecord(
        episode_id="ep1",
        step=0,
        phase=Phase.REFINEMENT,
        state_blob=blob,
        fitness_score=0.5,
        valid_fem=True,
        metadata=metadata
    )
    
    save_record(temp_db, record)
    
    # Init Dataset
    dataset = CantileverDataset(temp_db)
    
    # Test __len__
    assert len(dataset) == 1
    
    # Test __iter__
    for x, y in dataset:
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (5, 64, 32, 32) # Target shape
        assert y.item() == 100.0
        break

def test_dataset_empty(temp_db):
    """Test empty dataset handling."""
    dataset = CantileverDataset(temp_db)
    assert len(dataset) == 0
    
    count = 0
    for _ in dataset:
        count += 1
    assert count == 0
