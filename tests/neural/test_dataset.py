import pytest
import torch
import numpy as np
import json
import sqlite3
import pickle
from alphabuilder.src.logic.storage import (
    save_record, TrainingRecord, Phase, serialize_state
)
from alphabuilder.src.neural.dataset import AlphaBuilderDataset

def test_cantilever_dataset(temp_db):
    """Test AlphaBuilderDataset iteration."""
    # Create dummy data
    state = np.zeros((5, 64, 32, 32), dtype=np.float32) # 5 channels
    state_blob = serialize_state(state)
    
    policy = np.zeros((2, 64, 32, 32), dtype=np.float32)
    policy_blob = serialize_state(policy)
    
    metadata = {"compliance": 100.0}
    
    record = TrainingRecord(
        episode_id="ep1",
        step=0,
        phase=Phase.REFINEMENT,
        state_blob=state_blob,
        fitness_score=0.5,
        valid_fem=True,
        metadata=metadata,
        policy_blob=policy_blob
    )
    
    save_record(temp_db, record)
    
    # Init Dataset
    dataset = AlphaBuilderDataset(str(temp_db))
    
    # Test __len__
    assert len(dataset) == 1
    
    # Test __iter__
    for state_t, policy_t, value_t in dataset:
        assert isinstance(state_t, torch.Tensor)
        assert isinstance(policy_t, torch.Tensor)
        assert isinstance(value_t, torch.Tensor)
        assert state_t.shape == (5, 64, 32, 32)
        assert policy_t.shape == (2, 64, 32, 32)
        # Value is now log(fitness + epsilon)
        expected_val = np.log(0.5 + 1e-6)
        assert np.isclose(value_t.item(), expected_val, atol=1e-4)
        break

def test_dataset_empty(temp_db):
    """Test empty dataset handling."""
    dataset = AlphaBuilderDataset(str(temp_db))
    assert len(dataset) == 0
    
    count = 0
    for _ in dataset:
        count += 1
    assert count == 0
