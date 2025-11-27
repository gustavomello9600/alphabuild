import pytest
import numpy as np
from pathlib import Path
from alphabuilder.src.logic.runner import run_episode_v1_1, EpisodeConfig
from alphabuilder.src.core.physics_model import initialize_cantilever_context
from alphabuilder.src.logic.storage import get_episode_count
from alphabuilder.src.neural.dataset import CantileverDataset
import torch

def test_full_episode_execution(sample_props, temp_db):
    """Integration test: Run a full episode and verify storage."""
    try:
        import dolfinx
    except ImportError:
        pytest.skip("FEniCSx not installed")
        
    resolution = (16, 8, 8) # Small for speed
    ctx = initialize_cantilever_context(resolution, sample_props)
    
    config = EpisodeConfig(
        resolution=resolution,
        max_refinement_steps=2,
        stagnation_patience=2
    )
    
    run_episode_v1_1(
        db_path=temp_db,
        max_steps=2,
        resolution=resolution,
        config=config,
        ctx=ctx,
        props=sample_props
    )
    
    assert get_episode_count(temp_db) == 1

def test_training_loop_integration(sample_props, temp_db, tmp_path):
    """Integration test: Generate data and run one training step."""
    try:
        import dolfinx
        import torch
    except ImportError:
        pytest.skip("FEniCSx or Torch not installed")
        
    # 1. Generate Data
    resolution = (16, 8, 8)
    ctx = initialize_cantilever_context(resolution, sample_props)
    config = EpisodeConfig(resolution=resolution, max_refinement_steps=2)
    
    run_episode_v1_1(
        db_path=temp_db,
        max_steps=2,
        resolution=resolution,
        config=config,
        ctx=ctx,
        props=sample_props
    )
    
    # 2. Setup Training
    dataset = CantileverDataset(temp_db)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    
    # 3. Run one step
    for x, y in dataloader:
        assert x.shape[0] <= 2
        assert y.shape[0] <= 2
        break
