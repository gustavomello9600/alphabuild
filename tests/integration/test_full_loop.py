import pytest
import numpy as np
from pathlib import Path
from alphabuilder.src.logic.runner import run_episode, EpisodeConfig
from alphabuilder.src.core.physics_model import initialize_cantilever_context
from alphabuilder.src.neural.train import create_vit_regressor, create_dataset
from alphabuilder.src.logic.storage import get_episode_count

def test_full_episode_execution(sample_props, temp_db):
    """Integration test: Run a full episode and verify storage."""
    try:
        import dolfinx
    except ImportError:
        pytest.skip("FEniCSx not installed")
        
    resolution = (32, 16)
    ctx = initialize_cantilever_context(resolution, sample_props)
    
    config = EpisodeConfig(
        resolution=resolution,
        max_refinement_steps=5,
        stagnation_patience=10
    )
    
    episode_id = run_episode(ctx, sample_props, temp_db, config, seed=42)
    
    assert episode_id is not None
    assert get_episode_count(temp_db) == 1

def test_training_loop_integration(sample_props, temp_db, tmp_path):
    """Integration test: Generate data and run one training step."""
    try:
        import dolfinx
        import tensorflow as tf
    except ImportError:
        pytest.skip("FEniCSx or TensorFlow not installed")
        
    # 1. Generate Data
    resolution = (32, 16)
    ctx = initialize_cantilever_context(resolution, sample_props)
    config = EpisodeConfig(resolution=resolution, max_refinement_steps=5)
    run_episode(ctx, sample_props, temp_db, config, seed=42)
    
    # 2. Setup Training
    model = create_vit_regressor(input_shape=(16, 32, 3))
    model.compile(optimizer='adam', loss='mse')
    
    dataset = create_dataset(temp_db, batch_size=2)
    
    # 3. Run one step
    for x, y in dataset.take(1):
        loss = model.train_on_batch(x, y)
        assert loss >= 0
