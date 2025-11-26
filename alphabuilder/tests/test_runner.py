import pytest
import numpy as np
from pathlib import Path
from alphabuilder.src.logic.runner import run_episode_v1_1

def test_run_episode_integration(tmp_path):
    # Use a temporary directory for the database
    db_path = tmp_path / "test_training.db"
    
    # Run a very short episode with small resolution
    # Resolution (16, 8, 8) is small enough for fast testing
    run_episode_v1_1(
        db_path=db_path,
        max_steps=2,
        model=None, # Use random agent (MCTS with uniform prior)
        resolution=(16, 8, 8)
    )
    
    # Verify DB was created
    assert db_path.exists()
