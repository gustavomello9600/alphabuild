import pytest
import os
from pathlib import Path
from alphabuilder.src.neural.logger import TrainingLogger

def test_logger_initialization(tmp_path):
    """Test logger creates directory and csv."""
    log_dir = tmp_path / "logs"
    logger = TrainingLogger(log_dir)
    
    assert log_dir.exists()
    assert (log_dir / "training_log.csv").exists()

def test_logger_log_and_plot(tmp_path):
    """Test logging metrics and plotting."""
    log_dir = tmp_path / "logs"
    logger = TrainingLogger(log_dir)
    
    logger.log(step=1, epoch=0, loss=0.5)
    logger.log(step=2, epoch=0, loss=0.4)
    
    # Check CSV content
    with open(log_dir / "training_log.csv", 'r') as f:
        lines = f.readlines()
        assert len(lines) == 3 # Header + 2 rows
        
    # Test plotting
    logger.plot()
    assert (log_dir / "loss_plot.png").exists()
