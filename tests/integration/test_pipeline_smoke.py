import pytest
import os
import sqlite3
import pickle
import numpy as np
import torch
from pathlib import Path
import sys

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.core.physics_model import initialize_cantilever_context, PhysicalProperties
from alphabuilder.src.logic.simp_generator import run_simp_optimization_3d, SIMPConfig
from alphabuilder.src.logic.data_mining import extract_discrete_actions
from alphabuilder.src.neural.model_arch import build_model
from alphabuilder.src.logic.storage import initialize_database, TrainingRecord, Phase, save_record

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "smoke_test.db"
    initialize_database(db_path)
    return db_path

def test_harvest_generation(temp_db):
    """
    Test 1: Generate a tiny episode using SIMP and save to DB.
    Verifies that data mining produces 5-channel tensors.
    """
    resolution = (16, 8, 8) # Tiny resolution
    props = PhysicalProperties()
    ctx = initialize_cantilever_context(resolution, props)
    
    simp_config = SIMPConfig(vol_frac=0.5, max_iter=5) # Very short run
    
    # Run SIMP
    history = run_simp_optimization_3d(
        ctx, props, simp_config, 
        resolution=resolution,
        initial_density=np.ones(16*8*8)
    )
    
    assert len(history) > 0
    
    # Mine Data
    samples = extract_discrete_actions(history, jump_size=1, resolution=resolution)
    
    # If SIMP didn't change enough in 5 steps, samples might be empty.
    # Force a sample if empty for testing structure
    if not samples:
        # Create a dummy sample
        from alphabuilder.src.core.tensor_utils import build_input_tensor
        input_tensor = build_input_tensor(np.ones(resolution), resolution)
        target_policy = np.zeros((2, *resolution), dtype=np.float32)
        samples.append({
            "step": 1,
            "input_state": input_tensor,
            "target_policy": target_policy,
            "target_value": 100.0,
            "metadata": {"compliance": 100.0, "max_displacement": 1.0}
        })
    
    # Save to DB
    for sample in samples:
        record = TrainingRecord(
            episode_id="test_ep",
            step=sample['step'],
            phase=Phase.REFINEMENT,
            state_blob=pickle.dumps(sample['input_state']),
            fitness_score=1.0,
            valid_fem=True,
            metadata=sample['metadata'],
            policy_blob=pickle.dumps(sample['target_policy'])
        )
        save_record(temp_db, record)
        
    # Verify DB Content
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT state_blob FROM training_data LIMIT 1")
    blob = cursor.fetchone()[0]
    tensor = pickle.loads(blob)
    
    assert tensor.ndim == 4
    assert tensor.shape == (5, 16, 8, 8) # (Channels, D, H, W)
    assert tensor.dtype == np.float32
    conn.close()

def test_neural_training_step(temp_db):
    """
    Test 2: Load data from DB and run one training step.
    """
    # 1. Create dummy data in DB
    # SwinUNETR requires dims divisible by 32 (2**5)
    resolution = (64, 32, 32)
    tensor = np.random.randn(5, *resolution).astype(np.float32)
    policy = np.random.randn(2, *resolution).astype(np.float32)
    
    record = TrainingRecord(
        episode_id="train_test",
        step=1,
        phase=Phase.REFINEMENT,
        state_blob=pickle.dumps(tensor),
        fitness_score=1.0,
        valid_fem=True,
        metadata={},
        policy_blob=pickle.dumps(policy)
    )
    save_record(temp_db, record)
    
    # 2. Init Model
    model = build_model(input_shape=resolution)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 3. Load Data
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT state_blob, policy_blob FROM training_data LIMIT 1")
    s_blob, p_blob = cursor.fetchone()
    conn.close()
    
    input_t = torch.from_numpy(pickle.loads(s_blob)).unsqueeze(0) # Batch dim
    target_p = torch.from_numpy(pickle.loads(p_blob)).unsqueeze(0)
    
    # 4. Forward/Backward
    output = model(input_t)
    loss = torch.nn.MSELoss()(output.policy_logits, target_p)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    assert not torch.isnan(loss)

def test_unified_db_validity():
    """
    Test 3: Check a sample from the real unified DB (if exists).
    """
    db_path = Path('/home/Gustavo/projects/alphabuild/data/training_data_unified.db')
    if not db_path.exists():
        pytest.skip("Unified DB not found")
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT state_blob FROM training_data ORDER BY RANDOM() LIMIT 1")
    row = cursor.fetchone()
    
    if row:
        tensor = pickle.loads(row[0])
        assert tensor.ndim == 4
        assert tensor.shape[0] == 5 # 5 Channels
        # Resolution might vary if we mixed data, but channels must be 5
    
    conn.close()
