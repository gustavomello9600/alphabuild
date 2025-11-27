import pytest
import os
import sqlite3
import pickle
import numpy as np
import torch
from pathlib import Path
import sys
import logging

# Configure Logging to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PipelineTest")

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.core.physics_model import initialize_cantilever_context, PhysicalProperties
from alphabuilder.src.logic.simp_generator import run_simp_optimization_3d, SIMPConfig
from alphabuilder.src.logic.data_mining import extract_discrete_actions
from alphabuilder.src.logic.storage import initialize_database, TrainingRecord, Phase, save_record, generate_episode_id, serialize_state
from alphabuilder.src.neural.model_arch import build_model
from alphabuilder.src.neural.trainer import AlphaBuilderTrainer

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "new_episode_test.db"
    initialize_database(db_path)
    logger.info(f"Created temporary database at {db_path}")
    return db_path

def test_scenario_1_new_episode(temp_db):
    logger.info("="*60)
    logger.info("SCENARIO 1: Full Pipeline from New Episode Generation")
    logger.info("="*60)
    
    # 1. Harvest (Generation)
    logger.info("Step 1: Generating Data (Harvest)...")
    resolution = (64, 32, 8) # Using the problematic resolution to verify padding fix
    props = PhysicalProperties()
    ctx = initialize_cantilever_context(resolution, props)
    
    simp_config = SIMPConfig(vol_frac=0.5, max_iter=10) # Short run
    
    logger.info(f"Running SIMP optimization with resolution {resolution}...")
    history = run_simp_optimization_3d(
        ctx, props, simp_config, 
        resolution=resolution,
        initial_density=np.ones(resolution[0]*resolution[1]*resolution[2])
    )
    logger.info(f"SIMP completed with {len(history)} steps.")
    
    # 2. Data Mining
    logger.info("Step 2: Mining Discrete Actions...")
    samples = extract_discrete_actions(history, jump_size=2, resolution=resolution)
    logger.info(f"Extracted {len(samples)} training samples.")
    
    if not samples:
        pytest.fail("No samples extracted from SIMP history.")
        
    # 3. Storage
    logger.info("Step 3: Saving to Database...")
    episode_id = generate_episode_id()
    for sample in samples:
        # Policy needs to be serialized
        policy_blob = serialize_state(sample['target_policy'])
        state_blob = serialize_state(sample['input_state'])
        
        record = TrainingRecord(
            episode_id=episode_id,
            step=sample['step'],
            phase=Phase.REFINEMENT,
            state_blob=state_blob,
            fitness_score=sample['target_value'],
            valid_fem=True,
            metadata=sample['metadata'],
            policy_blob=policy_blob
        )
        save_record(temp_db, record)
    logger.info("Data saved successfully.")
    
    # 4. Training
    logger.info("Step 4: Training Neural Network...")
    model = build_model(input_shape=resolution) # Model handles padding internally
    
    # Initialize Trainer
    trainer = AlphaBuilderTrainer(
        model=model,
        db_path=str(temp_db),
        batch_size=2, # Small batch for test
        device="cpu" # Force CPU for test stability
    )
    
    # Run 1 Epoch
    logger.info("Starting training epoch...")
    metrics = trainer.train_epoch(epoch=1)
    logger.info(f"Training completed. Metrics: {metrics}")
    
    assert metrics['loss'] > 0
    logger.info("SCENARIO 1 COMPLETED SUCCESSFULLY.")

def test_scenario_2_unified_db():
    logger.info("="*60)
    logger.info("SCENARIO 2: Pipeline from Unified Database")
    logger.info("="*60)
    
    db_path = Path('/home/Gustavo/projects/alphabuild/data/training_data_unified.db')
    if not db_path.exists():
        pytest.skip("Unified DB not found at expected path.")
        
    logger.info(f"Using database: {db_path}")
    
    # 1. Load Model
    logger.info("Step 1: Initializing Model...")
    # Note: Model input shape arg is just for reference in current implementation, 
    # SwinUNETR adapts, but we pass expected resolution.
    model = build_model(input_shape=(64, 32, 8)) 
    
    # 2. Initialize Trainer
    logger.info("Step 2: Initializing Trainer...")
    trainer = AlphaBuilderTrainer(
        model=model,
        db_path=str(db_path),
        batch_size=4,
        device="cpu"
    )
    
    # 3. Run Training Step (1 Epoch on subset?)
    # Running full epoch on 8000 samples might take too long for a test.
    # Let's monkeypatch the dataloader or just run a few iterations.
    
    logger.info("Step 3: Running Training (Limited Batches)...")
    
    # We'll manually iterate the trainer's dataloader to limit steps
    model.train()
    optimizer = trainer.optimizer
    
    max_steps = 5
    steps = 0
    
    for batch_idx, (state, policy, value) in enumerate(trainer.dataloader):
        if steps >= max_steps:
            break
            
        logger.info(f"Batch {batch_idx}: Input Shape {state.shape}")
        
        optimizer.zero_grad()
        output = model(state)
        
        p_loss = trainer.policy_criterion(output.policy_logits, policy)
        v_loss = trainer.value_criterion(output.value_pred, value)
        loss = p_loss + 0.5 * v_loss
        
        loss.backward()
        optimizer.step()
        
        logger.info(f"  Loss: {loss.item():.4f} (P={p_loss.item():.4f}, V={v_loss.item():.4f})")
        steps += 1
        
    logger.info(f"Ran {steps} training steps successfully.")
    logger.info("SCENARIO 2 COMPLETED SUCCESSFULLY.")
