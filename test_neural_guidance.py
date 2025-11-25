#!/usr/bin/env python3
"""
Test script for Neural Guidance Integration.
Creates a dummy model and runs one episode to verify the pipeline.
"""

import sys
import os
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from alphabuilder.src.neural.model_arch import build_3d_vit
from alphabuilder.src.neural.trainer import save_model
from alphabuilder.src.logic.runner import run_episode, EpisodeConfig
from alphabuilder.src.core.physics_model import initialize_cantilever_context, PhysicalProperties
from alphabuilder.src.logic.storage import initialize_database

def main():
    print("=== Testing Neural Guidance Integration ===")
    
    # 1. Create Dummy Model
    print("\n1. Creating Dummy ViT Model...")
    model = build_3d_vit()
    
    # Save it
    checkpoint_path = "alphabuilder/checkpoints/test_model.keras"
    save_model(model, checkpoint_path)
    print(f"   Model saved to {checkpoint_path}")
    
    # 2. Setup Environment
    print("\n2. Setting up Environment...")
    db_path = Path("data/test_neural.db")
    if db_path.exists():
        db_path.unlink()
    initialize_database(db_path)
    
    props = PhysicalProperties()
    resolution = (16, 32)
    ctx = initialize_cantilever_context(resolution, props)
    
    # 3. Configure for Neural Guidance
    print("\n3. Configuring Episode...")
    config = EpisodeConfig(
        resolution=resolution,
        max_refinement_steps=5,  # Short run
        use_neural_guidance=True,
        neural_greedy_epsilon=0.0,  # Force neural choice
        exploration_strategy="neural_greedy"
    )
    
    # 4. Run Episode
    print("\n4. Running Episode with Neural Guidance...")
    try:
        episode_id = run_episode(
            ctx=ctx,
            props=props,
            db_path=db_path,
            config=config,
            seed=42,
            model=model
        )
        print(f"\n   Episode {episode_id} completed successfully!")
        print("   ✓ Integration verified.")
        
    except Exception as e:
        print(f"\n   ❌ Episode failed: {e}")
        import traceback
        traceback.print_exc()
        
    # Cleanup
    if db_path.exists():
        db_path.unlink()
    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()

if __name__ == "__main__":
    main()
