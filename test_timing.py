#!/usr/bin/env python3
"""
Quick test to estimate execution time for a single episode.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.core.physics_model import (
    initialize_cantilever_context,
    PhysicalProperties
)
from alphabuilder.src.logic.runner import run_episode, EpisodeConfig
from alphabuilder.src.logic.storage import initialize_database

def main():
    print("=" * 60)
    print("AlphaBuilder - Single Episode Timing Test")
    print("=" * 60)
    
    # Setup
    db_path = Path("data/test_timing.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Initializing database...")
    db_start = time.time()
    initialize_database(db_path)
    db_time = time.time() - db_start
    print(f"   ✓ Database init: {db_time:.3f}s")
    
    print("\n2. Initializing FEM context...")
    resolution = (32, 16)
    fem_start = time.time()
    ctx = initialize_cantilever_context(resolution)
    fem_time = time.time() - fem_start
    print(f"   ✓ FEM context init: {fem_time:.3f}s")
    
    props = PhysicalProperties()
    config = EpisodeConfig(
        resolution=resolution,
        max_refinement_steps=10,  # Reduced for quick test
        stagnation_threshold=1e-4,
        stagnation_patience=5
    )
    
    print("\n3. Running single episode (10 refinement steps)...")
    episode_start = time.time()
    episode_id = run_episode(
        ctx=ctx,
        props=props,
        db_path=db_path,
        config=config,
        seed=0
    )
    episode_time = time.time() - episode_start
    
    print(f"\n   ✓ Episode completed: {episode_time:.2f}s")
    print(f"   ✓ Episode ID: {episode_id[:8]}...")
    
    # Estimate for full run
    print("\n" + "=" * 60)
    print("ESTIMATES FOR FULL RUN (100 refinement steps, 50 episodes):")
    print("=" * 60)
    
    # Assume linear scaling with refinement steps
    estimated_per_episode = (episode_time / 10) * 100
    estimated_total = fem_time + (estimated_per_episode * 50)
    
    print(f"\nEstimated time per episode: {estimated_per_episode:.1f}s")
    print(f"Estimated total time: {estimated_total/60:.1f} minutes")
    print(f"  - FEM initialization (one-time): {fem_time:.1f}s")
    print(f"  - 50 episodes: {(estimated_per_episode * 50)/60:.1f} minutes")
    
    print("\n" + "=" * 60)
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()
        print(f"\n✓ Cleaned up test database")

if __name__ == "__main__":
    main()
