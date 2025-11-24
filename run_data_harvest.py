#!/usr/bin/env python3
"""
Data Harvest Script for AlphaBuilder Training Data Generation.

Runs 50 episodes using random exploration in Phase 2 and saves
all results to data/training_data.db.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.core.physics_model import (
    initialize_cantilever_context,
    PhysicalProperties
)
from alphabuilder.src.logic.runner import run_episode, EpisodeConfig
from alphabuilder.src.logic.storage import initialize_database, get_episode_count


def main():
    """Run 50 training episodes and save to database."""
    
    # Configuration
    num_episodes = 50
    db_path = Path("data/training_data.db")
    
    # Initialize database
    print("Initializing database...")
    initialize_database(db_path)
    existing_episodes = get_episode_count(db_path)
    print(f"Database initialized. Existing episodes: {existing_episodes}")
    
    # Initialize FEM context (reused across all episodes)
    print("\nInitializing FEM context...")
    resolution = (32, 16)  # (nx, ny)
    ctx = initialize_cantilever_context(resolution)
    print("FEM context initialized.")
    
    # Physical properties
    props = PhysicalProperties()
    
    # Episode configuration
    config = EpisodeConfig(
        resolution=resolution,
        max_refinement_steps=100,
        stagnation_threshold=1e-4,
        stagnation_patience=20
    )
    
    # Run episodes
    print(f"\nRunning {num_episodes} episodes...")
    print("=" * 60)
    
    for i in range(num_episodes):
        print(f"\nEpisode {i+1}/{num_episodes}")
        episode_id = run_episode(
            ctx=ctx,
            props=props,
            db_path=db_path,
            config=config,
            seed=i  # Use episode number as seed for reproducibility
        )
        
    print("\n" + "=" * 60)
    print("Data harvest complete!")
    
    # Final statistics
    final_count = get_episode_count(db_path)
    print(f"\nTotal episodes in database: {final_count}")
    print(f"Database location: {db_path.absolute()}")


if __name__ == "__main__":
    main()
