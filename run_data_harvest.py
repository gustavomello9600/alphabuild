#!/usr/bin/env python3
"""
Data Harvest Script for AlphaBuilder Training Data Generation.

Runs 50 episodes using random exploration in Phase 2 and saves
all results to data/training_data.db.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.core.physics_model import (
    initialize_cantilever_context,
    PhysicalProperties
)
from alphabuilder.src.logic.runner import run_episode, EpisodeConfig
from alphabuilder.src.logic.storage import initialize_database, get_episode_count


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    width = 80
    print("\n" + char * width)
    print(f"{text:^{width}}")
    print(char * width)


def print_section(text: str):
    """Print a section divider."""
    print(f"\n{'─' * 80}")
    print(f"► {text}")
    print('─' * 80)


def main():
    """Run 50 training episodes and save to database."""
    
    start_time = time.time()
    
    print_header("AlphaBuilder Data Harvest", "═")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration - optimized for data quality
    num_episodes = 100  # Increased from 50 for better dataset diversity
    db_path = Path("data/training_data.db")
    
    print_section("Configuration")
    print(f"  Number of episodes: {num_episodes}")
    print(f"  Database path: {db_path.absolute()}")
    print(f"  Resolution: 32x16 (512 cells)")
    print(f"  Max refinement steps: 50 (optimized for diversity)")
    print(f"  Stagnation threshold: Relative 0.1% or Absolute 1e-11")
    print(f"  Stagnation patience: 10 steps")
    print(f"  Growth strategy: A* pathfinding")
    print(f"  Exploration strategy: Mixed (40% random, 30% remove_weak, 20% add_support, 10% symmetry)")
    
    # Initialize database
    print_section("Database Initialization")
    print(f"  Initializing database at {db_path}...")
    db_start = time.time()
    initialize_database(db_path)
    db_time = time.time() - db_start
    existing_episodes = get_episode_count(db_path)
    print(f"  ✓ Database initialized in {db_time:.3f}s")
    print(f"  ✓ Existing episodes in database: {existing_episodes}")
    
    
    # Physical properties
    props = PhysicalProperties()
    
    # Initialize FEM context (reused across all episodes)
    print_section("FEM Context Initialization")
    print("  Initializing FEniCSx context...")
    print("  This may take a few seconds on first run...")
    resolution = (16, 32)  # (ny, nx) - height x width
    fem_start = time.time()
    ctx = initialize_cantilever_context(resolution, props)
    fem_time = time.time() - fem_start
    print(f"  ✓ FEM context initialized in {fem_time:.3f}s")
    print(f"  ✓ Mesh size: {resolution[1]}x{resolution[0]} = {resolution[0]*resolution[1]} cells")
    print(f"  ✓ Physical properties loaded")
    print(f"    - Young's modulus (solid): {props.E_solid}")
    print(f"    - Young's modulus (void): {props.E_void}")
    print(f"    - Poisson's ratio: {props.nu}")
    
    # Episode configuration - using new optimized defaults
    config = EpisodeConfig(
        resolution=resolution
        # All other parameters use optimized defaults:
        # - max_refinement_steps: 50
        # - use_relative_threshold: True
        # - stagnation_threshold_relative: 0.001 (0.1%)
        # - stagnation_threshold_absolute: 1e-11
        # - stagnation_patience: 10
        # - growth_strategy: "astar"
        # - exploration_strategy: "mixed"
    )
    
    # Run episodes
    print_header(f"Running {num_episodes} Episodes")
    
    episode_times = []
    
    for i in range(num_episodes):
        episode_start = time.time()
        
        print(f"\n{'═' * 80}")
        print(f"Episode {i+1}/{num_episodes} (Seed: {i})")
        print('═' * 80)
        
        episode_id = run_episode(
            ctx=ctx,
            props=props,
            db_path=db_path,
            config=config,
            seed=i  # Use episode number as seed for reproducibility
        )
        
        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        
        # Calculate statistics
        avg_time = sum(episode_times) / len(episode_times)
        remaining = num_episodes - (i + 1)
        eta_seconds = avg_time * remaining
        eta_minutes = eta_seconds / 60
        
        print(f"\n  ✓ Episode {i+1} completed in {episode_time:.2f}s")
        print(f"  ✓ Episode ID: {episode_id}")
        print(f"  📊 Average time per episode: {avg_time:.2f}s")
        print(f"  ⏱️  Estimated time remaining: {eta_minutes:.1f} minutes ({eta_seconds:.0f}s)")
        
        # Progress bar
        progress = (i + 1) / num_episodes
        bar_length = 50
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"  Progress: [{bar}] {progress*100:.1f}%")
    
    # Final statistics
    total_time = time.time() - start_time
    
    print_header("Data Harvest Complete!", "═")
    
    final_count = get_episode_count(db_path)
    new_episodes = final_count - existing_episodes
    
    print(f"\n📈 Statistics:")
    print(f"  Total episodes in database: {final_count}")
    print(f"  New episodes generated: {new_episodes}")
    print(f"  Database location: {db_path.absolute()}")
    print(f"  Database size: {db_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    print(f"\n⏱️  Timing:")
    print(f"  Total execution time: {total_time/60:.2f} minutes ({total_time:.1f}s)")
    print(f"  Average time per episode: {sum(episode_times)/len(episode_times):.2f}s")
    print(f"  Fastest episode: {min(episode_times):.2f}s")
    print(f"  Slowest episode: {max(episode_times):.2f}s")
    print(f"  FEM initialization: {fem_time:.3f}s")
    print(f"  Database initialization: {db_time:.3f}s")
    
    print(f"\n✓ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 80)


if __name__ == "__main__":
    main()
