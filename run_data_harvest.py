#!/usr/bin/env python3
"""
Data Harvest Script for AlphaBuilder Training Data Generation.

Runs episodes using random exploration in Phase 2 and saves
all results to data/training_data.db.
Supports command-line arguments for Colab usage.
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.core.physics_model import (
    initialize_cantilever_context,
    PhysicalProperties
)
from alphabuilder.src.logic.runner import run_episode_v1_1 as run_episode, EpisodeConfig
from alphabuilder.src.logic.storage import initialize_database, get_episode_count
from alphabuilder.src.utils.logger import TrainingLogger


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AlphaBuilder Data Harvest")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to run")
    parser.add_argument("--resolution", type=str, default="64x32x32", help="Mesh resolution (LxHxW), e.g., 64x32x32")
    parser.add_argument("--db-path", type=str, default="data/training_data.db", help="Path to SQLite database")
    parser.add_argument("--steps", type=int, default=100, help="Max refinement steps per episode")
    parser.add_argument("--strategy", type=str, default="balanced", choices=["random", "simp", "balanced"], help="Data generation strategy")
    return parser.parse_args()


def main():
    """Run training episodes and save to database."""
    args = parse_args()
    
    start_time = time.time()
    
    print_header("AlphaBuilder Data Harvest", "═")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    num_episodes = args.episodes
    db_path = Path(args.db_path)
    
    # Parse resolution
    try:
        parts = list(map(int, args.resolution.split('x')))
        if len(parts) == 3:
            resolution = tuple(parts)
        elif len(parts) == 2:
            # Default depth = width/2 or fixed? Let's assume depth=width/2 for 2D input
            w, h = parts
            resolution = (w, h, w//2)
        else:
            raise ValueError
    except ValueError:
        print(f"Error: Invalid resolution format '{args.resolution}'. Use LxHxW (e.g., 64x32x32).")
        sys.exit(1)
    
    print_section("Configuration")
    print(f"  Number of episodes: {num_episodes}")
    print(f"  Database path: {db_path.absolute()}")
    print(f"  Resolution: {resolution[0]}x{resolution[1]}x{resolution[2]} ({resolution[0]*resolution[1]*resolution[2]} cells)")
    print(f"  Max refinement steps: {args.steps}")
    print(f"  Stagnation threshold: 1e-4")
    print(f"  Stagnation patience: 20 steps")
    
    # Initialize database
    print_section("Database Initialization")
    print(f"  Initializing database at {db_path}...")
    db_start = time.time()
    initialize_database(db_path)
    db_time = time.time() - db_start
    existing_episodes = get_episode_count(db_path)
    print(f"  ✓ Database initialized in {db_time:.3f}s")
    print(f"  ✓ Existing episodes in database: {existing_episodes}")
    
    # Initialize Logger
    log_dir = db_path.parent / "logs"
    logger = TrainingLogger(
        log_dir=str(log_dir),
        filename="data_harvest_log.csv",
        headers=["episode", "duration", "steps", "strategy", "compliance", "max_disp", "volume_fraction"]
    )
    
    # Physical properties
    props = PhysicalProperties()
    
    # Initialize FEM context (reused across all episodes)
    print_section("FEM Context Initialization")
    print("  Initializing FEniCSx context...")
    print("  This may take a few seconds on first run...")
    fem_start = time.time()
    ctx = initialize_cantilever_context(resolution, props)
    fem_time = time.time() - fem_start
    print(f"  ✓ FEM context initialized in {fem_time:.3f}s")
    print(f"  ✓ Mesh size: {resolution[0]}x{resolution[1]}x{resolution[2]} = {resolution[0]*resolution[1]*resolution[2]} cells")
    
    print(f"  ✓ Physical properties loaded")
    print(f"    - Young's modulus (solid): {props.E}")
    print(f"    - Young's modulus (void): 1e-6")
    print(f"    - Poisson's ratio: {props.nu}")
    
    # Episode configuration
    config = EpisodeConfig(
        resolution=resolution,
        max_refinement_steps=args.steps,
        stagnation_threshold=1e-4,
        stagnation_patience=20
    )
    
    # Run episodes
    print_header(f"Running {num_episodes} Episodes")
    
    episode_times = []
    
    # Use existing count to offset seed, ensuring variety if run multiple times
    seed_offset = existing_episodes
    
    from tqdm import tqdm
    episode_pbar = tqdm(range(num_episodes), desc="Total Progress", unit="ep")
    
    for i in episode_pbar:
        episode_start = time.time()
        current_seed = seed_offset + i
        
        global_episode_num = existing_episodes + i + 1
        # print(f"\n{'═' * 80}")
        # print(f"Episode {i+1}/{num_episodes} (Global #{global_episode_num}, Seed: {current_seed})")
        # print('═' * 80)
        episode_pbar.set_description(f"Ep {i+1}/{num_episodes} (Global #{global_episode_num})")
        
        # Data Balancing Strategy (30% SIMP, 40% Guided, 30% Random)
        # We use the --strategy arg as a base, but if it's "balanced", we mix.
        # If user specifies "simp" or "random", we force that.
        
        current_strategy = args.strategy
        if args.strategy == "balanced":
            rand_val = np.random.random()
            if rand_val < 0.3:
                current_strategy = "simp"
            elif rand_val < 0.7:
                current_strategy = "guided"
            else:
                current_strategy = "random"
        
        if current_strategy == "simp":
            # Run SIMP Optimization
            from alphabuilder.src.logic.simp_generator import run_simp_optimization, SIMPConfig
            from alphabuilder.src.logic.storage import TrainingRecord, Phase, serialize_state, save_record, generate_episode_id
            
            # Randomize target volume fraction
            vol_frac = np.random.uniform(0.3, 0.7)
            
            simp_config = SIMPConfig(
                vol_frac=vol_frac,
                max_iter=50
            )
            
            # Run SIMP
            from alphabuilder.src.logic.simp_generator import run_simp_optimization_3d
            history = run_simp_optimization_3d(ctx, props, simp_config, resolution=resolution)
            
            episode_id = generate_episode_id()
            
            # Save history to DB
            for frame in history:
                state_blob = serialize_state(frame['binary_map'])
                
                # Calculate fitness using game formula
                omega_mat = np.sum(frame['binary_map'])
                penalty = max(0.0, frame['max_displacement'] - props.disp_limit)
                denominator = omega_mat + props.penalty_alpha * penalty
                fitness = 1.0 / denominator if denominator > 1e-9 else 0.0
                
                record = TrainingRecord(
                    episode_id=episode_id,
                    step=frame['step'],
                    phase=Phase.REFINEMENT, # Treat as refinement steps
                    state_blob=state_blob,
                    fitness_score=fitness,
                    valid_fem=True,
                    metadata={
                        "max_displacement": frame['max_displacement'],
                        "compliance": frame['compliance'],
                        "volume_fraction": omega_mat / (resolution[0]*resolution[1]),
                        "strategy": "simp"
                    }
                )
                save_record(db_path, record)
                
            # print(f"  ✓ SIMP Episode completed. Steps: {len(history)}, Final Disp: {history[-1]['max_displacement']:.4f}")
            
            # Log SIMP metrics
            logger.log({
                "episode": global_episode_num,
                "duration": time.time() - episode_start,
                "steps": len(history),
                "strategy": "simp",
                "compliance": history[-1]['compliance'],
                "max_disp": history[-1]['max_displacement'],
                "volume_fraction": vol_frac
            })
            
        else:
            # Standard Random/Game Episode
            # Configure exploration based on strategy
            episode_config = config # Copy default
            
            if current_strategy == "guided":
                # Guided: Heuristic Growth + Mixed Exploration
                episode_config = EpisodeConfig(
                    resolution=resolution,
                    max_refinement_steps=args.steps,
                    stagnation_threshold=1e-6,
                    stagnation_patience=20,
                    growth_strategy="smart_heuristic",
                    exploration_strategy="mixed"
                )
                # print("  Running Guided Episode (Smart Heuristic + Mixed Exploration)...")
                pass
                
            else: # random
                # Random: Random Pattern + Random Exploration
                episode_config = EpisodeConfig(
                    resolution=resolution,
                    max_refinement_steps=args.steps,
                    stagnation_threshold=1e-6,
                    stagnation_patience=20,
                    growth_strategy="random_pattern",
                    exploration_strategy="random"
                )
                # print("  Running Random Episode (Random Pattern + Random Exploration)...")

            episode_id = run_episode(
                ctx=ctx,
                props=props,
                db_path=db_path,
                config=episode_config,
                seed=current_seed
            )
            
            # Log Standard Episode metrics
            # Since run_episode doesn't return metrics, we log what we know.
            # Ideally run_episode should return a result object.
            # For now, we log basic info.
            logger.log({
                "episode": global_episode_num,
                "duration": time.time() - episode_start,
                "steps": args.steps, # Approximate
                "strategy": current_strategy,
                "compliance": 0.0, # Placeholder
                "max_disp": 0.0,   # Placeholder
                "volume_fraction": 0.0 # Placeholder
            })
        
        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        
        # Calculate statistics
        avg_time = sum(episode_times) / len(episode_times)
        remaining = num_episodes - (i + 1)
        eta_seconds = avg_time * remaining
        eta_minutes = eta_seconds / 60
        
        # print(f"\n  ✓ Episode {i+1} completed in {episode_time:.2f}s")
        # print(f"  ✓ Episode ID: {episode_id}")
        # print(f"  📊 Average time per episode: {avg_time:.2f}s")
        # print(f"  ⏱️  Estimated time remaining: {eta_minutes:.1f} minutes ({eta_seconds:.0f}s)")
        
        # Progress bar
        progress = (i + 1) / num_episodes
        # bar_length = 50
        # filled = int(bar_length * progress)
        # bar = '█' * filled + '░' * (bar_length - filled)
        # print(f"  Progress: [{bar}] {progress*100:.1f}%")
    
    # Final statistics
    total_time = time.time() - start_time
    
    print_header("Data Harvest Complete!", "═")
    
    final_count = get_episode_count(db_path)
    new_episodes = final_count - existing_episodes
    
    print(f"\n📈 Statistics:")
    print(f"  Total episodes in database: {final_count}")
    print(f"  New episodes generated: {new_episodes}")
    print(f"  Database location: {db_path.absolute()}")
    if db_path.exists():
        print(f"  Database size: {db_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    print(f"\n⏱️  Timing:")
    print(f"  Total execution time: {total_time/60:.2f} minutes ({total_time:.1f}s)")
    if episode_times:
        print(f"  Average time per episode: {sum(episode_times)/len(episode_times):.2f}s")
        print(f"  Fastest episode: {min(episode_times):.2f}s")
        print(f"  Slowest episode: {max(episode_times):.2f}s")
    print(f"  FEM initialization: {fem_time:.3f}s")
    print(f"  Database initialization: {db_time:.3f}s")
    
    print(f"\n✓ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 80)


if __name__ == "__main__":
    main()
