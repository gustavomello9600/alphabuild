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
import gc

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
    parser.add_argument('--resolution', type=str, default='64x32x8',
                        help='Grid resolution as LxHxD (e.g., 64x32x8 for cantilever beam with 2:1:0.25 ratio)')
    parser.add_argument("--db-path", type=str, default="data/training_data.db", help="Path to SQLite database")
    parser.add_argument("--steps", type=int, default=100, help="Max refinement steps per episode")
    parser.add_argument("--strategy", type=str, default="balanced", choices=["random", "simp", "balanced"], help="Data generation strategy")
    parser.add_argument("--seed-offset", type=int, default=0, help="Manual seed offset for parallel runs")
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
    
    # Use existing count OR manual offset to ensure variety
    seed_offset = args.seed_offset if args.seed_offset > 0 else existing_episodes
    
    from tqdm import tqdm
    episode_pbar = tqdm(range(num_episodes), desc="Total Progress", unit="ep", ncols=100, mininterval=0.5)
    
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
            # Run A* + SIMP Optimization
            # Phase 1: Build connectivity backbone with A*
            # Phase 2: Optimize with SIMP
            from alphabuilder.src.logic.simp_generator import run_simp_optimization_3d, SIMPConfig
            from alphabuilder.src.logic.storage import TrainingRecord, Phase, serialize_state, save_record, generate_episode_id
            from alphabuilder.src.logic.runner_astar import run_episode_astar
            
            # Build A* backbone first
            backbone_state = run_episode_astar(ctx, props, resolution=resolution)
            
            # Extract backbone density for SIMP initialization
            initial_density = backbone_state.density.flatten()
            initial_vol_frac = np.mean(initial_density)
            
            # EXPERT DIRECTIVE: SIMP should remove excess material from dilation
            # Set target volume to be LOWER than initial (e.g., 50-90% of initial)
            # This forces the "refinement" behavior we want to learn
            reduction_factor = np.random.uniform(0.5, 0.9)
            vol_frac = initial_vol_frac * reduction_factor
            
            # Ensure it's not too small (min 10%)
            vol_frac = max(0.1, vol_frac)
            
            print(f"  SIMP Target: Reduce volume from {initial_vol_frac:.2%} to {vol_frac:.2%}")
            
            simp_config = SIMPConfig(
                vol_frac=vol_frac,
                max_iter=50
            )
            
            # Run SIMP starting from A* backbone
            history = run_simp_optimization_3d(
                ctx, props, simp_config, 
                resolution=resolution,
                initial_density=initial_density
            )
            
            # EXPERT DIRECTIVE: Extract discrete actions from SIMP history
            from alphabuilder.src.logic.data_mining import extract_discrete_actions
            
            # Extract samples (State, Policy, Value)
            training_samples = extract_discrete_actions(history, jump_size=5)
            
            episode_id = generate_episode_id()
            
            # Save samples to DB
            for sample in training_samples:
                # Serialize State (Input)
                state_blob = serialize_state(sample['input_state'])
                
                # Serialize Policy (Target)
                policy_blob = serialize_state(sample['target_policy'])
                
                # Calculate fitness (Value)
                # Use the target value from the sample (final compliance)
                # Or compute local fitness? Expert says "target_value" is final compliance.
                # We'll store the final compliance as the score, normalized if needed.
                # For consistency with previous schema, let's use the game fitness formula
                # based on the sample's metadata if available, or just the target value.
                
                # Reconstruct fitness from metadata for consistency
                # But the sample has 'target_value' which is final_compliance.
                # Let's use that, but inverted for "fitness" (higher is better).
                compliance = sample['target_value']
                fitness = 1.0 / compliance if compliance > 1e-9 else 0.0
                
                record = TrainingRecord(
                    episode_id=episode_id,
                    step=sample['step'],
                    phase=Phase.REFINEMENT,
                    state_blob=state_blob,
                    fitness_score=fitness,
                    valid_fem=True,
                    metadata={
                        "compliance": sample['metadata']['compliance'],
                        "max_displacement": sample['metadata']['max_displacement'],
                        "strategy": "astar+simp+mining"
                    },
                    policy_blob=policy_blob # NEW
                )
                save_record(db_path, record)
                
            # print(f"  ✓ A*+SIMP Episode completed. Steps: {len(history)}, Extracted Samples: {len(training_samples)}")
            
            # Log metrics
            logger.log({
                "episode": global_episode_num,
                "duration": time.time() - episode_start,
                "steps": len(history),
                "samples": len(training_samples),
                "strategy": "astar+simp+mining",
                "compliance": history[-1]['compliance'],
                "max_displacement": history[-1]['max_displacement'],
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
        
        # Force GC to prevent OOM
        gc.collect()
    
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
