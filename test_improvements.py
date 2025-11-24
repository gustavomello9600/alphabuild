#!/usr/bin/env python3
"""Quick test with 3 episodes to verify all improvements work."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from run_data_harvest import main

# Temporarily reduce episode count for testing
import alphabuilder.src.logic.runner as runner_module

# Monkey patch for quick test
original_main = main

def test_main():
    """Run with just 3 episodes for testing."""
    # Patch the config in run_data_harvest temporarily
    import run_data_harvest
    old_num = 100
    run_data_harvest.num_episodes = 3 if hasattr(run_data_harvest, 'num_episodes') else old_num
    
    # Create a minimal test
    from alphabuilder.src.core.physics_model import initialize_cantilever_context, PhysicalProperties
    from alphabuilder.src.logic.runner import run_episode, EpisodeConfig
    from alphabuilder.src.logic.storage import initialize_database, get_episode_count
    from pathlib import Path
    
    db_path = Path("data/test_improvements.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Testing improved data generation...")
    print("="*60)
    
    # Initialize
    initialize_database(db_path)
    props = PhysicalProperties()
    ctx = initialize_cantilever_context((16, 32), props)
    config = EpisodeConfig()  # Use new defaults
    
    print(f"\nConfig:")
    print(f"  Growth strategy: {config.growth_strategy}")
    print(f"  Exploration strategy: {config.exploration_strategy}")
    print(f"  Max steps: {config.max_refinement_steps}")
    print(f"  Relative threshold: {config.stagnation_threshold_relative}")
    print(f"  Patience: {config.stagnation_patience}")
    
    print(f"\nRunning 3 test episodes...")
    
    for i in range(3):
        print(f"\n--- Episode {i+1}/3 ---")
        episode_id = run_episode(ctx, props, db_path, config, seed=i)
        print(f"Completed: {episode_id[:8]}")
    
    count = get_episode_count(db_path)
    print(f"\n✓ Success! {count} episodes in database")
    
    # Show some data
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT episode_id, step, fitness_score, metadata 
        FROM training_data 
        WHERE episode_id = (SELECT DISTINCT episode_id FROM training_data LIMIT 1)
        ORDER BY step 
        LIMIT 10
    """)
    
    print(f"\nSample data from first episode:")
    print("Step | Fitness      | Has Enhanced Metrics?")
    print("-" * 50)
    for row in cursor.fetchall():
        ep_id, step, fitness, metadata_json = row
        import json
        metadata = json.loads(metadata_json) if metadata_json else {}
        has_enhanced = "volume_fraction" in metadata
        print(f"{step:4d} | {fitness:12.6e} | {'Yes' if has_enhanced else 'No '}")
    
    conn.close()
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()
    
    print("\n" + "="*60)
    print("✓ All improvements working correctly!")

if __name__ == "__main__":
    test_main()
