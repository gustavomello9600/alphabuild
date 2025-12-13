#!/usr/bin/env python3
"""
Self-Play Runner for AlphaBuilder MCTS.

Runs complete self-play episodes that:
1. Start with EMPTY grid (only boundary conditions)
2. Use MCTS with neural guidance through Phase 1 (Growth) and Phase 2 (Refinement)
3. Record all steps for frontend replay and future training

Key differences from run_data_harvest.py:
- Starts with empty grid, not pre-constructed structure
- Uses MCTS for action selection, not SIMP
- Two-phase gameplay with connectivity-based phase transition
- Records MCTS statistics per step

Reference: specs/mcts_spec.md Sections 7-8
"""

import sys
import argparse
import random
import gc
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.logic.selfplay.storage import initialize_selfplay_db
from alphabuilder.src.neural.inference import AlphaBuilderInference
from alphabuilder.src.logic.selfplay.runner import (
    run_episode,
    resume_episode,
    EpisodeConfig
)

def parse_args():
    parser = argparse.ArgumentParser(description="AlphaBuilder Self-Play")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=600, help="Max steps per episode")
    parser.add_argument("--db-path", type=str, default="data/selfplay_games.db")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--resolution", type=str, default="64x32x8")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, xpu, cuda, cpu, npu")
    parser.add_argument("--resume", type=str, default=None, help="Resume game by ID")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Parse resolution
    resolution = tuple(int(x) for x in args.resolution.split("x"))
    
    # Setup database
    db_path = Path(args.db_path)
    initialize_selfplay_db(db_path)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = AlphaBuilderInference(args.checkpoint, device=args.device)
    print(f"Model loaded: {model}")
    
    # Episode config
    config = EpisodeConfig(
        max_steps=args.max_steps,
        resolution=resolution
    )
    
    # Run episodes
    results = []
    
    # Resume mode
    if args.resume:
        print(f"\n\n{'#'*60}")
        print(f"# Resuming Episode: {args.resume}")
        print(f"{'#'*60}")
        
        game_id, score, steps = resume_episode(model, db_path, args.resume, config)
        results.append((game_id, score, steps))
    else:
        # New episodes
        for i in range(args.episodes):
            print(f"\n\n{'#'*60}")
            print(f"# Episode {i+1}/{args.episodes}")
            print(f"{'#'*60}")
            
            game_id, score, steps = run_episode(model, db_path, config)
            results.append((game_id, score, steps))
            
            gc.collect()
    
    # Summary
    print(f"\n\n{'='*60}")
    print("Self-Play Complete!")
    print(f"{'='*60}")
    print(f"Episodes: {len(results)}")
    if results:
        print(f"Average score: {np.mean([r[1] for r in results]):.4f}")
        print(f"Average steps: {np.mean([r[2] for r in results]):.1f}")
    print(f"Database: {db_path}")


if __name__ == "__main__":
    main()
