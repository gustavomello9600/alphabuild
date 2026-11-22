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

from alphabuilder.src.logic.selfplay.storage import (
    initialize_selfplay_db,
    get_incomplete_games,
)
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
    parser.add_argument("--unguided-value", action="store_false", dest="guided_value", default=True,
                       help="Disable guided value (analytical bonuses in search)")
    
    # Resume functionality
    parser.add_argument("--resume", type=str, default=None, metavar="GAME_ID",
                       help="Resume a specific game by ID")
    parser.add_argument("--resume-latest", action="store_true",
                       help="Resume the most recent incomplete game")
    parser.add_argument("--list-incomplete", action="store_true",
                       help="List incomplete games that can be resumed")
    
    return parser.parse_args()


def list_incomplete_games(db_path: Path, max_steps: int):
    """Print a list of incomplete games that can be resumed."""
    games = get_incomplete_games(db_path, max_steps)
    
    if not games:
        print("No incomplete games found.")
        return
    
    print(f"\n{'='*80}")
    print(f"Incomplete Games (can be resumed)")
    print(f"{'='*80}")
    print(f"{'Game ID':<40} {'Steps':<8} {'Score':<10} {'Created At'}")
    print(f"{'-'*80}")
    
    for game in games:
        score_str = f"{game.final_score:.4f}" if game.final_score is not None else "---"
        print(f"{game.game_id:<40} {game.total_steps:<8} {score_str:<10} {game.created_at}")
    
    print(f"\nTo resume a game, use:")
    print(f"  python run_selfplay.py --resume <GAME_ID>")
    print(f"  python run_selfplay.py --resume-latest")


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
    
    # Handle --list-incomplete
    if args.list_incomplete:
        list_incomplete_games(db_path, args.max_steps)
        return
    
    # Episode config
    config = EpisodeConfig(
        max_steps=args.max_steps,
        resolution=resolution,
        use_guided_value=args.guided_value
    )
    
    # Handle resume modes
    if args.resume_latest:
        games = get_incomplete_games(db_path, args.max_steps)
        if not games:
            print("No incomplete games to resume. Starting new episode...")
            args.resume = None
        else:
            args.resume = games[0].game_id
            print(f"Resuming most recent incomplete game: {args.resume}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = AlphaBuilderInference(args.checkpoint, device=args.device)
    print(f"Model loaded: {model}")
    
    # Resume a specific game
    if args.resume:
        game_id, score, steps = resume_episode(model, db_path, args.resume, config)
        
        print(f"\n\n{'='*60}")
        print("Resume Complete!")
        print(f"{'='*60}")
        print(f"Game ID: {game_id}")
        print(f"Final score: {score:.4f}")
        print(f"Total steps: {steps}")
        print(f"Database: {db_path}")
        return
    
    # Run new episodes
    results = []
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

