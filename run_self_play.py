import sys
import argparse
import torch
from pathlib import Path
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.logic.runner import run_episode_v1_1, EpisodeConfig
from alphabuilder.src.neural.model_arch import build_model
from alphabuilder.src.logic.storage import initialize_database

def main():
    parser = argparse.ArgumentParser(description="AlphaBuilder Self-Play")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--db-path", type=str, default="data/self_play.db")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--resolution", type=str, default="64x32x32")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    
    # 1. Load Model
    print(f"Loading model from {args.checkpoint}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print("Model loaded successfully.")
    
    # 2. Parse Resolution
    try:
        parts = list(map(int, args.resolution.split('x')))
        if len(parts) == 3:
            resolution = tuple(parts)
        else:
            raise ValueError
    except ValueError:
        print("Invalid resolution. Use LxHxW (e.g. 64x32x32)")
        sys.exit(1)
        
    # 3. Configure Episode
    config = EpisodeConfig(
        resolution=resolution,
        max_refinement_steps=args.steps,
        growth_strategy="smart_heuristic", # Or 'neural' if implemented, but MCTS uses the model
        exploration_strategy="mixed" # Use model predictions + noise
    )
    
    # 4. Run Loop
    print(f"Starting {args.episodes} Self-Play Episodes...")
    initialize_database(Path(args.db_path))
    start_time = time.time()
    
    for i in range(args.episodes):
        print(f"\n--- Episode {i+1}/{args.episodes} ---")
        run_episode_v1_1(
            db_path=Path(args.db_path),
            max_steps=args.steps,
            model=model, # Pass the loaded model!
            resolution=resolution,
            config=config,
            seed=int(time.time()) + i
        )
        
    total_time = time.time() - start_time
    print(f"\nSelf-Play Complete. Total time: {total_time:.1f}s")

if __name__ == "__main__":
    main()
