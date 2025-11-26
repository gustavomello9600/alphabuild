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
from alphabuilder.src.logic.storage import initialize_database, get_episode_count
from alphabuilder.src.utils.logger import TrainingLogger
from alphabuilder.src.neural.train_v1 import train_epoch
from alphabuilder.src.neural.dataset import CantileverDataset
from torch.utils.data import DataLoader

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
    db_path = Path(args.db_path)
    initialize_database(db_path)
    
    # Initialize Logger
    log_dir = db_path.parent / "logs"
    logger = TrainingLogger(
        log_dir=str(log_dir),
        filename="self_play_log.csv",
        headers=["episode", "duration", "steps", "compliance", "max_disp", "training_loss"]
    )
    
    # Initialize Optimizer for Online Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    existing_count = get_episode_count(db_path)
    episodes_to_run = max(0, args.episodes - existing_count)
    
    print(f"Database has {existing_count} episodes.")
    if episodes_to_run == 0:
        print(f"Target of {args.episodes} episodes already reached. Exiting.")
        return

    print(f"Starting {episodes_to_run} Self-Play Episodes (Target: {args.episodes})...")
    start_time = time.time()
    
    # Offset seed by existing count to ensure variety
    seed_offset = existing_count
    
    for i in range(episodes_to_run):
        current_episode_num = existing_count + i + 1
        print(f"\n--- Episode {current_episode_num}/{args.episodes} ---")
        
        ep_start = time.time()
        
        # 1. Run Episode
        # Note: run_episode_v1_1 returns episode_id (string)
        episode_id = run_episode_v1_1(
            db_path=db_path,
            max_steps=args.steps,
            model=model, # Pass the loaded model!
            resolution=resolution,
            config=config,
            seed=int(time.time()) + seed_offset + i
        )
        
        ep_duration = time.time() - ep_start
        
        # 2. Online Training
        # Train on the data just generated (or all data in DB)
        # For efficiency, we train on the whole DB (Experience Replay) for 1 epoch
        # This ensures we don't forget past experiences
        print("  Running Online Training (1 Epoch)...")
        dataset = CantileverDataset(str(db_path))
        dataloader = DataLoader(dataset, batch_size=4) # Small batch for online updates
        
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"  Training Loss: {avg_loss:.4f}")
        
        # 3. Save Updated Checkpoint
        torch.save(model.state_dict(), args.checkpoint)
        print(f"  Updated model saved to {args.checkpoint}")
        
        # 4. Log Metrics
        # We need to fetch metrics from the last run. 
        # Since run_episode_v1_1 doesn't return metrics directly (only ID), 
        # we could query DB or assume success. 
        # For now, let's log what we have.
        logger.log({
            "episode": current_episode_num,
            "duration": ep_duration,
            "steps": args.steps, # Approximate if max reached
            "compliance": 0.0, # Placeholder - would need DB query to get actual
            "max_disp": 0.0,   # Placeholder
            "training_loss": avg_loss
        })
        
    total_time = time.time() - start_time
    print(f"\nSelf-Play Complete. Total time: {total_time:.1f}s")

if __name__ == "__main__":
    main()
