import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.logic.runner import run_episode_v1_1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--db-path", type=str, default="data/smoke_test.db")
    args = parser.parse_args()
    
    db_path = Path(args.db_path)
    # Ensure parent dir exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Running {args.episodes} episodes...")
    for i in range(args.episodes):
        print(f"Episode {i+1}")
        run_episode_v1_1(
            db_path=db_path,
            max_steps=5, # Short for smoke test
            resolution=(32, 16, 16) # Small for speed
        )
        
if __name__ == "__main__":
    main()
