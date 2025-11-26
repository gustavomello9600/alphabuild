import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.neural.dataset import CantileverDataset
from alphabuilder.src.neural.model_arch import build_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, default="data/smoke_test.db")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/swin_unetr_warmup.pt")
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.db_path}...")
    dataset = CantileverDataset(args.db_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    
    print("Building model...")
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    # Resume from checkpoint if exists
    ckpt_path = Path(args.checkpoint_path)
    if ckpt_path.exists():
        print(f"Resuming from checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print(f"No checkpoint found at {ckpt_path}. Starting from scratch.")
    
    print(f"Starting training for {args.epochs} epochs...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0.0
        steps = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            # data: (B, 5, D, H, W)
            # target: (B, 1)
            
            optimizer.zero_grad()
            output = model(data)
            
            # For Milestone 1, we only care about Value Head (Compliance prediction)
            # Policy head is auxiliary for now
            value_pred = output.value_pred # (B, 1)
            
            loss = criterion(value_pred, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            print(f"Step {steps}: Loss={loss.item():.4f}", end='\r')
            
        avg_loss = total_loss / steps if steps > 0 else 0.0
        print(f"\nEpoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
    print("Training Complete.")
    
    # Save Checkpoint
    ckpt_path = Path(args.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")

if __name__ == "__main__":
    main()
