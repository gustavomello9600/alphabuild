import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict
import time
from tqdm import tqdm

from .model_arch import AlphaBuilderSwinUNETR
from .dataset import AlphaBuilderDataset

class AlphaBuilderTrainer:
    def __init__(
        self,
        model: AlphaBuilderSwinUNETR,
        db_path: str,
        checkpoint_dir: str = "checkpoints",
        lr: float = 1e-4,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        self.dataset = AlphaBuilderDataset(db_path)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0) # 0 workers for safety
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        # Losses
        # Policy: Cross Entropy or BCE?
        # Policy output is logits for 2 channels (Add, Remove).
        # We can treat it as multi-label or separate BCEs.
        # Spec says: "Entropia Cruzada Ponderada".
        # Since channels are independent (Add vs Remove), BCEWithLogitsLoss is appropriate.
        self.policy_criterion = nn.BCEWithLogitsLoss()
        self.value_criterion = nn.MSELoss()
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        p_loss_total = 0.0
        v_loss_total = 0.0
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, (state, policy, value) in enumerate(pbar):
            state = state.to(self.device)
            policy = policy.to(self.device)
            value = value.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(state)
            
            # Calculate Losses
            # Policy Loss
            p_loss = self.policy_criterion(output.policy_logits, policy)
            
            # Value Loss
            v_loss = self.value_criterion(output.value_pred, value)
            
            # Total Loss (Weighted?)
            loss = p_loss + 0.5 * v_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            p_loss_total += p_loss.item()
            v_loss_total += v_loss.item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "P_Loss": f"{p_loss.item():.4f}", "V_Loss": f"{v_loss.item():.4f}"})
            
        avg_loss = total_loss / len(self.dataloader)
        return {
            "loss": avg_loss,
            "policy_loss": p_loss_total / len(self.dataloader),
            "value_loss": v_loss_total / len(self.dataloader)
        }
        
    def save_checkpoint(self, name: str):
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded checkpoint from {path}")

def train_loop(db_path: str, epochs: int = 10):
    from .model_arch import build_model
    
    # Init Model
    model = build_model()
    
    trainer = AlphaBuilderTrainer(model, db_path)
    
    for epoch in range(1, epochs + 1):
        metrics = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} Metrics: {metrics}")
        
        if epoch % 5 == 0:
            trainer.save_checkpoint(f"epoch_{epoch}")
            
    trainer.save_checkpoint("final")
