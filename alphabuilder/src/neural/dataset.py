import torch
import numpy as np
from torch.utils.data import IterableDataset
import sqlite3
import pickle
import json
from pathlib import Path
from typing import Tuple, Iterator

class CantileverDataset(IterableDataset):
    """
    Iterable Dataset that streams 5D tensors from SQLite.
    Hardcoded for Milestone 1: Cantilever Benchmark.
    """
    def __init__(self, db_path: str, max_steps: int = 10000):
        self.db_path = db_path
        self.max_steps = max_steps
        
    def __len__(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM training_data")
            count = cursor.fetchone()[0]
            conn.close()
            return min(count, self.max_steps)
        except:
            return 0
        
    def _deserialize_state(self, state_blob: bytes) -> np.ndarray:
        return pickle.loads(state_blob)

    def _process_record(self, state_blob: bytes, metadata_json: str) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Load Raw Topology (H, W) or (D, H, W)
        raw_state = self._deserialize_state(state_blob)
        metadata = json.loads(metadata_json)
        
        # Ensure 3D (D, H, W)
        if len(raw_state.shape) == 2:
            # Extrude 2D to 3D (Depth=16 for example, or just 1 channel)
            # For Milestone 1, let's assume we are training on 3D data directly 
            # OR extruding 2D data to fit the 3D network.
            # Let's extrude to D=64 to match the network input if needed, 
            # but for efficiency let's assume the network handles D=16 via padding or resizing.
            # Here we just add the dim.
            raw_state = np.expand_dims(raw_state, axis=0) # (1, H, W)
            # Repeat to fill depth? No, sparse is better.
            # Let's pad to (64, 32, 32)
            
        # 2. Create 5-Channel Tensor
        # Shape: (5, D, H, W)
        # Assuming raw_state is (D, H, W) or (1, H, W)
        D, H, W = raw_state.shape
        
        # Target Shape for Network
        TARGET_D, TARGET_H, TARGET_W = 64, 32, 32
        
        tensor = np.zeros((5, TARGET_D, TARGET_H, TARGET_W), dtype=np.float32)
        
        # Fill Ch0: Density
        # Simple center crop or pad
        d_end = min(D, TARGET_D)
        h_end = min(H, TARGET_H)
        w_end = min(W, TARGET_W)
        tensor[0, :d_end, :h_end, :w_end] = raw_state[:d_end, :h_end, :w_end]
        
        # Fill Ch1: Support (Left Wall)
        tensor[1, :, :, 0] = 1.0
        
        # Fill Ch2-4: Load (Tip Load at Right Center)
        # Fx=0, Fy=-1 (Gravity/Down), Fz=0
        # Load at (Right, Middle Height, Middle Depth)
        load_x = w_end - 1
        load_y = h_end // 2
        load_z = d_end // 2
        
        tensor[3, load_z, load_y, load_x] = -1.0 # Fy
        
        # 3. Target: Compliance (or derived fitness)
        # We want to minimize Compliance, so maybe predict Compliance directly.
        compliance = float(metadata.get("compliance", 100.0))
        target = torch.tensor([compliance], dtype=torch.float32)
        
        return torch.from_numpy(tensor), target

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Select all data
        cursor.execute("""
            SELECT state_blob, metadata 
            FROM training_data 
            ORDER BY RANDOM()
        """)
        
        count = 0
        while True:
            row = cursor.fetchone()
            if row is None or count >= self.max_steps:
                break
                
            state_blob, metadata_json = row
            try:
                yield self._process_record(state_blob, metadata_json)
                count += 1
            except Exception as e:
                # Skip malformed records
                continue
                
        conn.close()
