import torch
from torch.utils.data import Dataset
import sqlite3
import pickle
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class AlphaBuilderDataset(Dataset):
    """
    PyTorch Dataset for AlphaBuilder training data.
    Loads (State, Policy, Value) tuples from SQLite.
    """
    def __init__(self, db_path: str, transform=None):
        self.db_path = db_path
        self.transform = transform
        self.indices = self._index_db()
        
    def _index_db(self):
        """Get list of valid row IDs."""
        if not Path(self.db_path).exists():
            return []
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Only select rows with valid FEM and policy
        cursor.execute("SELECT rowid FROM training_data WHERE valid_fem=1 AND policy_blob IS NOT NULL")
        indices = [row[0] for row in cursor.fetchall()]
        conn.close()
        return indices
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        rowid = self.indices[idx]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT state_blob, policy_blob, fitness_score FROM training_data WHERE rowid=?", (rowid,))
        row = cursor.fetchone()
        conn.close()
        
        state_blob, policy_blob, fitness = row
        
        # Deserialize
        state = pickle.loads(state_blob).astype(np.float32) # (5, D, H, W)
        policy = pickle.loads(policy_blob).astype(np.float32) # (2, D, H, W)
        value = np.array([fitness], dtype=np.float32) # (1,)
        
        # Convert to Tensor
        state_t = torch.from_numpy(state)
        policy_t = torch.from_numpy(policy)
        value_t = torch.from_numpy(value)
        
        if self.transform:
            # Apply transforms if any
            pass
            
        return state_t, policy_t, value_t
