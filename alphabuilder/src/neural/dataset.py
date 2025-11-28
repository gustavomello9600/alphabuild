import torch
from torch.utils.data import Dataset
import sqlite3
import pickle
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import random

class PhysicsAugment:
    """Aplica transformações consistentes na Geometria E nos Vetores de Força."""
    
    def __call__(self, state: torch.Tensor, policy: torch.Tensor):
        # state: (5, D, H, W)
        # policy: (2, D, H, W)
        
        # 1. Random Flip X (Axis 1 - Depth/Length)
        if random.random() > 0.5:
            state = torch.flip(state, dims=[1])
            policy = torch.flip(policy, dims=[1])
            # Se inverter eixo X, força X inverte sinal (Channel 2 is Fx)
            state[2, ...] = -state[2, ...] 
            
        # 2. Random Flip Y (Axis 2 - Height)
        if random.random() > 0.5:
            state = torch.flip(state, dims=[2])
            policy = torch.flip(policy, dims=[2])
            # Se inverter eixo Y, força Y inverte sinal (Channel 3 is Fy)
            state[3, ...] = -state[3, ...]

        # 3. Random Flip Z (Axis 3 - Width)
        if random.random() > 0.5:
            state = torch.flip(state, dims=[3])
            policy = torch.flip(policy, dims=[3])
            # Se inverter eixo Z, força Z inverte sinal (Channel 4 is Fz)
            state[4, ...] = -state[4, ...]

        # Nota: Rotações de 90 graus são mais complexas para os vetores. 
        # Para Warm-up inicial, os Flips são suficientes para cobrir os 8 quadrantes.
        
        return state, policy

class AlphaBuilderDataset(Dataset):
    """
    PyTorch Dataset for AlphaBuilder training data.
    Loads (State, Policy, Value) tuples from SQLite.
    """
    def __init__(self, db_path: str, transform=None, augment: bool = False):
        self.db_path = db_path
        self.transform = transform
        self.augment = augment
        self.augmentor = PhysicsAugment() if augment else None
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
        # value = np.array([fitness], dtype=np.float32) # (1,) -> REMOVED
        
        # Convert to Tensor
        state_t = torch.from_numpy(state)
        policy_t = torch.from_numpy(policy)
        
        # Normalização Logarítmica do Value Target (Supondo Fitness = 1/Compliance)
        # Se Fitness for alto (bom), log(Fitness) é alto.
        # Target deve ser positivo para uma boa estrutura.
        epsilon = 1e-6
        # fitness comes from DB as float, convert to tensor
        value_t = torch.log(torch.tensor([fitness], dtype=torch.float32) + epsilon)

        # Aplicar Augmentation se estiver em modo de treino
        if self.augment and self.augmentor:
            state_t, policy_t = self.augmentor(state_t, policy_t)
            
        if self.transform:
            # Apply transforms if any
            pass
            
        return state_t, policy_t, value_t
