"""
PyTorch Dataset for AlphaBuilder v3.1.

Loads data from SQLite DB with on-the-fly augmentation.
"""
import sqlite3
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional
import json

from alphabuilder.src.logic.storage import deserialize_state
from alphabuilder.src.neural.augmentation import (
    rotate_90_z, flip_y, erosion_attack, 
    load_multiplier, sabotage, saboteur
)


class TopologyDatasetV31(Dataset):
    """
    Dataset for topology optimization training.
    
    Loads 7-channel states and 2-channel policies from SQLite DB.
    Applies augmentation on-the-fly based on sample metadata.
    """
    
    def __init__(
        self,
        db_path: Path,
        augment: bool = True,
        phase_filter: Optional[str] = None  # 'GROWTH' or 'REFINEMENT'
    ):
        """
        Args:
            db_path: Path to SQLite database
            augment: Whether to apply data augmentation
            phase_filter: Optional filter for specific phase
        """
        self.db_path = Path(db_path)
        self.augment = augment
        self.phase_filter = phase_filter
        
        # Load index of all records
        self._load_index()
    
    def _load_index(self):
        """Load record IDs and metadata from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT id, phase, metadata, fitness_score FROM training_data"
        if self.phase_filter:
            query += f" WHERE phase = '{self.phase_filter}'"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        self.record_ids = []
        self.phases = []
        self.metadata = []
        self.values = []
        
        for row in rows:
            self.record_ids.append(row[0])
            self.phases.append(row[1])
            self.metadata.append(json.loads(row[2]) if row[2] else {})
            self.values.append(row[3])
    
    def __len__(self):
        return len(self.record_ids)
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Returns:
            dict with:
                - state: (7, D, H, W) tensor
                - policy: (2, D, H, W) tensor
                - value: scalar
                - phase: 'GROWTH' or 'REFINEMENT'
                - is_final: bool
                - is_connected: bool
        """
        record_id = self.record_ids[idx]
        phase = self.phases[idx]
        meta = self.metadata[idx]
        value = self.values[idx]
        
        # Load blobs from DB
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT state_blob, policy_blob FROM training_data WHERE id = ?",
            (record_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        state = deserialize_state(row[0])
        policy = deserialize_state(row[1])
        
        # Get metadata flags
        is_final = meta.get('is_final_step', False)
        is_connected = meta.get('is_connected', False)
        
        # Apply augmentation if enabled
        if self.augment:
            state, policy, value = self._apply_augmentation(
                state, policy, value, is_final, is_connected
            )
        
        return {
            'state': torch.from_numpy(state).float(),
            'policy': torch.from_numpy(policy).float(),
            'value': torch.tensor([value], dtype=torch.float32),
            'phase': phase,
            'is_final': is_final,
            'is_connected': is_connected
        }
    
    def _apply_augmentation(self, state, policy, value, is_final, is_connected):
        """
        Apply augmentation based on spec v3.1.
        
        Priority order:
        1. Final step -> Erosion Attack (100%)
        2. Connected -> Load Multiplier (5%)
        3. General -> Sabotage (5%) or Saboteur (10%)
        4. Always: Random flip (50%) - rotation only if D==H
        """
        # Physical augmentation (always, 50% chance)
        if np.random.random() < 0.5:
            # Only rotate if D == H (to preserve shape for batching)
            D, H = state.shape[1], state.shape[2]
            if D == H and np.random.random() < 0.5:
                state, policy = rotate_90_z(state, policy)
            else:
                # Flip always preserves shape
                state, policy = flip_y(state, policy)
        
        # Negative sampling based on conditions
        if is_final:
            # Condition A: Final Step -> Erosion Attack (100%)
            state, policy, value = erosion_attack(state, policy, value)
        elif is_connected and np.random.random() < 0.05:
            # Condition B: Connected -> Load Multiplier (5%)
            state, policy, value = load_multiplier(state, policy, value, k=3.0)
        elif np.random.random() < 0.05:
            # Condition C: Sabotage (5%)
            state, policy, value = sabotage(state, policy, value)
        elif np.random.random() < 0.10:
            # Condition D: Saboteur (10%)
            state, policy, value = saboteur(state, policy, value)
        
        return state, policy, value

