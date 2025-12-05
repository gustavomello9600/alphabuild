"""
PyTorch Dataset for AlphaBuilder v3.1.

Supports both legacy (v1) and optimized (v2) schemas.
Loads data into RAM for fast access during training.
"""
import sqlite3
import pickle
import zlib
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json

from alphabuilder.src.neural.augmentation import (
    rotate_90_z, flip_y, random_pad_to_target, erosion_attack, 
    load_multiplier, sabotage, saboteur
)


def sparse_decode(indices: np.ndarray, values: np.ndarray, shape: tuple) -> np.ndarray:
    """Reconstruct dense array from sparse COO format."""
    flat = np.zeros(np.prod(shape), dtype=np.float32)
    if len(indices) > 0:
        flat[indices] = values
    return flat.reshape(shape)


def deserialize_sparse(blob: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Deserialize sparse arrays from compressed bytes."""
    data = pickle.loads(zlib.decompress(blob))
    n = data['n']
    indices = np.frombuffer(data['indices'], dtype=np.int32).copy()[:n]
    values = np.frombuffer(data['values'], dtype=np.float32).copy()[:n]
    return indices, values


def deserialize_array(blob: bytes) -> np.ndarray:
    """
    Deserialize numpy array from compressed bytes.
    
    Supports both formats:
    - New: np.save format (Kaggle-compatible, version independent)
    - Old: pickle format (legacy, NumPy version dependent)
    """
    decompressed = zlib.decompress(blob)
    
    # Try np.load format first (starts with numpy magic bytes)
    if decompressed[:6] == b'\x93NUMPY':
        import io
        buffer = io.BytesIO(decompressed)
        return np.load(buffer, allow_pickle=False)
    
    # Fall back to pickle format (legacy)
    return pickle.loads(decompressed)


def deserialize_state_legacy(blob: bytes) -> np.ndarray:
    """Legacy: deserialize from pickle with compression."""
    return deserialize_array(blob)  # Use unified function


def detect_schema_version(db_path: Path) -> str:
    """Detect which schema version the database uses."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM episodes")
        count = cursor.fetchone()[0]
        conn.close()
        return "v2" if count > 0 else "v1"
    except sqlite3.OperationalError:
        conn.close()
        return "v1"


class TopologyDatasetV31(Dataset):
    """
    Dataset for topology optimization training.
    
    Supports both v1 (legacy) and v2 (optimized) schemas.
    Loads all data into RAM for fast access during training.
    """
    
    def __init__(
        self,
        db_path: Path,
        augment: bool = True,
        phase_filter: Optional[str] = None,  # 'GROWTH' or 'REFINEMENT'
        preload_to_ram: bool = True
    ):
        """
        Args:
            db_path: Path to SQLite database
            augment: Whether to apply data augmentation
            phase_filter: Optional filter for specific phase
            preload_to_ram: Whether to load all data into RAM (recommended)
        """
        self.db_path = Path(db_path)
        self.augment = augment
        self.phase_filter = phase_filter
        self.preload_to_ram = preload_to_ram
        
        # Detect schema version
        self.schema_version = detect_schema_version(self.db_path)
        
        # Load index and optionally preload data
        if self.schema_version == "v2":
            self._load_v2()
        else:
            self._load_v1()
    
    def _load_v2(self):
        """Load data from v2 (optimized) schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # First, load all episode info
        cursor.execute("""
            SELECT episode_id, bc_masks_blob, forces_blob, resolution
            FROM episodes
        """)
        episode_rows = cursor.fetchall()
        
        self.episodes: Dict[str, dict] = {}
        for ep_id, bc_blob, forces_blob, resolution_json in episode_rows:
            self.episodes[ep_id] = {
                'bc_masks': deserialize_array(bc_blob),
                'forces': deserialize_array(forces_blob),
                'resolution': tuple(json.loads(resolution_json))
            }
        
        # Then load all records
        query = "SELECT episode_id, step, phase, density_blob, policy_add_blob, policy_remove_blob, fitness_score, is_final_step, is_connected FROM records"
        if self.phase_filter:
            query += f" WHERE phase = '{self.phase_filter}'"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        # Build index and optionally preload
        self.index = []
        self.data = [] if self.preload_to_ram else None
        
        for row in rows:
            ep_id, step, phase, density_blob, add_blob, rem_blob, fitness, is_final, is_conn = row
            
            if ep_id not in self.episodes:
                continue
            
            ep_info = self.episodes[ep_id]
            resolution = ep_info['resolution']
            
            if self.preload_to_ram:
                # Decode everything now
                density = deserialize_array(density_blob)
                
                # Decode sparse policy
                if add_blob:
                    add_idx, add_val = deserialize_sparse(add_blob)
                    policy_add = sparse_decode(add_idx, add_val, resolution)
                else:
                    policy_add = np.zeros(resolution, dtype=np.float32)
                
                if rem_blob:
                    rem_idx, rem_val = deserialize_sparse(rem_blob)
                    policy_remove = sparse_decode(rem_idx, rem_val, resolution)
                else:
                    policy_remove = np.zeros(resolution, dtype=np.float32)
                
                self.data.append({
                    'episode_id': ep_id,
                    'density': density,
                    'policy_add': policy_add,
                    'policy_remove': policy_remove,
                    'value': fitness,
                    'phase': phase,
                    'is_final': bool(is_final),
                    'is_connected': bool(is_conn)
                })
            
            self.index.append({
                'episode_id': ep_id,
                'step': step,
                'phase': phase,
                'value': fitness,
                'is_final': bool(is_final),
                'is_connected': bool(is_conn)
            })
    
    def _load_v1(self):
        """Load data from v1 (legacy) schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT id, episode_id, phase, metadata, fitness_score, state_blob, policy_blob FROM training_data"
        if self.phase_filter:
            query += f" WHERE phase = '{self.phase_filter}'"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        self.index = []
        self.data = [] if self.preload_to_ram else None
        self.episodes = {}  # Empty for v1
        
        for row in rows:
            rec_id, ep_id, phase, meta_json, fitness, state_blob, policy_blob = row
            
            meta = json.loads(meta_json) if meta_json else {}
            is_final = meta.get('is_final_step', False)
            is_conn = meta.get('is_connected', False)
            
            if self.preload_to_ram:
                state = deserialize_state_legacy(state_blob)
                policy = deserialize_state_legacy(policy_blob) if policy_blob else np.zeros((2,) + state.shape[1:], dtype=np.float32)
                
                self.data.append({
                    'state': state,
                    'policy': policy,
                    'value': fitness,
                    'phase': phase,
                    'is_final': is_final,
                    'is_connected': is_conn
                })
            
            self.index.append({
                'record_id': rec_id,
                'episode_id': ep_id,
                'phase': phase,
                'value': fitness,
                'is_final': is_final,
                'is_connected': is_conn,
                'metadata': meta
            })
    
    def __len__(self):
        return len(self.index)
    
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
        info = self.index[idx]
        
        if self.preload_to_ram:
            data = self.data[idx]
            
            if self.schema_version == "v2":
                # Reconstruct full state from density + episode BC/forces
                ep_info = self.episodes[data['episode_id']]
                state = np.concatenate([
                    data['density'][None],
                    ep_info['bc_masks'],
                    ep_info['forces']
                ], axis=0).astype(np.float32)
                policy = np.stack([data['policy_add'], data['policy_remove']], axis=0)
            else:
                state = data['state']
                policy = data['policy']
            
            value = data['value']
            phase = data['phase']
            is_final = data['is_final']
            is_connected = data['is_connected']
        else:
            # Load from DB on demand (slower)
            state, policy, value, phase, is_final, is_connected = self._load_from_db(idx)
        
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
    
    def _load_from_db(self, idx):
        """Load a single record from database (used when not preloaded)."""
        info = self.index[idx]
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if self.schema_version == "v2":
            ep_id = info['episode_id']
            step = info['step']
            
            cursor.execute("""
                SELECT density_blob, policy_add_blob, policy_remove_blob
                FROM records WHERE episode_id = ? AND step = ?
            """, (ep_id, step))
            row = cursor.fetchone()
            
            density = deserialize_array(row[0])
            
            # Load episode info
            cursor.execute("SELECT bc_masks_blob, forces_blob, resolution FROM episodes WHERE episode_id = ?", (ep_id,))
            ep_row = cursor.fetchone()
            bc_masks = deserialize_array(ep_row[0])
            forces = deserialize_array(ep_row[1])
            resolution = tuple(json.loads(ep_row[2]))
            
            state = np.concatenate([density[None], bc_masks, forces], axis=0).astype(np.float32)
            
            if row[1]:
                add_idx, add_val = deserialize_sparse(row[1])
                policy_add = sparse_decode(add_idx, add_val, resolution)
            else:
                policy_add = np.zeros(resolution, dtype=np.float32)
            
            if row[2]:
                rem_idx, rem_val = deserialize_sparse(row[2])
                policy_remove = sparse_decode(rem_idx, rem_val, resolution)
            else:
                policy_remove = np.zeros(resolution, dtype=np.float32)
            
            policy = np.stack([policy_add, policy_remove], axis=0)
        else:
            rec_id = info['record_id']
            cursor.execute("SELECT state_blob, policy_blob FROM training_data WHERE id = ?", (rec_id,))
            row = cursor.fetchone()
            state = deserialize_state_legacy(row[0])
            policy = deserialize_state_legacy(row[1]) if row[1] else np.zeros((2,) + state.shape[1:], dtype=np.float32)
        
        conn.close()
        
        return state, policy, info['value'], info['phase'], info['is_final'], info['is_connected']
    
    def _apply_augmentation(self, state, policy, value, is_final, is_connected):
        """
        Apply augmentation based on spec v3.1 + Aggressive D4 Symmetry + Translation.
        
        ORDER:
        1. Negative sampling (exclusive - only ONE effect, probabilistic)
        2. D4 Symmetry (ALWAYS applies uniformly: 4 rotations × 2 flips)
        3. Random translation (ALWAYS applies: shifts structure within volume)
        
        Combined augmentation ensures:
        - Probability of seeing original orientation: 1/8 (12.5%)
        - Network learns rotation and translation invariance
        - Batches have maximum statistical diversity
        """
        # STEP 1: Negative Sampling (exclusive - only ONE effect)
        if is_final:
            # Condition A: Erosion Attack (100% for final states)
            state, policy, value = erosion_attack(state, policy, value)
        elif is_connected and np.random.random() < 0.05:
            # Condition B: Load Multiplier (5% for connected states)
            state, policy, value = load_multiplier(state, policy, value, k=3.0)
        elif np.random.random() < 0.05:
            # Condition C: Sabotage (5% general)
            state, policy, value = sabotage(state, policy, value)
        elif np.random.random() < 0.10:
            # Condition D: Saboteur (10% general)
            state, policy, value = saboteur(state, policy, value)
        
        # STEP 2: D4 Symmetry - Aggressive (ALWAYS applies uniformly)
        # Applied to BOTH original and negative samples
        D, H = state.shape[1], state.shape[2]
        
        # Uniform rotation: 0°, 90°, 180°, or 270° (if D==H)
        if D == H:
            num_rotations = np.random.randint(0, 4)  # 0, 1, 2, or 3
            for _ in range(num_rotations):
                state, policy = rotate_90_z(state, policy)
        
        # Uniform flip: apply or not (50/50, but now part of D4 group)
        if np.random.random() < 0.5:
            state, policy = flip_y(state, policy)
        
        # STEP 3: Random Padding (ALWAYS applies)
        # Pads to target size (divisible by 32) with random positioning
        # This replaces deterministic padding in the model
        state, policy = random_pad_to_target(state, policy, divisor=32)
        
        return state, policy, value
    
    def get_memory_usage_mb(self) -> float:
        """Estimate RAM usage of preloaded data."""
        if not self.preload_to_ram or not self.data:
            return 0.0
        
        total_bytes = 0
        
        # Episode data (v2)
        for ep_info in self.episodes.values():
            total_bytes += ep_info['bc_masks'].nbytes
            total_bytes += ep_info['forces'].nbytes
        
        # Step data
        for data in self.data:
            if self.schema_version == "v2":
                total_bytes += data['density'].nbytes
                total_bytes += data['policy_add'].nbytes
                total_bytes += data['policy_remove'].nbytes
            else:
                total_bytes += data['state'].nbytes
                total_bytes += data['policy'].nbytes
        
        return total_bytes / (1024 * 1024)
