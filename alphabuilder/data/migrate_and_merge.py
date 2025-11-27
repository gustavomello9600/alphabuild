import sqlite3
import pickle
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.core.tensor_utils import build_input_tensor
from alphabuilder.src.logic.storage import initialize_database, TrainingRecord, Phase, serialize_state, save_record

def migrate_and_merge():
    data_dir = Path('/home/Gustavo/projects/alphabuild/data')
    
    # Source DBs
    sources = [
        data_dir / 'harvest_part1.db',
        data_dir / 'warmup_data.db'
    ]
    
    # Target DB
    target_db = data_dir / 'training_data_unified.db'
    
    print(f"Merging {sources} into {target_db}...")
    
    if target_db.exists():
        print(f"Target {target_db} already exists. Deleting...")
        os.remove(target_db)
        
    initialize_database(target_db)
    
    total_records = 0
    migrated_records = 0
    
    for src in sources:
        if not src.exists():
            print(f"Source {src} not found. Skipping.")
            continue
            
        print(f"Processing {src}...")
        conn = sqlite3.connect(src)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM training_data")
        rows = cursor.fetchall()
        
        # Get column names to map correctly
        col_names = [description[0] for description in cursor.description]
        
        for row in tqdm(rows, desc=f"Migrating {src.name}"):
            record_dict = dict(zip(col_names, row))
            
            try:
                # Deserialize state
                state_blob = record_dict['state_blob']
                state_tensor = pickle.loads(state_blob)
                
                # Check if it needs migration (if shape is 3D instead of 5D)
                if state_tensor.ndim == 3:
                    # It's just density (D, H, W)
                    # Reconstruct 5D tensor
                    # Assuming resolution matches tensor shape
                    resolution = state_tensor.shape
                    new_tensor = build_input_tensor(state_tensor, resolution)
                    new_state_blob = serialize_state(new_tensor)
                elif state_tensor.ndim == 4:
                     # Maybe (1, D, H, W)?
                     if state_tensor.shape[0] == 1:
                         resolution = state_tensor.shape[1:]
                         new_tensor = build_input_tensor(state_tensor[0], resolution)
                         new_state_blob = serialize_state(new_tensor)
                     elif state_tensor.shape[0] == 5:
                         # Already correct
                         new_state_blob = state_blob
                     else:
                         print(f"Unknown 4D shape: {state_tensor.shape}")
                         continue
                elif state_tensor.ndim == 5:
                    # Already correct
                    new_state_blob = state_blob
                else:
                    print(f"Unknown shape: {state_tensor.shape}")
                    continue

                # Handle Policy Blob (Target)
                # If it exists, it might also be just a mask (3D)
                policy_blob = record_dict.get('policy_blob')
                new_policy_blob = None
                if policy_blob:
                    policy_tensor = pickle.loads(policy_blob)
                    if policy_tensor.ndim == 3:
                        # Convert to (2, D, H, W) for Policy Head?
                        # Spec says: Channel 0 = Add, Channel 1 = Remove
                        # The mining logic produced a 'removal_action_map' (1 where removed)
                        # So this maps to Channel 1 (Remove).
                        # Channel 0 (Add) should be zeros for now (SIMP only removes).
                        
                        D, H, W = policy_tensor.shape
                        new_policy = np.zeros((2, D, H, W), dtype=np.float32)
                        new_policy[1] = policy_tensor # Channel 1 is Remove
                        new_policy_blob = serialize_state(new_policy)
                    else:
                        new_policy_blob = policy_blob # Assume correct if not 3D
                
                # Create Record
                # Map phase string to Enum
                phase_str = record_dict['phase']
                phase = Phase.REFINEMENT if 'REFINEMENT' in phase_str else Phase.GROWTH
                
                # Metadata might be JSON string or dict depending on how it was read?
                # SQLite returns string for TEXT columns.
                # save_record expects dict.
                import json
                metadata = record_dict['metadata']
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                new_record = TrainingRecord(
                    episode_id=record_dict['episode_id'],
                    step=record_dict['step'],
                    phase=phase,
                    state_blob=new_state_blob,
                    fitness_score=record_dict['fitness_score'],
                    valid_fem=bool(record_dict['valid_fem']),
                    metadata=metadata,
                    policy_blob=new_policy_blob
                )
                
                save_record(target_db, new_record)
                migrated_records += 1
                
            except Exception as e:
                print(f"Error migrating record {record_dict.get('id')}: {e}")
                
        conn.close()
        total_records += len(rows)
        
    print(f"\nMigration Complete.")
    print(f"Total source records: {total_records}")
    print(f"Migrated records: {migrated_records}")
    print(f"Unified DB: {target_db}")

if __name__ == "__main__":
    migrate_and_merge()
