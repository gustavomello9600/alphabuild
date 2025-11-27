import sqlite3
import pickle
import numpy as np
import os

db_path = '/home/Gustavo/projects/alphabuild/data/harvest_part1.db'

print(f"Inspecting: {db_path}")
if not os.path.exists(db_path):
    print("File does not exist.")
    exit()

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check row count
cursor.execute("SELECT COUNT(*) FROM training_data")
count = cursor.fetchone()[0]
print(f"Total rows: {count}")

# Check sample
cursor.execute("SELECT episode_id, step, phase, state_blob, fitness_score, valid_fem, policy_blob FROM training_data LIMIT 1")
row = cursor.fetchone()

if row:
    episode_id, step, phase, state_blob, fitness, valid, policy_blob = row
    print(f"Sample Row:")
    print(f"  Episode: {episode_id}")
    print(f"  Step: {step}")
    print(f"  Phase: {phase}")
    print(f"  Fitness: {fitness}")
    print(f"  Valid FEM: {valid}")
    
    # Inspect State Blob
    try:
        state_tensor = pickle.loads(state_blob)
        print(f"  State Tensor Shape: {state_tensor.shape}")
        print(f"  State Tensor Dtype: {state_tensor.dtype}")
        print(f"  State Tensor Min/Max: {state_tensor.min()}/{state_tensor.max()}")
        
        # Check channels
        if state_tensor.shape[0] == 5:
            print("  Channels: 5 (Correct)")
            print(f"  Density (Ch0) Unique: {np.unique(state_tensor[0])[:5]}")
            print(f"  Mask (Ch1) Unique: {np.unique(state_tensor[1])[:5]}")
            print(f"  Forces (Ch2-4) Max: {state_tensor[2:5].max()}")
        else:
            print(f"  WARNING: Incorrect channel count: {state_tensor.shape[0]}")
            
    except Exception as e:
        print(f"  Error deserializing state: {e}")

    # Inspect Policy Blob (if present)
    if policy_blob:
        try:
            policy_tensor = pickle.loads(policy_blob)
            print(f"  Policy Tensor Shape: {policy_tensor.shape}")
            # Spec says (2, D, H, W) for policy head output, but this might be target action mask?
            # Spec 5.2 says: "A ação correta é a diferença entre eles" -> likely a mask or probability map.
        except Exception as e:
            print(f"  Error deserializing policy: {e}")
    else:
        print("  Policy Blob: None")

conn.close()
