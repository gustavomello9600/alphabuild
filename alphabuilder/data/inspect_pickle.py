import sqlite3
import pickle
import numpy as np
import os

db_path = '/home/Gustavo/projects/alphabuild/data/local_test_harvest.db'

print(f"Inspecting: {db_path}")
if not os.path.exists(db_path):
    print("File does not exist.")
    exit()

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT state_blob, phase, fitness_score FROM training_data LIMIT 1")
row = cursor.fetchone()

if row:
    state_blob, phase, fitness = row
    try:
        state_tensor = pickle.loads(state_blob)
        print(f"Phase: {phase}")
        print(f"Fitness: {fitness}")
        print(f"State Tensor Type: {type(state_tensor)}")
        if isinstance(state_tensor, np.ndarray):
            print(f"Shape: {state_tensor.shape}")
            print(f"Dtype: {state_tensor.dtype}")
            print(f"Min: {state_tensor.min()}, Max: {state_tensor.max()}")
            
            # Check channels
            if state_tensor.ndim == 4: # Batch, C, D, H, W? Or just C, D, H, W?
                 # Spec says (5, D, H, W) for state.
                 pass
            
            # Check for NaN/Inf
            if np.isnan(state_tensor).any():
                print("WARNING: Tensor contains NaNs")
            if np.isinf(state_tensor).any():
                print("WARNING: Tensor contains Infs")
                
    except Exception as e:
        print(f"Error deserializing: {e}")
else:
    print("No rows found.")

conn.close()
