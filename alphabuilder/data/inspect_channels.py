import sqlite3
import pickle
import numpy as np
import os

db_path = '/home/Gustavo/projects/alphabuild/data/local_test_harvest.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT state_blob FROM training_data LIMIT 1")
row = cursor.fetchone()

if row:
    state_blob = row[0]
    state_tensor = pickle.loads(state_blob)
    
    print(f"Shape: {state_tensor.shape}")
    
    # Channel 0: Density
    rho = state_tensor[0]
    print(f"Channel 0 (Density) - Min: {rho.min()}, Max: {rho.max()}, Unique: {np.unique(rho)[:10]}")
    
    # Channel 1: Mask
    mask = state_tensor[1]
    print(f"Channel 1 (Mask) - Min: {mask.min()}, Max: {mask.max()}, Unique: {np.unique(mask)[:10]}")
    
    # Channel 2,3,4: Forces
    forces = state_tensor[2:5]
    print(f"Channels 2-4 (Forces) - Min: {forces.min()}, Max: {forces.max()}")
    
    # Check if forces are sparse (mostly 0)
    print(f"Forces non-zero count: {np.count_nonzero(forces)}")

conn.close()
