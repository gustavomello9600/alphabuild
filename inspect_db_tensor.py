import sqlite3
import pickle
import numpy as np
import sys

DB_PATH = "data/debug_episode.db"

print(f"Connecting to {DB_PATH}...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get the latest episode ID
cursor.execute("SELECT DISTINCT episode_id FROM training_data ORDER BY rowid DESC LIMIT 1")
row = cursor.fetchone()

if not row:
    print("No episodes found!")
    conn.close()
    exit()

latest_episode_id = row[0]
print(f"Inspecting latest episode: {latest_episode_id}")

# Get a random step from this episode
cursor.execute("SELECT state_blob, episode_id, step FROM training_data WHERE episode_id = ? ORDER BY RANDOM() LIMIT 1", (latest_episode_id,))
row = cursor.fetchone()

if not row:
    print("No steps found in episode!")
    conn.close()
    exit()

state_blob, episode_id, step = row
tensor = pickle.loads(state_blob)

# Extract channels
density = tensor[0]
supports = tensor[1]
loads = tensor[3]

print("\n" + "="*40)
print(f"RAW DB TENSOR LOADED (Ep: {episode_id[:8]}..., Step: {step})")
print("="*40)
print(f"Shape: {tensor.shape} (Channels, Dim0, Dim1, Dim2)")
print(f"Dim0 (Length?): {tensor.shape[1]}")
print(f"Dim1 (Height?): {tensor.shape[2]}")
print(f"Dim2 (Width?):  {tensor.shape[3]}")

print("\n--- Supports (Channel 1) ---")
supp_indices = np.where(supports > 0.5)
if len(supp_indices[0]) > 0:
    print(f"Indices: {supp_indices}")
    print(f"Dim0 Range: {supp_indices[0].min()} - {supp_indices[0].max()}")
    print(f"Dim1 Range: {supp_indices[1].min()} - {supp_indices[1].max()}")
    print(f"Dim2 Range: {supp_indices[2].min()} - {supp_indices[2].max()}")
else:
    print("No supports found in this step.")

print("\n--- Loads (Channel 3) ---")
load_indices = np.where(loads != 0)
if len(load_indices[0]) > 0:
    print(f"Indices: {load_indices}")
    print(f"Dim0 Range: {load_indices[0].min()} - {load_indices[0].max()}")
    print(f"Dim1 Range: {load_indices[1].min()} - {load_indices[1].max()}")
    print(f"Dim2 Range: {load_indices[2].min()} - {load_indices[2].max()}")

    print("\n--- Density Statistics ---")
    print(f"Global Mean: {np.mean(density):.4f}")
    print(f"Global Max: {np.max(density):.4f}")
    print(f"Global Min: {np.min(density):.4f}")

    print("\n--- Mass Projection ---")
    print(f"Density Shape: {density.shape}")
    
    # Sum along axes to see profile
    mass_x = np.sum(density, axis=(1, 2)) # Sum over Y, Z -> Profile along X
    mass_y = np.sum(density, axis=(0, 2)) # Sum over X, Z -> Profile along Y
    mass_z = np.sum(density, axis=(0, 1)) # Sum over X, Y -> Profile along Z
    
    print(f"Mass along X (first 10): {mass_x[:10]}")
    print(f"Mass along Y (first 10): {mass_y[:10]}")
    print(f"Mass along Z (all): {mass_z}")
    
    print(f"Peak Z index: {np.argmax(mass_z)}")
    
    print("\n--- Density Slices ---")
    # Assuming density shape (D, H, W) where D=Dim0, H=Dim1, W=Dim2
    # Print slices at Z = 0, 2, 4, 6
    for z in [0, 2, 4, 6]:
        if z >= density.shape[2]:
            continue
            
        slice_data = density[:, :, z] # (Dim0, Dim1)
        
        # Transpose for printing (Y vertical, X horizontal)
        # slice_data is (X, Y). We want Y rows, X cols.
        slice_data = slice_data.T # (Dim1, Dim0)
        
        # Flip Y to have Y=0 at bottom (standard plot)
        slice_data = np.flipud(slice_data)
        
        print(f"\nSlice at Z={z} (Mean: {np.mean(slice_data):.4f})")
        for row in slice_data:
            line = "".join(["#" if val > 0.5 else "." for val in row])
            print(line)
else:
    print("No loads found in this step.")

print("\n" + "="*40)
print("INSTRUCTIONS")
print("="*40)
print("Variables: tensor, density, supports, loads")

conn.close()
