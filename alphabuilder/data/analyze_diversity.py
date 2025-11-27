import sqlite3
import pickle
import numpy as np
import os
import random
import itertools

db_path = '/home/Gustavo/projects/alphabuild/data/harvest_part1.db'

def calculate_iou(vol1, vol2):
    """Intersection over Union for binary volumes."""
    # Threshold at 0.5 to ensure binary
    v1 = (vol1 > 0.5).astype(bool)
    v2 = (vol2 > 0.5).astype(bool)
    
    intersection = np.logical_and(v1, v2).sum()
    union = np.logical_or(v1, v2).sum()
    
    if union == 0:
        return 0.0
    return intersection / union

def calculate_mse(vol1, vol2):
    """Mean Squared Error for continuous volumes."""
    return np.mean((vol1 - vol2) ** 2)

print(f"Inspecting: {db_path}")
if not os.path.exists(db_path):
    print("File does not exist.")
    exit()

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get list of unique episodes
cursor.execute("SELECT DISTINCT episode_id FROM training_data")
episodes = [row[0] for row in cursor.fetchall()]
print(f"Total unique episodes: {len(episodes)}")

if len(episodes) < 10:
    print("Not enough episodes for 10 unique samples.")
    selected_episodes = episodes
else:
    selected_episodes = random.sample(episodes, 10)

samples = []
print(f"\nSampling {len(selected_episodes)} episodes...")

for ep_id in selected_episodes:
    # Get a random step from this episode (or the last one for final structure)
    # Let's get the LAST step to see the final optimized structure, which should be diverse.
    cursor.execute("SELECT state_blob, step FROM training_data WHERE episode_id=? ORDER BY step DESC LIMIT 1", (ep_id,))
    row = cursor.fetchone()
    if row:
        state_blob, step = row
        try:
            # We know it's just density (64, 32, 8) or similar based on previous check
            state_tensor = pickle.loads(state_blob)
            samples.append({'id': ep_id, 'step': step, 'tensor': state_tensor})
        except Exception as e:
            print(f"Error loading {ep_id}: {e}")

conn.close()

print(f"\nCalculating Similarity Metrics (Pairwise)...")
ious = []
mses = []

# Pairwise comparison
for i in range(len(samples)):
    for j in range(i + 1, len(samples)):
        s1 = samples[i]['tensor']
        s2 = samples[j]['tensor']
        
        # Ensure shapes match
        if s1.shape != s2.shape:
            print(f"Shape mismatch: {s1.shape} vs {s2.shape}")
            continue
            
        iou = calculate_iou(s1, s2)
        mse = calculate_mse(s1, s2)
        
        ious.append(iou)
        mses.append(mse)
        
        # print(f"  {samples[i]['id'][:8]} vs {samples[j]['id'][:8]}: IoU={iou:.4f}, MSE={mse:.4f}")

if ious:
    avg_iou = np.mean(ious)
    min_iou = np.min(ious)
    max_iou = np.max(ious)
    
    avg_mse = np.mean(mses)
    
    print(f"\n=== Diversity Report ===")
    print(f"Number of pairs compared: {len(ious)}")
    print(f"Average IoU: {avg_iou:.4f} (Lower is more diverse)")
    print(f"Min IoU: {min_iou:.4f}")
    print(f"Max IoU: {max_iou:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    
    if avg_iou > 0.9:
        print("\nCRITICAL: High similarity detected. Structures are nearly identical.")
    elif avg_iou > 0.7:
        print("\nWARNING: Moderate similarity. Check if BCs are too restrictive.")
    else:
        print("\nSUCCESS: Good diversity detected.")
        
else:
    print("No comparisons made.")
