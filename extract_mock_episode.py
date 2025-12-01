import sqlite3
import pickle
import json
import numpy as np
import random
from pathlib import Path

DB_PATH = "data/training_data.db"
OUTPUT_FILE = "alphabuilder/web/public/mock_episode.json"

def extract_episode():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get the latest episode ID
    cursor.execute("SELECT DISTINCT episode_id FROM training_data ORDER BY rowid DESC LIMIT 1")
    row = cursor.fetchone()
    if not row:
        print("No episodes found!")
        return
    
    episode_id = row[0]
    print(f"Extracting episode: {episode_id}")

    # Get all steps
    cursor.execute("""
        SELECT step, phase, state_blob, fitness_score, valid_fem, metadata, policy_blob 
        FROM training_data 
        WHERE episode_id = ? 
        ORDER BY step
    """, (episode_id,))
    
    steps = []
    rows = cursor.fetchall()
    
    for row in rows:
        step, phase, state_blob, fitness_score, valid_fem, metadata_json, policy_blob = row
        
        # Deserialize state
        tensor = pickle.loads(state_blob)
        # Tensor shape is likely (C, D, H, W) or (C, H, W) depending on 2D/3D
        # Frontend expects (4, D, H, W)
        
        # Deserialize metadata
        metadata = json.loads(metadata_json) if metadata_json else {}
        # Map vol_frac to volume_fraction for frontend
        if 'vol_frac' in metadata:
            metadata['volume_fraction'] = metadata['vol_frac']
        
        # Deserialize policy if present (mock if not)
        policy_heatmap = None
        if policy_blob:
            try:
                policy_tensor = pickle.loads(policy_blob)
                # Policy tensor shape is (2, D, H, W) -> [Add, Remove]
                # Frontend expects { add: number[][][], remove: number[][][] }
                policy_data = {
                    "add": policy_tensor[0].tolist(),
                    "remove": policy_tensor[1].tolist()
                }
            except:
                pass
        
        # Format for frontend
        game_state = {
            "episode_id": episode_id,
            "step": step,
            "phase": phase,
            "tensor": {
                "shape": list(tensor.shape),
                "data": tensor.flatten().tolist() # Flatten for JSON
            },
            "fitness_score": fitness_score,
            "valid_fem": bool(valid_fem),
            "metadata": metadata,
            "value_confidence": fitness_score, # Use fitness as proxy for value confidence
            "policy": policy_data
        }
        steps.append(game_state)

    # Extract load_config from first step metadata if available
    load_config = None
    if steps and 'metadata' in steps[0] and 'load_config' in steps[0]['metadata']:
        load_config = steps[0]['metadata']['load_config']

    # Wrap in MockEpisodeData structure
    output_data = {
        "episode_id": episode_id,
        "load_config": load_config,
        "frames": steps
    }

    # Save to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f)
    
    print(f"Saved {len(steps)} steps to {OUTPUT_FILE}")
    conn.close()

if __name__ == "__main__":
    extract_episode()
