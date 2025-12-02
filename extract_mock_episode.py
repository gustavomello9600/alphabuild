import sqlite3
import pickle
import json
import numpy as np
import random
from pathlib import Path

import argparse

def extract_episode(db_path, output_path, episode_id=None):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if episode_id:
        # Use provided episode_id (can be partial match)
        cursor.execute("SELECT DISTINCT episode_id FROM training_data WHERE episode_id LIKE ? LIMIT 1", (f"{episode_id}%",))
        row = cursor.fetchone()
        if not row:
            print(f"Episode {episode_id} not found!")
            return
        episode_id = row[0]
    else:
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
        policy_data = None
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
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f)
    
    print(f"Saved {len(steps)} steps to {output_path}")
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Extract mock episode from DB")
    parser.add_argument("--db-path", type=str, default="data/training_data.db", help="Path to database")
    parser.add_argument("--output", type=str, default="alphabuilder/web/public/mock_episode.json", help="Output JSON path")
    parser.add_argument("--episode-id", type=str, default=None, help="Specific episode ID (or prefix) to extract")
    args = parser.parse_args()

    extract_episode(args.db_path, args.output, args.episode_id)

if __name__ == "__main__":
    main()
