import sqlite3
import pickle
import json
import argparse
import numpy as np
from pathlib import Path

def extract_episode_to_json(db_path, output_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get the latest episode ID
    cursor.execute("SELECT episode_id FROM training_data ORDER BY id DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        print("No data found")
        return
    
    episode_id = result[0]
    print(f"Extracting episode: {episode_id}")
    
    # Get all steps for this episode
    cursor.execute("""
        SELECT step, phase, state_blob, fitness_score, metadata, policy_blob 
        FROM training_data 
        WHERE episode_id = ? 
        ORDER BY step ASC
    """, (episode_id,))
    
    rows = cursor.fetchall()
    
    frames = []
    
    for row in rows:
        step, phase, state_blob, fitness, metadata_json, policy_blob = row
        
        # Deserialize State (5 channels)
        state_tensor = pickle.loads(state_blob)
        # We only need density (Channel 0) for visualization
        density = state_tensor[0]
        
        # Deserialize Policy if present
        policy = None
        if policy_blob:
            policy_tensor = pickle.loads(policy_blob)
            # Channel 0: Add, Channel 1: Remove
            # Let's just store the max action or something simple?
            # Or just store the raw policy maps for visualization
            policy = {
                "add": policy_tensor[0].tolist(),
                "remove": policy_tensor[1].tolist()
            }
            
        # Metadata
        meta = json.loads(metadata_json) if metadata_json else {}
        
        frames.append({
            "step": step,
            "phase": phase,
            "density": density.tolist(), # 3D array
            "fitness": fitness,
            "compliance": meta.get("compliance", 0),
            "vol_frac": meta.get("vol_frac", 0),
            "policy": policy
        })
        
    # Extract load config from first frame metadata if available
    load_config = None
    if frames:
        first_meta = json.loads(rows[0][4]) if rows[0][4] else {}
        load_config = first_meta.get("load_config")
        
    output_data = {
        "episode_id": episode_id,
        "load_config": load_config,
        "frames": frames
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(output_data, f)
        
    print(f"Saved {len(frames)} frames to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract mock data from DB")
    parser.add_argument("--db-path", type=str, default="data/training_data.db", help="Path to database")
    parser.add_argument("--output", type=str, default="alphabuilder/web/src/data/mock_episode.json", help="Output JSON path")
    args = parser.parse_args()

    extract_episode_to_json(args.db_path, args.output)

if __name__ == "__main__":
    main()
