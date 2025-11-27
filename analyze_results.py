import sqlite3
import pandas as pd
import numpy as np
import json
from pathlib import Path

def analyze_logs(log_path):
    print(f"\n=== Training Log Analysis: {log_path} ===")
    try:
        df = pd.read_csv(log_path)
        print(f"Total Steps: {len(df)}")
        print(f"Epochs: {df['epoch'].max() + 1}")
        
        # Loss progression
        initial_loss = df.iloc[0]['loss']
        final_loss = df.iloc[-1]['loss']
        min_loss = df['loss'].min()
        
        print(f"Initial Loss: {initial_loss:.6f}")
        print(f"Final Loss:   {final_loss:.6f}")
        print(f"Min Loss:     {min_loss:.6f}")
        print(f"Reduction:    {(1 - final_loss/initial_loss)*100:.2f}%")
        
        # Check for instability (spikes)
        loss_std = df['loss'].std()
        print(f"Loss Std Dev: {loss_std:.6f}")
        
    except Exception as e:
        print(f"Error analyzing logs: {e}")

def analyze_db(db_path):
    print(f"\n=== Database Analysis: {db_path} ===")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. Basic Counts
        cursor.execute("SELECT COUNT(*) FROM training_data")
        total = cursor.fetchone()[0]
        print(f"Total Records: {total}")
        
        cursor.execute("SELECT COUNT(DISTINCT episode_id) FROM training_data")
        episodes = cursor.fetchone()[0]
        print(f"Total Episodes: {episodes}")
        
        # 2. Validity
        cursor.execute("SELECT COUNT(*) FROM training_data WHERE valid_fem=1")
        valid = cursor.fetchone()[0]
        print(f"Valid FEM Records: {valid} ({valid/total*100:.1f}%)")
        
        # 3. Fitness Stats (Valid only)
        cursor.execute("SELECT fitness_score FROM training_data WHERE valid_fem=1")
        fitness_scores = [x[0] for x in cursor.fetchall()]
        if fitness_scores:
            print(f"Fitness Score Stats:")
            print(f"  Mean: {np.mean(fitness_scores):.4f}")
            print(f"  Max:  {np.max(fitness_scores):.4f}")
            print(f"  Min:  {np.min(fitness_scores):.4f}")
            print(f"  Std:  {np.std(fitness_scores):.4f}")
            
        # 4. Metadata Analysis (Max Displacement)
        # We need to parse JSON for this
        cursor.execute("SELECT metadata FROM training_data WHERE valid_fem=1 LIMIT 1000")
        displacements = []
        compliances = []
        for row in cursor.fetchall():
            try:
                meta = json.loads(row[0])
                if 'max_displacement' in meta:
                    displacements.append(meta['max_displacement'])
                if 'compliance' in meta:
                    compliances.append(meta['compliance'])
            except:
                pass
                
        if displacements:
            print(f"Max Displacement Stats (Sample of 1000):")
            print(f"  Mean: {np.mean(displacements):.4f}")
            print(f"  Max:  {np.max(displacements):.4f}")
            print(f"  Min:  {np.min(displacements):.4f}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error analyzing DB: {e}")

if __name__ == "__main__":
    analyze_logs("logs/vit_run_1/training_log.csv")
    analyze_db("data/training_data.db")
