"""
Analyze Dataset Quality

Deep dive into the generated training data to verify:
1. Physical consistency (Displacement, Compliance)
2. Policy quality (Action masks)
3. Training suitability
"""

import sqlite3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

DB_PATH = "data/test_mining_final.db"

def analyze_dataset():
    print(f"Connecting to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get total count
    cursor.execute("SELECT count(*) FROM training_data")
    total_records = cursor.fetchone()[0]
    print(f"Total Records: {total_records}")
    
    if total_records == 0:
        print("Dataset is empty!")
        return

    # Fetch all records
    cursor.execute("""
        SELECT step, state_blob, policy_blob, fitness_score, metadata 
        FROM training_data 
        ORDER BY step ASC
    """)
    rows = cursor.fetchall()
    
    steps = []
    compliances = []
    max_disps = []
    removed_voxels_counts = []
    volume_fractions = []
    
    print("\nProcessing records...")
    
    for i, row in enumerate(rows):
        step, state_blob, policy_blob, fitness, metadata_json = row
        
        # Deserialize
        state = pickle.loads(state_blob)
        policy = pickle.loads(policy_blob) if policy_blob else None
        metadata = json.loads(metadata_json) if metadata_json else {}
        
        # Extract metrics
        comp = metadata.get('compliance', 0.0)
        disp = metadata.get('max_displacement', 0.0)
        
        # Calculate volume fraction of state
        vol_frac = np.mean(state)
        
        # Analyze Policy
        if policy is not None:
            removed_count = np.sum(policy)
            removed_voxels_counts.append(removed_count)
            
            # Sanity check: Policy should only remove existing voxels
            # policy=1 means REMOVE. state=1 means EXIST.
            # So policy should be subset of state?
            # Let's check overlap
            overlap = np.sum((policy == 1) & (state == 1))
            if overlap != removed_count:
                print(f"⚠️ Warning at step {step}: Policy removes {removed_count} voxels, but only {overlap} exist in state!")
        else:
            removed_voxels_counts.append(0)
            
        steps.append(step)
        compliances.append(comp)
        max_disps.append(disp)
        volume_fractions.append(vol_frac)
        
        if i % 10 == 0:
            print(f"Step {step}: C={comp:.2f}, Disp={disp:.2f}, Vol={vol_frac:.2%}, Removed={removed_voxels_counts[-1]}")

    conn.close()
    
    # --- Analysis ---
    
    print("\n" + "="*40)
    print("DATASET QUALITY REPORT")
    print("="*40)
    
    # 1. Physical Consistency
    print("\n1. Physical Consistency (Displacement & Compliance)")
    print(f"  Max Displacement Range: {min(max_disps):.4f} - {max(max_disps):.4f}")
    print(f"  Compliance Range:       {min(compliances):.4f} - {max(compliances):.4f}")
    
    # Check for explosion
    if max(max_disps) > 1e6:
        print("  ⚠️ WARNING: Extremely high displacement detected! (> 1e6)")
        print("     This might indicate disconnected structures or singular matrix issues.")
    elif max(max_disps) > 1000:
        print("  ℹ️ Note: High displacement (> 1000). Typical for soft/thin structures.")
    else:
        print("  ✅ Displacement values are within reasonable range.")

    # 2. Policy Quality
    print("\n2. Policy Quality (Action Masks)")
    avg_removal = np.mean(removed_voxels_counts)
    print(f"  Avg Voxels Removed per Step: {avg_removal:.2f}")
    print(f"  Total Voxels Removed:        {sum(removed_voxels_counts)}")
    
    if avg_removal < 1:
        print("  ⚠️ WARNING: Policy is very sparse (removing < 1 voxel/step).")
        print("     Network might struggle to learn meaningful actions.")
    elif avg_removal > 1000:
         print("  ℹ️ Note: Aggressive removal (> 1000 voxels/step).")
    else:
        print("  ✅ Policy granularity seems appropriate.")

    # 3. Volume Evolution
    print("\n3. Volume Evolution")
    print(f"  Initial Volume: {volume_fractions[0]:.2%}")
    print(f"  Final Volume:   {volume_fractions[-1]:.2%}")
    print(f"  Delta:          {volume_fractions[-1] - volume_fractions[0]:.2%}")
    
    if volume_fractions[-1] < volume_fractions[0]:
        print("  ✅ Volume decreased (Material Removal confirmed).")
    else:
        print("  ⚠️ WARNING: Volume did not decrease! SIMP might be adding material?")

    # 4. Training Suitability Verdict
    print("\n4. Verdict")
    if max(max_disps) < 1e8 and avg_removal > 0 and volume_fractions[-1] < volume_fractions[0]:
        print("  🟢 DATASET IS GOOD FOR TRAINING")
        print("     - Physics are stable")
        print("     - Actions are meaningful")
        print("     - Refinement behavior is captured")
    else:
        print("  🔴 DATASET REQUIRES INSPECTION")
        print("     Check warnings above.")

if __name__ == "__main__":
    analyze_dataset()
