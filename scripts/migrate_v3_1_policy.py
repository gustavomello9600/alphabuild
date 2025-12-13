#!/usr/bin/env python3
"""
Migration Script: Update Policy Targets to v3.1 Spec (Binary Masking + Max-Scaling).

This script iterates over all records in the training database, recalculates the policy targets
using the new logic (Binary Masking + Max-Scaling), and updates the records.
"""

import sys
import sqlite3
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.logic.storage import (
    deserialize_array, 
    serialize_sparse, 
    sparse_encode,
    sparse_decode,
    deserialize_sparse
)
from alphabuilder.src.logic.harvest.processing import generate_refinement_targets

DB_PATH = "data/training_data.db"

def migrate_database():
    db_path = Path(project_root) / DB_PATH
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    print(f"Migrating database at {db_path}...")
    
    conn = sqlite3.connect(db_path)
    # Use row factory to access columns by name if needed, but index is faster for simple iteration
    cursor = conn.cursor()
    
    # Get total count for progress bar
    cursor.execute("SELECT COUNT(*) FROM records WHERE phase = 'REFINEMENT'")
    total_records = cursor.fetchone()[0]
    
    print(f"Found {total_records} refinement records to update.")
    
    # Iterate over all refinement records
    # We need: id, density, episode_id, step (to find next step if needed, but here we just need to re-process policy?)
    # Wait, generate_refinement_targets needs CURRENT and NEXT density.
    # The current record only has CURRENT density.
    # The policy stored in the record was calculated from (Next - Current).
    # We can't easily reconstruct Next density from Current + Policy because of the old masking/clipping.
    #
    # HOWEVER, we can reconstruct the "Raw Difference" if we assume the old policy captured the changes.
    # But the old policy was also masked.
    #
    # BETTER APPROACH:
    # We need to load the entire episode history to have Current and Next densities.
    # Then we can re-run generate_refinement_targets.
    
    # 1. Get all episode IDs
    cursor.execute("SELECT DISTINCT episode_id FROM records WHERE phase = 'REFINEMENT'")
    episode_ids = [row[0] for row in cursor.fetchall()]
    
    print(f"Processing {len(episode_ids)} episodes...")
    
    updated_count = 0
    
    for ep_id in tqdm(episode_ids, desc="Episodes"):
        # Load all steps for this episode, ordered by step
        cursor.execute("""
            SELECT step, density_blob, id 
            FROM records 
            WHERE episode_id = ? AND phase = 'REFINEMENT' 
            ORDER BY step
        """, (ep_id,))
        
        rows = cursor.fetchall()
        if len(rows) < 2:
            continue
            
        # We need to process pairs (t, t+1)
        # The records in DB correspond to 'input_state' at step t.
        # The policy at step t targets the transition to t+1.
        # So we need density at t and density at t+1.
        
        # Note: The 'records' table might not be contiguous if we filtered steps?
        # But run_harvest_data saves a sequence.
        # Let's assume they are the sequence.
        
        for i in range(len(rows) - 1):
            curr_row = rows[i]
            next_row = rows[i+1]
            
            curr_step = curr_row[0]
            next_step = next_row[0]
            
            # Sanity check: are they consecutive in the chain?
            # If we skipped steps in harvest, this might be tricky.
            # But harvest logic says: "Iterate until second to last step... phase2_records.append..."
            # So the saved records should be the chain.
            
            curr_dens = deserialize_array(curr_row[1])
            next_dens = deserialize_array(next_row[1])
            record_id = curr_row[2]
            
            # Binary mask from current density
            # (Assuming density is continuous from SIMP, we threshold at 0.5 for the mask)
            curr_binary = (curr_dens > 0.5).astype(np.float32)
            
            # Generate NEW targets
            target_add, target_remove = generate_refinement_targets(curr_dens, next_dens, curr_binary)
            
            # Encode sparse
            add_indices, add_values = sparse_encode(target_add)
            rem_indices, rem_values = sparse_encode(target_remove)
            
            # Update DB
            cursor.execute("""
                UPDATE records 
                SET policy_add_blob = ?, policy_remove_blob = ?
                WHERE id = ?
            """, (
                serialize_sparse(add_indices, add_values),
                serialize_sparse(rem_indices, rem_values),
                record_id
            ))
            
            updated_count += 1
            
        conn.commit()
        
    conn.close()
    print(f"Migration complete. Updated {updated_count} records.")

if __name__ == "__main__":
    migrate_database()
