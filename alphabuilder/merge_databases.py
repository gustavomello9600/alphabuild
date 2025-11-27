import sqlite3
import argparse
from pathlib import Path
import shutil

def merge_databases(source_paths, target_path):
    target_path = Path(target_path)
    
    # Initialize target if not exists
    if not target_path.exists():
        print(f"Creating target database: {target_path}")
        # Copy the first source as base to preserve schema
        shutil.copy(source_paths[0], target_path)
        start_idx = 1
    else:
        start_idx = 0
        
    conn_target = sqlite3.connect(target_path)
    cursor_target = conn_target.cursor()
    
    total_merged = 0
    
    for src in source_paths[start_idx:]:
        src = Path(src)
        if not src.exists():
            print(f"Skipping missing source: {src}")
            continue
            
        print(f"Merging {src}...")
        
        # Attach source DB
        # Note: We can't use parameterized queries for ATTACH
        conn_target.execute(f"ATTACH DATABASE '{src}' AS src_db")
        
        try:
            # Copy data
            # Ignore duplicates (INSERT OR IGNORE) based on episode_id + step
            cursor_target.execute("""
                INSERT OR IGNORE INTO training_data 
                (episode_id, step, phase, state_blob, fitness_score, valid_fem, metadata, policy_blob, created_at)
                SELECT episode_id, step, phase, state_blob, fitness_score, valid_fem, metadata, policy_blob, created_at
                FROM src_db.training_data
            """)
            
            merged_count = cursor_target.rowcount
            print(f"  Imported {merged_count} records.")
            total_merged += merged_count
            
            conn_target.commit()
            
        except Exception as e:
            print(f"  Error merging {src}: {e}")
        finally:
            conn_target.execute("DETACH DATABASE src_db")
            
    conn_target.close()
    print(f"\nMerge Complete. Total records added: {total_merged}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge SQLite Databases")
    parser.add_argument("sources", nargs="+", help="Source DB files")
    parser.add_argument("--target", required=True, help="Target DB file")
    
    args = parser.parse_args()
    merge_databases(args.sources, args.target)
