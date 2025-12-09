
import sqlite3
import zlib
import pickle
import numpy as np
from pathlib import Path
import sys

# Add project root to path to ensure we can find data if needed, 
# though we'll use relative paths for now.

DB_PATH = Path("data/selfplay_games.db")

def deserialize_array(blob: bytes) -> np.ndarray:
    """Deserialize numpy array from compressed bytes."""
    return pickle.loads(zlib.decompress(blob))

def migrate_db():
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return

    print(f"Migrating database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Check if column exists
    cursor.execute("PRAGMA table_info(game_steps)")
    columns = [info[1] for info in cursor.fetchall()]
    
    if "volume_fraction" in columns:
        print("Column 'volume_fraction' already exists. Checking if backfill is needed...")
        # Optional: check if we have 0s that need update? 
        # For now, let's assume if it exists we might need to backfill if they are all 0/null
        # But `ALTER` adds it with default.
    else:
        print("Adding 'volume_fraction' column...")
        try:
            cursor.execute("ALTER TABLE game_steps ADD COLUMN volume_fraction REAL DEFAULT 0.0")
            print("Column added.")
        except sqlite3.OperationalError as e:
            print(f"Error adding column: {e}")
            return

    # 2. Backfill/Update values
    print("Calculating volume fractions from density blobs...")
    
    # Get all steps
    cursor.execute("SELECT id, density_blob FROM game_steps")
    rows = cursor.fetchall()
    
    total_rows = len(rows)
    print(f"Found {total_rows} steps to process.")
    
    updates = []
    
    for i, (row_id, density_blob) in enumerate(rows):
        try:
            density = deserialize_array(density_blob)
            vol_frac = float(np.mean(density))
            updates.append((vol_frac, row_id))
        except Exception as e:
            print(f"Error processing row {row_id}: {e}")
            
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{total_rows}...")

    # Batch update
    print("Executing updates...")
    cursor.executemany("UPDATE game_steps SET volume_fraction = ? WHERE id = ?", updates)
    conn.commit()
    
    print("Migration complete.")
    conn.close()

if __name__ == "__main__":
    migrate_db()
