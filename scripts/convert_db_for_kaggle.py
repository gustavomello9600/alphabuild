"""
Convert training database to Kaggle-compatible format.

This script re-exports the SQLite database using a serialization format
that is compatible across NumPy versions (1.x and 2.x).

Run locally before uploading to Kaggle.
"""

import sqlite3
import pickle
import zlib
import numpy as np
from pathlib import Path
import json
import struct

def serialize_array_compatible(arr: np.ndarray) -> bytes:
    """
    Serialize numpy array in a version-independent format.
    
    Format: 
    - 4 bytes: number of dimensions
    - 4 bytes per dim: shape
    - 1 byte: dtype code (0=float32, 1=float64, 2=int32, 3=bool)
    - rest: raw bytes, compressed with zlib
    """
    dtype_map = {
        np.float32: 0,
        np.float64: 1,
        np.int32: 2,
        bool: 3,
        np.bool_: 3,
    }
    
    # Ensure contiguous array
    arr = np.ascontiguousarray(arr.astype(np.float32))
    
    # Header: ndim + shape + dtype
    header = struct.pack('I', arr.ndim)
    for dim in arr.shape:
        header += struct.pack('I', dim)
    header += struct.pack('B', 0)  # float32
    
    # Compress data
    raw_bytes = arr.tobytes()
    compressed = zlib.compress(raw_bytes, level=6)
    
    return header + compressed


def deserialize_array_compatible(blob: bytes) -> np.ndarray:
    """Deserialize array from compatible format."""
    offset = 0
    
    # Read ndim
    ndim = struct.unpack('I', blob[offset:offset+4])[0]
    offset += 4
    
    # Read shape
    shape = []
    for _ in range(ndim):
        shape.append(struct.unpack('I', blob[offset:offset+4])[0])
        offset += 4
    
    # Read dtype
    dtype_code = struct.unpack('B', blob[offset:offset+1])[0]
    offset += 1
    dtype = [np.float32, np.float64, np.int32, np.bool_][dtype_code]
    
    # Decompress data
    compressed = blob[offset:]
    raw_bytes = zlib.decompress(compressed)
    arr = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)
    
    return arr.copy()


def deserialize_old_format(blob: bytes) -> np.ndarray:
    """Deserialize from old pickle format (local NumPy 2.x)."""
    return pickle.loads(zlib.decompress(blob))


def convert_database(input_path: Path, output_path: Path):
    """Convert database to Kaggle-compatible format."""
    
    print(f"Converting: {input_path}")
    print(f"Output: {output_path}")
    
    # Remove output if exists
    if output_path.exists():
        output_path.unlink()
    
    # Copy structure
    input_conn = sqlite3.connect(input_path)
    output_conn = sqlite3.connect(output_path)
    
    input_cursor = input_conn.cursor()
    output_cursor = output_conn.cursor()
    
    # Check schema version
    input_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in input_cursor.fetchall()]
    
    if 'episodes' in tables:
        print("Detected schema v2")
        convert_v2(input_conn, output_conn)
    else:
        print("Detected schema v1")
        convert_v1(input_conn, output_conn)
    
    output_conn.commit()
    output_conn.close()
    input_conn.close()
    
    # Report sizes
    input_size = input_path.stat().st_size / 1024**2
    output_size = output_path.stat().st_size / 1024**2
    print(f"\nInput size: {input_size:.1f} MB")
    print(f"Output size: {output_size:.1f} MB")
    print(f"Ratio: {output_size/input_size:.2f}x")


def convert_v2(input_conn, output_conn):
    """Convert v2 schema database."""
    
    input_cursor = input_conn.cursor()
    output_cursor = output_conn.cursor()
    
    # Create tables
    output_cursor.execute("""
        CREATE TABLE episodes (
            episode_id TEXT PRIMARY KEY,
            bc_masks_blob BLOB,
            forces_blob BLOB,
            resolution TEXT
        )
    """)
    
    output_cursor.execute("""
        CREATE TABLE records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT,
            step INTEGER,
            phase TEXT,
            density_blob BLOB,
            policy_add_blob BLOB,
            policy_remove_blob BLOB,
            fitness_score REAL,
            is_final_step INTEGER,
            is_connected INTEGER
        )
    """)
    
    # Convert episodes
    input_cursor.execute("SELECT episode_id, bc_masks_blob, forces_blob, resolution FROM episodes")
    episodes = input_cursor.fetchall()
    
    print(f"Converting {len(episodes)} episodes...")
    for ep_id, bc_blob, forces_blob, resolution in episodes:
        bc_masks = deserialize_old_format(bc_blob)
        forces = deserialize_old_format(forces_blob)
        
        new_bc = serialize_array_compatible(bc_masks)
        new_forces = serialize_array_compatible(forces)
        
        output_cursor.execute(
            "INSERT INTO episodes VALUES (?, ?, ?, ?)",
            (ep_id, new_bc, new_forces, resolution)
        )
    
    # Convert records
    input_cursor.execute("SELECT COUNT(*) FROM records")
    total_records = input_cursor.fetchone()[0]
    print(f"Converting {total_records} records...")
    
    batch_size = 1000
    offset = 0
    
    while True:
        input_cursor.execute(f"""
            SELECT episode_id, step, phase, density_blob, policy_add_blob, 
                   policy_remove_blob, fitness_score, is_final_step, is_connected
            FROM records
            LIMIT {batch_size} OFFSET {offset}
        """)
        rows = input_cursor.fetchall()
        
        if not rows:
            break
        
        for row in rows:
            ep_id, step, phase, density_blob, add_blob, rem_blob, fitness, is_final, is_conn = row
            
            density = deserialize_old_format(density_blob)
            new_density = serialize_array_compatible(density)
            
            # Sparse blobs need special handling
            new_add = None
            new_rem = None
            
            if add_blob:
                # Sparse format: just re-compress the pickle with new numpy
                try:
                    data = pickle.loads(zlib.decompress(add_blob))
                    # Re-serialize with raw bytes
                    new_add = zlib.compress(pickle.dumps(data, protocol=4), level=6)
                except:
                    new_add = add_blob
            
            if rem_blob:
                try:
                    data = pickle.loads(zlib.decompress(rem_blob))
                    new_rem = zlib.compress(pickle.dumps(data, protocol=4), level=6)
                except:
                    new_rem = rem_blob
            
            output_cursor.execute(
                "INSERT INTO records (episode_id, step, phase, density_blob, policy_add_blob, policy_remove_blob, fitness_score, is_final_step, is_connected) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ep_id, step, phase, new_density, new_add, new_rem, fitness, is_final, is_conn)
            )
        
        offset += batch_size
        print(f"  Processed {min(offset, total_records)}/{total_records} records", end='\r')
    
    print()


def convert_v1(input_conn, output_conn):
    """Convert v1 schema database."""
    
    input_cursor = input_conn.cursor()
    output_cursor = output_conn.cursor()
    
    # Create table
    output_cursor.execute("""
        CREATE TABLE training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT,
            phase TEXT,
            metadata TEXT,
            fitness_score REAL,
            state_blob BLOB,
            policy_blob BLOB
        )
    """)
    
    input_cursor.execute("SELECT COUNT(*) FROM training_data")
    total = input_cursor.fetchone()[0]
    print(f"Converting {total} records...")
    
    batch_size = 500
    offset = 0
    
    while True:
        input_cursor.execute(f"""
            SELECT episode_id, phase, metadata, fitness_score, state_blob, policy_blob
            FROM training_data
            LIMIT {batch_size} OFFSET {offset}
        """)
        rows = input_cursor.fetchall()
        
        if not rows:
            break
        
        for row in rows:
            ep_id, phase, meta, fitness, state_blob, policy_blob = row
            
            state = deserialize_old_format(state_blob)
            new_state = serialize_array_compatible(state)
            
            new_policy = None
            if policy_blob:
                policy = deserialize_old_format(policy_blob)
                new_policy = serialize_array_compatible(policy)
            
            output_cursor.execute(
                "INSERT INTO training_data (episode_id, phase, metadata, fitness_score, state_blob, policy_blob) VALUES (?, ?, ?, ?, ?, ?)",
                (ep_id, phase, meta, fitness, new_state, new_policy)
            )
        
        offset += batch_size
        print(f"  Processed {min(offset, total)}/{total} records", end='\r')
    
    print()


if __name__ == "__main__":
    input_path = Path("data/warm_up_data/warmup_data.db")
    output_path = Path("data/warm_up_data/warmup_data_kaggle.db")
    
    convert_database(input_path, output_path)
    print("\n✅ Conversion complete!")
    print(f"Upload {output_path} to Kaggle.")
