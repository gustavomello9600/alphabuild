import tensorflow as tf
import sqlite3
import numpy as np
import pickle
import json
from pathlib import Path

def deserialize_state(state_blob):
    """Deserialize numpy array from bytes."""
    return pickle.loads(state_blob)

def get_dataset_shape(db_path):
    """
    Peek at the database to determine the input shape (H, W, 3).
    Returns (H, W, 3) or None if DB is empty/invalid.
    """
    if not Path(db_path).exists():
        return None
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT state_blob FROM training_data LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return None
        
    state = deserialize_state(row[0])
    # State is (H, W) or (H, W, 1)
    H, W = state.shape[:2]
    return (H, W, 3)

def data_generator(db_path):
    """
    Generator function that yields (state, max_displacement) pairs from the database.
    """
    # Allow cross-thread access for tf.data pipeline
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    # Select all valid records from Refinement phase
    # We need metadata to extract max_displacement
    cursor.execute("""
        SELECT state_blob, metadata 
        FROM training_data 
        WHERE phase = 'REFINEMENT' AND valid_fem = 1
        ORDER BY RANDOM()
    """)
    
    while True:
        row = cursor.fetchone()
        if row is None:
            break
            
        state_blob, metadata_json = row
        state = deserialize_state(state_blob)
        metadata = json.loads(metadata_json)
        
        # Target: Max Displacement (Crucial Context #1)
        max_disp = float(metadata.get("max_displacement", 0.0))
        
        # Ensure state is float32
        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=-1)
            
        H, W = state.shape[:2]
        full_state = np.zeros((H, W, 3), dtype=np.float32)
        
        # Ch0: Topology
        full_state[:, :, 0] = state[:, :, 0]
        
        # Ch1: Supports (Left edge)
        full_state[:, 0, 1] = 1.0
        
        # Ch2: Loads (Right edge middle)
        load_y = H // 2
        load_x = W - 1
        full_state[load_y, load_x, 2] = 1.0
        
        yield full_state, max_disp
        
    conn.close()

def create_dataset(db_path, batch_size=32, buffer_size=1000):
    """
    Create a tf.data.Dataset from the SQLite database.
    """
    return tf.data.Dataset.from_generator(
        lambda: data_generator(db_path),
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    ).shuffle(buffer_size).padded_batch(batch_size).prefetch(tf.data.AUTOTUNE)
