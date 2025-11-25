import tensorflow as tf
import sqlite3
import numpy as np
import pickle
from pathlib import Path

def deserialize_state(state_blob):
    """Deserialize numpy array from bytes."""
    return pickle.loads(state_blob)

def data_generator(db_path):
    """
    Generator function that yields (state, fitness) pairs from the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Select all valid records from Refinement phase
    cursor.execute("""
        SELECT state_blob, fitness_score 
        FROM training_data 
        WHERE phase = 'REFINEMENT' AND valid_fem = 1
        ORDER BY RANDOM()
    """)
    
    while True:
        row = cursor.fetchone()
        if row is None:
            break
            
        state_blob, fitness = row
        state = deserialize_state(state_blob)
        
        # Ensure state is float32 and normalized if needed
        # State is binary (0, 1), so float32 is fine.
        # Shape is (H, W). ViT expects (H, W, C).
        # We need to add channel dimension if missing.
        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=-1)
            
        # Add extra channels for BCs and Loads if we had them
        # For now, just topology channel.
        # Ideally we should construct the full 3-channel tensor here:
        # Ch0: Topology
        # Ch1: Supports (Fixed x=0)
        # Ch2: Loads (Point at L, H/2)
        
        H, W = state.shape[:2]
        full_state = np.zeros((H, W, 3), dtype=np.float32)
        
        # Ch0: Topology
        full_state[:, :, 0] = state[:, :, 0]
        
        # Ch1: Supports (Left edge)
        full_state[:, 0, 1] = 1.0
        
        # Ch2: Loads (Right edge middle)
        # Assuming 64x32 or 32x16
        load_y = H // 2
        load_x = W - 1
        full_state[load_y, load_x, 2] = 1.0
        
        yield full_state, float(fitness)
        
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
    ).shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
