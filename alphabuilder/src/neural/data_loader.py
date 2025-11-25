import sqlite3
import pickle
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import List, Tuple

# Constants
MAX_DEPTH = 16
MAX_HEIGHT = 64
MAX_WIDTH = 128
CHANNELS = 3

@dataclass(frozen=True)
class VolumetricInput:
    """
    Immutable container for the input tensor.
    Shape: (Batch, D, H, W, C)
    """
    tensor: tf.Tensor

@dataclass(frozen=True)
class TrainingBatch:
    """
    Pair (Input, Target) for the training loop.
    """
    inputs: VolumetricInput
    targets: tf.Tensor  # Shape: (Batch, 1) -> Fitness Real

def prepare_volumetric_batch(grids_2d: List[np.ndarray], thicknesses: List[int]) -> VolumetricInput:
    """
    Extrudes 2D grids into 3D volumetric tensors.
    
    Args:
        grids_2d: List of 2D numpy arrays (H, W, C) or (H, W).
                  If (H, W), assumes it's the topology channel and others are 0? 
                  Actually blueprint says input is (H, W, 3).
        thicknesses: List of integer thicknesses for each grid.
        
    Returns:
        VolumetricInput with tensor of shape (Batch, MAX_DEPTH, MAX_HEIGHT, MAX_WIDTH, 3)
    """
    batch_size = len(grids_2d)
    # Initialize batch with zeros
    batch_volume = np.zeros((batch_size, MAX_DEPTH, MAX_HEIGHT, MAX_WIDTH, CHANNELS), dtype=np.float32)
    
    for i, (grid, thickness) in enumerate(zip(grids_2d, thicknesses)):
        # Ensure grid matches spatial dimensions
        # Note: We assume grid comes in (H, W, C) or (H, W)
        # If (H, W), we might need to handle channels. 
        # Blueprint says "Tensor de Estado Universal" is (H, W, 3).
        
        h, w = grid.shape[:2]
        
        # Clip thickness to MAX_DEPTH
        t = min(thickness, MAX_DEPTH)
        
        # Place the 2D grid into the first 't' slices of the depth axis
        # We broadcast the (H, W, C) grid to (t, H, W, C)
        # batch_volume[i, :t, :h, :w, :] = grid[None, :h, :w, :] # This assumes grid is (H, W, C)
        
        # Handle potential shape mismatches if grid is smaller than MAX_HEIGHT/WIDTH
        # We slice the destination to match grid size
        dest_h = min(h, MAX_HEIGHT)
        dest_w = min(w, MAX_WIDTH)
        
        # Also handle if grid has no channel dim (though it should)
        if grid.ndim == 2:
             # If just topology, maybe we shouldn't support this based on strict types, 
             # but for robustness:
             # Assume channel 0 is topology, others 0? Or expand?
             # Let's assume strict (H, W, 3) as per blueprint.
             pass
        
        # Copy data
        # We copy to :t depth slices.
        # grid is (H, W, 3). 
        # We assign to batch_volume[i, z, :dest_h, :dest_w, :]
        
        # Handle grid dimensions
        if grid.ndim == 2:
            # Assume it's just topology (Channel 0)
            # Expand to (H, W, 1)
            grid = grid[:, :, np.newaxis]
            
        # If grid has fewer channels than CHANNELS, pad with zeros
        if grid.shape[2] < CHANNELS:
            # Pad with zeros for missing channels
            padding = np.zeros((grid.shape[0], grid.shape[1], CHANNELS - grid.shape[2]), dtype=grid.dtype)
            grid = np.concatenate([grid, padding], axis=2)
            
        # Vectorized assignment for depth
        batch_volume[i, :t, :dest_h, :dest_w, :] = grid[:dest_h, :dest_w, :]
        
    return VolumetricInput(tensor=tf.convert_to_tensor(batch_volume))

def load_training_data(db_path: str, batch_size: int = 32) -> List[TrainingBatch]:
    """
    Loads training data from SQLite database.
    
    Args:
        db_path: Path to the SQLite database.
        batch_size: Size of batches to create.
        
    Returns:
        List of TrainingBatch objects.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query for valid refinement steps with fitness
    # Schema assumed from blueprint: 
    # state_blob (pickle), fitness_score (float)
    # We filter for valid_fem=1 (implied by having a fitness score? blueprint says valid_fem column)
    
    query = """
        SELECT state_blob, fitness_score 
        FROM training_data 
        WHERE valid_fem = 1
    """
    
    # Use generator to avoid loading everything into memory
    
    # We can't easily shuffle everything if we don't load everything.
    # But we can shuffle the query results if we read them all (might still be big?)
    # Or we can read in chunks and shuffle within chunks.
    # For now, let's read all rows (metadata) which is smaller than full blobs if blobs are huge?
    # Actually blobs are in the row.
    # Better to use `ORDER BY RANDOM()` in SQL if we want global shuffle, but that's slow.
    # Or just iterate cursor and shuffle buffer.
    
    # Let's try to fetch all rows first? If blobs are large, this causes OOM.
    # So we should NOT fetchall().
    
    # Use iterator
    cursor.execute(query)
    
    while True:
        chunk = cursor.fetchmany(batch_size)
        if not chunk:
            break
            
        grids = []
        thicknesses = []
        targets = []
        
        for blob, fitness in chunk:
            try:
                grid = pickle.loads(blob)
                thickness = 1 
                grids.append(grid)
                thicknesses.append(thickness)
                targets.append(fitness)
            except Exception as e:
                print(f"Error decoding row: {e}")
                continue
        
        if not grids:
            continue
            
        vol_input = prepare_volumetric_batch(grids, thicknesses)
        target_tensor = tf.convert_to_tensor(np.array(targets, dtype=np.float32).reshape(-1, 1))
        
        yield TrainingBatch(inputs=vol_input, targets=target_tensor)
        
    conn.close()

if __name__ == "__main__":
    # Smoke Test
    print("Running Smoke Test for Data Loader...")
    
    # Mock Data
    mock_grids = [
        np.random.rand(64, 128, 3).astype(np.float32),
        np.random.rand(64, 128, 3).astype(np.float32)
    ]
    mock_thicknesses = [1, 5]
    
    # Pipeline Check
    vol_batch = prepare_volumetric_batch(mock_grids, mock_thicknesses)
    
    print(f"Output Shape: {vol_batch.tensor.shape}")
    
    expected_shape = (2, 16, 64, 128, 3)
    assert vol_batch.tensor.shape == expected_shape, f"Expected {expected_shape}, got {vol_batch.tensor.shape}"
    
    # Check extrusion logic (depth 0 should match input)
    # grid 0 (thickness 1)
    assert np.allclose(vol_batch.tensor[0, 0, :, :, :], mock_grids[0]), "Layer 0 mismatch for grid 0"
    assert np.all(vol_batch.tensor[0, 1:, :, :, :] == 0), "Padding mismatch for grid 0"
    
    # grid 1 (thickness 5)
    assert np.allclose(vol_batch.tensor[1, 4, :, :, :], mock_grids[1]), "Layer 4 mismatch for grid 1"
    assert np.all(vol_batch.tensor[1, 5:, :, :, :] == 0), "Padding mismatch for grid 1"
    
    print("Data Loader Smoke Test Passed!")
