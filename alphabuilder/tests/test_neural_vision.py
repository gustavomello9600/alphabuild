import pytest
import numpy as np
import tensorflow as tf
from alphabuilder.src.neural.data_loader import prepare_volumetric_batch, load_training_data, VolumetricInput, TrainingBatch
from alphabuilder.src.neural.model_arch import build_3d_vit
from alphabuilder.src.neural.trainer import train_step

# --- Data Loader Tests ---

def test_prepare_volumetric_batch_shapes():
    # Mock data: 2 grids of size (64, 128)
    # Case 1: 3 channels
    grid1 = np.random.rand(64, 128, 3).astype(np.float32)
    # Case 2: 2D grid (topology only), should be expanded
    grid2 = np.random.rand(64, 128).astype(np.float32)
    
    grids = [grid1, grid2]
    thicknesses = [1, 5]
    
    vol_input = prepare_volumetric_batch(grids, thicknesses)
    
    # Expected shape: (Batch, Depth, Height, Width, Channels)
    # Depth=16, Height=64, Width=128, Channels=3
    assert vol_input.tensor.shape == (2, 16, 64, 128, 3)
    
    # Check extrusion for grid 1 (thickness 1)
    # Slice 0 should match grid1
    assert np.allclose(vol_input.tensor[0, 0, :, :, :], grid1)
    # Slice 1 should be zero
    assert np.all(vol_input.tensor[0, 1:, :, :, :] == 0)
    
    # Check extrusion for grid 2 (thickness 5)
    # It was 2D, so it should have been expanded to 3 channels (ch0=topo, ch1,2=0)
    # Slice 4 should have data
    assert np.any(vol_input.tensor[1, 4, :, :, 0] != 0)
    # Slice 5 should be zero
    assert np.all(vol_input.tensor[1, 5:, :, :, :] == 0)
    # Channel 1 and 2 should be zero (padding)
    assert np.all(vol_input.tensor[1, :, :, :, 1:] == 0)

def test_load_training_data_generator(tmp_path):
    # Setup mock DB
    import sqlite3
    import pickle
    
    db_path = tmp_path / "test_training.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE training_data (state_blob BLOB, fitness_score REAL, valid_fem INTEGER)")
    
    # Insert mock data
    for i in range(5):
        grid = np.random.rand(64, 128, 3).astype(np.float32)
        blob = pickle.dumps(grid)
        cursor.execute("INSERT INTO training_data VALUES (?, ?, 1)", (blob, float(i)))
        
    conn.commit()
    conn.close()
    
    # Test generator
    batch_size = 2
    gen = load_training_data(str(db_path), batch_size=batch_size)
    
    batch1 = next(gen)
    assert batch1.inputs.tensor.shape[0] == 2
    assert batch1.targets.shape == (2, 1)
    
    batch2 = next(gen)
    assert batch2.inputs.tensor.shape[0] == 2
    
    batch3 = next(gen)
    assert batch3.inputs.tensor.shape[0] == 1 # Remaining 1
    
    with pytest.raises(StopIteration):
        next(gen)

# --- Model Arch Tests ---

def test_build_3d_vit_output_shape():
    model = build_3d_vit()
    
    # Mock input
    mock_input = tf.random.normal((2, 16, 64, 128, 3))
    output = model(mock_input)
    
    assert output.shape == (2, 1)

def test_build_3d_vit_positivity():
    model = build_3d_vit()
    
    # Mock input
    mock_input = tf.random.normal((5, 16, 64, 128, 3))
    output = model(mock_input)
    
    # Softplus output should be non-negative
    assert np.all(output.numpy() >= 0)

# --- Trainer Tests ---

def test_train_step_gradient_update():
    model = build_3d_vit()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    # Mock Batch
    inputs = tf.random.normal((2, 16, 64, 128, 3))
    targets = tf.constant([[1.0], [2.0]], dtype=tf.float32)
    
    # Get initial weights
    initial_weights = [w.numpy() for w in model.trainable_variables]
    
    # Run step
    metrics = train_step(model, inputs, targets, optimizer, loss_fn)
    
    assert "loss" in metrics
    assert "mae" in metrics
    
    # Check if weights changed
    final_weights = [w.numpy() for w in model.trainable_variables]
    
    # At least some weights should change
    weights_changed = False
    for w_init, w_final in zip(initial_weights, final_weights):
        if not np.allclose(w_init, w_final):
            weights_changed = True
            break
            
    assert weights_changed, "Weights did not update after training step"
