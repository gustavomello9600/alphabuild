import tensorflow as tf
import numpy as np
import os
from typing import Dict, List
from .data_loader import TrainingBatch, prepare_volumetric_batch

@tf.function
def train_step(model: tf.keras.Model, inputs: tf.Tensor, targets: tf.Tensor, optimizer: tf.keras.optimizers.Optimizer, loss_fn: tf.keras.losses.Loss) -> Dict[str, tf.Tensor]:
    """
    Executes one training step using Gradient Descent.
    
    Args:
        model: Keras model (ViT 3D)
        inputs: Input tensor (Batch, D, H, W, C)
        targets: Target tensor (Batch, 1)
        optimizer: Keras optimizer
        loss_fn: Keras loss function
        
    Returns:
        Dictionary of metrics (loss, mae)
    """
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(inputs, training=True)
        
        # Compute loss
        loss = loss_fn(targets, predictions)
        
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Compute metrics
    mae = tf.reduce_mean(tf.abs(targets - predictions))
    
    return {"loss": loss, "mae": mae}

def predict_fitness(model: tf.keras.Model, grids: List[np.ndarray], thicknesses: List[int]) -> np.ndarray:
    """
    Predicts fitness for a list of 2D grids.
    This is the main interface for the MCTS agent.
    
    Args:
        model: Trained Keras model
        grids: List of 2D numpy arrays
        thicknesses: List of thicknesses
        
    Returns:
        Numpy array of predicted fitness scores (Batch, 1)
    """
    # Preprocess
    vol_input = prepare_volumetric_batch(grids, thicknesses)
    
    # Inference
    predictions = model(vol_input.tensor, training=False)
    
    return predictions.numpy()

def save_model(model: tf.keras.Model, path: str):
    """
    Saves the model to the specified path.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}")

def load_model(path: str) -> tf.keras.Model:
    """
    Loads the model from the specified path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
        
    # We might need to register custom objects if we used custom layers like PatchEncoder
    # But since PatchEncoder is defined in model_arch, we need to import it or pass custom_objects
    from .model_arch import PatchEncoder
    
    return tf.keras.models.load_model(path, custom_objects={"PatchEncoder": PatchEncoder})
