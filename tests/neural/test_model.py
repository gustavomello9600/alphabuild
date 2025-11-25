import pytest
import tensorflow as tf
import numpy as np
from alphabuilder.src.neural.model import create_vit_regressor, UniversalPatchEncoder

def test_create_vit_regressor():
    """Test full model creation and output shape."""
    # Create model with flexible input (Rank 5)
    model = create_vit_regressor(
        input_shape=(None, None, None, 3),
        patch_size=4,
        projection_dim=32,
        transformer_layers=1
    )
    
    # Test with 2D-like input (Depth=1)
    # Shape: (1, 1, 32, 64, 3)
    x = tf.random.normal((1, 1, 32, 64, 3))
    y = model(x)
    assert y.shape == (1, 1)

def test_variable_resolution_2d():
    """Test that model handles different 2D resolutions (as Depth=1 3D)."""
    model = create_vit_regressor(input_shape=(None, None, None, 3))
    
    # Resolution 1: 32x64 (Depth=1)
    x1 = tf.random.normal((1, 1, 32, 64, 3))
    y1 = model(x1)
    assert y1.shape == (1, 1)
    
    # Resolution 2: 16x32 (Depth=1)
    x2 = tf.random.normal((1, 1, 16, 32, 3))
    y2 = model(x2)
    assert y2.shape == (1, 1)

def test_universal_vit_3d():
    """Test that model handles 3D volumetric input."""
    model = create_vit_regressor(input_shape=(None, None, None, 3))
    
    # Volume: 16x16x16
    x = tf.random.normal((1, 16, 16, 16, 3))
    y = model(x)
    assert y.shape == (1, 1)
