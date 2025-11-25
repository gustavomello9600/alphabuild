import pytest
import tensorflow as tf
import numpy as np
from alphabuilder.src.neural.model import create_vit_regressor, UniversalPatchEncoder

def test_create_vit_regressor():
    """Test full model creation and output shape."""
    # Create model with flexible input
    model = create_vit_regressor(
        input_shape=(None, None, 3),
        patch_size=4,
        projection_dim=32,
        transformer_layers=1
    )

    assert isinstance(model, tf.keras.Model)

    # Test forward pass with 32x64 (Standard)
    inputs1 = tf.random.normal((1, 32, 64, 3))
    output1 = model(inputs1)
    assert output1.shape == (1, 1)

    # Test forward pass with 16x32 (Small) - Variable Resolution Check
    inputs2 = tf.random.normal((1, 16, 32, 3))
    output2 = model(inputs2)
    assert output2.shape == (1, 1)
    
    # Check output name (layer name, not tensor name)
    # assert model.layers[-1].name == "max_displacement_output"
    pass

def test_universal_vit_3d():
    """Test Universal ViT with 3D volumetric input."""
    # Create model for 3D (4 channels: D, H, W, C)
    model = create_vit_regressor(
        input_shape=(None, None, None, 3),
        patch_size=4,
        projection_dim=32,
        transformer_layers=1
    )
    
    # Test forward pass with 16x16x16 Volume
    inputs = tf.random.normal((1, 16, 16, 16, 3))
    output = model(inputs)
    assert output.shape == (1, 1)
