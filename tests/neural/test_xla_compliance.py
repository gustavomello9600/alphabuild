import pytest
import tensorflow as tf
import numpy as np
from alphabuilder.src.neural.model import create_vit_regressor

def test_xla_compilation_dynamic_shapes():
    """
    Verify that the model can be compiled with XLA and handle dynamic shapes.
    This reproduces the 'Reading input as constant from a dynamic tensor' error.
    """
    # Create model with fully dynamic shape (Rank 5)
    model = create_vit_regressor(input_shape=(None, None, None, 3))
    
    # Define a compiled function
    @tf.function(jit_compile=True)
    def predict_step(inputs):
        return model(inputs, training=False)
        
    # Test 1: 2D Input (Simulated as Depth=1)
    # Shape: (1, 1, 32, 64, 3)
    x_2d = tf.random.normal((1, 1, 32, 64, 3))
    try:
        out_2d = predict_step(x_2d)
        assert out_2d.shape == (1, 1)
    except Exception as e:
        pytest.fail(f"XLA compilation failed for 2D (Depth=1) input: {e}")
        
    # Test 2: 3D Input
    # Shape: (1, 16, 16, 16, 3)
    x_3d = tf.random.normal((1, 16, 16, 16, 3))
    try:
        out_3d = predict_step(x_3d)
        assert out_3d.shape == (1, 1)
    except Exception as e:
        pytest.fail(f"XLA compilation failed for 3D input: {e}")

if __name__ == "__main__":
    test_xla_compilation_dynamic_shapes()
