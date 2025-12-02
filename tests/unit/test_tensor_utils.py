"""
Unit tests for tensor_utils module.
"""
import pytest
import numpy as np
from alphabuilder.src.core.tensor_utils import build_input_tensor_v31


class TestBuildInputTensorV31:
    """Tests for 7-channel input tensor construction."""
    
    def test_output_shape(self):
        """Output should have shape (7, D, H, W)."""
        density = np.zeros((64, 32, 8))
        load_config = {'x': 63, 'y': 16, 'z_start': 3, 'z_end': 5}
        
        tensor = build_input_tensor_v31(density, (64, 32, 8), load_config)
        
        assert tensor.shape == (7, 64, 32, 8)
    
    def test_output_dtype(self):
        """Output should be float32."""
        density = np.zeros((64, 32, 8))
        load_config = {'x': 63, 'y': 16, 'z_start': 3, 'z_end': 5}
        
        tensor = build_input_tensor_v31(density, (64, 32, 8), load_config)
        
        assert tensor.dtype == np.float32
    
    def test_density_channel(self):
        """Channel 0 should contain density values."""
        density = np.random.rand(64, 32, 8)
        load_config = {'x': 63, 'y': 16, 'z_start': 3, 'z_end': 5}
        
        tensor = build_input_tensor_v31(density, (64, 32, 8), load_config)
        
        np.testing.assert_array_almost_equal(tensor[0], density.astype(np.float32))
    
    def test_full_clamp_masks(self):
        """Full clamp should set all masks at X=0."""
        density = np.zeros((64, 32, 8))
        load_config = {'x': 63, 'y': 16, 'z_start': 3, 'z_end': 5}
        
        tensor = build_input_tensor_v31(density, (64, 32, 8), load_config, 'full_clamp')
        
        # All masks should be 1 at X=0
        assert tensor[1, 0, :, :].sum() == 32 * 8  # Mask X
        assert tensor[2, 0, :, :].sum() == 32 * 8  # Mask Y
        assert tensor[3, 0, :, :].sum() == 32 * 8  # Mask Z
        
        # All masks should be 0 elsewhere
        assert tensor[1, 1:, :, :].sum() == 0
        assert tensor[2, 1:, :, :].sum() == 0
        assert tensor[3, 1:, :, :].sum() == 0
    
    def test_roller_y_masks(self):
        """Roller Y should only set mask_y at X=0."""
        density = np.zeros((64, 32, 8))
        load_config = {'x': 63, 'y': 16, 'z_start': 3, 'z_end': 5}
        
        tensor = build_input_tensor_v31(density, (64, 32, 8), load_config, 'roller_y')
        
        assert tensor[1, 0, :, :].sum() == 0      # Mask X = 0
        assert tensor[2, 0, :, :].sum() == 32 * 8  # Mask Y = 1
        assert tensor[3, 0, :, :].sum() == 0      # Mask Z = 0
    
    def test_roller_z_masks(self):
        """Roller Z should only set mask_z at X=0."""
        density = np.zeros((64, 32, 8))
        load_config = {'x': 63, 'y': 16, 'z_start': 3, 'z_end': 5}
        
        tensor = build_input_tensor_v31(density, (64, 32, 8), load_config, 'roller_z')
        
        assert tensor[1, 0, :, :].sum() == 0      # Mask X = 0
        assert tensor[2, 0, :, :].sum() == 0      # Mask Y = 0
        assert tensor[3, 0, :, :].sum() == 32 * 8  # Mask Z = 1
    
    def test_force_channel_location(self):
        """Force should be applied at correct location."""
        density = np.zeros((64, 32, 8))
        load_config = {'x': 60, 'y': 16, 'z_start': 3, 'z_end': 5}
        
        tensor = build_input_tensor_v31(density, (64, 32, 8), load_config)
        
        # Force in -Y direction (channel 5)
        assert tensor[5].min() == -1.0
        
        # Force should be at specified location
        assert tensor[5, 60, :, :].min() == -1.0
    
    def test_force_is_negative_y(self):
        """Default force should be in -Y direction."""
        density = np.zeros((64, 32, 8))
        load_config = {'x': 63, 'y': 16, 'z_start': 3, 'z_end': 5}
        
        tensor = build_input_tensor_v31(density, (64, 32, 8), load_config)
        
        # Fx and Fz should be zero
        assert tensor[4].sum() == 0  # Fx
        assert tensor[6].sum() == 0  # Fz
        
        # Fy should have negative values
        assert tensor[5].min() < 0
    
    def test_1d_density_reshape(self):
        """1D density input should be reshaped correctly."""
        density_1d = np.zeros(64 * 32 * 8)
        density_1d[0] = 1.0  # Set first element
        load_config = {'x': 63, 'y': 16, 'z_start': 3, 'z_end': 5}
        
        tensor = build_input_tensor_v31(density_1d, (64, 32, 8), load_config)
        
        assert tensor.shape == (7, 64, 32, 8)
        assert tensor[0, 0, 0, 0] == 1.0

