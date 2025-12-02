"""
Unit tests for neural network model.
"""
import pytest
import torch
import numpy as np

from alphabuilder.src.neural.model import AlphaBuilderV31


class TestAlphaBuilderV31:
    """Tests for the AlphaBuilder v3.1 neural network."""
    
    @pytest.fixture
    def model(self):
        """Create model with simple backbone for testing."""
        return AlphaBuilderV31(in_channels=7, use_swin=False)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        # (B, C, D, H, W) = (2, 7, 64, 32, 8)
        return torch.randn(2, 7, 64, 32, 8)
    
    def test_forward_output_shapes(self, model, sample_input):
        """Forward pass should return correct shapes."""
        model.eval()
        with torch.no_grad():
            policy, value = model(sample_input)
        
        B, C, D, H, W = sample_input.shape
        
        # Policy: (B, 2, D, H, W)
        assert policy.shape == (B, 2, D, H, W)
        
        # Value: (B, 1)
        assert value.shape == (B, 1)
    
    def test_policy_output_range(self, model, sample_input):
        """Policy output should be logits (unbounded)."""
        model.eval()
        with torch.no_grad():
            policy, _ = model(sample_input)
        
        # Logits can be any value (before sigmoid)
        # Just check it's not all zeros
        assert policy.abs().sum() > 0
    
    def test_value_output_range(self, model, sample_input):
        """Value output should be in [-1, 1] (tanh)."""
        model.eval()
        with torch.no_grad():
            _, value = model(sample_input)
        
        assert value.min() >= -1.0
        assert value.max() <= 1.0
    
    def test_gradient_flow(self, model, sample_input):
        """Gradients should flow through both heads."""
        model.train()
        
        policy, value = model(sample_input)
        
        # Create dummy loss
        loss = policy.sum() + value.sum()
        loss.backward()
        
        # Check gradients exist in encoder
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "No gradients found in model parameters"
    
    def test_batch_independence(self, model):
        """Samples in batch should be processed independently."""
        model.eval()
        
        # Create two different inputs
        input1 = torch.randn(1, 7, 64, 32, 8)
        input2 = torch.randn(1, 7, 64, 32, 8)
        
        with torch.no_grad():
            # Process separately
            policy1, value1 = model(input1)
            policy2, value2 = model(input2)
            
            # Process together
            combined = torch.cat([input1, input2], dim=0)
            policy_combined, value_combined = model(combined)
        
        # Results should be the same
        torch.testing.assert_close(policy1, policy_combined[0:1], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(policy2, policy_combined[1:2], rtol=1e-4, atol=1e-4)
    
    def test_deterministic_eval(self, model, sample_input):
        """Model should be deterministic in eval mode."""
        model.eval()
        
        with torch.no_grad():
            policy1, value1 = model(sample_input)
            policy2, value2 = model(sample_input)
        
        torch.testing.assert_close(policy1, policy2)
        torch.testing.assert_close(value1, value2)
    
    def test_dynamic_resolution(self, model):
        """Model should handle different resolutions."""
        model.eval()
        
        resolutions = [
            (1, 7, 32, 16, 4),
            (1, 7, 64, 32, 8),
            (1, 7, 48, 24, 6),
        ]
        
        for shape in resolutions:
            x = torch.randn(*shape)
            with torch.no_grad():
                policy, value = model(x)
            
            assert policy.shape == (shape[0], 2, *shape[2:])
            assert value.shape == (shape[0], 1)


class TestModelDualHead:
    """Tests for dual-head architecture (policy from decoder, value from bottleneck)."""
    
    @pytest.fixture
    def model(self):
        return AlphaBuilderV31(in_channels=7, use_swin=False)
    
    def test_value_head_exists(self, model):
        """Value head should exist as separate component."""
        assert hasattr(model, 'value_head'), "Model should have value_head"
    
    def test_encoder_exists(self, model):
        """Encoder should exist for feature extraction."""
        assert hasattr(model, 'encoder'), "Model should have encoder"
    
    def test_policy_decoder_exists(self, model):
        """Policy decoder should exist (not necessarily named policy_head)."""
        # SimpleBackbone uses decoder for policy reconstruction
        has_decoder = hasattr(model, 'decoder') or hasattr(model, 'policy_decoder')
        assert has_decoder, "Model should have decoder for policy"
    
    def test_value_head_trainable(self, model):
        """Value head should have trainable parameters."""
        value_params = sum(p.numel() for p in model.value_head.parameters() if p.requires_grad)
        assert value_params > 0, "Value head should have trainable parameters"


class TestModelParameterCount:
    """Tests for model parameter counts."""
    
    def test_simple_backbone_params(self):
        """Simple backbone should have reasonable parameter count."""
        model = AlphaBuilderV31(in_channels=7, use_swin=False)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should be less than 10M for simple backbone
        assert total_params < 10_000_000, f"Too many parameters: {total_params:,}"
        
        # Should be at least 100K
        assert total_params > 100_000, f"Too few parameters: {total_params:,}"
    
    def test_parameter_breakdown(self):
        """Check parameter distribution across components."""
        model = AlphaBuilderV31(in_channels=7, use_swin=False)
        
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        value_params = sum(p.numel() for p in model.value_head.parameters())
        
        # Main components should have parameters
        assert encoder_params > 0, "Encoder should have parameters"
        assert value_params > 0, "Value head should have parameters"
        
        # Total params should be reasonable
        total = sum(p.numel() for p in model.parameters())
        assert total > 100_000, f"Model should have > 100K params, has {total:,}"

