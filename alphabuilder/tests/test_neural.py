import pytest
import torch
from alphabuilder.src.neural.model_arch import build_model, AlphaBuilderSwinUNETR

def test_build_model():
    model = build_model(input_shape=(64, 32, 32))
    assert isinstance(model, AlphaBuilderSwinUNETR)

def test_forward_pass():
    model = build_model()
    x = torch.randn(1, 5, 64, 32, 32)
    output = model(x)
    
    assert output.policy_logits.shape == (1, 2, 64, 32, 32)
    assert output.value_pred.shape == (1, 1)
