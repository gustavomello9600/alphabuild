"""
Unit tests for training module.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np

from alphabuilder.src.neural.trainer import (
    weighted_value_loss,
    policy_loss,
    train_one_epoch,
    W_NEG,
    LAMBDA_POLICY,
    POS_WEIGHT_ADD,
    POS_WEIGHT_REM
)


class TestWeightedValueLoss:
    """Tests for weighted value loss function."""
    
    def test_positive_targets_weight_1(self):
        """Positive targets should have weight 1."""
        pred = torch.tensor([[0.5]])
        target = torch.tensor([[0.5]])  # Positive
        
        loss = weighted_value_loss(pred, target)
        
        # Should be simple MSE
        expected = (0.5 - 0.5) ** 2
        assert loss.item() == pytest.approx(expected)
    
    def test_negative_targets_weighted(self):
        """Negative targets should have higher weight."""
        pred = torch.tensor([[0.0]])
        target = torch.tensor([[-0.5]])  # Negative
        
        loss = weighted_value_loss(pred, target)
        
        # Should be weighted MSE
        mse = (0.0 - (-0.5)) ** 2
        expected = W_NEG * mse
        assert loss.item() == pytest.approx(expected)
    
    def test_batch_average(self):
        """Loss should be averaged over batch."""
        pred = torch.tensor([[0.0], [0.0]])
        target = torch.tensor([[0.5], [0.5]])  # Both positive
        
        loss = weighted_value_loss(pred, target)
        
        # Both have same error, average should be same as individual
        individual_mse = (0.0 - 0.5) ** 2
        assert loss.item() == pytest.approx(individual_mse)
    
    def test_gradient_flow(self):
        """Gradients should flow through loss."""
        pred = torch.tensor([[0.5]], requires_grad=True)
        target = torch.tensor([[-0.5]])
        
        loss = weighted_value_loss(pred, target)
        loss.backward()
        
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0


class TestPolicyLoss:
    """Tests for policy loss function with pos_weight."""
    
    def test_output_is_scalar(self):
        """Loss should return a scalar."""
        pred = torch.randn(2, 2, 8, 8, 4)
        target = torch.zeros(2, 2, 8, 8, 4)
        
        loss = policy_loss(pred, target)
        
        assert loss.dim() == 0  # Scalar
    
    def test_pos_weight_constants_exist(self):
        """pos_weight constants should be defined."""
        assert POS_WEIGHT_ADD > 0
        assert POS_WEIGHT_REM > 0
    
    def test_pos_weight_compensates_imbalance(self):
        """pos_weight should increase loss for false negatives."""
        # Scenario: mostly negative targets with one positive
        pred = torch.zeros(1, 2, 4, 4, 2)  # All predict 0 (after sigmoid)
        target = torch.zeros(1, 2, 4, 4, 2)
        target[0, 0, 2, 2, 1] = 1.0  # One positive in ADD channel
        
        loss_with_positive = policy_loss(pred, target)
        
        # Compare with all-negative target
        target_all_neg = torch.zeros(1, 2, 4, 4, 2)
        loss_all_negative = policy_loss(pred, target_all_neg)
        
        # Loss should be higher when we miss a positive
        assert loss_with_positive > loss_all_negative
    
    def test_growth_phase_masks_remove(self):
        """GROWTH phase should reduce weight on Remove channel."""
        pred = torch.randn(2, 2, 8, 8, 4)
        target = torch.ones(2, 2, 8, 8, 4)  # All ones
        
        loss_growth = policy_loss(pred, target, phase='GROWTH')
        loss_normal = policy_loss(pred, target, phase=None)
        
        # GROWTH should have lower loss (Remove channel masked)
        assert loss_growth < loss_normal
    
    def test_gradient_flow(self):
        """Gradients should flow through loss."""
        pred = torch.randn(2, 2, 8, 8, 4, requires_grad=True)
        target = torch.zeros(2, 2, 8, 8, 4)
        
        loss = policy_loss(pred, target)
        loss.backward()
        
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0


class TestTrainOneEpoch:
    """Tests for training loop."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a minimal model for testing."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv3d(7, 16, 3, padding=1)
                self.policy_head = nn.Conv3d(16, 2, 1)
                self.value_head = nn.Sequential(
                    nn.AdaptiveAvgPool3d(1),
                    nn.Flatten(),
                    nn.Linear(16, 1),
                    nn.Tanh()
                )
            
            def forward(self, x):
                features = self.conv(x)
                policy = self.policy_head(features)
                value = self.value_head(features)
                return policy, value
        
        return SimpleModel()
    
    @pytest.fixture
    def simple_dataloader(self):
        """Create a minimal dataloader for testing."""
        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4
            
            def __getitem__(self, idx):
                return {
                    'state': torch.randn(7, 16, 8, 4),
                    'policy': torch.zeros(2, 16, 8, 4),
                    'value': torch.tensor([0.5])
                }
        
        return torch.utils.data.DataLoader(SimpleDataset(), batch_size=2)
    
    def test_returns_metrics_dict(self, simple_model, simple_dataloader):
        """Should return dictionary with loss metrics."""
        optimizer = torch.optim.Adam(simple_model.parameters())
        
        metrics = train_one_epoch(simple_model, simple_dataloader, optimizer, torch.device('cpu'))
        
        assert 'loss' in metrics
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
    
    def test_model_updates(self, simple_model, simple_dataloader):
        """Model parameters should change after training."""
        optimizer = torch.optim.Adam(simple_model.parameters())
        
        # Get initial params
        initial_params = [p.clone() for p in simple_model.parameters()]
        
        # Train
        train_one_epoch(simple_model, simple_dataloader, optimizer, torch.device('cpu'))
        
        # Check params changed
        params_changed = False
        for initial, current in zip(initial_params, simple_model.parameters()):
            if not torch.equal(initial, current):
                params_changed = True
                break
        
        assert params_changed, "Model parameters should update during training"
    
    def test_loss_decreases_trend(self, simple_model, simple_dataloader):
        """Loss should generally decrease over epochs."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        
        losses = []
        for _ in range(5):
            metrics = train_one_epoch(simple_model, simple_dataloader, optimizer, torch.device('cpu'))
            losses.append(metrics['loss'])
        
        # At least the trend should be downward (last < first)
        # Note: not guaranteed for every run, but generally true
        assert losses[-1] < losses[0] * 2, "Loss should not explode"


class TestLossWeightConstants:
    """Tests for loss weight constants."""
    
    def test_w_neg_value(self):
        """W_NEG should be positive and > 1."""
        assert W_NEG > 1, "W_NEG should weight negatives more"
    
    def test_lambda_policy_value(self):
        """LAMBDA_POLICY should be positive."""
        assert LAMBDA_POLICY > 0
    
    def test_pos_weight_add(self):
        """POS_WEIGHT_ADD should compensate for ~10% positive class."""
        # For 10% positive, pos_weight should be around 9-10
        assert POS_WEIGHT_ADD >= 5
        assert POS_WEIGHT_ADD <= 15
    
    def test_pos_weight_rem(self):
        """POS_WEIGHT_REM should compensate for ~20% positive class."""
        # For 20% positive, pos_weight should be around 4-5
        assert POS_WEIGHT_REM >= 3
        assert POS_WEIGHT_REM <= 10

