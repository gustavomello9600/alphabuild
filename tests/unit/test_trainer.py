"""
Unit tests for v3.3 loss functions and training step.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphabuilder.src.neural.model import AlphaBuilderV31
from alphabuilder.src.neural.trainer import (
    HUBER_BETA,
    LAMBDA_BLUEPRINT_WARMUP,
    LAMBDA_POLICY,
    LAMBDA_VALUE,
    W_NEG,
    blueprint_loss,
    compute_backbone_grad_ratio,
    compute_total_loss,
    policy_loss,
    train_one_epoch,
    weighted_value_loss,
)


class TestWeightedValueLoss:
    def test_huber_zero_at_exact(self):
        pred = torch.tensor([[0.5]])
        target = torch.tensor([[0.5]])
        assert weighted_value_loss(pred, target).item() == pytest.approx(0.0)

    def test_negative_targets_weighted_and_huber_scaled(self):
        pred = torch.tensor([[0.0]])
        target = torch.tensor([[-0.5]])
        expected = W_NEG * F.smooth_l1_loss(pred, target, reduction="none", beta=HUBER_BETA).item()
        assert weighted_value_loss(pred, target).item() == pytest.approx(expected)

    def test_loss_constants(self):
        assert 1.0 < W_NEG <= 2.0
        assert HUBER_BETA == pytest.approx(0.2)
        assert LAMBDA_POLICY == pytest.approx(1.0)
        assert LAMBDA_VALUE == pytest.approx(2.0)
        assert LAMBDA_BLUEPRINT_WARMUP == pytest.approx(0.50)


class TestMaskedPolicyLoss:
    def test_zero_kl_at_optimal_logits(self):
        legal = torch.zeros(1, 2, 4, 4, 2)
        legal[0, 0, 1, 1, 0] = 1.0
        legal[0, 0, 2, 2, 0] = 1.0
        target = torch.zeros_like(legal)
        target[0, 0, 1, 1, 0] = 0.75
        target[0, 0, 2, 2, 0] = 0.25

        pred = torch.full_like(legal, 100.0)  # Huge invalid logits must be ignored!
        pred[0, 0, 1, 1, 0] = float(torch.log(torch.tensor(0.75)))
        pred[0, 0, 2, 2, 0] = float(torch.log(torch.tensor(0.25)))

        loss = policy_loss(pred, target, legal_mask=legal)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_ignores_stationary_policy_mask_zero(self):
        pred = torch.randn(2, 2, 4, 4, 2, requires_grad=True)
        legal = torch.ones_like(pred)
        target = torch.zeros_like(pred)
        target[0, 0, 1, 1, 0] = 1.0
        pmask = torch.tensor([0.0, 0.0])

        loss = policy_loss(pred, target, legal_mask=legal, policy_mask=pmask)
        assert loss.item() == pytest.approx(0.0)
        loss.backward()
        assert torch.all(pred.grad == 0)

    def test_float16_safe_no_nan(self):
        pred = torch.randn(2, 2, 8, 8, 4, dtype=torch.float16, requires_grad=True)
        legal = torch.zeros(2, 2, 8, 8, 4)
        legal[:, 0, 2:4, 2:4, 1] = 1.0
        target = torch.zeros(2, 2, 8, 8, 4)
        target[:, 0, 2, 2, 1] = 1.0

        loss = policy_loss(pred, target, legal_mask=legal)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(pred.grad).all()


class TestBlueprintAndTotalLoss:
    def test_blueprint_loss_masking(self):
        pred = torch.zeros(2, 1, 4, 4, 2)
        target = torch.zeros_like(pred)
        target[:, :, 1:3, 1:3, :] = 1.0

        loss_masked_all_zero = blueprint_loss(pred, target, blueprint_mask=torch.tensor([0.0, 0.0]))
        assert loss_masked_all_zero.item() == pytest.approx(0.0)

        loss_active = blueprint_loss(pred, target, blueprint_mask=torch.tensor([1.0, 1.0]))
        assert loss_active.item() > 0.0

    def test_compute_total_loss_and_grad_ratio(self):
        model = AlphaBuilderV31(in_channels=7, feature_size=16, use_swin=False)
        x = torch.randn(2, 7, 16, 8, 4)
        legal = torch.ones(2, 2, 16, 8, 4)
        t_pol = torch.zeros(2, 2, 16, 8, 4)
        t_pol[:, 0, 4, 4, 2] = 1.0
        t_val = torch.tensor([[0.5], [-0.4]])
        t_bp = torch.zeros(2, 1, 16, 8, 4)
        t_bp[:, :, 2:12, 3:5, 1:3] = 1.0

        p_pred, v_pred, bp_pred = model(x, return_blueprint=True)
        p_l = policy_loss(p_pred, t_pol, legal_mask=legal)
        bp_l = blueprint_loss(bp_pred, t_bp)
        ratio = compute_backbone_grad_ratio(model, p_l, bp_l, lambda_bp=0.50)
        assert ratio is not None and ratio > 0.0

        tot, metrics = compute_total_loss(
            p_pred,
            v_pred,
            t_pol,
            t_val,
            legal_mask=legal,
            pred_blueprint=bp_pred,
            target_blueprint=t_bp,
        )
        assert tot.item() == pytest.approx(
            LAMBDA_POLICY * metrics["policy_loss"]
            + LAMBDA_VALUE * metrics["value_loss"]
            + LAMBDA_BLUEPRINT_WARMUP * metrics["blueprint_loss"],
            rel=1e-4,
        )


class TestTrainOneEpoch:
    @pytest.fixture
    def simple_dataloader(self):
        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                pol = torch.zeros(2, 16, 8, 4)
                pol[0, 3, 3, 1] = 1.0
                bp = torch.zeros(1, 16, 8, 4)
                bp[0, :8, 2:6, 1:3] = 1.0
                return {
                    "state": torch.randn(7, 16, 8, 4),
                    "policy": pol,
                    "legal_mask": torch.ones(2, 16, 8, 4),
                    "policy_mask": torch.tensor(1.0),
                    "value": torch.tensor([0.5]),
                    "blueprint": bp,
                    "blueprint_mask": torch.tensor(1.0),
                }

        return torch.utils.data.DataLoader(SimpleDataset(), batch_size=2)

    def test_2head_and_3head_models_update(self, simple_dataloader):
        # 2-head backward compatibility
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv3d(7, 16, 3, padding=1)
                self.policy_head = nn.Conv3d(16, 2, 1)
                self.value_head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(16, 1), nn.Tanh())

            def forward(self, x):
                feat = self.conv(x)
                return self.policy_head(feat), self.value_head(feat)

        m2 = SimpleModel()
        opt2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
        metrics2 = train_one_epoch(m2, simple_dataloader, opt2, torch.device("cpu"))
        assert "loss" in metrics2 and "policy_loss" in metrics2 and "value_loss" in metrics2

        # 3-head AlphaBuilderV31
        m3 = AlphaBuilderV31(in_channels=7, feature_size=16, use_swin=False)
        opt3 = torch.optim.Adam(m3.parameters(), lr=1e-3)
        initial_bp_w = m3.blueprint_head[-1].weight.clone()
        metrics3 = train_one_epoch(m3, simple_dataloader, opt3, torch.device("cpu"), lambda_bp=0.50)
        assert metrics3["blueprint_loss"] > 0.0
        assert not torch.equal(initial_bp_w, m3.blueprint_head[-1].weight)
