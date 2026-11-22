"""
Unit tests for AlphaBuilder v3.3 neural network architectures.
"""
import pytest
import torch
import torch.nn as nn

import alphabuilder.src.neural.model as model_mod
from alphabuilder.src.neural.model import AlphaBuilderV31, HAS_MONAI


class TestAlphaBuilderV31:
    """Core contract tests for AlphaBuilderV31 (SimpleBackbone)."""

    @pytest.fixture
    def model(self):
        return AlphaBuilderV31(in_channels=7, use_swin=False)

    @pytest.fixture
    def sample_input(self):
        return torch.randn(2, 7, 64, 32, 8)

    def test_forward_output_shapes(self, model, sample_input):
        model.eval()
        with torch.no_grad():
            policy, value = model(sample_input)

        B, _, D, H, W = sample_input.shape
        assert policy.shape == (B, 2, D, H, W)
        assert value.shape == (B, 1)

    def test_policy_output_range(self, model, sample_input):
        model.eval()
        with torch.no_grad():
            policy, _ = model(sample_input)
        assert policy.abs().sum() > 0

    def test_value_output_range(self, model, sample_input):
        model.eval()
        with torch.no_grad():
            _, value = model(sample_input)
        assert value.min() >= -1.0
        assert value.max() <= 1.0

    def test_gradient_flow(self, model, sample_input):
        model.train()
        policy, value = model(sample_input)
        loss = policy.sum() + value.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_batch_independence(self, model):
        model.eval()
        input1 = torch.randn(1, 7, 64, 32, 8)
        input2 = torch.randn(1, 7, 64, 32, 8)

        with torch.no_grad():
            policy1, value1 = model(input1)
            policy2, value2 = model(input2)
            policy_comb, value_comb = model(torch.cat([input1, input2], dim=0))

        torch.testing.assert_close(policy1, policy_comb[0:1], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(policy2, policy_comb[1:2], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(value1, value_comb[0:1], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(value2, value_comb[1:2], rtol=1e-4, atol=1e-4)

    def test_deterministic_eval(self, model, sample_input):
        model.eval()
        with torch.no_grad():
            p1, v1 = model(sample_input)
            p2, v2 = model(sample_input)
        torch.testing.assert_close(p1, p2)
        torch.testing.assert_close(v1, v2)

    def test_dynamic_resolution(self, model):
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
    @pytest.fixture
    def model(self):
        return AlphaBuilderV31(in_channels=7, use_swin=False)

    def test_heads_and_components_exist(self, model):
        assert hasattr(model, "encoder")
        assert hasattr(model, "policy_decoder")
        assert hasattr(model, "value_head")
        assert hasattr(model, "policy_head")
        assert hasattr(model, "blueprint_head")

    def test_value_head_trainable(self, model):
        value_params = sum(p.numel() for p in model.value_head.parameters() if p.requires_grad)
        assert value_params > 0


class TestModelParameterCount:
    def test_simple_backbone_params(self):
        model = AlphaBuilderV31(in_channels=7, use_swin=False)
        total_params = sum(p.numel() for p in model.parameters())
        assert 100_000 < total_params < 10_000_000, f"Unexpected param count: {total_params:,}"


class TestBlocoBArchitecture:
    """Tests validating Bloco B architectural upgrades."""

    def test_uses_groupnorm_no_instancenorm(self):
        simple = AlphaBuilderV31(in_channels=7, use_swin=False)
        inst_norms = [m for m in simple.modules() if isinstance(m, nn.InstanceNorm3d)]
        grp_norms = [m for m in simple.modules() if isinstance(m, nn.GroupNorm)]
        assert len(inst_norms) == 0, "InstanceNorm3d must not be used anywhere"
        assert len(grp_norms) > 0, "GroupNorm should be used"

        if HAS_MONAI:
            swin = AlphaBuilderV31(in_channels=7, use_swin=True)
            assert not any(isinstance(m, nn.InstanceNorm3d) for m in swin.modules())
            assert any(isinstance(m, nn.GroupNorm) for m in swin.modules())

    def test_scale_and_bc_sensitivity(self):
        """Force multiplier (1x -> 3x) and FULL_CLAMP vs RAIL_XY must produce distinct outputs."""
        torch.manual_seed(7)
        model = AlphaBuilderV31(in_channels=7, use_swin=False)
        model.eval()

        base = torch.zeros(1, 7, 64, 32, 8)
        base[0, 0, :, 14:18, 2:6] = 1.0
        base[0, 1:4, 0, :, :] = 1.0  # FULL_CLAMP
        base[0, 5, 63, 15:18, 3:6] = -1.0

        scaled_3x = base.clone()
        scaled_3x[0, 4:7] *= 3.0

        rail_xy = base.clone()
        rail_xy[0, 3] = 0.0  # Free Z on X=0 -> RAIL_XY

        with torch.no_grad():
            pol_base, val_base = model(base)
            pol_3x, val_3x = model(scaled_3x)
            pol_rail, val_rail = model(rail_xy)

        assert not torch.allclose(val_base, val_3x, atol=1e-4)
        assert not torch.allclose(pol_base, pol_3x, atol=1e-4)
        assert not torch.allclose(val_base, val_rail, atol=1e-4)
        assert not torch.allclose(pol_base, pol_rail, atol=1e-4)

    def test_simple_backbone_receptive_field_ge_64(self):
        """Perturbing x=0 must influence policy output at x=63 (RF >= 64 along D)."""
        torch.manual_seed(11)
        model = AlphaBuilderV31(in_channels=7, use_swin=False)
        model.eval()

        x1 = torch.zeros(1, 7, 64, 32, 8)
        x1[0, 0, 10:55, 14:18, 3:5] = 1.0
        x2 = x1.clone()
        x2[0, 1, 0, 16, 4] = 10.0  # Perturb strictly at X=0

        with torch.no_grad():
            p1, v1 = model(x1)
            p2, v2 = model(x2)

        tip_diff = (p1[0, :, 63, :, :] - p2[0, :, 63, :, :]).abs().max().item()
        assert tip_diff > 1e-6, f"Support at X=0 did not affect tip X=63 (diff={tip_diff})"
        assert (v1 - v2).abs().item() > 1e-6

    def test_blueprint_head_isolated_and_trainable(self):
        model = AlphaBuilderV31(in_channels=7, use_swin=False)
        model.train()
        x = torch.randn(2, 7, 64, 32, 8)

        out_default = model(x)
        assert isinstance(out_default, tuple) and len(out_default) == 2

        policy, value, bp = model(x, return_blueprint=True)
        assert policy.shape == (2, 2, 64, 32, 8)
        assert value.shape == (2, 1)
        assert bp.shape == (2, 1, 64, 32, 8)

        model.zero_grad()
        bp.sum().backward()
        enc_grad = sum(
            p.grad.abs().sum().item() for p in model.encoder.parameters() if p.grad is not None
        )
        assert enc_grad > 0.0

    def test_swin_requires_monai(self, monkeypatch):
        monkeypatch.setattr(model_mod, "HAS_MONAI", False)
        with pytest.raises(ImportError):
            AlphaBuilderV31(in_channels=7, use_swin=True)

    @pytest.mark.skipif(not HAS_MONAI, reason="MONAI not installed")
    def test_swin_unetr_enc0_and_rot90_shapes(self):
        model = AlphaBuilderV31(in_channels=7, feature_size=24, use_swin=True)
        assert hasattr(model.encoder, "encoder0")
        assert hasattr(model.policy_decoder, "decoder0")

        model.train()
        for shape in [(1, 7, 64, 32, 8), (1, 7, 32, 64, 8)]:
            x = torch.randn(*shape)
            pol, val, bp = model(x, return_blueprint=True)
            assert pol.shape == (1, 2, *shape[2:])
            assert val.shape == (1, 1)
            assert bp.shape == (1, 1, *shape[2:])

        model.zero_grad()
        (pol.sum() + val.sum() + bp.sum()).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.encoder0.parameters())
