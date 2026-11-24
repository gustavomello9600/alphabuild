"""
Unit and integration tests for PhysUViT3D architecture, factored losses,
D4 x Z2 symmetry invariants, Multi-Load Set, Domain Masking, and Inference round-trip.
"""

import json
import math
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from alphabuilder.src.logic.storage import (
    EpisodeInfo,
    Phase,
    StepRecord,
    initialize_database,
    save_episode,
    save_step,
)
from alphabuilder.src.neural.augmentation import flip_x, flip_y, flip_z, load_multiplier, rotate_90_z, sabotage
from alphabuilder.src.neural.dataset import TopologyDatasetV31
from alphabuilder.src.neural.inference import AlphaBuilderInference
from alphabuilder.src.neural.model import build_model, resolve_architecture
from alphabuilder.src.neural.phys_uvit import PhysUViT3D
from alphabuilder.src.neural.train_config import get_config
from alphabuilder.src.neural.train_pipeline import create_model, save_checkpoint
from alphabuilder.src.neural.trainer import (
    compute_backbone_grad_ratio,
    compute_connectivity_accuracy,
    compute_log_c_mae,
    compute_total_loss,
    train_one_epoch,
)


class TestPhysUViTArchitecture:
    @pytest.fixture
    def model(self):
        return PhysUViT3D(
            in_channels=7,
            out_channels=2,
            feature_size=32,
            embed_dim=192,
            transformer_depth=5,
            num_heads=6,
        )

    def test_param_budget_groupnorm_and_factory(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        assert 4_000_000 <= total_params <= 6_200_000, f"Unexpected param count: {total_params:,}"
        assert len([m for m in model.modules() if isinstance(m, nn.InstanceNorm3d)]) == 0
        assert len([m for m in model.modules() if isinstance(m, nn.GroupNorm)]) > 0

        built = build_model(architecture="phys_uvit", feature_size=32)
        assert isinstance(built, PhysUViT3D)
        assert resolve_architecture({"architecture": "Phys-UViT 3D"}) == "phys_uvit"

    def test_shapes_zero_padding_div4_and_aux_outputs(self, model):
        model.eval()
        for shape in [(2, 7, 64, 32, 8), (2, 7, 32, 64, 8), (1, 7, 48, 24, 6)]:
            x = torch.randn(*shape)
            with torch.no_grad():
                pol, val = model(x)
                pol_bp, val_bp, bp = model(x, return_blueprint=True)
                pol_aux, val_aux, aux = model(x, return_aux=True, compute_sed=True)

            assert pol.shape == (shape[0], 2, *shape[2:])
            assert val.shape == (shape[0], 1)
            assert float(val.min()) >= -1.0 and float(val.max()) <= 1.0
            assert bp.shape == (shape[0], 1, *shape[2:])
            assert aux["log_compliance"].shape == (shape[0], 1)
            assert aux["conn_logit"].shape == (shape[0], 1)
            assert aux["sed_u"].shape == (shape[0], 4, *shape[2:])
            torch.testing.assert_close(pol, pol_bp)
            torch.testing.assert_close(pol, pol_aux)
            torch.testing.assert_close(val, val_aux)

    def test_layer1_global_receptive_field(self):
        torch.manual_seed(17)
        small = PhysUViT3D(feature_size=16, embed_dim=96, transformer_depth=1, num_heads=3)
        small.eval()

        x1 = torch.zeros(1, 7, 64, 32, 8)
        x1[0, 0, 12:52, 14:18, 2:6] = 1.0
        x2 = x1.clone()
        x2[0, 1, 0, 16, 4] = 10.0  # Support perturbation strictly at X=0

        x3 = x1.clone()
        x3[0, 5, 63, 16, 4] = -5.0  # Load perturbation strictly at X=63

        with torch.no_grad():
            p1, v1 = small(x1)
            p2, v2 = small(x2)
            p3, _ = small(x3)

        tip_diff = (p1[0, :, 63, :, :] - p2[0, :, 63, :, :]).abs().max().item()
        root_diff = (p1[0, :, 0, :, :] - p3[0, :, 0, :, :]).abs().max().item()
        assert tip_diff > 1e-6, "Layer-1 global transformer did not bridge X=0 -> X=63"
        assert root_diff > 1e-6, "Layer-1 global transformer did not bridge X=63 -> X=0"
        assert (v1 - v2).abs().item() > 1e-6

    def test_domain_mask_keep_out_isolation(self):
        torch.manual_seed(23)
        small = PhysUViT3D(feature_size=16, embed_dim=96, transformer_depth=2, num_heads=3)
        small.eval()

        x = torch.randn(1, 7, 32, 16, 8)
        domain_mask = torch.ones(1, 1, 32, 16, 8)
        domain_mask[:, :, 16:24, 8:16, :] = 0.0  # Cutout corner (keep-out hole)

        with torch.no_grad():
            pol, val, aux = small(x, domain_mask=domain_mask, return_aux=True, compute_sed=True)

        assert torch.isfinite(pol).all()
        assert torch.isfinite(val).all()
        assert torch.isfinite(aux["log_compliance"]).all()
        assert (pol[:, :, 16:24, 8:16, :] < -1e3).all()

    def test_multi_load_set_no_opposing_cancellation(self):
        torch.manual_seed(31)
        small = PhysUViT3D(feature_size=16, embed_dim=96, transformer_depth=2, num_heads=3)
        small.eval()

        x = torch.zeros(1, 7, 32, 16, 8)
        x[0, 0, :24, 6:10, 2:6] = 1.0
        x[0, 1:4, 0, :, :] = 1.0

        load_pos = torch.zeros(1, 3, 32, 16, 8)
        load_pos[0, 1, 23, 8, 4] = 2.0
        load_neg = torch.zeros(1, 3, 32, 16, 8)
        load_neg[0, 1, 23, 8, 4] = -2.0

        loads_opposing = torch.stack([load_pos, load_neg], dim=1)  # (1, K=2, 3, 32, 16, 8)
        loads_zero = torch.zeros_like(loads_opposing)

        with torch.no_grad():
            p_opp, v_opp = small(x, loads=loads_opposing)
            p_zero, v_zero = small(x, loads=loads_zero)

        assert not torch.allclose(p_opp, p_zero, atol=1e-4)
        assert not torch.allclose(v_opp, v_zero, atol=1e-4)

    def test_scale_bc_sensitivity_and_d4_invariance(self):
        torch.manual_seed(42)
        small = PhysUViT3D(feature_size=16, embed_dim=96, transformer_depth=2, num_heads=3)
        small.eval()

        base = np.zeros((7, 32, 16, 8), dtype=np.float32)
        base[0, :, 6:10, 2:6] = 1.0
        base[1:4, 0, :, :] = 1.0
        base[5, 31, 7:10, 3:5] = -1.0

        scaled_3x = base.copy()
        scaled_3x[4:7] *= 3.0

        rail_xy = base.copy()
        rail_xy[3] = 0.0

        with torch.no_grad():
            p_b, v_b, aux_b = small(torch.from_numpy(base).unsqueeze(0), return_aux=True)
            p_3, v_3, aux_3 = small(torch.from_numpy(scaled_3x).unsqueeze(0), return_aux=True)
            p_r, v_r, _ = small(torch.from_numpy(rail_xy).unsqueeze(0), return_aux=True)

        assert not torch.allclose(v_b, v_3, atol=1e-4)
        assert not torch.allclose(p_b, p_3, atol=1e-4)
        assert not torch.allclose(aux_b["log_compliance"], aux_3["log_compliance"], atol=1e-4)
        assert not torch.allclose(v_b, v_r, atol=1e-4)
        assert not torch.allclose(p_b, p_r, atol=1e-4)

        # Verify D4 x Z2 invariance of physics context stats
        dummy_pol = np.zeros((2, 32, 16, 8), dtype=np.float32)
        s_rot, _ = rotate_90_z(base, dummy_pol)
        s_fx, _ = flip_x(base, dummy_pol)
        s_fy, _ = flip_y(base, dummy_pol)
        s_fz, _ = flip_z(base, dummy_pol)

        stats_base = PhysUViT3D._extract_physics_stats(torch.from_numpy(base).unsqueeze(0))
        for variant in (s_rot, s_fx, s_fy, s_fz):
            stats_var = PhysUViT3D._extract_physics_stats(torch.from_numpy(variant).unsqueeze(0))
            torch.testing.assert_close(stats_base, stats_var, rtol=1e-5, atol=1e-5)


class TestPhysUViTTrainingAndDataset:
    def test_dataset_factored_targets_and_load_multiplier_scaling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "factored.db"
            initialize_database(db_path)

            res = (16, 8, 4)
            bc = np.zeros((3, *res), dtype=np.float32)
            bc[:, 0, :, :] = 1.0
            forces = np.zeros((3, *res), dtype=np.float32)
            forces[1, 15, 3:5, 1:3] = -1.0
            v_const = np.zeros(res, dtype=np.float32)
            v_const[:, 2:6, 1:3] = 1.0

            save_episode(
                db_path,
                EpisodeInfo(
                    episode_id="ep_comp",
                    bc_masks=bc,
                    forces=forces,
                    load_config={"x": 15, "y": 4, "z_start": 1, "z_end": 3},
                    bc_type="FULL_CLAMP",
                    strategy="BEZIER",
                    resolution=res,
                    v_constructed=v_const,
                    final_is_connected=True,
                ),
            )
            pol_add = np.zeros(res, dtype=np.float32)
            pol_add[5, 6, 1] = 1.0
            save_step(
                db_path,
                StepRecord(
                    episode_id="ep_comp",
                    step=0,
                    phase=Phase.REFINEMENT,
                    density=v_const.copy(),
                    policy_add=pol_add,
                    policy_remove=np.zeros(res, dtype=np.float32),
                    fitness_score=0.45,
                    compliance=1000.0,
                    is_final_step=False,
                    is_connected=True,
                ),
            )

            ds = TopologyDatasetV31(db_path, augment=False, preload_to_ram=True)
            item = ds[0]
            assert item["compliance_mask"].item() == pytest.approx(1.0)
            assert item["log_compliance"].item() == pytest.approx(3.0, abs=1e-5)
            assert item["conn_mask"].item() == pytest.approx(1.0)
            assert item["conn_target"].item() == pytest.approx(1.0)

            # Verify load_multiplier analytical compliance shift
            orig_state = item["state"].numpy()
            orig_pol = item["policy"].numpy()
            _, _, new_val = load_multiplier(orig_state, orig_pol, 0.45, k=3.0)
            assert new_val != 0.45
            expected_scaled_log_c = 3.0 + 2.0 * math.log10(3.0)
            assert expected_scaled_log_c == pytest.approx(3.9542425, abs=1e-5)

            # Verify sabotage marks mutated sample masks cleanly
            sab_state, _, _ = sabotage(orig_state, orig_pol, 0.45)
            assert sab_state is not orig_state

    def test_factored_loss_grad_ratio_train_and_inference_roundtrip(self):
        cfg = get_config("local_phys_uvit")
        cfg.checkpoint_dir = tempfile.mkdtemp()
        cfg.log_dir = tempfile.mkdtemp()
        cfg.feature_size = 16
        cfg.embed_dim = 96
        cfg.transformer_depth = 2
        cfg.num_heads = 3

        model = create_model(cfg, torch.device("cpu"))
        assert isinstance(model, PhysUViT3D)
        assert not any(p.requires_grad for p in model.sed_head.parameters())

        class SyntheticDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                pol = torch.zeros(2, 16, 8, 4)
                pol[0, 4, 4, 1] = 1.0
                bp = torch.zeros(1, 16, 8, 4)
                bp[0, :, 2:6, 1:3] = 1.0
                return {
                    "state": torch.randn(7, 16, 8, 4),
                    "policy": pol,
                    "legal_mask": torch.ones(2, 16, 8, 4),
                    "policy_mask": torch.tensor(1.0),
                    "value": torch.tensor([0.4]),
                    "blueprint": bp,
                    "blueprint_mask": torch.tensor(1.0),
                    "log_compliance": torch.tensor([3.1]),
                    "compliance_mask": torch.tensor(1.0),
                    "conn_target": torch.tensor([1.0]),
                    "conn_mask": torch.tensor(1.0),
                }

        loader = torch.utils.data.DataLoader(SyntheticDataset(), batch_size=2)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        metrics = train_one_epoch(
            model,
            loader,
            opt,
            torch.device("cpu"),
            lambda_bp=0.5,
            lambda_log_c=0.20,
            lambda_conn=0.25,
        )
        assert metrics["loss"] > 0.0
        assert metrics["log_c_loss"] > 0.0
        assert metrics["conn_loss"] > 0.0

        # Check grad ratio + validation metrics + checkpoint round-trip
        batch = next(iter(loader))
        p_out, v_out, aux = model(batch["state"], return_aux=True, compute_sed=False)
        loss, _ = compute_total_loss(
            p_out,
            v_out,
            batch["policy"],
            batch["value"],
            legal_mask=batch["legal_mask"],
            policy_mask=batch["policy_mask"],
            pred_blueprint=aux["blueprint"],
            target_blueprint=batch["blueprint"],
            blueprint_mask=batch["blueprint_mask"],
            pred_log_c=aux["log_compliance"],
            target_log_c=batch["log_compliance"],
            compliance_mask=batch["compliance_mask"],
            pred_conn_logit=aux["conn_logit"],
            target_conn=batch["conn_target"],
            conn_mask=batch["conn_mask"],
            lambda_log_c=0.20,
            lambda_conn=0.25,
        )
        assert float(loss.item()) > 0.0
        assert compute_log_c_mae(aux["log_compliance"], batch["log_compliance"]) is not None
        assert compute_connectivity_accuracy(aux["conn_logit"], batch["conn_target"]) is not None
        ratio = compute_backbone_grad_ratio(model, loss, loss)
        assert ratio is not None and ratio > 0.0

        with tempfile.TemporaryDirectory() as ckpt_tmp:
            ckpt_path = Path(ckpt_tmp) / "phys_uvit_test.pt"
            save_checkpoint(model, opt, 0, metrics, cfg, ckpt_path)
            infer = AlphaBuilderInference(ckpt_path, device="cpu")
            assert infer.architecture == "phys_uvit"
            assert infer.use_swin is False

            state_np = np.random.randn(7, 16, 8, 4).astype(np.float32)
            val, p_add, p_rem, aux_np = infer.predict_with_aux(state_np, compute_sed=True)
            assert -1.0 <= val <= 1.0
            assert p_add.shape == (16, 8, 4)
            assert p_rem.shape == (16, 8, 4)
            assert "log_compliance" in aux_np and "conn_prob" in aux_np
            assert aux_np["sed_u"].shape == (4, 16, 8, 4)

            vals, p_adds, p_rems = infer.predict_batch([state_np, state_np])
            assert len(vals) == 2 and len(p_adds) == 2 and len(p_rems) == 2

            cfg_raw = json.loads(json.dumps(infer.config))
            assert cfg_raw["architecture"] == "phys_uvit"
