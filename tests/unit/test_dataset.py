"""
Unit tests for TopologyDatasetV31 and 16x symmetry DataLoader collate (Bloco B).
"""
import json
from pathlib import Path
import pickle
import tempfile
import zlib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from alphabuilder.src.logic.storage import (
    EpisodeInfo,
    Phase,
    StepRecord,
    TrainingRecord,
    initialize_database,
    save_episode,
    save_record,
    save_step,
    serialize_state,
)
from alphabuilder.src.neural.dataset import (
    TopologyDatasetV31,
    deserialize_array,
    deserialize_state_legacy,
    make_symmetry_collate_fn,
)


class TestDatasetDeserialization:
    def test_deserialize_state_legacy_compressed(self):
        original = np.random.rand(5, 5).astype(np.float32)
        blob = zlib.compress(pickle.dumps(original))
        np.testing.assert_array_equal(original, deserialize_state_legacy(blob))

    def test_deserialize_array_compressed(self):
        original = np.random.rand(5, 5).astype(np.float32)
        blob = zlib.compress(pickle.dumps(original))
        np.testing.assert_array_equal(original, deserialize_array(blob))


class TestTopologyDatasetV31:
    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            initialize_database(db_path)
            yield db_path

    def test_load_v1_legacy(self, temp_db):
        state = np.zeros((7, 10, 10, 10), dtype=np.float32)
        policy = np.zeros((2, 10, 10, 10), dtype=np.float32)
        record = TrainingRecord(
            episode_id="ep1",
            step=0,
            phase=Phase.GROWTH,
            state_blob=serialize_state(state),
            policy_blob=serialize_state(policy),
            fitness_score=0.5,
            valid_fem=True,
            metadata={"is_final_step": True},
        )
        save_record(temp_db, record)

        dataset = TopologyDatasetV31(temp_db, augment=False, preload_to_ram=True)
        assert len(dataset) == 1
        sample = dataset[0]
        assert sample["state"].shape == (7, 10, 10, 10)
        assert sample["policy"].shape == (2, 10, 10, 10)
        assert sample["value"].item() == pytest.approx(0.5)
        assert sample["is_final"] is True

    def test_load_v2_optimized(self, temp_db):
        resolution = (10, 10, 10)
        ep_info = EpisodeInfo(
            episode_id="ep2",
            bc_masks=np.zeros((3,) + resolution, dtype=np.float32),
            forces=np.zeros((3,) + resolution, dtype=np.float32),
            load_config={"type": "test"},
            bc_type="CLAMP",
            strategy="BEZIER",
            resolution=resolution,
            final_compliance=100.0,
            final_volume=0.2,
        )
        save_episode(temp_db, ep_info)

        step_rec = StepRecord(
            episode_id="ep2",
            step=0,
            phase=Phase.REFINEMENT,
            density=np.zeros(resolution, dtype=np.float32),
            policy_add=np.zeros(resolution, dtype=np.float32),
            policy_remove=np.zeros(resolution, dtype=np.float32),
            fitness_score=0.8,
            is_final_step=False,
            is_connected=True,
        )
        save_step(temp_db, step_rec)

        dataset = TopologyDatasetV31(temp_db, augment=False, preload_to_ram=True)
        assert len(dataset) == 1
        sample = dataset[0]
        assert sample["state"].shape == (7, 10, 10, 10)
        assert sample["policy"].shape == (2, 10, 10, 10)
        assert sample["value"].item() == pytest.approx(0.8)
        assert sample["phase"] == "REFINEMENT"
        assert sample["is_connected"] is True

    def test_phase_filter(self, temp_db):
        state = np.zeros((7, 10, 10, 10), dtype=np.float32)
        save_record(temp_db, TrainingRecord("ep1", 0, Phase.GROWTH, serialize_state(state), 0.0, True))
        save_record(temp_db, TrainingRecord("ep1", 1, Phase.REFINEMENT, serialize_state(state), 0.0, True))

        ds_growth = TopologyDatasetV31(temp_db, augment=False, phase_filter="GROWTH")
        assert len(ds_growth) == 1
        assert ds_growth[0]["phase"] == "GROWTH"

        ds_ref = TopologyDatasetV31(temp_db, augment=False, phase_filter="REFINEMENT")
        assert len(ds_ref) == 1
        assert ds_ref[0]["phase"] == "REFINEMENT"

    def test_v33_compressed_ram_blueprint_and_legal_masks(self, temp_db):
        res = (16, 8, 4)
        bc = np.zeros((3, *res), dtype=np.float32)
        bc[:, 0, :, :] = 1.0
        forces = np.zeros((3, *res), dtype=np.float32)
        forces[1, 15, 3:5, 1:3] = -1.0

        v_const = np.zeros(res, dtype=np.float32)
        v_const[:, 2:6, 1:3] = 1.0

        for ep_idx in range(2):
            ep_id = f"ep_{ep_idx}"
            save_episode(
                temp_db,
                EpisodeInfo(
                    episode_id=ep_id,
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
            dens = v_const.copy()
            # Step 0: non-stationary legal add move
            pol_add = np.zeros(res, dtype=np.float32)
            pol_add[5, 6, 1] = 2.0  # Adjacent neighbor of solid
            save_step(
                temp_db,
                StepRecord(
                    episode_id=ep_id,
                    step=0,
                    phase=Phase.REFINEMENT,
                    density=dens,
                    policy_add=pol_add,
                    policy_remove=np.zeros(res, dtype=np.float32),
                    fitness_score=0.55,
                    is_final_step=False,
                    is_connected=True,
                ),
            )
            # Step 1: stationary final step -> policy_mask == 0
            save_step(
                temp_db,
                StepRecord(
                    episode_id=ep_id,
                    step=1,
                    phase=Phase.REFINEMENT,
                    density=dens,
                    policy_add=np.zeros(res, dtype=np.float32),
                    policy_remove=np.zeros(res, dtype=np.float32),
                    fitness_score=0.72,
                    is_final_step=True,
                    is_connected=True,
                ),
            )

        split_path = temp_db.parent / "split.json"
        split_path.write_text(
            json.dumps({"train_episode_ids": ["ep_0"], "val_episode_ids": ["ep_1"]}),
            encoding="utf-8",
        )

        train_ds = TopologyDatasetV31(temp_db, augment=True, preload_to_ram=True, split_json=split_path, split="train")
        val_ds = TopologyDatasetV31(temp_db, augment=False, preload_to_ram=True, split_json=split_path, split="val")

        assert train_ds.get_episode_ids() == ["ep_0"]
        assert val_ds.get_episode_ids() == ["ep_1"]
        assert train_ds.get_memory_usage_mb() > 0.0

        s0 = val_ds[0]
        assert s0["policy_mask"].item() == 1.0
        assert s0["policy"].sum().item() == pytest.approx(1.0)
        assert s0["blueprint_mask"].item() == 1.0
        assert s0["blueprint"].shape == (1, *res)
        np.testing.assert_array_equal(s0["blueprint"][0].numpy(), v_const)

        s1 = val_ds[1]
        assert s1["policy_mask"].item() == 0.0
        assert s1["blueprint_mask"].item() == 1.0

        # Symmetry collate + multiprocessing worker test
        collate = make_symmetry_collate_fn(enable_rot90=True)
        loader = DataLoader(train_ds, batch_size=2, num_workers=2, collate_fn=collate)
        batch = next(iter(loader))
        assert batch["state"].shape[2:] in [(16, 8, 4), (8, 16, 4)]
        assert batch["blueprint"].shape[2:] == batch["state"].shape[2:]
        assert batch["legal_mask"].shape[2:] == batch["state"].shape[2:]
