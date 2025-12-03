"""
Unit tests for dataset module.
"""
import pytest
import numpy as np
import tempfile
import sqlite3
import pickle
import zlib
from pathlib import Path

from alphabuilder.src.neural.dataset import (
    TopologyDatasetV31,
    deserialize_state_legacy,
    deserialize_array,
    sparse_decode
)
from alphabuilder.src.logic.storage import (
    initialize_database,
    save_record,
    save_episode,
    save_step,
    serialize_state,
    serialize_array,
    TrainingRecord,
    EpisodeInfo,
    StepRecord,
    Phase
)

class TestDatasetDeserialization:
    """Tests for dataset deserialization helpers."""
    
    def test_deserialize_state_legacy_compressed(self):
        """Should correctly deserialize zlib-compressed pickle (legacy)."""
        original = np.random.rand(5, 5).astype(np.float32)
        # Manually compress like storage.py does
        blob = zlib.compress(pickle.dumps(original))
        
        restored = deserialize_state_legacy(blob)
        np.testing.assert_array_equal(original, restored)

    def test_deserialize_array_compressed(self):
        """Should correctly deserialize zlib-compressed pickle (v2)."""
        original = np.random.rand(5, 5).astype(np.float32)
        blob = zlib.compress(pickle.dumps(original))
        
        restored = deserialize_array(blob)
        np.testing.assert_array_equal(original, restored)


class TestTopologyDatasetV31:
    """Tests for TopologyDatasetV31."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            initialize_database(db_path)
            yield db_path

    def test_load_v1_legacy(self, temp_db):
        """Should load data from legacy v1 schema."""
        # Create legacy data
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
            metadata={"is_final_step": True}
        )
        save_record(temp_db, record)
        
        # Load dataset
        dataset = TopologyDatasetV31(temp_db, augment=False, preload_to_ram=True)
        assert len(dataset) == 1
        
        sample = dataset[0]
        assert sample['state'].shape == (7, 10, 10, 10)
        assert sample['policy'].shape == (2, 10, 10, 10)
        assert sample['value'].item() == 0.5
        assert sample['is_final'] is True

    def test_load_v2_optimized(self, temp_db):
        """Should load data from optimized v2 schema."""
        resolution = (10, 10, 10)
        
        # Create episode info
        ep_info = EpisodeInfo(
            episode_id="ep2",
            bc_masks=np.zeros((3,) + resolution, dtype=np.float32),
            forces=np.zeros((3,) + resolution, dtype=np.float32),
            load_config={"type": "test"},
            bc_type="CLAMP",
            strategy="BEZIER",
            resolution=resolution,
            final_compliance=100.0,
            final_volume=0.2
        )
        save_episode(temp_db, ep_info)
        
        # Create step record
        density = np.zeros(resolution, dtype=np.float32)
        policy_add = np.zeros(resolution, dtype=np.float32)
        policy_remove = np.zeros(resolution, dtype=np.float32)
        
        step_rec = StepRecord(
            episode_id="ep2",
            step=0,
            phase=Phase.REFINEMENT,
            density=density,
            policy_add=policy_add,
            policy_remove=policy_remove,
            fitness_score=0.8,
            is_final_step=False,
            is_connected=True
        )
        save_step(temp_db, step_rec)
        
        # Load dataset
        dataset = TopologyDatasetV31(temp_db, augment=False, preload_to_ram=True)
        assert len(dataset) == 1
        
        sample = dataset[0]
        # State should be 7 channels (1 density + 3 masks + 3 forces)
        assert sample['state'].shape == (7, 10, 10, 10)
        assert sample['policy'].shape == (2, 10, 10, 10)
        assert sample['value'].item() == pytest.approx(0.8)
        assert sample['phase'] == "REFINEMENT"
        assert sample['is_connected'] is True

    def test_phase_filter(self, temp_db):
        """Should filter by phase."""
        # Save one GROWTH and one REFINEMENT (using v1 for simplicity)
        state = np.zeros((7, 10, 10, 10), dtype=np.float32)
        
        rec1 = TrainingRecord("ep1", 0, Phase.GROWTH, serialize_state(state), 0.0, True)
        rec2 = TrainingRecord("ep1", 1, Phase.REFINEMENT, serialize_state(state), 0.0, True)
        
        save_record(temp_db, rec1)
        save_record(temp_db, rec2)
        
        ds_growth = TopologyDatasetV31(temp_db, phase_filter="GROWTH")
        assert len(ds_growth) == 1
        assert ds_growth[0]['phase'] == "GROWTH"
        
        ds_ref = TopologyDatasetV31(temp_db, phase_filter="REFINEMENT")
        assert len(ds_ref) == 1
        assert ds_ref[0]['phase'] == "REFINEMENT"
