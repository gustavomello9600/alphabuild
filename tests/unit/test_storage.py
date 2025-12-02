"""
Unit tests for storage module.
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path

from alphabuilder.src.logic.storage import (
    Phase,
    TrainingRecord,
    initialize_database,
    save_record,
    serialize_state,
    deserialize_state,
    get_episode_count,
    generate_episode_id
)


class TestPhaseEnum:
    """Tests for Phase enumeration."""
    
    def test_growth_value(self):
        """GROWTH should have correct string value."""
        assert Phase.GROWTH.value == "GROWTH"
    
    def test_refinement_value(self):
        """REFINEMENT should have correct string value."""
        assert Phase.REFINEMENT.value == "REFINEMENT"


class TestSerializeDeserialize:
    """Tests for state serialization."""
    
    def test_roundtrip_float32(self):
        """Float32 tensor should survive serialization."""
        original = np.random.rand(7, 64, 32, 8).astype(np.float32)
        
        blob = serialize_state(original)
        restored = deserialize_state(blob)
        
        np.testing.assert_array_equal(original, restored)
    
    def test_roundtrip_float64(self):
        """Float64 tensor should survive serialization."""
        original = np.random.rand(2, 64, 32, 8).astype(np.float64)
        
        blob = serialize_state(original)
        restored = deserialize_state(blob)
        
        np.testing.assert_array_equal(original, restored)
    
    def test_preserves_dtype(self):
        """Serialization should preserve dtype."""
        original_f32 = np.zeros((5, 5), dtype=np.float32)
        original_f64 = np.zeros((5, 5), dtype=np.float64)
        
        restored_f32 = deserialize_state(serialize_state(original_f32))
        restored_f64 = deserialize_state(serialize_state(original_f64))
        
        assert restored_f32.dtype == np.float32
        assert restored_f64.dtype == np.float64
    
    def test_preserves_shape(self):
        """Serialization should preserve shape."""
        original = np.zeros((3, 10, 20, 30))
        
        restored = deserialize_state(serialize_state(original))
        
        assert restored.shape == (3, 10, 20, 30)


class TestDatabaseOperations:
    """Tests for database operations."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            initialize_database(db_path)
            yield db_path
    
    def test_initialize_creates_file(self, temp_db):
        """Initialize should create database file."""
        assert temp_db.exists()
    
    def test_initialize_creates_table(self, temp_db):
        """Initialize should create training_data table."""
        import sqlite3
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_data'")
        result = cursor.fetchone()
        
        conn.close()
        assert result is not None
    
    def test_save_and_count(self, temp_db):
        """Saving records should increase count."""
        assert get_episode_count(temp_db) == 0
        
        state = np.zeros((7, 64, 32, 8), dtype=np.float32)
        policy = np.zeros((2, 64, 32, 8), dtype=np.float32)
        
        record = TrainingRecord(
            episode_id="test-episode-1",
            step=0,
            phase=Phase.GROWTH,
            state_blob=serialize_state(state),
            policy_blob=serialize_state(policy),
            fitness_score=0.5,
            valid_fem=True,
            metadata={"test": True}
        )
        
        save_record(temp_db, record)
        
        assert get_episode_count(temp_db) == 1
    
    def test_save_multiple_steps_same_episode(self, temp_db):
        """Multiple steps in same episode should count as 1 episode."""
        state = np.zeros((7, 64, 32, 8), dtype=np.float32)
        
        for step in range(5):
            record = TrainingRecord(
                episode_id="test-episode-1",
                step=step,
                phase=Phase.GROWTH,
                state_blob=serialize_state(state),
                fitness_score=0.5,
                valid_fem=True
            )
            save_record(temp_db, record)
        
        assert get_episode_count(temp_db) == 1
    
    def test_save_multiple_episodes(self, temp_db):
        """Different episode IDs should count separately."""
        state = np.zeros((7, 64, 32, 8), dtype=np.float32)
        
        for ep in range(3):
            record = TrainingRecord(
                episode_id=f"test-episode-{ep}",
                step=0,
                phase=Phase.GROWTH,
                state_blob=serialize_state(state),
                fitness_score=0.5,
                valid_fem=True
            )
            save_record(temp_db, record)
        
        assert get_episode_count(temp_db) == 3
    
    def test_metadata_serialization(self, temp_db):
        """Metadata should be stored as JSON."""
        import sqlite3
        import json
        
        state = np.zeros((7, 64, 32, 8), dtype=np.float32)
        metadata = {"compliance": 100.5, "vol_frac": 0.1, "nested": {"key": "value"}}
        
        record = TrainingRecord(
            episode_id="test-episode",
            step=0,
            phase=Phase.GROWTH,
            state_blob=serialize_state(state),
            fitness_score=0.5,
            valid_fem=True,
            metadata=metadata
        )
        save_record(temp_db, record)
        
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT metadata FROM training_data WHERE episode_id='test-episode'")
        stored_json = cursor.fetchone()[0]
        conn.close()
        
        stored_metadata = json.loads(stored_json)
        assert stored_metadata == metadata


class TestGenerateEpisodeId:
    """Tests for episode ID generation."""
    
    def test_returns_string(self):
        """Should return a string."""
        episode_id = generate_episode_id()
        assert isinstance(episode_id, str)
    
    def test_unique_ids(self):
        """Should generate unique IDs."""
        ids = [generate_episode_id() for _ in range(100)]
        assert len(set(ids)) == 100
    
    def test_uuid_format(self):
        """Should be valid UUID format."""
        import uuid
        episode_id = generate_episode_id()
        # Should not raise
        uuid.UUID(episode_id)

