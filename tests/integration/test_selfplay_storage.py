
import pytest
import numpy as np
import tempfile
import sqlite3
from pathlib import Path
import json

from alphabuilder.src.logic.selfplay.storage import (
    initialize_selfplay_db,
    save_game,
    record_step,
    load_game,
    load_all_game_steps,
    list_games,
    GameInfo,
    GameStep,
    Phase,
    SelectedAction,
    MCTSStats
)
from typing import Dict, List, Any
from dataclasses import dataclass

@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "selfplay.db"
        initialize_selfplay_db(db_path)
        yield db_path

class TestSelfPlayStorage:

    def test_save_and_load_game(self, temp_db):
        """Test saving and loading game metadata."""
        game_id = "test-game-1"
        game = GameInfo(
            game_id=game_id,
            neural_engine="simple",
            checkpoint_version="v1",
            bc_masks=np.zeros((3, 10, 10, 10)),
            forces=np.zeros((3, 10, 10, 10)),
            load_config={'x': 5},
            bc_type="fixed",
            resolution=(10, 10, 10),
            final_score=0.8,
            final_compliance=100.0,
            final_volume=0.3,
            total_steps=50
        )
        
        save_game(temp_db, game)
        
        loaded = load_game(temp_db, game_id)
        assert loaded is not None
        assert loaded.game_id == game_id
        assert loaded.final_score == 0.8
        assert loaded.resolution == (10, 10, 10)
        np.testing.assert_array_equal(loaded.bc_masks, game.bc_masks)

    def test_save_and_load_step(self, temp_db):
        """Test saving and loading a game step with new reward fields."""
        game_id = "test-game-1"
        resolution = (10, 10, 10)
        
        # Must save game first for foreign key constraint
        save_game(temp_db, GameInfo(
            game_id=game_id,
            neural_engine="test",
            checkpoint_version="v1",
            bc_masks=np.zeros((3, 10, 10, 10)),
            forces=np.zeros((3, 10, 10, 10)),
            load_config={},
            bc_type="test",
            resolution=resolution
        ))
        
        # Mock State
        @dataclass
        class MockState:
            game_id: str
            current_step: int
            phase: Phase
            density: np.ndarray
        
        state = MockState(
            game_id=game_id,
            current_step=1,
            phase=Phase.GROWTH,
            density=np.zeros(resolution)
        )
        
        # Mock Result
        @dataclass
        class MockResult:
            visit_distribution: Dict
            actions: List
            num_simulations: int
            root: Any = None
            
        result = MockResult(
            visit_distribution={(0, 1, 1, 1): 10},
            actions=[(0, 1, 1, 1)],
            num_simulations=100
        )
        
        # Call record_step
        record_step(
            db_path=temp_db,
            state=state,
            result=result,
            value=0.7,
            policy_add=np.zeros(resolution),
            policy_remove=np.zeros(resolution),
            island_analysis={
                'n_islands': 2,
                'loose_voxels': 5,
                'is_connected': False
            },
            compliance_fem=123.45,
            max_displacement=10.0,
            island_penalty=0.1,
            volume_fraction=0.25,
            executed_actions=[(0, 1, 1, 1)] # Simulate PV sequence
        )
        
        loaded_steps = load_all_game_steps(temp_db, game_id, resolution)
        assert len(loaded_steps) == 1
        loaded = loaded_steps[0]
        
        assert loaded.step == 1
        assert loaded.n_islands == 2
        assert loaded.loose_voxels == 5
        assert loaded.is_connected == False
        assert loaded.compliance_fem == pytest.approx(123.45)
        assert loaded.max_displacement == pytest.approx(10.0)
        assert loaded.island_penalty == pytest.approx(0.1)
        assert loaded.volume_fraction == pytest.approx(0.25)
        
        # Check visits from visit_distribution or executed_actions
        # record_step logic: if executed_actions provided, it records them.
        # But visit count logic tries to find them in tree or uses visit_dist fallback.
        # Our mock visit_dist has counts.
        assert loaded.selected_actions[0].visits == 10

    def test_list_games(self, temp_db):
        """Test listing games."""
        save_game(temp_db, GameInfo(
            game_id="g1", neural_engine="e1", checkpoint_version="v1",
            bc_masks=np.zeros((3,5,5,5)), forces=np.zeros((3,5,5,5)),
            load_config={}, bc_type="t", resolution=(5,5,5),
            final_score=0.9
        ))
        save_game(temp_db, GameInfo(
            game_id="g2", neural_engine="e1", checkpoint_version="v2",
            bc_masks=np.zeros((3,5,5,5)), forces=np.zeros((3,5,5,5)),
            load_config={}, bc_type="t", resolution=(5,5,5),
            final_score=0.8
        ))
        
        games = list_games(temp_db)
        assert len(games) == 2
        ids = sorted([g.game_id for g in games])
        assert ids == ["g1", "g2"]
        assert games[0].neural_engine == "e1"

