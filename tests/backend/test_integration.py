
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys
import shutil

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alphabuilder.web.backend.main import app
from alphabuilder.src.logic.storage import initialize_database, save_episode, EpisodeInfo
from alphabuilder.src.logic.selfplay.storage import initialize_selfplay_db, save_game, GameInfo
import numpy as np

client = TestClient(app)

@pytest.fixture
def test_db_path(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d / "test_training.db"

@pytest.fixture
def test_selfplay_db_path(tmp_path):
    d = tmp_path / "data" 
    d.mkdir()
    # Mock the selfplay path in the backend configuration
    # Since we can't easily patch the constant in main.py directly if it's imported at top level,
    # we rely on overriding finding logic or mocking.
    # For now, let's create a temporary DB and use dependency injection or patching if needed.
    # But `fastapi_utils` resolves paths. We can mock `fastapi_utils.SELFPLAY_DB_PATH`.
    return d / "selfplay_games.db"


def test_list_databases(test_db_path):
    # Setup dummy DB
    initialize_database(test_db_path)
    
    # We need to temporarily patch DATA_SEARCH_PATHS to include tmp_path/data
    from alphabuilder.web.backend import fastapi_utils
    original_paths = fastapi_utils.DATA_SEARCH_PATHS
    fastapi_utils.DATA_SEARCH_PATHS = [test_db_path.parent]
    
    try:
        response = client.get("/databases")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert any(d['name'] == 'test_training' for d in data)
    finally:
        fastapi_utils.DATA_SEARCH_PATHS = original_paths

def test_get_episode_metadata(test_db_path):
    initialize_database(test_db_path)
    
    # Create a dummy episode
    ep_id = "test-ep-1"
    info = EpisodeInfo(
        episode_id=ep_id,
        bc_masks=np.zeros((3, 10, 10, 10)),
        forces=np.zeros((3, 10, 10, 10)),
        load_config={},
        bc_type="test",
        strategy="test",
        resolution=(10, 10, 10),
        final_compliance=1.0,
        final_volume=0.2
    )
    save_episode(test_db_path, info)
    
    from alphabuilder.web.backend import fastapi_utils
    original_paths = fastapi_utils.DATA_SEARCH_PATHS
    fastapi_utils.DATA_SEARCH_PATHS = [test_db_path.parent]
    
    try:
        # First list to get sure endpoint works
        resp_list = client.get(f"/databases/{test_db_path.name}/episodes")
        assert resp_list.status_code == 200
        
        # Then get metadata
        resp_meta = client.get(f"/databases/{test_db_path.name}/episodes/{ep_id}/metadata")
        assert resp_meta.status_code == 200
        data = resp_meta.json()
        assert data['episode_id'] == ep_id
        assert data['final_compliance'] == 1.0
        
    finally:
        fastapi_utils.DATA_SEARCH_PATHS = original_paths

def test_selfplay_endpoints(tmp_path):
    # Setup
    db_path = tmp_path / "selfplay_games.db"
    initialize_selfplay_db(db_path)
    
    # Patch path
    from alphabuilder.web.backend import fastapi_utils
    orig_path = fastapi_utils.SELFPLAY_DB_PATH
    fastapi_utils.SELFPLAY_DB_PATH = db_path
    
    try:
        # 1. List (Empty)
        resp = client.get("/selfplay/games")
        assert resp.status_code == 200
        assert resp.json() == []
        
        # 2. Add Game
        game_id = "game-1"
        game = GameInfo(
            game_id=game_id,
            neural_engine="test",
            checkpoint_version="v1",
            bc_masks=np.zeros((3, 8, 8, 8)),
            forces=np.zeros((3, 8, 8, 8)),
            load_config={},
            bc_type="test",
            resolution=(8, 8, 8)
        )
        save_game(db_path, game)
        
        # 3. List (Found)
        resp = client.get("/selfplay/games")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]['game_id'] == game_id
        
        # 4. Metadata
        resp = client.get(f"/selfplay/games/{game_id}/metadata")
        assert resp.status_code == 200
        meta = resp.json()
        assert meta['resolution'] == [8, 8, 8]
        
    finally:
        fastapi_utils.SELFPLAY_DB_PATH = orig_path
