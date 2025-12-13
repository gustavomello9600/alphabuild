
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
import struct
import numpy as np

# Adjust path to import backend
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "alphabuilder" / "web" / "backend"))

from main import app

client = TestClient(app)

def test_binary_endpoint_structure():
    # 1. List games to get a valid ID
    response = client.get("/selfplay/games?limit=1")
    if response.status_code != 200 or len(response.json()) == 0:
        pytest.skip("No games available for testing")
        
    game_id = response.json()[0]['game_id']
    
    # 2. Fetch Binary Data (first step)
    response = client.get(f"/selfplay/games/{game_id}/steps/binary?start=0&end=1")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    
    data = response.content
    assert len(data) > 40 # At least header
    
    # 3. Parse Header
    # <10i = 40 bytes
    header_size = 40
    header_data = data[:header_size]
    header = struct.unpack('<10i', header_data)
    
    step_idx = header[0]
    phase = header[1]
    D, H, W = header[2], header[3], header[4]
    data_length = header[5]
    
    assert step_idx == 0
    assert D > 0 and H > 0 and W > 0
    assert data_length > 0
    
    # 4. Verify Data Size matches claimed length
    # Total size should be header_size + data_length (since we requested 1 step)
    assert len(data) == header_size + data_length

def test_binary_vs_json_consistency():
    # Verify that binary data matches JSON data for the same step
    response = client.get("/selfplay/games?limit=1")
    if response.status_code != 200 or len(response.json()) == 0:
        pytest.skip("No games available for testing")
        
    game_id = response.json()[0]['game_id']
    
    # Fetch JSON
    json_resp = client.get(f"/selfplay/games/{game_id}/steps?start=0&end=1")
    json_step = json_resp.json()[0]
    
    # Fetch Binary
    bin_resp = client.get(f"/selfplay/games/{game_id}/steps/binary?start=0&end=1")
    data = bin_resp.content
    
    # Parse Binary Header
    header = struct.unpack('<10i', data[:40])
    D, H, W = header[2], header[3], header[4]
    
    # Parse Tensor Data
    # Tensor is immediately after header
    # Shape: (7, D, H, W) -> float32
    tensor_size = 7 * D * H * W * 4 # 4 bytes per float
    tensor_data_bytes = data[40 : 40 + tensor_size]
    tensor_arr = np.frombuffer(tensor_data_bytes, dtype=np.float32)
    
    # Compare with JSON
    # JSON tensor_data is flat list
    assert np.allclose(tensor_arr, np.array(json_step['tensor_data'], dtype=np.float32), atol=1e-5)
