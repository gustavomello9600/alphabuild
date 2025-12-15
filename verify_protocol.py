
import requests
import json
import struct
import io

# Simulate the Request
# from alphabuilder.web.backend.routers.selfplay import get_selfplay_game_steps_binary

from alphabuilder.src.logic.selfplay.storage import list_games
# Mock or import fastapi_utils
try:
    from fastapi_utils import get_selfplay_db_path
except ImportError:
    from alphabuilder.web.backend.fastapi_utils import get_selfplay_db_path

def verify_protocol():
    print("Listing games...")
    db_path = get_selfplay_db_path(False)
    games = list_games(db_path, limit=1)
    
    if not games:
        print("No games found to verify.")
        return

    game_id = games[0].game_id
    print(f"Verifying for Game ID: {game_id}")

    # Call Real Backend
    url = f"http://localhost:8000/selfplay/games/{game_id}/steps/binary?start=0&end=1"
    print(f"Requesting: {url}")
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return
        
    data = response.content
    print(f"Received {len(data)} bytes")
    
    # Parse Header
    # 40 bytes: step(i), phase(i), D(i), H(i), W(i), len(i), sims(i), val(i), conn(i), vol(i)
    header = struct.unpack('<10i', data[:40])
    print(f"Header: {header}")
    
    step_index = header[0]
    data_length = header[5]
    
    # Check bounds
    if len(data) < 40 + data_length:
        print("Error: Data shorter than expected")
        return

    # Skip to extension block
    # We need to calculate offsets roughly or just parse everything
    D, H, W = header[2], header[3], header[4]
    spatial = D * H * W
    
    # Offsets
    offset = 40
    
    # Tensor (7 * spatial * 4)
    tensor_size = 7 * spatial * 4
    offset += tensor_size
    
    # Policies (4 * spatial * 4)
    offset += 4 * spatial * 4
    
    # Selected Actions
    num_actions = struct.unpack('<i', data[offset:offset+4])[0]
    offset += 4
    offset += num_actions * 24 # 5 ints + 1 float
    
    # Extension Block (20 bytes)
    # n_islands, loose, is_connected, compliance, penalty
    ext = struct.unpack('<3i2f', data[offset:offset+20])
    print(f"Extension: {ext}")
    offset += 20
    
    # NEW: Reward Components
    rc_len = struct.unpack('<i', data[offset:offset+4])[0]
    offset += 4
    
    print(f"Reward Components Length: {rc_len}")
    
    if rc_len > 0:
        rc_bytes = data[offset:offset+rc_len]
        rc_json = rc_bytes.decode('utf-8')
        print(f"Reward Components JSON: {rc_json}")
        rc_obj = json.loads(rc_json)
        print("SUCCESS: Parsed Reward Components")
    else:
        print("WARNING: No reward components found (length 0). This might be valid if they are None.")
        
    print("Verification Complete")

if __name__ == "__main__":
    verify_protocol()
