import json
import os

path = 'alphabuilder/web/public/mock_episode.json'

if not os.path.exists(path):
    print(f"File not found: {path}")
    exit(1)

with open(path, 'r') as f:
    data = json.load(f)

print(f"Episode ID: {data.get('episode_id')}")
frames = data.get('frames', [])
print(f"Frames: {len(frames)}")

if len(frames) > 0:
    first = frames[0]
    print(f"Frame 0 keys: {first.keys()}")
    print(f"Frame 0 fitness: {first.get('fitness')}")
    print(f"Frame 0 policy present: {'policy' in first}")
    
    if 'policy' in first:
        p = first['policy']
        print(f"Policy keys: {p.keys()}")
        if 'add' in p:
            # Check dimensions of add
            add = p['add']
            print(f"Policy add type: {type(add)}")
            if isinstance(add, list):
                print(f"Policy add len (D): {len(add)}")
                if len(add) > 0 and isinstance(add[0], list):
                    print(f"Policy add[0] len (H): {len(add[0])}")
                    if len(add[0]) > 0 and isinstance(add[0][0], list):
                        print(f"Policy add[0][0] len (W): {len(add[0][0])}")

    # Check a middle frame too
    mid = frames[len(frames)//2]
    print(f"Frame {len(frames)//2} fitness: {mid.get('fitness')}")
    print(f"Frame {len(frames)//2} policy present: {'policy' in mid}")
