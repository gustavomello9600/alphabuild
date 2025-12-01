import json
import numpy as np

def analyze():
    with open('alphabuilder/web/src/api/mock_episode.json', 'r') as f:
        data = json.load(f)
    
    first_step = data[0]
    tensor_data = first_step['tensor']['data']
    shape = first_step['tensor']['shape'] # [C, D, H, W]
    
    C, D, H, W = shape
    print(f"Shape: {shape}")
    
    tensor = np.array(tensor_data).reshape(C, D, H, W)
    
    # Channel 1: Support
    supports = np.where(tensor[1] > 0.5)
    if len(supports[0]) > 0:
        print("\nSupports (Channel 1):")
        print(f"  D range: {np.min(supports[0])} - {np.max(supports[0])}")
        print(f"  H range: {np.min(supports[1])} - {np.max(supports[1])}")
        print(f"  W range: {np.min(supports[2])} - {np.max(supports[2])}")
        print(f"  Count: {len(supports[0])}")
    else:
        print("\nNo Supports found!")

    # Channel 3: Load
    loads = np.where(tensor[3] != 0)
    if len(loads[0]) > 0:
        print("\nLoads (Channel 3):")
        print(f"  D range: {np.min(loads[0])} - {np.max(loads[0])}")
        print(f"  H range: {np.min(loads[1])} - {np.max(loads[1])}")
        print(f"  W range: {np.min(loads[2])} - {np.max(loads[2])}")
        print(f"  Count: {len(loads[0])}")
        
        # Check centering
        d_center = np.mean(loads[0])
        h_center = np.mean(loads[1])
        print(f"  Center (D, H): ({d_center:.1f}, {h_center:.1f})")
        print(f"  Expected Center: ({D/2 - 0.5:.1f}, {H/2 - 0.5:.1f})")
    else:
        print("\nNo Loads found!")

if __name__ == "__main__":
    analyze()
