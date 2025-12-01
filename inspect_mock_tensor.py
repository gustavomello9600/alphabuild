import json
import numpy as np
import sys

# Load data
path = 'alphabuilder/web/src/api/mock_episode.json'
print(f"Loading {path}...")
with open(path, 'r') as f:
    data = json.load(f)

# Extract tensor
step = data[0]
raw_data = step['tensor']['data']
shape = step['tensor']['shape']
tensor = np.array(raw_data).reshape(shape)

# Extract channels
density = tensor[0]
supports = tensor[1]
loads = tensor[3]

print("\n" + "="*40)
print("MOCK TENSOR LOADED")
print("="*40)
print(f"Shape: {tensor.shape} (Channels, Depth, Height, Width)")
print(f"Density Channel Shape: {density.shape}")
print(f"Density Range: [{density.min():.4f}, {density.max():.4f}]")
print(f"Active Voxels (>0.1): {np.sum(density > 0.1)}")

print("\n" + "="*40)
print("INSTRUCTIONS FOR INSPECTION")
print("="*40)
print("The following variables are available in the global scope:")
print("  - tensor:   The full 4D tensor (C, D, H, W)")
print("  - density:  Channel 0 (Density)")
print("  - supports: Channel 1 (Boundary Conditions)")
print("  - loads:    Channel 3 (Forces)")
print("\nExample commands:")
print("  >>> density[32, 16, :]  # Inspect a slice at Depth=32, Height=16")
print("  >>> np.where(supports > 0.5)  # Find support coordinates")
print("\n" + "="*40)
