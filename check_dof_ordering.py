
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.core.physics_model import initialize_cantilever_context, PhysicalProperties

def check_ordering():
    print("Initializing Context...")
    # Use the same resolution as the harvest script
    resolution = (64, 32, 8)
    ctx = initialize_cantilever_context(resolution, PhysicalProperties())
    
    # Get DG0 space (material field)
    V_mat = ctx.material_field.function_space
    
    # Get coordinates of all DOFs
    # tabulate_dof_coordinates() returns [num_dofs, 3]
    coords = V_mat.tabulate_dof_coordinates()
    
    print(f"Total DOFs: {coords.shape[0]}")
    print(f"Expected: {64*32*8}")
    
    print("\nFirst 10 coordinates:")
    for i in range(10):
        print(f"  {i}: {coords[i]}")
        
    print("\nChecking strides...")
    # Calculate differences between consecutive DOFs
    diffs = np.diff(coords, axis=0)
    
    # Check which axis changes most frequently (non-zero diff in first few steps)
    # We look at the first few changes
    for i in range(5):
        d = coords[i+1] - coords[i]
        changed_axis = np.where(np.abs(d) > 1e-6)[0]
        axis_names = ['X', 'Y', 'Z']
        changes = [axis_names[idx] for idx in changed_axis]
        print(f"  Step {i}->{i+1}: Changed {changes} (Delta: {d})")

    # Determine fastest axis
    # If X changes every step, it's X-fast.
    # If Z changes every step, it's Z-fast.
    
    d0 = coords[1] - coords[0]
    if abs(d0[0]) > 1e-6:
        print("\nCONCLUSION: X varies fastest (X-fast).")
    elif abs(d0[2]) > 1e-6:
        print("\nCONCLUSION: Z varies fastest (Z-fast).")
    elif abs(d0[1]) > 1e-6:
        print("\nCONCLUSION: Y varies fastest (Y-fast).")
    else:
        print("\nCONCLUSION: Unknown ordering.")

if __name__ == "__main__":
    check_ordering()
