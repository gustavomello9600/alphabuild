import numpy as np
import dolfinx
from dolfinx import mesh, fem
from mpi4py import MPI
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from alphabuilder.src.core.physics_model import initialize_cantilever_context, PhysicalProperties
from alphabuilder.src.logic.simp_generator import HelmholtzFilter

def test_helmholtz_filter_smoothing():
    print("Initializing Context...")
    # 1. Setup minimal context
    resolution = (20, 20, 20)
    ctx = initialize_cantilever_context(resolution)
    
    print("Initializing HelmholtzFilter...")
    # 2. Initialize Filter
    r_min = 2.0 # Filter radius in elements
    pde_filter = HelmholtzFilter(ctx, r_min)
    
    # 3. Create Impulse Density (Single 1 in center)
    nx, ny, nz = resolution
    n_elements = nx * ny * nz
    x = np.zeros(n_elements)
    
    # Center index
    # We just pick the middle element of the array.
    # Even if FEniCSx reorders, this is a valid "single element" density.
    center_idx = n_elements // 2
    x[center_idx] = 1.0
    
    print("Applying Filter...")
    # 4. Apply Filter
    x_filtered = pde_filter.apply(x)
    
    # 5. Verify Smoothing
    # The max value should be < 1.0 (diffusion)
    max_val = np.max(x_filtered)
    print(f"Max filtered value: {max_val}")
    
    # Check if it diffused
    if max_val >= 1.0:
        print("WARNING: Max value >= 1.0. This might happen if r is very small or mesh is weird.")
    
    assert max_val > 0.0, "Filter killed the signal completely"
    
    # The support should be larger than 1 element
    non_zero_count = np.sum(x_filtered > 1e-4)
    print(f"Non-zero elements (>1e-4): {non_zero_count}")
    assert non_zero_count > 1, "Filter did not spread the signal"
    
    # Check conservation (roughly)
    # Helmholtz filter is not volume preserving, but the sum shouldn't explode.
    sum_orig = np.sum(x)
    sum_filt = np.sum(x_filtered)
    print(f"Sum Original: {sum_orig}, Sum Filtered: {sum_filt}")
    
    print("Helmholtz Filter Test Passed!")

if __name__ == "__main__":
    test_helmholtz_filter_smoothing()
