"""
FEM Solver Smoke Test - Full Solid Cantilever Beam

Tests the solver with a simple case:
- Full solid beam (all voxels = 1)
- Fixed left face
- Point load at tip center
- Expected: Analytical solution for cantilever deflection
"""

import numpy as np
from alphabuilder.src.core.physics_model import initialize_cantilever_context, PhysicalProperties
from alphabuilder.src.core.solver import solve_topology_3d

def test_full_solid_cantilever():
    """
    Smoke test: Full solid cantilever with tip load.
    
    Analytical solution for cantilever beam:
    δ_max = (F * L^3) / (3 * E * I)
    
    For rectangular cross-section:
    I = (b * h^3) / 12
    
    Where:
    - F = 1.0 (load)
    - L = 2.0 (length)
    - b = 1.0 (width/depth)
    - h = 1.0 (height)
    - E = 1.0 (Young's modulus)
    
    I = (1.0 * 1.0^3) / 12 = 0.0833
    δ_max = (1.0 * 2.0^3) / (3 * 1.0 * 0.0833) = 8.0 / 0.25 = 32.0
    
    Compliance C = F * δ = 1.0 * 32.0 = 32.0
    """
    
    print("="*80)
    print("FEM Solver Smoke Test - Full Solid Cantilever")
    print("="*80)
    
    # Test with different resolutions
    resolutions = [
        (16, 8, 4),   # Coarse
        (32, 16, 8),  # Medium
        (64, 32, 8),  # Fine (current default)
    ]
    
    for resolution in resolutions:
        print(f"\n{'─'*80}")
        print(f"Resolution: {resolution[0]}x{resolution[1]}x{resolution[2]} ({np.prod(resolution)} cells)")
        print(f"{'─'*80}")
        
        # Initialize context
        ctx = initialize_cantilever_context(resolution=resolution)
        props = PhysicalProperties(
            E=1.0,
            nu=0.3,
            disp_limit=100.0
        )
        
        # Create full solid tensor
        D, H, W = resolution
        tensor = np.zeros((5, D, H, W), dtype=np.float32)
        
        # Channel 0: Full solid (all 1s)
        tensor[0, :, :, :] = 1.0
        
        # Channel 1: Support (left face)
        tensor[1, :, :, 0] = 1.0
        
        # Channel 3: Load at tip center (F_y = -1.0)
        tensor[3, D//2, H//2, W-1] = -1.0
        
        # Solve
        result = solve_topology_3d(tensor, ctx, props)
        
        # Analytical solution
        L, b, h = 2.0, 1.0, 1.0
        I = (b * h**3) / 12
        delta_analytical = (1.0 * L**3) / (3 * 1.0 * I)
        compliance_analytical = 1.0 * delta_analytical
        
        # Results
        print(f"\n📊 Results:")
        print(f"  Max Displacement (FEM):        {result.max_displacement:.4f}")
        print(f"  Max Displacement (Analytical): {delta_analytical:.4f}")
        print(f"  Ratio (FEM/Analytical):        {result.max_displacement/delta_analytical:.4f}")
        print(f"")
        print(f"  Compliance (FEM):              {result.compliance:.4f}")
        print(f"  Compliance (Analytical):       {compliance_analytical:.4f}")
        print(f"  Ratio (FEM/Analytical):        {result.compliance/compliance_analytical:.4f}")
        print(f"")
        print(f"  Valid:                         {result.valid}")
        print(f"  Volume Fraction:               {np.mean(tensor[0]):.4f}")
        
        # Check if reasonable
        if 0.8 < result.max_displacement/delta_analytical < 1.2:
            print(f"\n✅ PASS: Displacement within 20% of analytical")
        else:
            print(f"\n❌ FAIL: Displacement deviates significantly from analytical")
        
        if 0.8 < result.compliance/compliance_analytical < 1.2:
            print(f"✅ PASS: Compliance within 20% of analytical")
        else:
            print(f"❌ FAIL: Compliance deviates significantly from analytical")
    
    print(f"\n{'='*80}")
    print("Smoke Test Complete")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    test_full_solid_cantilever()
