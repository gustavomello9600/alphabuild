"""
FEM Solver Verification Test.

Verifies that the FEniCSx-based FEM solver produces correct results
by comparing against analytical solution for a cantilever beam benchmark.
"""
import numpy as np
from mpi4py import MPI
from dolfinx import mesh

from alphabuilder.src.logic.fenitop.fem import form_fem


def run_cantilever_verification():
    """Run cantilever beam verification test."""
    # 1. Analytic Parameters
    L, H, W = 20.0, 4.0, 4.0
    nx, ny, nz = 20, 4, 4  # Coarse mesh for speed
    E = 1000.0
    nu = 0.3
    TotalForceY = -10.0
    traction_y = TotalForceY / (H * W)
    
    print(f"--- FEM Verification ---")
    print(f"Dimensions: {L}x{H}x{W}")
    print(f"Mesh: {nx}x{ny}x{nz}")
    print(f"E: {E}, nu: {nu}")
    print(f"Total Load: {TotalForceY} (Traction: {traction_y})")

    # 2. Analytical Solution (Euler-Bernoulli Beam Theory)
    I = (W * (H**3)) / 12.0
    delta_analytic = (TotalForceY * (L**3)) / (3 * E * I)
    print(f"Moment of Inertia I: {I:.4f}")
    print(f"Expected Tip Displacement (Analytic): {delta_analytic:.6f}")

    # 3. FEM Setup
    comm = MPI.COMM_WORLD
    domain = mesh.create_box(
        comm, 
        [np.array([0.0, 0.0, 0.0]), np.array([L, H, W])],
        [nx, ny, nz],
        mesh.CellType.hexahedron
    )
    
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
        
    def load_boundary(x):
        return np.isclose(x[0], L)

    fem_config = {
        "mesh": domain,
        "young's modulus": E,
        "poisson's ratio": nu,
        "quadrature_degree": 2,
        "body_force": (0, 0, 0),
        "petsc_options": {"ksp_type": "preonly", "pc_type": "lu"},
        "bcs": [([0.0, 0.0, 0.0], left_boundary, None)],
        "traction_bcs": [((0, traction_y, 0), load_boundary)]
    }
    
    opt_config = {
        "penalty": 1.0,
        "epsilon": 1e-6,
        "opt_compliance": True,
        "in_spring": None,
        "out_spring": None
    }
    
    # 4. Form and Solve
    linear_problem, u_field, _, _, rho_phys_field = form_fem(fem_config, opt_config)
    rho_phys_field.x.array[:] = 1.0
    
    print("Solving FEM...")
    linear_problem.solve_fem()
    
    # 5. Measure Max Y Displacement
    vals = u_field.x.array.reshape((-1, 3))
    min_y = np.min(vals[:, 1])
    print(f"FEM Tip Displacement: {min_y:.6f}")

    # 6. Compare
    rel_error = abs(min_y - delta_analytic) / abs(delta_analytic)
    print(f"Relative Error: {rel_error*100:.2f}%")
    
    # Assert (10% margin for coarse mesh)
    if rel_error > 0.10:
        print("FAIL: Error too high")
        return False
    else:
        print("PASS: Analytical match confirmed")
        return True


if __name__ == "__main__":
    try:
        success = run_cantilever_verification()
        exit(0 if success else 1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
