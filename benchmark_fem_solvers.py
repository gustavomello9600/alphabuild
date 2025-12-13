"""
FEM Solver Benchmark - Compare configurations against analytical solution
Uses cantilever beam with known analytical displacement at tip
"""

import time
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.mesh import create_box, CellType
from dolfinx.fem import functionspace, Function, dirichletbc, locate_dofs_topological, Constant, form
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
import ufl
import basix.ufl

# Small mesh for quick testing (cantilever beam)
nx, ny, nz = 20, 4, 2
Lx, Ly, Lz = 10.0, 2.0, 1.0

# Material properties
E = 1000.0
nu = 0.3
force = -1.0

# Analytical solution for cantilever beam tip displacement (Euler-Bernoulli)
I = Lz * (Ly**3) / 12
delta_analytical = abs(force) * (Lx**3) / (3 * E * I)
print("=== Cantilever Beam Benchmark ===")
print(f"Mesh: {nx}x{ny}x{nz} = {nx*ny*nz} elements")
print(f"DOFs: ~{3*(nx+1)*(ny+1)*(nz+1)} (vector)")
print(f"Analytical tip displacement: {delta_analytical:.6f}")
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
print()

# Create mesh
comm = MPI.COMM_WORLD
domain = create_box(comm, [[0, 0, 0], [Lx, Ly, Lz]], [nx, ny, nz], CellType.hexahedron)

# Function spaces
v_element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(domain.geometry.dim,))
V = functionspace(domain, v_element)

# Material
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

def epsilon(u): 
    return ufl.sym(ufl.grad(u))

def sigma(u): 
    return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx

# Body force (as if distributed load at tip - simplified)
f = Constant(domain, np.array([0.0, force/(Ly*Lz), 0.0], dtype=PETSc.ScalarType))
L = ufl.dot(f, v) * ufl.dx

# BC: Fix x=0 face
def left_boundary(x): 
    return np.isclose(x[0], 0.0)

fdim = domain.topology.dim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
u_zero = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
bc = dirichletbc(u_zero, locate_dofs_topological(V, fdim, left_facets), V)

# Solver configurations to test
SOLVERS = [
    # (name, ksp_type, pc_type, extra_options)
    ("CG+Jacobi", "cg", "jacobi", {}),
    ("CG+SOR", "cg", "sor", {}),
    ("CG+ILU", "cg", "ilu", {}),
    ("CG+GAMG", "cg", "gamg", {"mg_levels_ksp_type": "chebyshev", "mg_levels_pc_type": "jacobi"}),
    ("CG+GAMG-SOR", "cg", "gamg", {"mg_levels_ksp_type": "chebyshev", "mg_levels_pc_type": "sor"}),
    ("GMRES+Jacobi", "gmres", "jacobi", {}),
    ("GMRES+ILU", "gmres", "ilu", {}),
    ("GMRES+GAMG", "gmres", "gamg", {"mg_levels_ksp_type": "chebyshev", "mg_levels_pc_type": "jacobi"}),
    ("MINRES+Jacobi", "minres", "jacobi", {}),
    ("MINRES+GAMG", "minres", "gamg", {"mg_levels_ksp_type": "chebyshev", "mg_levels_pc_type": "jacobi"}),
    ("LU-MUMPS", "preonly", "lu", {"pc_factor_mat_solver_type": "mumps"}),
    ("Chol-MUMPS", "preonly", "cholesky", {"pc_factor_mat_solver_type": "mumps"}),
]

header = f"{'Solver':<18} {'Time(s)':<10} {'TipDisp':<12} {'Err%':<10}"
print(header)
print("-" * len(header))

results = []
for idx, (name, ksp, pc, extra) in enumerate(SOLVERS):
    try:
        opts = {
            "ksp_type": ksp, 
            "pc_type": pc, 
            "ksp_rtol": 1e-8, 
            "ksp_max_it": 1000, 
            **extra
        }
        
        t0 = time.time()
        problem = LinearProblem(a, L, bcs=[bc], petsc_options=opts, petsc_options_prefix=f"bench{idx}")
        u_sol = problem.solve()
        elapsed = time.time() - t0
        
        # Get tip displacement
        u_array = u_sol.x.array.reshape(-1, 3)
        coords = V.tabulate_dof_coordinates()
        tip_mask = np.isclose(coords[:, 0], Lx)
        tip_disp = np.max(np.abs(u_array[tip_mask, 1])) if np.any(tip_mask) else 0.0
        
        # Error relative to analytical
        error_pct = 100 * abs(tip_disp - delta_analytical) / delta_analytical
        
        print(f"{name:<18} {elapsed:<10.4f} {tip_disp:<12.6f} {error_pct:<10.2f}")
        results.append((name, elapsed, tip_disp, error_pct))
        
    except Exception as e:
        print(f"{name:<18} FAILED: {str(e)[:35]}")
        results.append((name, float("inf"), 0, 100))

print()
print("=== Summary ===")
valid = [(n, t, d, e) for n, t, d, e in results if t < 100]
if valid:
    fastest = min(valid, key=lambda x: x[1])
    most_accurate = min(valid, key=lambda x: x[3])
    print(f"Fastest: {fastest[0]} ({fastest[1]:.4f}s, {fastest[3]:.2f}% error)")
    print(f"Most accurate: {most_accurate[0]} ({most_accurate[3]:.2f}% error, {most_accurate[1]:.4f}s)")
    
    # Find best trade-off (fast AND accurate)
    scores = [(n, t + e/10, t, e) for n, t, d, e in valid]  # Weighted: time + error/10
    best_overall = min(scores, key=lambda x: x[1])
    print(f"Best overall: {best_overall[0]} ({best_overall[2]:.4f}s, {best_overall[3]:.2f}% error)")
