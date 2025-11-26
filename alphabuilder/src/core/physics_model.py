import numpy as np
import dolfinx
from dolfinx import mesh, fem
from mpi4py import MPI
from dataclasses import dataclass
import ufl
from petsc4py import PETSc

@dataclass
class PhysicalProperties:
    E: float = 1.0  # Young's Modulus (Normalized)
    nu: float = 0.3 # Poisson's Ratio
    rho: float = 1.0 # Density
    disp_limit: float = 100.0 # Loose limit for initial training
    penalty_epsilon: float = 0.1
    penalty_alpha: float = 10.0

import dolfinx.fem.petsc

@dataclass
class FEMContext:
    domain: mesh.Mesh
    V: fem.FunctionSpace
    bc: fem.DirichletBC
    problem: dolfinx.fem.petsc.LinearProblem
    u_sol: fem.Function
    dof_map: np.ndarray # Map from Grid Voxel to FEM Cell
    material_field: fem.Function # DG0 field for density

@dataclass
class SimulationResult:
    fitness: float
    max_displacement: float
    compliance: float
    valid: bool
    displacement_array: np.ndarray

def initialize_cantilever_context(resolution=(64, 32, 32), props: PhysicalProperties = None) -> FEMContext:
    """
    Initialize a 3D FEM context for a Cantilever Beam.
    Resolution: (L, H, W) -> (x, y, z)
    Physical Size: 2.0 x 1.0 x 1.0
    """
    L, H, W = 2.0, 1.0, 1.0
    nx, ny, nz = resolution
    
    # 1. Create 3D Hexahedral Mesh
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([L, H, W])],
        [nx, ny, nz],
        cell_type=mesh.CellType.hexahedron
    )
    
    # 2. Function Space (Vector Element for Displacement)
    # FEniCSx v0.8+: Use ufl.VectorElement and fem.FunctionSpace
    # FEniCSx v0.8+: Use basix.ufl and fem.functionspace
    import basix.ufl
    element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(domain.geometry.dim,))
    V = fem.functionspace(domain, element)
    
    # 3. Boundary Conditions (Fix Left Face: x=0)
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
        
    fdim = domain.topology.dim - 1
    left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
    
    # Fix all components (0, 1, 2) -> (x, y, z)
    u_zero = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
    bc = fem.dirichletbc(u_zero, fem.locate_dofs_topological(V, fdim, left_facets), V)
    
    # 4. Variational Problem (Linear Elasticity)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Material Field (Density 0 or 1)
    # We use a DG0 space (constant per cell) to map voxels to elements
    V_mat = fem.functionspace(domain, ("DG", 0))
    material_field = fem.Function(V_mat)
    material_field.x.array[:] = 1.0 # Start full
    
    # SIMP Interpolation: E(rho) = E_min + (E_0 - E_min) * rho^p
    # For binary (0/1), this simplifies to E_0 * rho (plus small epsilon to avoid singularity)
    if props is None:
        props = PhysicalProperties()
        
    E_0 = props.E
    E_min = 1e-6
    rho = material_field
    E = E_min + (E_0 - E_min) * rho # Linear for binary is fine, or rho**3 for SIMP
    nu = props.nu
    
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
        
    def sigma(u):
        return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2.0 * mu * epsilon(u)
        
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(fem.Constant(domain, np.zeros(3, dtype=PETSc.ScalarType)), v) * ufl.dx
    
    # 5. Linear Problem Setup
    # Use Iterative Solver (CG + GAMG) for 3D Elasticity to save memory
    # LU is too expensive for 64x32x32 (200k DOFs) on Colab
    solver_options = {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-6,
        "ksp_atol": 1e-10,
        "ksp_max_it": 1000
    }
    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options=solver_options, petsc_options_prefix="cantilever")
    
    # 6. DOF Map (Voxel -> Cell Index)
    # In FEniCSx with structured box mesh, cell indices usually follow a pattern.
    # However, to be safe, we rely on the fact that we update 'material_field' which is DG0.
    # The DG0 DoFs correspond 1-to-1 with cells.
    # We assume the grid passed to solver matches the mesh resolution (nx, ny, nz).
    # We need a map if the ordering differs, but for BoxMesh, it's usually lexicographical.
    # Let's create a placeholder map.
    dof_map = np.arange(nx * ny * nz, dtype=np.int32)
    
    u_sol = fem.Function(V)
    
    return FEMContext(
        domain=domain,
        V=V,
        bc=bc,
        problem=problem,
        u_sol=u_sol,
        dof_map=dof_map,
        material_field=material_field
    )
