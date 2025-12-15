"""
FEM Evaluator for Self-Play MCTS.

Provides a simple interface to calculate compliance and max displacement
for a given density grid using FEniCSx / FEniTop.

This is used in Phase 2 of self-play to get ground truth rewards.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

try:
    from mpi4py import MPI
    from dolfinx import mesh, fem
    from dolfinx.mesh import create_box, CellType
    from dolfinx.fem import functionspace, Function, dirichletbc, locate_dofs_topological, Constant, form
    from dolfinx.fem.petsc import LinearProblem
    from petsc4py import PETSc
    import ufl
    import basix.ufl
    HAS_FENICS = True
except ImportError:
    HAS_FENICS = False
    print("WARNING: FEniCSx not available. FEM evaluation disabled.")


@dataclass
class FEMResult:
    """Result from FEM evaluation."""
    compliance: float
    max_displacement: float
    vol_frac: float
    valid: bool
    displacement_map: Optional[np.ndarray] = None
    error_msg: Optional[str] = None


def evaluate_fem(
    density: np.ndarray,
    load_config: Dict[str, Any],
    E: float = 100.0,  # Match harvest data: Young's modulus = 100
    nu: float = 0.25,  # Match harvest data: Poisson's ratio = 0.25
    force_magnitude: float = -2.0  # Match harvest data: traction = (0, -2.0, 0)
) -> FEMResult:
    """
    Evaluate a density grid using FEM to get compliance and max displacement.
    
    Uses the same FEM parameters as the harvest data optimization (optimization.py):
    - E = 100, nu = 0.25
    - Surface traction load at right boundary (x=Lx) in load region
    - Fixed displacement BC at left boundary (x=0)
    
    Args:
        density: Density grid (nx, ny, nz) with values in [0, 1]
        load_config: Dictionary with 'x', 'y', 'z_start', 'z_end' keys
        E: Young's modulus (default 100 to match harvest)
        nu: Poisson's ratio (default 0.25 to match harvest)
        force_magnitude: Applied traction force (default -2.0 to match harvest)
        
    Returns:
        FEMResult with compliance, max_displacement, and validity
    """
    if not HAS_FENICS:
        return FEMResult(
            compliance=float('inf'),
            max_displacement=float('inf'),
            vol_frac=0.0,
            valid=False,
            error_msg="FEniCSx not available"
        )
    
    try:
        comm = MPI.COMM_WORLD
        nx, ny, nz = density.shape
        
        # Volume fraction
        vol_frac = float(np.mean(density > 0.5))
        
        if vol_frac < 0.001:
            return FEMResult(
                compliance=float('inf'),
                max_displacement=float('inf'),
                vol_frac=vol_frac,
                valid=False,
                error_msg="Empty structure"
            )
        
        # Physical dimensions (1:1 voxel mapping)
        Lx, Ly, Lz = float(nx), float(ny), float(nz)
        
        # Create mesh
        domain = create_box(
            comm, 
            [[0, 0, 0], [Lx, Ly, Lz]], 
            [nx, ny, nz], 
            CellType.hexahedron
        )
        
        # Vector function space for displacement
        v_element = basix.ufl.element(
            "Lagrange", 
            domain.topology.cell_name(), 
            1, 
            shape=(domain.geometry.dim,)
        )
        V = functionspace(domain, v_element)
        
        # Scalar function space for density (CG1 - like FEniTop's rho_phys_field)
        # Using CG1 instead of DG0 provides proper interpolation across element boundaries
        # This matches FEniTop's approach and produces consistent compliance values
        s_element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
        S = functionspace(domain, s_element)
        
        # Material field (CG1)
        rho = Function(S)
        
        # Map density grid to CG1 field using node coordinates
        # This samples the voxel grid at node locations, providing smooth interpolation
        coords = S.tabulate_dof_coordinates()
        x_idx = np.floor(coords[:, 0]).astype(int)
        y_idx = np.floor(coords[:, 1]).astype(int)
        z_idx = np.floor(coords[:, 2]).astype(int)
        
        # Clip to valid range
        x_idx = np.clip(x_idx, 0, nx - 1)
        y_idx = np.clip(y_idx, 0, ny - 1)
        z_idx = np.clip(z_idx, 0, nz - 1)
        
        # Sample density at node locations
        rho.x.array[:] = density[x_idx, y_idx, z_idx]
        
        # SIMP penalization (p=3, same as harvest) - FEniTop formula
        p = 3.0
        eps = 1e-6  # Match FEniTop epsilon
        E_eff = (eps + (1 - eps) * rho**p) * E
        
        # Lame parameters
        mu = E_eff / (2 * (1 + nu))
        lmbda = E_eff * nu / ((1 + nu) * (1 - 2 * nu))
        
        # Strain and stress
        def epsilon(u):
            return ufl.sym(ufl.grad(u))
        
        def sigma(u):
            return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Bilinear form
        a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        
        # Load region parameters (match harvest data)
        ly = float(load_config.get('y', ny // 2))
        lz_s = float(load_config.get('z_start', 0))
        lz_e = float(load_config.get('z_end', nz))
        lz_center = (lz_s + lz_e) / 2.0
        load_half_width = 1.0
        
        # Surface traction at right boundary (x=Lx) in load region
        # This matches the harvest data setup exactly (UPDATED to match load_config.x)
        load_x_pos = float(load_config.get('x')) + 1.0
        def right_load_boundary(x):
            """Right face (x=Lx) within load region."""
            return (
                np.isclose(x[0], load_x_pos) &
                (x[1] >= ly - load_half_width - 0.5) & (x[1] <= ly + load_half_width + 0.5) &
                (x[2] >= lz_center - load_half_width - 0.5) & (x[2] <= lz_center + load_half_width + 0.5)
            )
        
        # Locate facets for traction BC
        fdim = domain.topology.dim - 1
        load_facets = mesh.locate_entities_boundary(domain, fdim, right_load_boundary)
        
        # Create facet tags for load region
        facet_indices = np.array(load_facets, dtype=np.int32)
        facet_markers = np.ones_like(facet_indices, dtype=np.int32)  # Tag = 1 for load region
        sorted_indices = np.argsort(facet_indices)
        facet_tags = mesh.meshtags(domain, fdim, facet_indices[sorted_indices], facet_markers[sorted_indices])
        
        # Define exterior facet measure with tags
        ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
        
        # Traction vector (0, force_magnitude, 0) - matches harvest: (0, -2.0, 0)
        T = Constant(domain, np.array([0.0, force_magnitude, 0.0], dtype=PETSc.ScalarType))
        
        # Linear form: surface traction on load region (tag=1)
        L = ufl.dot(T, v) * ds(1)
        
        # Boundary condition (fix X=0 plane) - same as harvest
        def left_boundary(x):
            return np.isclose(x[0], 0.0)
        
        fdim = domain.topology.dim - 1
        left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
        u_zero = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
        bc = dirichletbc(u_zero, locate_dofs_topological(V, fdim, left_facets), V)
        
        # Solve using CG + GAMG (Algebraic Multigrid) - optimized for elasticity
        # GAMG provides 31x speedup over Jacobi for this problem size
        # OpenMP threading uses all CPU cores regardless of MPI processes
        problem = LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": "cg",
                "pc_type": "gamg",
                "ksp_rtol": 1e-6,
                "ksp_max_it": 500,
                "mg_levels_ksp_type": "chebyshev",
                "mg_levels_pc_type": "jacobi",
            },
            petsc_options_prefix="fem_eval"
        )
        
        u_sol = problem.solve()
        
        # Compliance: C = u^T * K * u = F^T * u (energy)
        # Using C = ∫ σ(u):ε(u) dx
        compliance_form = form(ufl.inner(sigma(u_sol), epsilon(u_sol)) * ufl.dx)
        compliance = comm.allreduce(
            fem.assemble_scalar(compliance_form), 
            op=MPI.SUM
        )
        
        # Max displacement
        u_array = u_sol.x.array.reshape(-1, 3)
        displacement_magnitudes = np.linalg.norm(u_array, axis=1)
        max_displacement = float(np.max(displacement_magnitudes))
        
        # --- Gather Global Displacement Map ---
        # Only include displacement values where there is actual material
        # Use np.maximum.at to handle multiple FEM nodes mapping to same grid cell
        local_grid = np.zeros((nx, ny, nz), dtype=np.float32)
        node_densities = density[x_idx, y_idx, z_idx]
        has_material = node_densities > 0.5
        
        # Use maximum to aggregate (handles multiple nodes per cell correctly)
        np.maximum.at(
            local_grid,
            (x_idx[has_material], y_idx[has_material], z_idx[has_material]),
            displacement_magnitudes[has_material].astype(np.float32)
        )
        
        # Reduce to get global grid (max is more appropriate than sum for overlapping regions)
        global_displacement_map = np.zeros_like(local_grid)
        comm.Allreduce(local_grid, global_displacement_map, op=MPI.MAX)
        
        return FEMResult(
            compliance=float(compliance),
            max_displacement=max_displacement,
            vol_frac=vol_frac,
            valid=True,
            displacement_map=global_displacement_map
        )
        
    except Exception as e:
        return FEMResult(
            compliance=float('inf'),
            max_displacement=float('inf'),
            vol_frac=0.0,
            valid=False,
            error_msg=str(e)
        )


def evaluate_main_island(
    density: np.ndarray,
    load_config: Dict[str, Any],
    island_mask: np.ndarray,
    **kwargs
) -> FEMResult:
    """
    Evaluate only the main island of a structure.
    
    This masks out disconnected islands and evaluates only the
    connected component that links support to load.
    
    Args:
        density: Full density grid
        load_config: Load configuration
        island_mask: Boolean mask of the main island
        **kwargs: Additional args passed to evaluate_fem
        
    Returns:
        FEMResult for the main island only
    """
    # Create density with only main island
    main_density = np.where(island_mask, density, 0.0).astype(np.float32)
    return evaluate_fem(main_density, load_config, **kwargs)
