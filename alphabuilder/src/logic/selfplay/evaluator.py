"""
FEM Evaluator for Self-Play MCTS (v3.3).

Aligned physically 1-to-1 with `alphabuilder/src/logic/harvest/optimization.py`:
- Same E = 100.0, nu = 0.25, SIMP p = 3.0, eps = 1e-6
- Exact DG0 element density mapping without boundary truncation
- Exact FULL_CLAMP vs RAIL_XY Dirichlet BCs + Z-translation NullSpace + 6-mode RBM NearNullSpace
- Exact volumetric body force `-1.125` on `get_load_region_bounds` (works for interior & tip loads)
- Same symmetric SOR Chebyshev GAMG solver + automatic GMRES fallback
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

from alphabuilder.src.logic.harvest.generators import get_load_region_bounds

try:
    from mpi4py import MPI
    from dolfinx import fem
    from dolfinx.mesh import CellType, create_box, locate_entities_boundary
    from dolfinx.fem import Constant, Function, dirichletbc, form, functionspace, locate_dofs_topological
    from petsc4py import PETSc
    import ufl
    import basix.ufl
    from alphabuilder.src.logic.fenitop.utility import LinearProblem as FEniTopLinearProblem

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
    E: float = 100.0,
    nu: float = 0.25,
    force_magnitude: float = -1.125,
    bc_type: Optional[str] = None,
) -> FEMResult:
    """
    Evaluate a voxel density grid using FEniCSx linear elasticity matching FEniTop harvest physics.
    """
    if not HAS_FENICS:
        return FEMResult(
            compliance=float("inf"),
            max_displacement=float("inf"),
            vol_frac=0.0,
            valid=False,
            error_msg="FEniCSx not available",
        )

    linear_problem = None
    try:
        comm = MPI.COMM_SELF
        nx, ny, nz = density.shape
        vol_frac = float(np.mean(density > 0.5))

        if vol_frac < 0.001:
            return FEMResult(
                compliance=float("inf"),
                max_displacement=float("inf"),
                vol_frac=vol_frac,
                valid=False,
                error_msg="Empty structure",
            )

        Lx, Ly, Lz = float(nx), float(ny), float(nz)
        domain = create_box(comm, [[0.0, 0.0, 0.0], [Lx, Ly, Lz]], [nx, ny, nz], CellType.hexahedron)

        v_element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(domain.geometry.dim,))
        V = functionspace(domain, v_element)

        s0_element = basix.ufl.element("DG", domain.topology.cell_name(), 0)
        S0 = functionspace(domain, s0_element)

        rho = Function(S0)
        cell_coords = S0.tabulate_dof_coordinates()
        cx = np.clip(np.floor(cell_coords[:, 0]).astype(int), 0, nx - 1)
        cy = np.clip(np.floor(cell_coords[:, 1]).astype(int), 0, ny - 1)
        cz = np.clip(np.floor(cell_coords[:, 2]).astype(int), 0, nz - 1)
        rho.x.array[:] = density[cx, cy, cz].astype(np.float64)

        p = 3.0
        eps = 1e-6
        E_eff = (eps + (1.0 - eps) * rho**p) * float(E)
        mu = E_eff / (2.0 * (1.0 + nu))
        lmbda = E_eff * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        def epsilon(u):
            return ufl.sym(ufl.grad(u))

        def sigma(u):
            return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2.0 * mu * epsilon(u)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        u_sol = Function(V)
        lam_sol = Function(V)

        fdim = domain.topology.dim - 1
        resolved_bc = str(bc_type or load_config.get("bc_type", "FULL_CLAMP")).upper()
        left_facets = locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))

        if resolved_bc == "RAIL_XY":
            dofs_x = locate_dofs_topological(V.sub(0), fdim, left_facets)
            dofs_y = locate_dofs_topological(V.sub(1), fdim, left_facets)
            bcs = [
                dirichletbc(Constant(domain, 0.0), dofs_x, V.sub(0)),
                dirichletbc(Constant(domain, 0.0), dofs_y, V.sub(1)),
            ]
            nullspace_axes = (2,)
        else:
            dofs_all = locate_dofs_topological(V, fdim, left_facets)
            bcs = [
                dirichletbc(Constant(domain, np.zeros(3, dtype=np.float64)), dofs_all, V),
            ]
            nullspace_axes = None

        lx, y_min, y_max, z_min, z_max = get_load_region_bounds((nx, ny, nz), load_config)
        body_force = Function(V)
        body_force.interpolate(
            lambda x: np.stack(
                [
                    np.zeros(x.shape[1]),
                    force_magnitude
                    * (
                        (x[0] >= float(lx) - 1e-6)
                        & (x[0] <= float(lx + 1) + 1e-6)
                        & (x[1] >= float(y_min) - 1e-6)
                        & (x[1] <= float(y_max) + 1e-6)
                        & (x[2] >= float(z_min) - 1e-6)
                        & (x[2] <= float(z_max) + 1e-6)
                    ),
                    np.zeros(x.shape[1]),
                ]
            )
        )

        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2})
        lhs = ufl.inner(sigma(u), epsilon(v)) * dx
        rhs = ufl.dot(body_force, v) * dx

        petsc_options = {
            "ksp_type": "cg",
            "pc_type": "gamg",
            "ksp_rtol": 1e-5,
            "ksp_max_it": 500,
            "pc_gamg_type": "agg",
            "pc_gamg_agg_nsmooths": 1,
            "pc_gamg_threshold": 0.02,
            "mg_levels_ksp_type": "chebyshev",
            "mg_levels_pc_type": "sor",
            "mg_levels_pc_sor_symmetric": None,
        }

        linear_problem = FEniTopLinearProblem(
            u_sol,
            lam_sol,
            lhs,
            rhs,
            l_vec=None,
            spring_vec=None,
            bcs=bcs,
            petsc_options=petsc_options,
            nullspace_axes=nullspace_axes,
        )
        linear_problem.solve_fem()

        compliance_form = form(ufl.inner(sigma(u_sol), epsilon(u_sol)) * dx)
        compliance = float(fem.assemble_scalar(compliance_form))

        u_array = u_sol.x.array.reshape(-1, 3)
        displacement_magnitudes = np.linalg.norm(u_array, axis=1)
        max_displacement = float(np.max(displacement_magnitudes)) if displacement_magnitudes.size > 0 else 0.0

        # Map nodal CG1 displacements cleanly to exact (nx, ny, nz) hexahedral cell maxima
        v_coords = V.tabulate_dof_coordinates()
        vx = np.clip(np.rint(v_coords[:, 0]).astype(int), 0, nx)
        vy = np.clip(np.rint(v_coords[:, 1]).astype(int), 0, ny)
        vz = np.clip(np.rint(v_coords[:, 2]).astype(int), 0, nz)

        disp_nodal = np.zeros((nx + 1, ny + 1, nz + 1), dtype=np.float32)
        disp_nodal[vx, vy, vz] = displacement_magnitudes.astype(np.float32)
        cell_disp = np.maximum.reduce(
            [
                disp_nodal[:-1, :-1, :-1],
                disp_nodal[1:, :-1, :-1],
                disp_nodal[:-1, 1:, :-1],
                disp_nodal[:-1, :-1, 1:],
                disp_nodal[1:, 1:, :-1],
                disp_nodal[1:, :-1, 1:],
                disp_nodal[:-1, 1:, 1:],
                disp_nodal[1:, 1:, 1:],
            ]
        )
        displacement_map = np.where(density > 0.5, cell_disp, 0.0).astype(np.float32)

        if not np.isfinite(compliance) or not np.isfinite(max_displacement):
            return FEMResult(
                compliance=float("inf"),
                max_displacement=float("inf"),
                vol_frac=vol_frac,
                valid=False,
                error_msg="Non-finite FEM result",
            )

        return FEMResult(
            compliance=compliance,
            max_displacement=max_displacement,
            vol_frac=vol_frac,
            valid=True,
            displacement_map=displacement_map,
        )

    except Exception as e:
        return FEMResult(
            compliance=float("inf"),
            max_displacement=float("inf"),
            vol_frac=0.0,
            valid=False,
            error_msg=str(e),
        )
    finally:
        if linear_problem is not None:
            for attr in ("solver", "lhs_mat", "rhs_vec", "spring_vec", "l_vec", "near_nullspace", "nullspace"):
                obj = getattr(linear_problem, attr, None)
                if obj is not None:
                    try:
                        obj.destroy()
                    except Exception:
                        pass
                    setattr(linear_problem, attr, None)


def evaluate_main_island(
    density: np.ndarray,
    load_config: Dict[str, Any],
    island_mask: np.ndarray,
    **kwargs,
) -> FEMResult:
    """Evaluate only the main connected structural component."""
    main_density = np.where(island_mask, density, 0.0).astype(np.float32)
    return evaluate_fem(main_density, load_config, **kwargs)
