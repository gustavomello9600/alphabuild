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
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from alphabuilder.src.logic.harvest.generators import get_load_region_bounds

try:
    import pyamg  # noqa: F401
    HAS_FAST_FEM = True
except ImportError:
    HAS_FAST_FEM = False

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
    if not HAS_FAST_FEM:
        print("WARNING: Neither FEniCSx nor PyAMG available. FEM evaluation disabled.")


def resolve_fem_backend(backend: Optional[str] = None) -> str:
    """Resolve active FEM backend ('dolfinx', 'fast_amg', or 'auto')."""
    choice = (backend or os.environ.get("AB_FEM_BACKEND", "auto")).strip().lower()
    if choice in ("fast", "fast_amg", "pyamg"):
        return "fast_amg"
    if choice in ("dolfinx", "fenics", "fenicsx"):
        return "dolfinx"
    if HAS_FENICS:
        return "dolfinx"
    if HAS_FAST_FEM:
        return "fast_amg"
    return "dolfinx"


@dataclass
class FEMResult:
    """Result from FEM evaluation."""

    compliance: float
    max_displacement: float
    vol_frac: float
    valid: bool
    displacement_map: Optional[np.ndarray] = None
    sed_map: Optional[np.ndarray] = None
    error_msg: Optional[str] = None


def _bc_masks_to_nodal_dofs(
    bc_masks: np.ndarray,
    resolution: Tuple[int, int, int],
) -> np.ndarray:
    """Map `(3, nx, ny, nz)` support masks cleanly to `(nx+1, ny+1, nz+1, 3)` nodal flags."""
    nx, ny, nz = resolution
    nodal = np.zeros((nx + 1, ny + 1, nz + 1, 3), dtype=bool)
    masks_bin = np.asarray(bc_masks) > 0.5
    for axis in range(3):
        m = masks_bin[axis]
        for ix, iy, iz in np.argwhere(m):
            xs = [0] if (ix == 0 and (nx == 1 or not m[1, iy, iz])) else (
                [nx] if (ix == nx - 1 and (nx == 1 or not m[nx - 2, iy, iz])) else [ix, ix + 1]
            )
            ys = [0] if (iy == 0 and (ny == 1 or not m[ix, 1, iz])) else (
                [ny] if (iy == ny - 1 and (ny == 1 or not m[ix, ny - 2, iz])) else [iy, iy + 1]
            )
            zs = [0] if (iz == 0 and (nz == 1 or not m[ix, iy, 1])) else (
                [nz] if (iz == nz - 1 and (nz == 1 or not m[ix, iy, nz - 2])) else [iz, iz + 1]
            )
            nodal[np.ix_(xs, ys, zs, [axis])] = True
    return nodal


def _forces_grid_to_nodal_values(
    forces: np.ndarray,
    resolution: Tuple[int, int, int],
    force_scale: float = 1.125,
) -> np.ndarray:
    """Map `(3, nx, ny, nz)` force tensor to `(nx+1, ny+1, nz+1, 3)` nodal values."""
    nx, ny, nz = resolution
    f_arr = np.asarray(forces, dtype=np.float64) * float(force_scale)
    f_sum = np.zeros((nx + 1, ny + 1, nz + 1, 3), dtype=np.float64)
    f_cnt = np.zeros((nx + 1, ny + 1, nz + 1), dtype=np.float64)
    active = (np.sum(np.abs(f_arr), axis=0) > 1e-8).astype(np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                f_cnt[dx : dx + nx, dy : dy + ny, dz : dz + nz] += active
                for c in range(3):
                    f_sum[dx : dx + nx, dy : dy + ny, dz : dz + nz, c] += f_arr[c] * active
    nodal_f = np.zeros_like(f_sum)
    nz_mask = f_cnt > 0.0
    nodal_f[nz_mask] = f_sum[nz_mask] / f_cnt[nz_mask, None]
    return nodal_f


def evaluate_fem(
    density: np.ndarray,
    load_config: Dict[str, Any],
    E: float = 100.0,
    nu: float = 0.25,
    force_magnitude: float = -1.125,
    bc_type: Optional[str] = None,
    backend: Optional[str] = None,
    bc_masks: Optional[np.ndarray] = None,
    forces: Optional[np.ndarray] = None,
    compute_sed: bool = False,
) -> FEMResult:
    """
    Evaluate a voxel density grid using linear elasticity matching FEniTop harvest physics.
    """
    selected_backend = resolve_fem_backend(backend)
    if selected_backend == "fast_amg":
        if not HAS_FAST_FEM:
            return FEMResult(
                compliance=float("inf"),
                max_displacement=float("inf"),
                vol_frac=0.0,
                valid=False,
                error_msg="PyAMG fast_amg backend requested but pyamg is not installed",
            )
        from .fast_fem import evaluate_fem_fast

        return evaluate_fem_fast(
            density=density,
            load_config=load_config,
            E=E,
            nu=nu,
            force_magnitude=force_magnitude,
            bc_type=bc_type,
            bc_masks=bc_masks,
            forces=forces,
        )

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

        if bc_masks is not None:
            nodal_bc = _bc_masks_to_nodal_dofs(bc_masks, (nx, ny, nz))
            v_sub0, _ = V.sub(0).collapse()
            from dolfinx.fem import locate_dofs_geometrical

            bcs = []
            free_axes: List[int] = []
            for axis in range(3):
                if np.any(nodal_bc[:, :, :, axis]):
                    ax_mask = nodal_bc[:, :, :, axis]

                    def _loc(x, m=ax_mask):
                        vx_i = np.clip(np.rint(x[0]).astype(int), 0, nx)
                        vy_i = np.clip(np.rint(x[1]).astype(int), 0, ny)
                        vz_i = np.clip(np.rint(x[2]).astype(int), 0, nz)
                        return m[vx_i, vy_i, vz_i]

                    dofs_sub = locate_dofs_geometrical((V.sub(axis), v_sub0), _loc)[0]
                    bcs.append(dirichletbc(Constant(domain, 0.0), dofs_sub, V.sub(axis)))
                else:
                    free_axes.append(axis)
            nullspace_axes = tuple(free_axes) if free_axes else None
        else:
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

        body_force = Function(V)
        if forces is not None:
            nodal_f = _forces_grid_to_nodal_values(forces, (nx, ny, nz), force_scale=abs(float(force_magnitude)))

            def _interp_force(x):
                vx_i = np.clip(np.rint(x[0]).astype(int), 0, nx)
                vy_i = np.clip(np.rint(x[1]).astype(int), 0, ny)
                vz_i = np.clip(np.rint(x[2]).astype(int), 0, nz)
                return np.stack(
                    [
                        nodal_f[vx_i, vy_i, vz_i, 0],
                        nodal_f[vx_i, vy_i, vz_i, 1],
                        nodal_f[vx_i, vy_i, vz_i, 2],
                    ]
                )

            body_force.interpolate(_interp_force)
        else:
            lx, y_min, y_max, z_min, z_max = get_load_region_bounds((nx, ny, nz), load_config)
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

        sed_map: Optional[np.ndarray] = None
        if compute_sed:
            w_dg0 = ufl.TestFunction(S0)
            sed_form = form(ufl.inner(sigma(u_sol), epsilon(u_sol)) * w_dg0 * dx)
            sed_vec = fem.assemble_vector(sed_form)
            sed_grid = np.zeros((nx, ny, nz), dtype=np.float32)
            sed_grid[cx, cy, cz] = np.maximum(0.0, sed_vec.array[:]).astype(np.float32)
            sed_map = sed_grid

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
            sed_map=sed_map,
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
