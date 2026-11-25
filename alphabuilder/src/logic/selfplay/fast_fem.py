"""
Fast Cached Structured Q1-Hex8 FEM Evaluator using PyAMG (Smoothed Aggregation AMG + CG).

Aligned mathematically 1-to-1 with `alphabuilder/src/logic/selfplay/evaluator.py`
(`dolfinx` / FEniTop):
- Unit hexahedral voxel mesh `[0, nx] x [0, ny] x [0, nz]` (`CG1` displacement, `DG0` density)
- Same `E = 100.0, nu = 0.25, SIMP p = 3.0, eps = 1e-6`
- Exact `2x2x2` Gauss-Legendre Q1 stiffness (`24x24`) & consistent mass (`8x8`)
- Supports both legacy `load_config`/`bc_type` AND arbitrary 3D `bc_masks` + 3D `forces`
- Exact vectorized 8-slot direct BSR `(3x3)` assembly + `coarse_solver='splu'`
- Computes per-element Strain Energy Density `sed_map` (`e_eff * u_e^T Ke u_e`) in <2 ms
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.sparse import bsr_matrix
import pyamg

from alphabuilder.src.logic.harvest.generators import get_load_region_bounds


@dataclass
class _BCData:
    bc_mask: np.ndarray        # (n_dof,) bool: Dirichlet DOFs (incl. auto nullspace pin)
    bsr_keep: np.ndarray       # (n_uniq_blocks, 3, 3) bool: ~bc_row & ~bc_col
    bsr_diag_bc: np.ndarray    # (n_uniq_blocks, 3, 3) bool: diagonal BC entries -> 1.0
    rbm_modes: np.ndarray      # (n_dof, <=6) float64 near-nullspace zeroed on BC DOFs
    free_translation_axes: Tuple[int, ...]


def bc_masks_to_nodal_dofs(
    bc_masks: np.ndarray,
    resolution: Tuple[int, int, int],
) -> np.ndarray:
    """
    Map voxel-level Dirichlet masks `(3, nx, ny, nz)` cleanly onto nodal DOF flags
    `(nx+1, ny+1, nz+1, 3)` consistently across both `fast_fem` and `dolfinx`.

    Voxels on an exterior support plane (`x=0`, `x=nx-1`, `y=0`, `y=ny-1`, `z=0`, `z=nz-1`)
    without interior neighbors along that normal constrain only that exterior plane's nodes,
    matching FEniTop's face-boundary Dirichlet conditions (`x == 0`, etc.).
    """
    nx, ny, nz = resolution
    nodal = np.zeros((nx + 1, ny + 1, nz + 1, 3), dtype=bool)
    masks_bin = np.asarray(bc_masks) > 0.5

    for axis in range(3):
        m = masks_bin[axis]
        coords = np.argwhere(m)
        for ix, iy, iz in coords:
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


def forces_grid_to_nodal_values(
    forces: np.ndarray,
    resolution: Tuple[int, int, int],
    force_scale: float = 1.125,
) -> np.ndarray:
    """
    Map voxel force tensor `(3, nx, ny, nz)` consistently onto nodal `CG1` body-force values
    `(nx+1, ny+1, nz+1, 3)` (matching `body_force.interpolate` on unit patches).
    """
    nx, ny, nz = resolution
    f_arr = np.asarray(forces, dtype=np.float64) * float(force_scale)
    f_sum = np.zeros((nx + 1, ny + 1, nz + 1, 3), dtype=np.float64)
    f_cnt = np.zeros((nx + 1, ny + 1, nz + 1), dtype=np.float64)

    active = np.sum(np.abs(f_arr), axis=0) > 1e-8  # (nx, ny, nz)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = active.astype(np.float64)
                f_cnt[dx : dx + nx, dy : dy + ny, dz : dz + nz] += w
                for c in range(3):
                    f_sum[dx : dx + nx, dy : dy + ny, dz : dz + nz, c] += f_arr[c] * w

    nodal_f = np.zeros_like(f_sum)
    nz_mask = f_cnt > 0.0
    nodal_f[nz_mask] = f_sum[nz_mask] / f_cnt[nz_mask, None]
    return nodal_f


def _hex8_unit_matrices(nu: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute exact unit-cube Q1 Hex8 stiffness matrix (24x24, E=1.0) and consistent
    scalar mass matrix (8x8) using 2x2x2 Gauss-Legendre quadrature.
    Local node ordering is lexicographic (dx, dy, dz) in {0, 1}^3.
    """
    corners = np.array(
        [(dx, dy, dz) for dx in (0.0, 1.0) for dy in (0.0, 1.0) for dz in (0.0, 1.0)],
        dtype=np.float64,
    )

    gp_1d = 0.5 * (1.0 + np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=np.float64))
    w_1d = np.array([0.5, 0.5], dtype=np.float64)

    fac = 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu))
    C = fac * np.array(
        [
            [1.0 - nu, nu, nu, 0.0, 0.0, 0.0],
            [nu, 1.0 - nu, nu, 0.0, 0.0, 0.0],
            [nu, nu, 1.0 - nu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * nu), 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * nu), 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)],
        ],
        dtype=np.float64,
    )

    Ke = np.zeros((24, 24), dtype=np.float64)
    Me = np.zeros((8, 8), dtype=np.float64)

    for ix, xi in enumerate(gp_1d):
        for iy, eta in enumerate(gp_1d):
            for iz, zeta in enumerate(gp_1d):
                w = w_1d[ix] * w_1d[iy] * w_1d[iz]
                N = np.empty(8, dtype=np.float64)
                dN = np.empty((8, 3), dtype=np.float64)
                for a in range(8):
                    cx, cy, cz = corners[a]
                    fx = xi if cx > 0.5 else (1.0 - xi)
                    fy = eta if cy > 0.5 else (1.0 - eta)
                    fz = zeta if cz > 0.5 else (1.0 - zeta)
                    dfx = 1.0 if cx > 0.5 else -1.0
                    dfy = 1.0 if cy > 0.5 else -1.0
                    dfz = 1.0 if cz > 0.5 else -1.0
                    N[a] = fx * fy * fz
                    dN[a, 0] = dfx * fy * fz
                    dN[a, 1] = fx * dfy * fz
                    dN[a, 2] = fx * fy * dfz

                B = np.zeros((6, 24), dtype=np.float64)
                for a in range(8):
                    c = 3 * a
                    B[0, c] = dN[a, 0]
                    B[1, c + 1] = dN[a, 1]
                    B[2, c + 2] = dN[a, 2]
                    B[3, c] = dN[a, 1]
                    B[3, c + 1] = dN[a, 0]
                    B[4, c + 1] = dN[a, 2]
                    B[4, c + 2] = dN[a, 1]
                    B[5, c] = dN[a, 2]
                    B[5, c + 2] = dN[a, 0]

                Ke += w * (B.T @ C @ B)
                Me += w * np.outer(N, N)

    Ke = 0.5 * (Ke + Ke.T)
    Me = 0.5 * (Me + Me.T)
    return Ke, Me


class StructuredHex8FEM:
    """Precomputed static BSR topology, 8-slot assembly, and PyAMG solver for voxel Hex8 FEM."""

    def __init__(self, resolution: Tuple[int, int, int], nu: float = 0.25):
        self.resolution = (int(resolution[0]), int(resolution[1]), int(resolution[2]))
        self.nu = float(nu)
        self.nx, self.ny, self.nz = self.resolution
        self.nnx, self.nny, self.nnz = self.nx + 1, self.ny + 1, self.nz + 1
        self.n_nodes = self.nnx * self.nny * self.nnz
        self.n_dof = 3 * self.n_nodes
        self.n_elem = self.nx * self.ny * self.nz

        self.Ke, self.Me = _hex8_unit_matrices(nu=self.nu)

        ex = np.arange(self.nx, dtype=np.int32)[:, None, None]
        ey = np.arange(self.ny, dtype=np.int32)[None, :, None]
        ez = np.arange(self.nz, dtype=np.int32)[None, None, :]
        sy = self.nny * self.nnz
        sz = self.nnz
        base_node = (ex * sy + ey * sz + ez).reshape(-1)

        offsets_node = np.array(
            [dx * sy + dy * sz + dz for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)],
            dtype=np.int32,
        )
        self.elem_nodes = base_node[:, None] + offsets_node[None, :]  # (n_elem, 8)
        self.elem_dofs = np.empty((self.n_elem, 24), dtype=np.int32)
        self.elem_dofs[:, 0::3] = 3 * self.elem_nodes
        self.elem_dofs[:, 1::3] = 3 * self.elem_nodes + 1
        self.elem_dofs[:, 2::3] = 3 * self.elem_nodes + 2

        pair_rows = np.repeat(self.elem_nodes, 8, axis=1).reshape(-1)
        pair_cols = np.tile(self.elem_nodes, (1, 8)).reshape(-1)
        keys = pair_rows.astype(np.int64) * self.n_nodes + pair_cols.astype(np.int64)
        uniq_keys, elem_pair_to_uniq = np.unique(keys, return_inverse=True)
        uniq_rn = (uniq_keys // self.n_nodes).astype(np.int32)
        uniq_cn = (uniq_keys % self.n_nodes).astype(np.int32)
        bsr_indptr = np.zeros(self.n_nodes + 1, dtype=np.int32)
        np.add.at(bsr_indptr[1:], uniq_rn, 1)
        np.cumsum(bsr_indptr, out=bsr_indptr)

        self.n_uniq_blocks = len(uniq_keys)
        self.bsr_indptr = bsr_indptr
        self.bsr_indices = uniq_cn
        self.bsr_row_nodes = uniq_rn
        self._is_diag_block = uniq_rn == uniq_cn
        self._eye3 = np.eye(3, dtype=bool)[None, :, :]
        self._ke_flat = self.Ke.reshape(8, 3, 8, 3).transpose(0, 2, 1, 3).reshape(64, 9).copy()

        # Precompute 8-slot vectorized gather/scatter schedule into BSR blocks (~10x faster assembly)
        ep = elem_pair_to_uniq.astype(np.int32)
        order = np.argsort(ep, kind="mergesort")
        u_sorted = ep[order]
        e_sorted = (order // 64).astype(np.int32)
        p_sorted = (order % 64).astype(np.int32)
        diff = np.ones(len(u_sorted), dtype=bool)
        diff[1:] = u_sorted[1:] != u_sorted[:-1]
        group_starts = np.flatnonzero(diff)
        counts = np.diff(np.append(group_starts, len(u_sorted)))
        within = np.arange(len(u_sorted), dtype=np.int32) - np.repeat(group_starts, counts)
        self._asm_slots: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for k in range(int(within.max()) + 1):
            m = within == k
            self._asm_slots.append((u_sorted[m].astype(np.int32), e_sorted[m], p_sorted[m]))

        gx, gy, gz = np.meshgrid(
            np.arange(self.nnx, dtype=np.float64),
            np.arange(self.nny, dtype=np.float64),
            np.arange(self.nnz, dtype=np.float64),
            indexing="ij",
        )
        self.node_coords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])  # (n_nodes, 3)

        x, y, z = self.node_coords[:, 0], self.node_coords[:, 1], self.node_coords[:, 2]
        raw = np.zeros((self.n_dof, 6), dtype=np.float64)
        raw[0::3, 0] = 1.0
        raw[1::3, 1] = 1.0
        raw[2::3, 2] = 1.0
        raw[0::3, 3] = -y
        raw[1::3, 3] = x
        raw[0::3, 4] = z
        raw[2::3, 4] = -x
        raw[1::3, 5] = -z
        raw[2::3, 5] = y
        self._raw_rbm = raw

        self._bc_cache: Dict[str, _BCData] = {}
        self._bc_cache["FULL_CLAMP"] = self._build_bc_data_legacy("FULL_CLAMP")
        self._bc_cache["RAIL_XY"] = self._build_bc_data_legacy("RAIL_XY")

    def _finalize_bc_data(self, nodal_bc: np.ndarray) -> _BCData:
        """Build BSR masks, auto rigid-translation pins, and zeroed RBM near-nullspace."""
        nodal = nodal_bc.copy()
        free_axes: List[int] = []
        for axis in range(3):
            if not np.any(nodal[:, :, :, axis]):
                free_axes.append(axis)
                nodal[0, 0, 0, axis] = True  # Remove constant rigid translation mode safely

        bc_mask = nodal.reshape(-1)
        bc_nd = bc_mask.reshape(self.n_nodes, 3)
        bsr_keep = (~bc_nd[self.bsr_row_nodes, :, None]) & (~bc_nd[self.bsr_indices, None, :])
        bsr_diag_bc = (
            self._is_diag_block[:, None, None]
            & self._eye3
            & (bc_nd[self.bsr_row_nodes, :, None] | bc_nd[self.bsr_indices, None, :])
        )

        raw = self._raw_rbm.copy()
        raw[bc_mask, :] = 0.0
        cols = []
        for j in range(6):
            v = raw[:, j].copy()
            for q in cols:
                v -= np.dot(v, q) * q
            nrm = float(np.linalg.norm(v))
            if nrm > 1e-10:
                cols.append(v / nrm)
        rbm = np.column_stack(cols) if cols else np.ones((self.n_dof, 1), dtype=np.float64)

        return _BCData(
            bc_mask=bc_mask,
            bsr_keep=bsr_keep,
            bsr_diag_bc=bsr_diag_bc,
            rbm_modes=rbm,
            free_translation_axes=tuple(free_axes),
        )

    def _build_bc_data_legacy(self, bc_type: str) -> _BCData:
        nodal = np.zeros((self.nnx, self.nny, self.nnz, 3), dtype=bool)
        if bc_type == "RAIL_XY":
            nodal[0, :, :, 0] = True
            nodal[0, :, :, 1] = True
        else:
            nodal[0, :, :, :] = True
        return self._finalize_bc_data(nodal)

    def _get_bc_data(self, bc_type: str, bc_masks: Optional[np.ndarray]) -> _BCData:
        if bc_masks is None:
            key = bc_type if bc_type in ("FULL_CLAMP", "RAIL_XY") else "FULL_CLAMP"
            return self._bc_cache[key]
        h = hashlib.md5((bc_masks > 0.5).astype(np.uint8).tobytes()).hexdigest()
        cached = self._bc_cache.get(h)
        if cached is not None:
            return cached
        nodal = bc_masks_to_nodal_dofs(bc_masks, self.resolution)
        data = self._finalize_bc_data(nodal)
        if len(self._bc_cache) < 32:
            self._bc_cache[h] = data
        return data

    def _assemble_bsr_blocks(self, e_eff: np.ndarray, bc: _BCData) -> np.ndarray:
        """Vectorized 8-slot assembly directly into BSR `(n_uniq_blocks, 3, 3)` blocks."""
        blk_flat = np.zeros((self.n_uniq_blocks, 9), dtype=np.float64)
        for b_idx, e_idx, p_idx in self._asm_slots:
            blk_flat[b_idx] += e_eff[e_idx, None] * self._ke_flat[p_idx]
        blk3 = blk_flat.reshape(self.n_uniq_blocks, 3, 3)
        blk3[~bc.bsr_keep] = 0.0
        blk3[bc.bsr_diag_bc] = 1.0
        return blk3

    def _assemble_rhs(
        self,
        load_config: Dict[str, Any],
        force_magnitude: float,
        forces: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Assemble exact CG1-interpolated body-force RHS matching `dolfinx` evaluator."""
        rhs = np.zeros(self.n_dof, dtype=np.float64)
        if forces is not None:
            nodal_f = forces_grid_to_nodal_values(
                forces, self.resolution, force_scale=abs(float(force_magnitude))
            ).reshape(self.n_nodes, 3)
            for c in range(3):
                if np.any(nodal_f[:, c] != 0.0):
                    elem_fc = nodal_f[self.elem_nodes, c]
                    elem_rhs = elem_fc @ self.Me
                    rhs[c::3] = np.bincount(
                        self.elem_nodes.ravel(),
                        weights=elem_rhs.ravel(),
                        minlength=self.n_nodes,
                    )
            return rhs

        lx, y_min, y_max, z_min, z_max = get_load_region_bounds(self.resolution, load_config)
        cx, cy, cz = self.node_coords[:, 0], self.node_coords[:, 1], self.node_coords[:, 2]
        in_patch = (
            (cx >= float(lx) - 1e-6)
            & (cx <= float(lx + 1) + 1e-6)
            & (cy >= float(y_min) - 1e-6)
            & (cy <= float(y_max) + 1e-6)
            & (cz >= float(z_min) - 1e-6)
            & (cz <= float(z_max) + 1e-6)
        )
        nodal_fy = np.where(in_patch, float(force_magnitude), 0.0)
        elem_fy = nodal_fy[self.elem_nodes]
        elem_rhs_y = elem_fy @ self.Me
        rhs[1::3] = np.bincount(
            self.elem_nodes.ravel(),
            weights=elem_rhs_y.ravel(),
            minlength=self.n_nodes,
        )
        return rhs

    def solve(
        self,
        density: np.ndarray,
        load_config: Dict[str, Any],
        E: float = 100.0,
        force_magnitude: float = -1.125,
        bc_type: Optional[str] = None,
        bc_masks: Optional[np.ndarray] = None,
        forces: Optional[np.ndarray] = None,
        rtol: float = 1e-5,
        maxiter: int = 400,
    ) -> Tuple[float, float, np.ndarray, np.ndarray, bool]:
        """Solve 3D linear elasticity via BSR PyAMG + splu coarse + CG; return (C, u_max, disp_map, sed_map, valid)."""
        rho_elem = density.reshape(-1).astype(np.float64)
        p = 3.0
        eps = 1e-6
        e_eff = (eps + (1.0 - eps) * (rho_elem ** p)) * float(E)

        resolved_bc = str(bc_type or load_config.get("bc_type", "FULL_CLAMP")).upper()
        bc = self._get_bc_data(resolved_bc, bc_masks)

        blk3 = self._assemble_bsr_blocks(e_eff, bc)
        A = bsr_matrix((blk3, self.bsr_indices, self.bsr_indptr), shape=(self.n_dof, self.n_dof))

        rhs = self._assemble_rhs(load_config, force_magnitude, forces=forces)
        rhs[bc.bc_mask] = 0.0

        rhs_norm = float(np.linalg.norm(rhs))
        if rhs_norm <= 1e-18:
            z = np.zeros(self.resolution, dtype=np.float32)
            return float("inf"), float("inf"), z, z.copy(), False

        ml = pyamg.smoothed_aggregation_solver(
            A,
            B=bc.rbm_modes,
            symmetry="symmetric",
            strength=("symmetric", {"theta": 0.02}),
            aggregate="standard",
            smooth=("jacobi", {"omega": 4.0 / 3.0}),
            presmoother=("block_gauss_seidel", {"sweep": "symmetric", "iterations": 1}),
            postsmoother=("block_gauss_seidel", {"sweep": "symmetric", "iterations": 1}),
            max_coarse=300,
            coarse_solver="splu",
        )
        u = ml.solve(rhs, tol=rtol, maxiter=maxiter, accel="cg")

        rel_res = float(np.linalg.norm(A @ u - rhs) / rhs_norm)
        if not np.isfinite(rel_res) or rel_res > 1e-3:
            u = ml.solve(rhs, x0=u if np.all(np.isfinite(u)) else None, tol=1e-7, maxiter=800, accel="cg")
            rel_res = float(np.linalg.norm(A @ u - rhs) / rhs_norm)
            if not np.isfinite(rel_res) or rel_res > 1e-2:
                z = np.zeros(self.resolution, dtype=np.float32)
                return float("inf"), float("inf"), z, z.copy(), False

        for free_ax in bc.free_translation_axes:
            u[free_ax::3] -= float(np.mean(u[free_ax::3]))

        compliance = float(np.dot(u, rhs))

        # Exact element-wise Strain Energy Density: SED_e = E_eff[e] * u_e^T Ke u_e
        u_e = u[self.elem_dofs]  # (n_elem, 24)
        sed_elem = e_eff * np.einsum("ei,ij,ej->e", u_e, self.Ke, u_e)
        sed_map = np.maximum(0.0, sed_elem).reshape(self.resolution).astype(np.float32)

        u_vec = u.reshape(-1, 3)
        u_mag = np.linalg.norm(u_vec, axis=1)
        max_disp = float(np.max(u_mag))

        disp_nodal = u_mag.reshape(self.nnx, self.nny, self.nnz).astype(np.float32)
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

        if not np.isfinite(compliance) or not np.isfinite(max_disp) or compliance <= 0.0:
            z = np.zeros(self.resolution, dtype=np.float32)
            return float("inf"), float("inf"), z, z.copy(), False

        return compliance, max_disp, displacement_map, sed_map, True


_FEM_CACHE: Dict[Tuple[Tuple[int, int, int], float], StructuredHex8FEM] = {}


def _get_solver(resolution: Tuple[int, int, int], nu: float = 0.25) -> StructuredHex8FEM:
    key = ((int(resolution[0]), int(resolution[1]), int(resolution[2])), round(float(nu), 6))
    solver = _FEM_CACHE.get(key)
    if solver is None:
        solver = StructuredHex8FEM(resolution=key[0], nu=key[1])
        _FEM_CACHE[key] = solver
    return solver


def evaluate_fem_fast(
    density: np.ndarray,
    load_config: Dict[str, Any],
    E: float = 100.0,
    nu: float = 0.25,
    force_magnitude: float = -1.125,
    bc_type: Optional[str] = None,
    bc_masks: Optional[np.ndarray] = None,
    forces: Optional[np.ndarray] = None,
):
    """Evaluate a voxel density grid using the fast cached PyAMG Hex8 solver."""
    from alphabuilder.src.logic.selfplay.evaluator import FEMResult

    vol_frac = float(np.mean(density > 0.5))
    if vol_frac < 0.001:
        return FEMResult(
            compliance=float("inf"),
            max_displacement=float("inf"),
            vol_frac=vol_frac,
            valid=False,
            error_msg="Empty structure",
        )

    try:
        solver = _get_solver(density.shape, nu=nu)  # type: ignore[arg-type]
        comp, max_disp, disp_map, sed_map, valid = solver.solve(
            density=density,
            load_config=load_config,
            E=E,
            force_magnitude=force_magnitude,
            bc_type=bc_type,
            bc_masks=bc_masks,
            forces=forces,
        )
        if not valid:
            return FEMResult(
                compliance=float("inf"),
                max_displacement=float("inf"),
                vol_frac=vol_frac,
                valid=False,
                error_msg="PyAMG FEM solver failed to converge or non-finite result",
            )
        return FEMResult(
            compliance=float(comp),
            max_displacement=float(max_disp),
            vol_frac=vol_frac,
            valid=True,
            displacement_map=disp_map,
            sed_map=sed_map,
        )
    except Exception as exc:
        return FEMResult(
            compliance=float("inf"),
            max_displacement=float("inf"),
            vol_frac=vol_frac,
            valid=False,
            error_msg=str(exc),
        )
