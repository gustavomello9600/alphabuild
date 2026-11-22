"""
FEniTop Integration for Topology Optimization.
"""
import sys
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from mpi4py import MPI
import basix.ufl

from dolfinx.mesh import create_box, CellType
from dolfinx.fem import functionspace

from alphabuilder.src.core.physics_model import PhysicalProperties
from alphabuilder.src.logic.fenitop.topopt import topopt
from alphabuilder.src.logic.harvest.config import SIMPConfig
from alphabuilder.src.logic.harvest.generators import get_load_region_bounds
from alphabuilder.src.logic.selfplay.reward import calculate_fem_score


def _cg1_nodal_to_cell_mean(nodal_grid: np.ndarray) -> np.ndarray:
    """Average 8 hexahedral corner nodal values to produce exact cell-wise (nx, ny, nz) grid."""
    return 0.125 * (
        nodal_grid[:-1, :-1, :-1]
        + nodal_grid[1:, :-1, :-1]
        + nodal_grid[:-1, 1:, :-1]
        + nodal_grid[:-1, :-1, 1:]
        + nodal_grid[1:, 1:, :-1]
        + nodal_grid[1:, :-1, 1:]
        + nodal_grid[:-1, 1:, 1:]
        + nodal_grid[1:, 1:, 1:]
    )


def run_fenitop_optimization(
    resolution: Tuple[int, int, int],
    props: PhysicalProperties,
    simp_config: SIMPConfig,
    initial_density: np.ndarray = None,
    strategy: str = 'BEZIER',
    record_displacement_map: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run Topology Optimization using FEniTop Core.
    """
    comm = MPI.COMM_WORLD
    nx, ny, nz = resolution
    
    # Physical dimensions (1:1 mapping with voxels)
    Lx, Ly, Lz = float(nx), float(ny), float(nz)
    
    # --- Mesh Creation ---
    mesh = create_box(comm, [[0, 0, 0], [Lx, Ly, Lz]], [nx, ny, nz], CellType.hexahedron)
    
    if comm.rank == 0:
        mesh_serial = create_box(MPI.COMM_SELF, [[0, 0, 0], [Lx, Ly, Lz]], [nx, ny, nz], CellType.hexahedron)
    else:
        mesh_serial = None
    
    # Synchronize all ranks
    comm.Barrier()
    
    # --- Grid Mapper for DOF -> Voxel conversion ---
    grid_mapper = None
    if comm.rank == 0 and mesh_serial is not None:
        element = basix.ufl.element("Lagrange", mesh_serial.topology.cell_name(), 1)
        V_serial = functionspace(mesh_serial, element)
        coords = V_serial.tabulate_dof_coordinates()
        
        x_idx = np.rint(coords[:, 0]).astype(int)
        y_idx = np.rint(coords[:, 1]).astype(int)
        z_idx = np.rint(coords[:, 2]).astype(int)
        
        x_idx = np.clip(x_idx, 0, nx)
        y_idx = np.clip(y_idx, 0, ny)
        z_idx = np.clip(z_idx, 0, nz)
        
        grid_mapper = (x_idx, y_idx, z_idx)
    
    # --- Load & BC Configuration ---
    load_cfg = simp_config.load_config
    lx, y_min, y_max, z_min, z_max = get_load_region_bounds(resolution, load_cfg)
    bc_type = str(load_cfg.get('bc_type', 'FULL_CLAMP')).upper()

    if bc_type == 'RAIL_XY':
        # X and Y fixed at X=0; Z remains free (sliding rail) with explicit rigid-body TZ nullspace
        bcs = [
            (0.0, lambda x: np.isclose(x[0], 0.0), 0),
            (0.0, lambda x: np.isclose(x[0], 0.0), 1),
        ]
        nullspace_axes = (2,)
    else:
        # FULL_CLAMP: all 3 displacement DOFs fixed at X=0
        bcs = [
            ([0.0, 0.0, 0.0], lambda x: np.isclose(x[0], 0.0), None),
        ]
        nullspace_axes = None
    
    # --- FEM Parameters ---
    fem = {
        "mesh": mesh,
        "mesh_serial": mesh_serial,
        "young's modulus": 100,
        "poisson's ratio": 0.25,
        "bcs": bcs,
        "nullspace_axes": nullspace_axes,
        "traction_bcs": [],
        "body_force": lambda x: np.stack([
            np.zeros(x.shape[1]),
            -1.125 * (
                (x[0] >= float(lx) - 1e-6) & (x[0] <= float(lx + 1) + 1e-6) &
                (x[1] >= float(y_min) - 1e-6) & (x[1] <= float(y_max) + 1e-6) &
                (x[2] >= float(z_min) - 1e-6) & (x[2] <= float(z_max) + 1e-6)
            ),
            np.zeros(x.shape[1])
        ]),
        "quadrature_degree": 2,
        "petsc_options": {
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
        },
    }
    
    # --- Optimization Parameters ---
    if strategy == 'FULL_DOMAIN':
        max_iter = min(simp_config.max_iter, 120)
        beta_interval = 30
        move_limit = 0.02
    else:
        max_iter = min(simp_config.max_iter, 100)
        beta_interval = 25
        move_limit = 0.02
    
    filter_r = max(simp_config.r_min, 1.2)
    
    opt = {
        "max_iter": max_iter,
        "opt_tol": 1e-4,
        "vol_frac": simp_config.vol_frac,
        "solid_zone": lambda x: (
            (x[0] >= float(lx)) & (x[0] <= float(lx + 1)) &
            (x[1] >= float(y_min)) & (x[1] <= float(y_max)) &
            (x[2] >= float(z_min)) & (x[2] <= float(z_max))
        ),
        "void_zone": lambda x: np.full(x.shape[1], False),
        "penalty": 3.0,
        "epsilon": 1e-6,
        "filter_radius": filter_r,
        "beta_interval": beta_interval,
        "beta_max": 128,
        "use_oc": True,
        "move": move_limit,
        "opt_compliance": True,
    }
    
    # --- History Recording ---
    history = []
    pbar = None
    if comm.rank == 0:
        pbar = tqdm(
            total=max_iter,
            desc="SIMP",
            leave=False,
            unit="it",
            disable=not sys.stderr.isatty(),
        )
    
    def record_step(data):
        if comm.rank == 0:
            rho_flat = data['density']
            
            if grid_mapper is None:
                rho_3d = np.full((nx, ny, nz), simp_config.vol_frac, dtype=np.float32)
            elif len(rho_flat) != len(grid_mapper[0]):
                print("  WARNING: Size mismatch in record_step")
                rho_3d = np.full((nx, ny, nz), simp_config.vol_frac, dtype=np.float32)
            else:
                rho_nodal = np.zeros((nx + 1, ny + 1, nz + 1), dtype=np.float32)
                rho_nodal[grid_mapper[0], grid_mapper[1], grid_mapper[2]] = rho_flat
                rho_3d = _cg1_nodal_to_cell_mean(rho_nodal).astype(np.float32)
                
            # Process displacement magnitude without truncating free boundary nodes
            max_disp = 0.0
            disp_3d = None
            if 'displacement_map' in data and data['displacement_map'] is not None:
                disp_flat = data['displacement_map']
                max_disp = float(np.max(disp_flat)) if len(disp_flat) > 0 else 0.0
                if record_displacement_map and grid_mapper is not None and len(disp_flat) == len(grid_mapper[0]):
                    disp_nodal = np.zeros((nx + 1, ny + 1, nz + 1), dtype=np.float32)
                    disp_nodal[grid_mapper[0], grid_mapper[1], grid_mapper[2]] = disp_flat
                    disp_3d = np.maximum.reduce([
                        disp_nodal[:-1, :-1, :-1],
                        disp_nodal[1:, :-1, :-1],
                        disp_nodal[:-1, 1:, :-1],
                        disp_nodal[:-1, :-1, 1:],
                        disp_nodal[1:, 1:, :-1],
                        disp_nodal[1:, :-1, 1:],
                        disp_nodal[:-1, 1:, 1:],
                        disp_nodal[1:, 1:, 1:],
                    ]).astype(np.float32)

            # Calculate metrics
            cur_iter = data['iter']
            comp = float(data['compliance'])
            vol = float(data['vol_frac'])
            fem_score = calculate_fem_score(comp, vol)
            mixing_factor = min(1.0, cur_iter / float(max_iter))

            history.append({
                'step': cur_iter,
                'density_map': rho_3d,
                'compliance': comp,
                'vol_frac': vol,
                'max_displacement': max_disp,
                'fem_score': fem_score,
                'mixing_factor': mixing_factor,
                'displacement_map': disp_3d,
                'beta': data.get('beta', 1)
            })
            
            if cur_iter % 10 == 0 and pbar:
                pbar.set_postfix({
                    'C': f"{comp:.2f}", 
                    'V': f"{vol:.2f}",
                    'S': f"{fem_score:.2f}"
                })
            
            if pbar:
                pbar.update(1)

    # --- Run Optimization ---
    if comm.rank == 0:
        print(f"  FEniTop: iter={max_iter}, V={simp_config.vol_frac:.2f}, r={filter_r:.1f}, strategy={strategy}, bc={bc_type}")
    topopt(fem, opt, initial_density=initial_density, callback=record_step)
    
    if comm.rank == 0 and pbar:
        pbar.close()
        
    # --- Post-Process: Add Final Score to all steps ---
    if comm.rank == 0 and history:
        final_step = history[-1]
        final_fem_score = final_step['fem_score']
        final_compliance = final_step['compliance']
        
        for step_data in history:
            step_data['final_fem_score'] = final_fem_score
            step_data['final_compliance'] = final_compliance

    return history
