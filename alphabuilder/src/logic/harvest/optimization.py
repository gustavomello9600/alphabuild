"""
FEniTop Integration for Topology Optimization.
"""
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
from alphabuilder.src.logic.selfplay.reward import calculate_fem_score

def run_fenitop_optimization(
    resolution: Tuple[int, int, int],
    props: PhysicalProperties,
    simp_config: SIMPConfig,
    initial_density: np.ndarray = None,
    strategy: str = 'BEZIER'
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
    
    # --- Load Configuration ---
    load_cfg = simp_config.load_config
    load_y_center = float(load_cfg['y'])
    load_z_center = (float(load_cfg['z_start']) + float(load_cfg['z_end'])) / 2.0
    load_half_width = 1.0
    
    # --- FEM Parameters ---
    fem = {
        "mesh": mesh,
        "mesh_serial": mesh_serial,
        "young's modulus": 100,
        "poisson's ratio": 0.25,
        "disp_bc": lambda x: np.isclose(x[0], 0),
        "traction_bcs": [],
        "body_force": lambda x: np.stack([
            np.zeros(x.shape[1]),
            -2.0 * (
                (x[0] >= float(load_cfg['x'])) & (x[0] <= float(load_cfg['x']) + 1.0) &
                (x[1] >= load_y_center - load_half_width - 0.5) & (x[1] <= load_y_center + load_half_width + 0.5) &
                (x[2] >= load_z_center - load_half_width - 0.5) & (x[2] <= load_z_center + load_half_width + 0.5)
            ),
            np.zeros(x.shape[1])
        ]),
        "quadrature_degree": 2,
        "petsc_options": {
            "ksp_type": "cg",
            "pc_type": "gamg",
            "ksp_rtol": 1e-4,
            "ksp_max_it": 1000,
            "pc_gamg_type": "agg",
            "pc_gamg_agg_nsmooths": 1,
            "pc_gamg_threshold": 0.01,
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
        "solid_zone": lambda x: np.full(x.shape[1], False),
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
        pbar = tqdm(total=max_iter, desc="SIMP", leave=False, unit="it")
    
    def record_step(data):
        if comm.rank == 0:
            rho_flat = data['density']
            
            if grid_mapper is None:
                rho_3d = np.full((nx, ny, nz), simp_config.vol_frac, dtype=np.float32)
            elif len(rho_flat) != len(grid_mapper[0]):
                print(f"  WARNING: Size mismatch in record_step")
                rho_3d = np.full((nx, ny, nz), simp_config.vol_frac, dtype=np.float32)
            else:
                rho_nodal = np.zeros((nx+1, ny+1, nz+1), dtype=np.float32)
                rho_nodal[grid_mapper[0], grid_mapper[1], grid_mapper[2]] = rho_flat
                rho_3d = rho_nodal[:-1, :-1, :-1]
                
            # Process displacement map if available
            disp_3d = None
            if 'displacement_map' in data and data['displacement_map'] is not None:
                disp_flat = data['displacement_map']
                if grid_mapper is not None and len(disp_flat) == len(grid_mapper[0]):
                    disp_nodal = np.zeros((nx+1, ny+1, nz+1), dtype=np.float32)
                    disp_nodal[grid_mapper[0], grid_mapper[1], grid_mapper[2]] = disp_flat
                    disp_3d = disp_nodal[:-1, :-1, :-1]

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
        print(f"  FEniTop: iter={max_iter}, V={simp_config.vol_frac:.2f}, r={filter_r:.1f}, strategy={strategy}")
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
