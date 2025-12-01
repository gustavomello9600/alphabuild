import numpy as np
import dolfinx
import dolfinx.fem as fem
import ufl
from petsc4py import PETSc
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import time
from tqdm import tqdm
import scipy.ndimage

from alphabuilder.src.core.physics_model import FEMContext, PhysicalProperties, SimulationResult

@dataclass
class SIMPConfig:
    """Configuration for SIMP optimization."""
    vol_frac: float = 0.3         # Target volume fraction (Updated for stability)
    penal: float = 3.0            # Penalization power (p)
    r_min: float = 3.5            # Filter radius (in elements) (Increased for better connectivity)
    max_iter: int = 50            # Max iterations
    change_tol: float = 0.01      # Convergence tolerance (change in density)
    move_limit: float = 0.2       # Max density change per step
    adaptive_penal: bool = False  # Enable adaptive penalty schedule (delayed continuation)
    adaptive_penal: bool = False  # Enable adaptive penalty schedule (delayed continuation)
    load_config: Optional[Dict] = None # Custom load configuration
    debug_log_path: Optional[str] = None # Path to save detailed CSV log

def apply_sensitivity_filter(
    x: np.ndarray, 
    dc: np.ndarray, 
    H: np.ndarray, 
    Hs: np.ndarray
) -> np.ndarray:
    """
    Apply mesh-independent sensitivity filter.
    """
    # Simple convolution filter: dc_new = (H * (x * dc)) / (Hs * x)
    x_safe = np.maximum(x, 1e-3)
    numerator = H.dot(x_safe * dc)
    denominator = Hs * x_safe
    return numerator / denominator

def apply_heaviside_projection(x: np.ndarray, beta: float, eta: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Heaviside projection to force 0/1 values.
    Returns: (x_projected, derivative)
    """
    if beta < 1e-6:
        return x, np.ones_like(x)
        
    tanh_beta_eta = np.tanh(beta * eta)
    tanh_beta_1_eta = np.tanh(beta * (1 - eta))
    denom = tanh_beta_eta + tanh_beta_1_eta
    
    tanh_val = np.tanh(beta * (x - eta))
    x_new = (tanh_beta_eta + tanh_val) / denom
    
    # Derivative: d(x_new)/dx
    # d/dx tanh(u) = sech^2(u) * u' = (1 - tanh^2(u)) * u'
    dtanh = beta * (1 - tanh_val**2)
    derivative = dtanh / denom
    
    return x_new, derivative

class HelmholtzFilter:
    """
    Helmholtz PDE Filter for Topology Optimization.
    Solves: -r^2 * laplace(u) + u = x
    """
    def __init__(self, ctx: FEMContext, r_min: float):
        self.ctx = ctx
        self.r_min = r_min
        
        # Filter radius parameter (r = r_min / (2*sqrt(3)) for equivalent support?)
        # Common heuristic: r = r_min / 3.5 or similar. 
        # For SIMP, r_min is usually the diameter of the filter.
        # The PDE length scale r is related to r_min.
        # If r_min is the radius of the cone filter, r ~ r_min / 2.5.
        # Let's use r = r_min / 3.0 as a starting point.
        self.r = r_min / 3.0
        
        # Create Function Spaces
        # Input/Output is DG0 (Element-wise)
        # Filter solution is CG1 (Continuous)
        self.V_dg = ctx.material_field.function_space # DG0
        self.V_cg = fem.functionspace(ctx.domain, ("CG", 1))
        
        # Trial and Test Functions for CG1
        u = ufl.TrialFunction(self.V_cg)
        v = ufl.TestFunction(self.V_cg)
        
        # Variational Form: (r^2 * grad(u) . grad(v) + u * v) * dx
        self.a = (self.r**2 * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(u, v)) * ufl.dx
        
        # Assemble Matrix A (Constant)
        self.A = fem.petsc.assemble_matrix(fem.form(self.a))
        self.A.assemble()
        
        # Create Solver
        self.solver = PETSc.KSP().create(ctx.domain.comm)
        self.solver.setOperators(self.A)
        self.solver.setType(PETSc.KSP.Type.CG)
        self.solver.getPC().setType(PETSc.PC.Type.GAMG)
        self.solver.setFromOptions()
        
        self.u_sol = fem.Function(self.V_cg)
        
    def apply(self, x_dens: np.ndarray) -> np.ndarray:
        # 1. Map x_dens (numpy) to DG0 Function
        # We assume x_dens is ordered same as DG0 dofs.
        # If not, we need the map. But usually for DG0 it matches cell index.
        # Let's assume it matches for now (standard for FEniCSx if we extract correctly)
        
        # Create input function
        f_in = fem.Function(self.V_dg)
        f_in.x.array[:] = x_dens
        
        # 2. Solve PDE
        # L = f_in * v * dx
        v = ufl.TestFunction(self.V_cg)
        L = ufl.inner(f_in, v) * ufl.dx
        b = fem.petsc.assemble_vector(fem.form(L))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        
        self.solver.solve(b, self.u_sol.x.petsc_vec)
        self.u_sol.x.scatter_forward()
        
        # 3. Map result (CG1) back to DG0 (Element-wise average/centroid)
        # We can just interpolate CG1 -> DG0
        # Create a temp DG0 function to hold result
        result_dg = fem.Function(self.V_dg)
        result_dg.interpolate(self.u_sol)
        
        return result_dg.x.array.copy()

def run_simp_optimization_3d(
    ctx: FEMContext,
    props: PhysicalProperties,
    config: SIMPConfig,
    resolution: Tuple[int, int, int],
    seed: int = 0,
    initial_density: Optional[np.ndarray] = None,
    passive_mask: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    
    nx, ny, nz = resolution
    n_elements = nx * ny * nz
    
    # Initialize density
    if initial_density is not None:
         if initial_density.ndim == 3:
             if not hasattr(ctx, 'dof_indices'):
                V = ctx.material_field.function_space
                coords = V.tabulate_dof_coordinates()
                # Adaptive Physical Domain (Step 1)
                voxel_size = 1.0
                phys_L = nx * voxel_size
                phys_H = ny * voxel_size
                phys_W = nz * voxel_size
                
                dx = phys_L / nx
                dy = phys_H / ny
                dz = phys_W / nz
                ix = np.floor((coords[:, 0]) / dx).astype(np.int32)
                iy = np.floor((coords[:, 1]) / dy).astype(np.int32)
                iz = np.floor((coords[:, 2]) / dz).astype(np.int32)
                ix = np.clip(ix, 0, nx - 1)
                iy = np.clip(iy, 0, ny - 1)
                iz = np.clip(iz, 0, nz - 1)
                objectsetattr(ctx, 'dof_indices', (ix, iy, iz))
             
             ix, iy, iz = ctx.dof_indices
             x = initial_density[ix, iy, iz]
         else:
             x = initial_density.copy()
    else:
        x = np.ones(n_elements) * config.vol_frac
    
    # Ensure dof_indices are computed
    if not hasattr(ctx, 'dof_indices'):
         V = ctx.material_field.function_space
         coords = V.tabulate_dof_coordinates()
         # Adaptive Physical Domain (Step 1)
         voxel_size = 1.0
         phys_L = nx * voxel_size
         phys_H = ny * voxel_size
         phys_W = nz * voxel_size
         
         dx = phys_L / nx
         dy = phys_H / ny
         dz = phys_W / nz
         ix = np.floor((coords[:, 0]) / dx).astype(np.int32)
         iy = np.floor((coords[:, 1]) / dy).astype(np.int32)
         iz = np.floor((coords[:, 2]) / dz).astype(np.int32)
         ix = np.clip(ix, 0, nx - 1)
         iy = np.clip(iy, 0, ny - 1)
         iz = np.clip(iz, 0, nz - 1)
         objectsetattr(ctx, 'dof_indices', (ix, iy, iz))

    # Initialize Helmholtz Filter
    pde_filter = HelmholtzFilter(ctx, config.r_min)
    
    history = []

    # Helper to record state
    def record_state(step_num, current_x, current_compliance=0.0):
        if not hasattr(ctx, 'dof_indices'):
             pass
             
        ix, iy, iz = ctx.dof_indices
        d_map = np.zeros((nx, ny, nz), dtype=np.float32)
        d_map[ix, iy, iz] = current_x
        b_map = (d_map > 0.5).astype(np.int32)
        history.append({
            'step': step_num,
            'density_map': d_map,
            'binary_map': b_map,
            'fitness': 0.0,
            'max_displacement': 0.0,
            'compliance': current_compliance,
            'valid': True
        })

    record_state(0, x)

    loop = 0
    change = 1.0
    b = ctx.problem.b
    u = ctx.u_sol
    
    # Pre-assemble Load Vector b (Constant Force)
    b.zeroEntries()
    dolfinx.fem.petsc.assemble_vector(b, ctx.problem.L)
    
    # Apply Load (Dynamic based on config)
    if hasattr(config, 'load_config') and config.load_config:
        lc = config.load_config
        
        voxel_size = 1.0
        grid_x = lc.get('x', nx-1)
        grid_y = lc.get('y', ny//2)
        grid_z_start = lc.get('z_start', nz//2)
        grid_z_end = lc.get('z_end', grid_z_start + 3)
        
        px = (grid_x + 0.5) * voxel_size
        py = (grid_y + 0.5) * voxel_size
        pz_start = grid_z_start * voxel_size
        pz_end = (grid_z_end + 1) * voxel_size
        tol = voxel_size * 0.6 
        
        def load_locator(p):
            return np.logical_and(
                np.isclose(p[0], px, atol=tol),
                np.logical_and(
                    np.isclose(p[1], py, atol=tol),
                    np.logical_and(p[2] >= pz_start, p[2] <= pz_end)
                )
            )
        
        V = ctx.V
        load_dofs = fem.locate_dofs_geometrical(V, load_locator)
        bs = V.dofmap.index_map_bs
        with b.localForm() as b_local:
            f_total = lc.get('magnitude', -1.0)
            n_nodes = len(load_dofs) // bs if bs > 0 else 1
            if n_nodes > 0:
                f_node = f_total / n_nodes
                for dof in load_dofs:
                    if dof % bs == 1: # Y component
                        b_local.setValues([dof], [f_node], addv=PETSc.InsertMode.ADD_VALUES)
    else:
        # Fallback: Point Load
        max_coords = ctx.domain.geometry.x.max(axis=0)
        tip_x = max_coords[0]
        mid_y = max_coords[1] / 2.0
        mid_z = max_coords[2] / 2.0
        def point_load_locator(p):
            return np.logical_and(
                np.isclose(p[0], tip_x, atol=0.1),
                np.logical_and(
                    np.isclose(p[1], mid_y, atol=0.1),
                    np.isclose(p[2], mid_z, atol=0.1)
                )
            )
        V = ctx.V
        load_dofs = fem.locate_dofs_geometrical(V, point_load_locator)
        bs = V.dofmap.index_map_bs
        with b.localForm() as b_local:
            f_total = -1.0
            n_nodes = len(load_dofs) // bs if bs > 0 else 1
            if n_nodes > 0:
                f_node = f_total / n_nodes
                for dof in load_dofs:
                    if dof % bs == 1:
                        b_local.setValues([dof], [f_node], addv=PETSc.InsertMode.ADD_VALUES)
        
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, ctx.problem.bcs)

    from tqdm import tqdm
    pbar = tqdm(total=config.max_iter, desc="SIMP Optimization", unit="iter", leave=False, ncols=100, mininterval=0.5)
    
    # Initialize Debug CSV
    if config.debug_log_path:
        import csv
        with open(config.debug_log_path, 'w', newline='') as csvfile:
            fieldnames = [
                'iter', 'compliance', 'vol_frac', 'target_vol', 'change', 
                'penalty', 'beta', 
                'dens_min', 'dens_max', 'dens_mean', 'dens_std',
                'gray_frac', 'solid_frac', 'void_frac',
                'sens_min', 'sens_max', 'sens_mean',
                'max_disp', 'n_components',
                't_solve', 't_filter', 't_update'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    while change > config.change_tol and loop < config.max_iter:
        loop += 1
        
        # Continuation Schedule (Penalty & Beta)
        current_penal = config.penal
        beta = 1.0
        
        if config.adaptive_penal:
            # Delayed continuation:
            # Iter 0-20: p=1.0 (Convex)
            # Iter 21-40: p -> 3.0
            # Iter 41+: Beta increase
            
            if loop <= 20:
                current_penal = 1.0
                beta = 1.0
            elif loop <= 40:
                # Ramp p from 1.0 to config.penal
                progress = (loop - 20) / 20.0
                current_penal = 1.0 + (config.penal - 1.0) * progress
                beta = 1.0
            else:
                current_penal = config.penal
                # Start Beta continuation late
                beta = 2.0 ** ((loop - 40) / 5.0)
                beta = min(beta, 64.0)
        else:
            # Standard Beta Schedule
            if loop <= 10:
                beta = 1.0
            elif loop <= 40:
                beta = 2.0 ** ((loop - 10) / 5.0)
            else:
                beta = 64.0
            
        # Filtering (Three-Field Scheme)
        t_filter_start = time.time()
        x_tilde = pde_filter.apply(x)
        
        # Project
        x_phys, dx_proj_dx = apply_heaviside_projection(x_tilde, beta)
        
        # Update FEM Material Field
        x_phys = np.clip(x_phys, 0.0, 1.0)
        penalized_density = x_phys ** current_penal
        ctx.material_field.x.array[:] = penalized_density
        ctx.material_field.x.scatter_forward()
        t_filter = time.time() - t_filter_start
        
        # Assemble A
        t_asm_start = time.time()
        A = ctx.problem.A
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, ctx.problem.a, bcs=ctx.problem.bcs)
        A.assemble()
        t_asm = time.time() - t_asm_start
        
        # Solve
        t0 = time.time()
        solver = ctx.problem.solver
        solver.solve(b, u.x.petsc_vec)
        u.x.scatter_forward()
        t_sol = time.time() - t0
        
        # Compliance and Sensitivity
        # C = f^T u = u^T K u
        # Sensitivity: dC/dx = -p * x^(p-1) * (u^T K_e u)
        
        # Calculate Strain Energy Density (Element-wise)
        # We can use UFL to compute strain energy density field
        # W = 0.5 * sigma : epsilon
        # Compliance density = 2 * W
        
        # But we need it per element (DG0)
        # Let's compute it efficiently.
        # For SIMP, dC/drho = -p * rho^(p-1) * (u_e^T k_0 u_e)
        # We can compute (u^T k_0 u) by assembling a form with E=1 (or E_0)
        
        # Define strain energy form for full material
        E_0 = props.E
        E_min = 1e-6
        nu = props.nu
        mu_0 = E_0 / (2 * (1 + nu))
        lmbda_0 = E_0 * nu / ((1 + nu) * (1 - 2 * nu))
        
        def sigma_0(u):
            return lmbda_0 * ufl.tr(ufl.sym(ufl.grad(u))) * ufl.Identity(len(u)) + 2.0 * mu_0 * ufl.sym(ufl.grad(u))
            
        # Strain energy density (scalar field)
        strain_energy = ufl.inner(sigma_0(u), ufl.sym(ufl.grad(u))) # This is 2*W if E=E_0
        
        # Project strain energy to DG0 space to get element values
        # This gives us (u^T k_0 u) per cell approximately
        W_dg = fem.Function(ctx.material_field.function_space)
        
        # Projection solver
        # We can use a simple projection since it's DG0 (just integrate over cell and divide by volume)
        # But fem.Expression is easier
        expr = fem.Expression(strain_energy, ctx.material_field.function_space.element.interpolation_points)
        W_dg.interpolate(expr)
        strain_energy_values = W_dg.x.array
        
        # Compliance
        # C = integral(rho^p * strain_energy)
        # But we computed strain_energy using u, which comes from penalized stiffness.
        # So C = b.dot(u.vector)
        compliance = b.dot(u.x.petsc_vec)
        
        # Sensitivity
        # dC/dx_phys = -p * x_phys^(p-1) * strain_energy_values
        # Note: strain_energy_values is roughly u^T k_0 u
        # Actually, if we use E_min, it's slightly more complex, but for E_min << E_0, this holds.
        
        dc = -current_penal * (x_phys ** (current_penal - 1)) * strain_energy_values
        
        # Chain Rule for Projection: dC/dx_tilde = dC/dx_phys * dx_phys/dx_tilde
        dc = dc * dx_proj_dx
        
        # Chain Rule for Filter: dC/dx = Filter^T * dC/dx_tilde
        # Since filter is self-adjoint (Helmholtz), we just apply filter again
        dc = pde_filter.apply(dc)
        
        # Optimality Criteria Update
        # x_new = x * ( -dc / lambda )^eta
        # We need to find lambda (Lagrange multiplier) such that sum(x_new) = vol_frac
        
        def optimality_criteria(x, dc, target_vol, limit):
            l1 = 0.0
            l2 = 1e9
            move = limit
            
            # Damping
            damping = 0.5
            
            x_new = np.zeros_like(x)
            
            while (l2 - l1) / (l1 + l2 + 1e-9) > 1e-3:
                l_mid = 0.5 * (l2 + l1)
                
                # OC Update Rule
                # B_e = -dc / lambda
                # x_new = max(0, max(x - move, min(1, min(x + move, x * sqrt(B_e)))))
                
                # Avoid division by zero
                # dc is negative, so -dc is positive.
                # We want B_e = (-dc) / l_mid
                
                factor = np.sqrt(np.maximum(1e-10, -dc) / l_mid)
                
                x_trial = x * factor
                
                x_lower = np.maximum(0.0, x - move)
                x_upper = np.minimum(1.0, x + move)
                
                x_new = np.clip(x_trial, x_lower, x_upper)
                
                if np.mean(x_new) > target_vol:
                    l1 = l_mid
                else:
                    l2 = l_mid
            return x_new

        # Dynamic Move Limit?
        current_move_limit = config.move_limit
        
        x_new = optimality_criteria(x, dc, config.vol_frac, current_move_limit)
        t_update = time.time() - t0 # Fix timer approximation
        
        change = np.max(np.abs(x_new - x))
        x = x_new
        
        # Record
        record_state(loop, x_phys, compliance)
        
        pbar.set_postfix({
            'Chg': f"{change:.3f}", 
            'C': f"{compliance:.1f}",
            'T_sol': f"{t_sol:.2f}s",
            'T_asm': f"{t_asm:.2f}s"
        })
        pbar.update(1)

        # Detailed Logging
        if config.debug_log_path:
            # Calculate extra stats
            # Density Stats
            d_flat = x_phys.flatten()
            dens_min, dens_max = np.min(d_flat), np.max(d_flat)
            dens_mean, dens_std = np.mean(d_flat), np.std(d_flat)
            gray_frac = np.mean((d_flat > 0.1) & (d_flat < 0.9))
            solid_frac = np.mean(d_flat >= 0.9)
            void_frac = np.mean(d_flat <= 0.1)
            
            # Sensitivity Stats
            s_flat = dc.flatten()
            sens_min, sens_max, sens_mean = np.min(s_flat), np.max(s_flat), np.mean(s_flat)
            
            # Max Displacement
            # u is a vector function. We need magnitude.
            # u.x.petsc_vec is the vector.
            # We can get it from u.x.array (which is flattened dofs)
            # This is approximate but fast.
            u_arr = u.x.array
            # Reshape to (N, 3)? No, it's blocked by dof.
            # Just take max abs value of any component as a proxy, or norm if possible.
            # For logging, max component is fine, or we can compute norm properly if cheap.
            # Let's use max absolute value of any DOF for speed.
            max_disp = np.max(np.abs(u_arr))
            
            # Connectivity (Expensive but requested)
            # Reconstruct 3D map for connectivity
            if hasattr(ctx, 'dof_indices'):
                ix, iy, iz = ctx.dof_indices
                d_map_temp = np.zeros((nx, ny, nz), dtype=np.float32)
                d_map_temp[ix, iy, iz] = x_phys
                labeled, n_components = scipy.ndimage.label(d_map_temp > 0.5)
            else:
                n_components = -1

            with open(config.debug_log_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'iter': loop,
                    'compliance': compliance,
                    'vol_frac': np.mean(x_phys),
                    'target_vol': config.vol_frac,
                    'change': change,
                    'penalty': current_penal,
                    'beta': beta,
                    'dens_min': dens_min,
                    'dens_max': dens_max,
                    'dens_mean': dens_mean,
                    'dens_std': dens_std,
                    'gray_frac': gray_frac,
                    'solid_frac': solid_frac,
                    'void_frac': void_frac,
                    'sens_min': sens_min,
                    'sens_max': sens_max,
                    'sens_mean': sens_mean,
                    'max_disp': max_disp,
                    'n_components': n_components,
                    't_solve': t_sol,
                    't_filter': t_filter,
                    't_update': t_update
                })
        
    pbar.close()
    return history

def objectsetattr(obj, name, value):
    """Helper to set attribute on frozen dataclass or object"""
    object.__setattr__(obj, name, value)
