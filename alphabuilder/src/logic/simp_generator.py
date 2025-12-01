import numpy as np
import dolfinx
import dolfinx.fem as fem
import ufl
from petsc4py import PETSc
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import time
from tqdm import tqdm

from alphabuilder.src.core.physics_model import FEMContext, PhysicalProperties, SimulationResult

@dataclass
class SIMPConfig:
    """Configuration for SIMP optimization."""
    vol_frac: float = 0.3         # Target volume fraction (Updated for stability)
    penal: float = 3.0            # Penalization power (p)
    r_min: float = 2.0            # Filter radius (in elements) (Updated for cubic voxels)
    max_iter: int = 50            # Max iterations
    change_tol: float = 0.01      # Convergence tolerance (change in density)
    move_limit: float = 0.2       # Max density change per step
    load_config: Optional[Dict] = None # Custom load configuration

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

# ... (precompute_filter_3d remains same)

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
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        
        # Placeholders
        self.u_sol = fem.Function(self.V_cg)
        self.x_dg = fem.Function(self.V_dg) # Helper for input
        
        # Pre-compile Linear Form
        v = ufl.TestFunction(self.V_cg)
        self.L = ufl.inner(self.x_dg, v) * ufl.dx
        self.L_form = fem.form(self.L)
        
        # Create RHS Vector
        # Use assemble_vector to create the vector initially
        self.b_vec = fem.petsc.assemble_vector(self.L_form)
        
    def apply(self, x_array: np.ndarray) -> np.ndarray:
        """
        Apply filter: DG0 -> CG1 (Solve) -> DG0 (Interpolate/Project)
        """
        # 1. Load input into DG0 function
        self.x_dg.x.array[:] = x_array
        self.x_dg.x.scatter_forward() # Ensure ghost values are updated
        
        # 2. Assemble RHS: L = x_dg * v * dx
        # Zero out the vector first
        with self.b_vec.localForm() as b_loc:
            b_loc.set(0.0)
            
        fem.petsc.assemble_vector(self.b_vec, self.L_form)
        self.b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        
        # 3. Solve
        self.solver.solve(self.b_vec, self.u_sol.x.petsc_vec)
        self.u_sol.x.scatter_forward()
        
        # 4. Return to DG0
        # We can interpolate CG1 -> DG0 directly
        # Or project? Interpolation is faster and usually sufficient.
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
    # Note: r_min is in element units. We need physical units?
    # If mesh coordinates are physical, r_min should be physical.
    # Our mesh is 1.0 per voxel? Yes, we scaled it.
    # Wait, in physics_model.py we might have scaled the mesh?
    # Let's assume coords are physical.
    # If r_min=2.0 (voxels), and voxel_size=1.0, then r_min_phys=2.0.
    
    # Check if we need to scale r_min
    # In run_data_harvest, we set r_min=2.0.
    # If we use Helmholtz, we use that directly.
    
    # Initialize Filter
    pde_filter = HelmholtzFilter(ctx, config.r_min)
    
    history = []

    # Helper to record state
    def record_state(step_num, current_x, current_compliance=0.0):
        # With Helmholtz, we don't rely on dof_indices for the map anymore!
        # We can just reshape the array if ordering matches?
        # NO! FEniCSx ordering is still different from grid ordering.
        # We still need dof_indices to map back to (nx, ny, nz) for visualization.
        
        if not hasattr(ctx, 'dof_indices'):
             # Compute mapping if missing (should be there from init block)
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
    # Assemble b once
    b.zeroEntries()
    dolfinx.fem.petsc.assemble_vector(b, ctx.problem.L)
    
    # Apply Load (Dynamic based on config)
    if hasattr(config, 'load_config') and config.load_config:
        lc = config.load_config
        
        # Adaptive Physical Domain (Step 1)
        voxel_size = 1.0
        phys_L = nx * voxel_size
        phys_H = ny * voxel_size
        phys_W = nz * voxel_size
        
        grid_x = lc.get('x', nx-1)
        grid_y = lc.get('y', ny//2)
        grid_z_start = lc.get('z_start', nz//2)
        grid_z_end = lc.get('z_end', grid_z_start + 3)
        
        # Step 2: Synchronize Masks
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
            f_total = -1.0
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
    
    while change > config.change_tol and loop < config.max_iter:
        loop += 1
        
        # Beta Continuation Schedule
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
        penalized_density = x_phys ** config.penal
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
        
        # b is already assembled and constant!
        
        # Solve
        t0 = time.time()
        solver = ctx.problem.solver
        solver.solve(b, u.x.petsc_vec)
        u.x.scatter_forward()
        t_solve = time.time() - t0
        
        # Compute Compliance and Sensitivity
        t1 = time.time()
        compliance = b.dot(u.x.petsc_vec)
        
        # Sensitivity Calculation
        if not hasattr(ctx, 'energy_expr'):
             E_solid = props.E
             E_void = 1e-6
             E_var = E_void + (E_solid - E_void) * ctx.material_field
             mu = E_var / (2 * (1 + props.nu))
             lmbda = E_var * props.nu / ((1 + props.nu) * (1 - 2 * props.nu))
             eps = ufl.sym(ufl.grad(u))
             sig = 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(len(u))
             comp_dens = ufl.inner(sig, eps)
             W = fem.functionspace(ctx.domain, ("DG", 0))
             expr = fem.Expression(comp_dens, W.element.interpolation_points)
             objectsetattr(ctx, 'energy_expr', expr)
             objectsetattr(ctx, 'W_space', W)
             energy_vals = fem.Function(W)
             objectsetattr(ctx, 'energy_vals', energy_vals)
             
        energy_vals = ctx.energy_vals
        energy_vals.interpolate(ctx.energy_expr)
        energies = energy_vals.x.array
        
        x_phys_safe = np.maximum(x_phys, 1e-3)
        dc_phys = -config.penal * energies / x_phys_safe
        
        dc_tilde = dc_phys * dx_proj_dx
        
        # Filter Sensitivity
        dc = pde_filter.apply(dc_tilde)
        
        t_sens = time.time() - t1
        
        # Update Design
        t3 = time.time()
        
        if loop <= 5:
            current_move_limit = 0.05
        else:
            current_move_limit = config.move_limit
        
        x_new = optimality_criteria(x, dc, config.vol_frac, current_move_limit)
        t_update = time.time() - t3
        
        change = np.max(np.abs(x_new - x))
        x = x_new
        
        # Enforce passive elements
        if passive_mask is None and hasattr(config, 'load_config') and config.load_config:
             lc = config.load_config
             if hasattr(ctx, 'dof_indices'):
                 ix, iy, iz = ctx.dof_indices
                 grid_x = lc.get('x', nx-1)
                 grid_y = lc.get('y', ny//2)
                 grid_z_start = lc.get('z_start', nz//2)
                 grid_z_end = lc.get('z_end', grid_z_start + 3)
                 
                 local_mask = (np.abs(ix - grid_x) <= 1) & \
                              (np.abs(iy - grid_y) <= 1) & \
                              (iz >= grid_z_start - 1) & (iz <= grid_z_end + 1)
                 
                 x[local_mask] = 1.0
        
        elif passive_mask is not None:
             if not hasattr(ctx, 'flat_passive_mask_arg'):
                 ix, iy, iz = ctx.dof_indices
                 flat_mask = passive_mask[ix, iy, iz].astype(bool)
                 objectsetattr(ctx, 'flat_passive_mask_arg', flat_mask)
             x[ctx.flat_passive_mask_arg] = 1.0
        
        # History
        u_vals = u.x.array.reshape(-1, 3)
        max_disp = np.max(np.linalg.norm(u_vals, axis=1))
        
        ix, iy, iz = ctx.dof_indices
        density_map = np.zeros((nx, ny, nz), dtype=np.float32)
        density_map[ix, iy, iz] = x
        binary_map = (density_map > 0.5).astype(np.int32)
        
        history.append({
            'step': loop,
            'density_map': density_map.copy(),
            'binary_map': binary_map,
            'fitness': 0.0,
            'max_displacement': max_disp,
            'compliance': compliance,
            'valid': True
        })
        
        if loop % 10 == 0 or loop == 1:
            pbar.set_postfix({
                "Chg": f"{change:.3f}", 
                "C": f"{compliance:.1f}", 
                "T_sol": f"{t_solve:.2f}s",
                "T_asm": f"{t_asm:.2f}s",
                "T_sen": f"{t_sens:.2f}s"
            })
            
        pbar.update(1)
            
    pbar.close()
    return history

# Removed precompute_filter_3d (Replaced by HelmholtzFilter)

def optimality_criteria(
    x: np.ndarray, 
    dc: np.ndarray, 
    vol_frac: float, 
    move_limit: float
) -> np.ndarray:
    """
    Optimality Criteria (OC) update scheme.
    """
    l1, l2 = 0.0, 1e9
    x_new = np.zeros_like(x)
    
    while (l2 - l1) / (l1 + l2 + 1e-9) > 1e-3:
        l_mid = 0.5 * (l2 + l1)
        
        term = np.sqrt(np.maximum(0, -dc / l_mid))
        x_trial = x * term
        
        x_lower = np.maximum(0.0, x - move_limit)
        x_upper = np.minimum(1.0, x + move_limit)
        
        x_new = np.clip(x_trial, x_lower, x_upper)
        
        if np.mean(x_new) > vol_frac:
            l1 = l_mid
        else:
            l2 = l_mid
            
    return x_new


# Helper to allow setting attributes on frozen dataclass (for caching)
def objectsetattr(obj, name, value):
    object.__setattr__(obj, name, value)




