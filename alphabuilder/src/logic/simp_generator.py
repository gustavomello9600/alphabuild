import numpy as np
import dolfinx
from dolfinx import fem
import ufl
from petsc4py import PETSc
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass

from alphabuilder.src.core.physics_model import FEMContext, PhysicalProperties, SimulationResult

@dataclass
class SIMPConfig:
    """Configuration for SIMP optimization."""
    vol_frac: float = 0.5         # Target volume fraction
    penal: float = 3.0            # Penalization power (p)
    r_min: float = 1.5            # Filter radius (in elements)
    max_iter: int = 50            # Max iterations
    change_tol: float = 0.01      # Convergence tolerance (change in density)
    move_limit: float = 0.2       # Max density change per step

def apply_sensitivity_filter(
    x: np.ndarray, 
    dc: np.ndarray, 
    H: np.ndarray, 
    Hs: np.ndarray
) -> np.ndarray:
    """
    Apply mesh-independent sensitivity filter.
    
    Args:
        x: Current density array.
        dc: Sensitivity (derivative of compliance).
        H: Convolution kernel (precomputed).
        Hs: Kernel sum (precomputed).
        
    Returns:
        Filtered sensitivity.
    """
    # Simple convolution filter: dc_new = (H * (x * dc)) / (Hs * x)
    # Note: x must be > 0 to avoid division by zero
    x_safe = np.maximum(x, 1e-3)
    numerator = H.dot(x_safe * dc)
    denominator = Hs * x_safe
    return numerator / denominator

def precompute_filter(nx: int, ny: int, r_min: float) -> Tuple[Any, np.ndarray]:
    """
    Precompute filter matrix H and vector Hs.
    Uses scipy.sparse for efficiency.
    """
    from scipy.sparse import lil_matrix
    
    nelx, nely = nx, ny
    n_elements = nelx * nely
    
    H = lil_matrix((n_elements, n_elements))
    Hs = np.zeros(n_elements)
    
    # r_min in element units
    r_min_sq = r_min**2
    
    # This can be slow for large meshes, but is done once per resolution
    # Optimized loop
    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = i1 * nely + j1
            
            # Search window
            i_min = max(0, int(i1 - np.ceil(r_min)))
            i_max = min(nelx, int(i1 + np.ceil(r_min) + 1))
            j_min = max(0, int(j1 - np.ceil(r_min)))
            j_max = min(nely, int(j1 + np.ceil(r_min) + 1))
            
            for i2 in range(i_min, i_max):
                for j2 in range(j_min, j_max):
                    e2 = i2 * nely + j2
                    
                    dist = np.sqrt((i1 - i2)**2 + (j1 - j2)**2)
                    
                    if dist < r_min:
                        val = r_min - dist
                        H[e1, e2] = val
                        Hs[e1] += val
                        
    return H.tocsr(), Hs

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
    
    # Bisection method for Lagrange multiplier
    while (l2 - l1) / (l1 + l2 + 1e-9) > 1e-3:
        l_mid = 0.5 * (l2 + l1)
        
        # B_e = -dc / lambda
        # x_new = x * sqrt(B_e)
        # But limited by move_limit and [0, 1]
        
        # Avoid division by zero and sqrt of negative
        # dc is usually negative (compliance decreases as density increases)
        # So -dc is positive.
        
        # Heuristic update rule:
        # x_new = max(0, max(x - move, min(1, min(x + move, x * sqrt(-dc/l_mid)))))
        
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

def run_simp_optimization(
    ctx: FEMContext,
    props: PhysicalProperties,
    config: SIMPConfig,
    seed: int = 0
) -> List[Dict[str, Any]]:
    """
    Run a full SIMP optimization loop.
    
    Returns:
        List of dictionaries containing history of the optimization.
        Each dict has: 'density', 'fitness', 'compliance', 'max_disp', 'valid'
    """
    # 1. Setup
    # Extract resolution from dof_map shape (ny, nx)
    ny, nx = ctx.dof_map.shape
    n_elements = nx * ny
    
    # Initialize density x (uniform = vol_frac)
    x = np.ones(n_elements) * config.vol_frac
    
    # Precompute filter
    # Note: For production, cache this based on (nx, ny, r_min)
    H, Hs = precompute_filter(nx, ny, config.r_min)
    
    history = []
    
    loop = 0
    change = 1.0
    
    # Reuse solver structures
    b = ctx.problem.b
    u = ctx.u_sol
    
    while change > config.change_tol and loop < config.max_iter:
        loop += 1
        
        # 2. Update FEM Material Field
        # Map 1D density x back to 2D grid for FEM
        # Note: Our dof_map logic in physics_model maps (row, col) to index
        # We need to be careful with ordering.
        # x is flat (nx * ny). Let's assume row-major or col-major consistent with filter.
        # In precompute_filter: e = i * nely + j  (where i is x, j is y)
        # So x is ordered by x then y (column-major effectively if x is col index)
        
        # Let's construct the 2D density map
        density_map = np.zeros((ny, nx))
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                # FEM coordinates: row = ny - 1 - j, col = i
                # Wait, let's check physics_model.py dof_map logic:
                # col = int(x/dx), row = ny - 1 - int(y/dy)
                # So j corresponds to y-index (0 at bottom), row corresponds to matrix row (0 at top)
                row = ny - 1 - j
                col = i
                density_map[row, col] = x[idx]
        
        # Update FEM context
        # We use the same logic as solver.py but with continuous values
        mask = ctx.dof_map >= 0
        valid_indices = ctx.dof_map[mask]
        
        # We need to extract values from density_map in the order of valid_indices
        # This is tricky because dof_map is (ny, nx).
        # Let's just iterate the map
        # Or simpler:
        # valid_values = density_map[mask] -> This works if mask follows (ny, nx) order
        valid_values = density_map[mask].astype(dolfinx.default_scalar_type)
        
        # Apply penalization E = E_min + x^p * (E_0 - E_min)
        # But wait, physics_model.py does: E = E_void + (E_solid - E_void) * material_field
        # So we should set material_field = x^p
        # SIMP uses penalized density for stiffness
        penalized_density = valid_values ** config.penal
        
        ctx.material_field.x.array[valid_indices] = penalized_density
        ctx.material_field.x.scatter_forward()
        
        # 3. Solve FEM
        # We can reuse logic from solver.py, but we need gradients (compliance sensitivity)
        # solver.py doesn't return gradients. We need to implement solve here or refactor.
        # Let's implement minimal solve here to get compliance and u.
        
        # Assemble A
        A = ctx.problem.A
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, ctx.problem.a, bcs=ctx.problem.bcs)
        A.assemble()
        
        # Assemble b (Load)
        b.zeroEntries()
        dolfinx.fem.petsc.assemble_vector(b, ctx.problem.L)
        
        # Apply Point Load (Hardcoded for now as in solver.py - (2.0, 0.5))
        # Ideally this should be configurable or extracted from a "ProblemDefinition"
        V = ctx.V
        def point_load_locator(p):
            return np.logical_and(np.isclose(p[0], 2.0, atol=1e-3), np.isclose(p[1], 0.5, atol=1e-3))
            
        load_dofs = fem.locate_dofs_geometrical(V, point_load_locator)
        if len(load_dofs) > 0:
            V_y, vy_map = V.sub(1).collapse()
            load_dofs_y_local = fem.locate_dofs_geometrical(V_y, point_load_locator)
            load_dofs_y = vy_map[load_dofs_y_local]
            with b.localForm() as b_local:
                for dof in load_dofs_y:
                    b_local.setValues([dof], [-1.0], addv=PETSc.InsertMode.ADD_VALUES)
                    
        dolfinx.fem.petsc.apply_lifting(b, [ctx.problem.a], bcs=[ctx.problem.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, ctx.problem.bcs)
        
        # Solve
        solver = ctx.problem.solver
        solver.solve(b, u.x.petsc_vec)
        u.x.scatter_forward()
        
        # 4. Compute Compliance and Sensitivity
        # Compliance c = u^T K u = u^T f
        compliance = b.dot(u.x.petsc_vec)
        
        # Sensitivity dc/dx_e = -p * x_e^(p-1) * u_e^T k_0 u_e
        # This is element-wise strain energy.
        # In FEniCSx, we can compute this by projecting strain energy density.
        # But simpler: Total Strain Energy = 1/2 * Compliance
        # We need element-wise contributions.
        
        # Let's use UFL to compute strain energy density field
        # W = 0.5 * sigma : epsilon
        # But we want u^T k_0 u which is 2 * Strain Energy of element with density=1
        
        # E_e = E_min + x_e^p (E_0 - E_min)
        # dE/dx = p * x_e^(p-1) * (E_0 - E_min)
        # dc/dx = - dE/dx * (strain energy density / E_e) * Volume ?
        # Standard formula: dc/dx_e = -p * x_e^(p-1) * (u_e k0 u_e)
        
        # Let's calculate Strain Energy Density field
        # We define a DG0 function for energy density
        W = fem.functionspace(ctx.mesh, ("DG", 0))
        
        # E_sim = E_void + (E_solid - E_void) * material_field (where material_field is x^p)
        # We want the "Strain Energy Density"
        # strain_energy = 0.5 * inner(sigma(u), epsilon(u))
        # But sigma depends on E.
        
        # Let's compute the raw strain energy density using the CURRENT E
        # Then we can back-calculate sensitivity.
        
        # Define variational form for projection
        # This is expensive to do every step.
        # Alternative: Approximate using cell centers?
        
        # Let's use the standard approach:
        # dc_e = -p * x_e^(p-1) * (Strain Energy of Element / x_e^p)
        #      = -p * (Strain Energy) / x_e
        
        # Calculate element-wise strain energy
        # We can integrate 'inner(sigma, epsilon)' over each cell.
        # Since we use DG0 for material, we can just project the energy density.
        
        # E_field = props.E_void + (props.E_solid - props.E_void) * ctx.material_field
        # mu = E_field / (2 * (1 + props.nu))
        # lmbda = E_field * props.nu / ((1 + props.nu) * (1 - 2 * props.nu))
        # def epsilon(u): return ufl.sym(ufl.grad(u))
        # def sigma(u): return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u))
        # energy_density = ufl.inner(sigma(u), epsilon(u)) # This is 2*Strain Energy Density
        
        # Project energy_density to DG0 space
        # This gives us the average energy density per cell
        
        # Optimization: Don't create new solver every time for projection
        # Just use expression evaluation at cell centers if possible
        
        # Let's define the expression once
        if not hasattr(ctx, 'energy_expr'):
            # This is a hack to attach to ctx, but safe for this script
            E_var = props.E_void + (props.E_solid - props.E_void) * ctx.material_field
            mu = E_var / (2 * (1 + props.nu))
            lmbda = E_var * props.nu / ((1 + props.nu) * (1 - 2 * props.nu))
            eps = ufl.sym(ufl.grad(u))
            sig = 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(len(u))
            # Compliance density = sigma : epsilon
            comp_dens = ufl.inner(sig, eps)
            
            # Create expression
            # In dolfinx 0.8+, we use fem.Expression
            # We need to compile it for the DG0 space
            expr = fem.Expression(comp_dens, W.element.interpolation_points)
            objectsetattr(ctx, 'energy_expr', expr)
            objectsetattr(ctx, 'W_space', W)
            
        energy_vals = fem.Function(ctx.W_space)
        energy_vals.interpolate(ctx.energy_expr)
        
        # Now we have compliance density per cell in 'energy_vals'
        # We need to map this back to our 'x' vector (nx * ny)
        # The mapping is the reverse of what we did for density_map
        
        # Extract array from energy_vals
        # This array is ordered by dof index of DG0 space
        # We need to map it to our (i, j) grid
        
        # Create a grid for sensitivities
        dc_grid = np.zeros((ny, nx))
        
        # Map energy values to grid
        # energy_vals.x.array corresponds to valid_indices in dof_map
        # (Since D and W are both DG0 on the same mesh)
        
        # Get raw energy values
        raw_energies = energy_vals.x.array
        
        # Map back to grid
        dc_grid[mask] = raw_energies
        
        # Now flatten to x vector order
        dc = np.zeros(n_elements)
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                row = ny - 1 - j
                col = i
                
                # Sensitivity calculation
                # dc = -p * x^(p-1) * (Energy / x^p) = -p * Energy / x
                dens = x[idx]
                energy = dc_grid[row, col]
                
                if dens > 1e-3:
                    dc[idx] = -config.penal * energy / dens
                else:
                    dc[idx] = 0.0
                    
        # 5. Filtering
        dc = apply_sensitivity_filter(x, dc, H, Hs)
        
        # 6. Update Design (OC)
        x_new = optimality_criteria(x, dc, config.vol_frac, config.move_limit)
        
        # 7. Check Convergence
        change = np.max(np.abs(x_new - x))
        x = x_new
        
        # 8. Record History
        # Calculate Max Disp for metadata
        u_vals = u.x.array.reshape(-1, 2)
        max_disp = np.max(np.linalg.norm(u_vals, axis=1))
        
        # Binarize for storage (threshold 0.5)
        # Or store continuous? The game uses binary.
        # Let's store binary representation of current step
        binary_map = (density_map > 0.5).astype(np.int32)
        
        # Calculate fitness using game rules
        # omega_mat = sum(binary_map)
        # penalty = max(0, max_disp - props.disp_limit)
        # fitness = 1 / (omega_mat + 0.5*penalty)
        
        # Store
        history.append({
            'step': loop,
            'density_map': density_map.copy(), # Continuous
            'binary_map': binary_map,          # Binary
            'fitness': 0.0, # Placeholder, calculated later if needed
            'max_displacement': max_disp,
            'compliance': compliance,
            'valid': True
        })
        
        # Print progress
        if loop % 10 == 0 or loop == 1:
            print(f"  SIMP Iter {loop}: Change={change:.4f}, Compliance={compliance:.4f}, MaxDisp={max_disp:.4f}")
            
    return history

# Helper to allow setting attributes on frozen dataclass (for caching)
def objectsetattr(obj, name, value):
    object.__setattr__(obj, name, value)
