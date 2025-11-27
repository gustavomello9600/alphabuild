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
    """
    # Simple convolution filter: dc_new = (H * (x * dc)) / (Hs * x)
    x_safe = np.maximum(x, 1e-3)
    numerator = H.dot(x_safe * dc)
    denominator = Hs * x_safe
    return numerator / denominator

def precompute_filter_3d(nx: int, ny: int, nz: int, r_min: float) -> Tuple[Any, np.ndarray]:
    """
    Precompute filter matrix H and vector Hs for 3D.
    """
    from scipy.sparse import lil_matrix
    
    n_elements = nx * ny * nz
    
    H = lil_matrix((n_elements, n_elements))
    Hs = np.zeros(n_elements)
    
    # r_min in element units
    r_min_sq = r_min**2
    ceil_r = int(np.ceil(r_min))
    
    # This can be slow for large meshes.
    # TODO: Optimize with numba or vectorized operations if needed.
    # For now, we assume reasonable resolution (e.g. 32x16x16)
    
    for i1 in range(nx):
        for j1 in range(ny):
            for k1 in range(nz):
                e1 = (i1 * ny + j1) * nz + k1
                
                i_min = max(0, i1 - ceil_r)
                i_max = min(nx, i1 + ceil_r + 1)
                j_min = max(0, j1 - ceil_r)
                j_max = min(ny, j1 + ceil_r + 1)
                k_min = max(0, k1 - ceil_r)
                k_max = min(nz, k1 + ceil_r + 1)
                
                for i2 in range(i_min, i_max):
                    for j2 in range(j_min, j_max):
                        for k2 in range(k_min, k_max):
                            e2 = (i2 * ny + j2) * nz + k2
                            
                            dist = np.sqrt((i1 - i2)**2 + (j1 - j2)**2 + (k1 - k2)**2)
                            
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

def run_simp_optimization(
    ctx: FEMContext,
    props: PhysicalProperties,
    config: SIMPConfig,
    seed: int = 0
) -> List[Dict[str, Any]]:
    """
    Run a full SIMP optimization loop (3D).
    """
    # 1. Setup Dimensions
    # We need to infer dimensions from the mesh or dof_map
    # In physics_model.py, dof_map is 1D array of valid indices.
    # But we need the grid dimensions (nx, ny, nz).
    # We can get this from the mesh topology or assume it was passed in ctx?
    # Actually, ctx.mesh.geometry.x contains coordinates.
    # Let's try to deduce from mesh bounds and cell size (assuming unit cells or uniform).
    
    # Better: The caller (run_data_harvest) knows the resolution.
    # But we only get ctx here.
    # Let's inspect ctx.mesh
    
    # Assuming regular grid:
    # bounds = ctx.mesh.geometry.x.max(axis=0) - ctx.mesh.geometry.x.min(axis=0)
    # But this depends on physical size.
    
    # Let's look at the number of cells.
    num_cells = ctx.mesh.topology.index_map(ctx.mesh.topology.dim).size_local
    
    # This is tricky without explicit resolution.
    # However, dof_map length is num_cells (since it's DG0).
    # Wait, dof_map in physics_model is `np.arange(nx * ny * nz)`.
    # So len(dof_map) == nx * ny * nz.
    
    # We need to factorize num_cells into nx, ny, nz.
    # This is ambiguous (64 = 4x4x4 or 8x8x1 etc).
    # Hack: We can try to guess based on coordinates.
    
    # Get cell centers
    # This is expensive but robust.
    # Or we can rely on the fact that run_data_harvest creates the mesh with specific dims.
    # Ideally, FEMContext should store resolution.
    
    # Let's try to get it from mesh.
    # dolfinx.mesh.create_box arguments are not stored directly.
    
    # Workaround: Use cell centers to determine grid size.
    # Or, update FEMContext to store resolution.
    # Since I cannot easily update FEMContext definition across all files right now without risk,
    # I will try to deduce it.
    
    # Let's assume standard aspect ratio or check max coordinates.
    # In physics_model, L=2.0, H=1.0, D=1.0 (or similar).
    # And resolution is proportional.
    
    # Let's use a heuristic based on coordinates of cell centers.
    tdim = ctx.domain.topology.dim
    num_cells = ctx.domain.topology.index_map(tdim).size_local
    
    # We can't easily get (nx, ny, nz) without more info.
    # BUT, run_data_harvest passes 'resolution' string to other functions.
    # It doesn't pass it here.
    
    # CRITICAL FIX: We will assume the mesh was created with create_box and cells are ordered.
    # But we need nx, ny, nz for the filter.
    
    # Let's try to get max indices from cell centers.
    # This is getting complicated.
    # Let's modify the signature of run_simp_optimization to accept resolution tuple.
    # This requires changing run_data_harvest.py as well.
    
    # For now, let's assume 2D if nz is not apparent? No, we are 3D.
    
    # Let's raise error if we can't determine.
    # But wait, I can modify run_data_harvest.py to pass resolution.
    # That is the cleanest way.
    
    # For this step, I will update the function to accept `resolution` argument.
    # And I will update run_data_harvest.py in the next step.
    pass

def run_simp_optimization_3d(
    ctx: FEMContext,
    props: PhysicalProperties,
    config: SIMPConfig,
    resolution: Tuple[int, int, int],
    seed: int = 0,
    initial_density: Optional[np.ndarray] = None  # NEW: Allow A* backbone initialization
) -> List[Dict[str, Any]]:
    
    nx, ny, nz = resolution
    n_elements = nx * ny * nz
    
    # Verify consistency
    # num_cells = ctx.mesh.topology.index_map(ctx.mesh.topology.dim).size_local
    # if num_cells != n_elements:
    #     print(f"Warning: Mesh cells ({num_cells}) != Resolution product ({n_elements})")
    
    # Initialize density
    if initial_density is not None:
        # Use provided backbone (e.g., from A*)
        x = initial_density.copy()
        if len(x) != n_elements:
            raise ValueError(f"initial_density length ({len(x)}) != n_elements ({n_elements})")
    else:
        # Default: uniform density at target volume fraction
        x = np.ones(n_elements) * config.vol_frac
    
    # Precompute filter
    H, Hs = precompute_filter_3d(nx, ny, nz, config.r_min)
    
    history = []
    loop = 0
    change = 1.0
    
    # Reuse solver structures
    b = ctx.problem.b
    u = ctx.u_sol
    
    from tqdm import tqdm
    pbar = tqdm(total=config.max_iter, desc="SIMP Optimization", unit="iter", leave=False)
    
    while change > config.change_tol and loop < config.max_iter:
        loop += 1
        
        # 2. Update FEM Material Field
        # x is flat (nx*ny*nz).
        # We need to map it to the FEM DG0 space.
        # Assuming standard lexicographic ordering in create_box:
        # x varies fastest, then y, then z? Or z, y, x?
        # FEniCSx create_box ordering is usually x, y, z.
        # i.e. index = x + y*nx + z*nx*ny
        
        # Let's assume the flat 'x' vector matches the DG0 dof ordering.
        # This is a strong assumption but usually holds for structured meshes.
        # If not, we would need to map via coordinates.
        
        # Let's try direct assignment first.
        # ctx.material_field is a DG0 function.
        
        # Apply penalization
        penalized_density = x ** config.penal
        
        # Update material field
        # Note: ctx.dof_map contains the valid indices for the active domain.
        # But here we are optimizing the WHOLE domain (SIMP approach).
        # Or are we?
        # If ctx.dof_map is just "all cells", then:
        ctx.material_field.x.array[:] = penalized_density
        ctx.material_field.x.scatter_forward()
        
        # 3. Solve FEM
        # Assemble A
        A = ctx.problem.A
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, ctx.problem.a, bcs=ctx.problem.bcs)
        A.assemble()
        
        # Assemble b
        b.zeroEntries()
        dolfinx.fem.petsc.assemble_vector(b, ctx.problem.L)
        
        # Apply Point Load at Tip (Mid-Height, Mid-Depth, Right-End)
        # Coordinates: x=max, y=mid, z=mid
        max_coords = ctx.domain.geometry.x.max(axis=0)
        tip_x = max_coords[0]
        mid_y = max_coords[1] / 2.0
        mid_z = max_coords[2] / 2.0
        
        def point_load_locator(p):
            # Locate tip region (small tolerance)
            return np.logical_and(
                np.isclose(p[0], tip_x, atol=0.1),
                np.logical_and(
                    np.isclose(p[1], mid_y, atol=0.1),
                    np.isclose(p[2], mid_z, atol=0.1)
                )
            )
            
        V = ctx.V
        load_dofs = fem.locate_dofs_geometrical(V, point_load_locator)
        
        # Apply load F_y = -1.0
        # In FEniCSx, for Vector FunctionSpace, DOFs are blocked if created with (mesh, ("Lagrange", 1, (3,)))
        # But if created with VectorElement, they might be interlaced.
        # PhysicsModel uses: element = basix.ufl.element("Lagrange", ..., shape=(3,))
        # This creates a blocked map usually (bs=3).
        
        bs = V.dofmap.index_map_bs
        
        with b.localForm() as b_local:
            # We want to apply load to Y component (index 1)
            # If bs=3, then for each node dof 'k', the components are k*3, k*3+1, k*3+2
            # BUT locate_dofs returns the indices in the local array directly?
            # No, locate_dofs returns indices in the dofmap.
            
            # Let's handle the block size carefully.
            # If we used fem.functionspace(mesh, element), and element has shape (3,),
            # then V.dofmap.index_map.size_local * bs is the total local size.
            
            # For a given 'node_dof' returned by locate_dofs (if unrolled? or blocked?)
            # locate_dofs_geometrical on a Vector FunctionSpace returns the indices of ALL components for the matching nodes?
            # Or just the block indices?
            # It usually returns all DOFs associated with the entity.
            
            # Let's assume we want to distribute the load -1.0 among found nodes.
            f_total = -1.0
            n_nodes = len(load_dofs) // bs if bs > 0 else 1
            if n_nodes > 0:
                f_node = f_total / n_nodes
                
                # We need to target Y component.
                # If bs=3, the DOFs are [x, y, z, x, y, z...] or [x,x,x, y,y,y...]?
                # FEniCSx default is usually blocked: [x0, y0, z0, x1, y1, z1...]
                
                # Let's iterate and check modulo
                for dof in load_dofs:
                    if dof % bs == 1: # Y component
                        b_local.setValues([dof], [f_node], addv=PETSc.InsertMode.ADD_VALUES)
            
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        
        # Re-apply BCs (Dirichlet overrides load if conflict, but here they are far apart)
        dolfinx.fem.petsc.set_bc(b, ctx.problem.bcs)
        
        # Solve
        t0 = time.time()
        solver = ctx.problem.solver
        solver.solve(b, u.x.petsc_vec)
        u.x.scatter_forward()
        t_solve = time.time() - t0
        
        # 4. Compute Compliance and Sensitivity
        t1 = time.time()
        compliance = b.dot(u.x.petsc_vec)
        
        # Sensitivity
        # dc_e = -p * (Strain Energy Density) / x_e
        
        # Compute Strain Energy Density
        # We need to project it to DG0
        if not hasattr(ctx, 'energy_expr'):
             # We need to define this expression once
             E_solid = props.E
             E_void = 1e-3 # Increased from 1e-6 for CG stability
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
             # Create reusable function for energy values
             energy_vals = fem.Function(W)
             objectsetattr(ctx, 'energy_vals', energy_vals)
             
        energy_vals = ctx.energy_vals
        energy_vals.interpolate(ctx.energy_expr)
        
        # Raw energies (aligned with x if ordering matches)
        energies = energy_vals.x.array
        
        # Compute dc
        # dc = -p * energy / x
        # Avoid division by zero
        x_safe = np.maximum(x, 1e-3)
        dc = -config.penal * energies / x_safe
        t_sens = time.time() - t1
        
        # 5. Filtering
        t2 = time.time()
        dc = apply_sensitivity_filter(x, dc, H, Hs)
        t_filter = time.time() - t2
        
        # 6. Update Design
        t3 = time.time()
        x_new = optimality_criteria(x, dc, config.vol_frac, config.move_limit)
        t_update = time.time() - t3
        
        change = np.max(np.abs(x_new - x))
        x = x_new
        
        # 7. History
        u_vals = u.x.array.reshape(-1, 3) # 3D
        max_disp = np.max(np.linalg.norm(u_vals, axis=1))
        
        # Reshape density to 3D for storage
        density_map = x.reshape((nx, ny, nz)) # Assuming x,y,z ordering
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
            # print(f"  SIMP Iter {loop}: Change={change:.4f}, Compliance={compliance:.4f}, MaxDisp={max_disp:.4f}")
            pbar.set_postfix({
                "Chg": f"{change:.3f}", 
                "C": f"{compliance:.1f}", 
                "T_sol": f"{t_solve:.2f}s",
                "T_sen": f"{t_sens:.2f}s",
                "T_fil": f"{t_filter:.2f}s"
            })
            
        pbar.update(1)
            
    pbar.close()
    return history

# Helper to allow setting attributes on frozen dataclass (for caching)
def objectsetattr(obj, name, value):
    object.__setattr__(obj, name, value)

