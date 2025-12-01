# Relatório Técnico: Validação do Solver SIMP (MBB Beam 3D)

## 1. Visão Geral
Este relatório detalha a implementação atual do solver de Otimização Topológica (SIMP) no projeto AlphaBuilder e o teste de validação realizado utilizando o clássico problema "MBB Beam" em 3D. O objetivo foi validar a integridade do solver FEM, a eficácia do filtro de sensibilidade e a capacidade do algoritmo de convergir para estruturas de treliça otimizadas.

## 2. Metodologia e Implementação

### 2.1. O Problema MBB Beam
O "Messerschmitt-Bölkow-Blohm" (MBB) Beam é um benchmark padrão em otimização topológica. Representa uma viga simplesmente apoiada com carga central. Devido à simetria, modelamos apenas metade da viga.

**Condições de Contorno (3D):**
*   **Simetria (Face Esquerda, x=0):** Deslocamento em X travado ($u_x = 0$).
*   **Apoio (Aresta Inferior Direita, x=L, y=0):** Deslocamento em Y travado ($u_y = 0$).
*   **Estabilidade (Aresta Inferior Direita):** Deslocamento em Z travado ($u_z = 0$) para evitar movimento de corpo rígido.
*   **Carga:** Carga linear distribuída ao longo do eixo Z no topo da face esquerda (x=0, y=H).

### 2.2. Algoritmo SIMP
A implementação utiliza o método *Solid Isotropic Material with Penalization* (SIMP) com as seguintes características:
*   **Filtro de Helmholtz:** PDE-based filter para suavizar o campo de densidades e evitar o problema de "checkerboard".
*   **Projeção de Heaviside:** Para binarizar a solução (0 ou 1) e garantir contornos definidos.
*   **Penalização Adaptativa:** Um esquema de continuação ("continuation method") foi implementado para evitar mínimos locais prematuros:
    *   Iterações 0-20: Penalidade $p=1.0$ (problema convexo).
    *   Iterações 21-40: $p$ aumenta gradualmente até 3.0.
    *   Iterações 41+: A projeção de Heaviside ($\beta$) torna-se mais agressiva.

### 2.3. Parâmetros de Refinamento
Para obter a estrutura de treliça detalhada, utilizamos:
*   **Resolução:** $90 \times 30 \times 6$ elementos.
*   **Fração de Volume:** 40% ($0.4$).
*   **Raio do Filtro ($r_{min}$):** $1.2$ elementos (pequeno para permitir diagonais finas).

---

## 3. Código Fonte Completo

Abaixo estão os arquivos principais utilizados para executar este teste.

### 3.1. Lógica do Solver (`alphabuilder/src/logic/simp_generator.py`)
Este arquivo contém o núcleo do algoritmo SIMP, incluindo o filtro de Helmholtz, a projeção e o loop de otimização.

```python
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
        # So C = b.dot(u.x.petsc_vec)
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
        
    pbar.close()
    return history

def objectsetattr(obj, name, value):
    """Helper to set attribute on frozen dataclass or object"""
    object.__setattr__(obj, name, value)
```

### 3.2. Script de Teste MBB (`test_mbb_validation.py`)
Este script configura o ambiente FEM específico para o MBB Beam, define as condições de contorno e executa a otimização.

```python
import numpy as np
import dolfinx
from dolfinx import mesh, fem
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt
import os
import sys

from alphabuilder.src.core.physics_model import FEMContext, PhysicalProperties
from alphabuilder.src.logic.simp_generator import SIMPConfig, run_simp_optimization_3d

def initialize_mbb_context(resolution=(60, 20, 4), props: PhysicalProperties = None) -> FEMContext:
    """
    Initialize a 3D FEM context for the MBB Beam.
    Resolution: (60, 20, 4)
    BCs:
    - Left Face (x=0): ux = 0 (Symmetry)
    - Bottom Right Edge (x=L, y=0): uy = 0 (Roller)
    - Fixed Z at support to prevent rigid body motion
    """
    nx, ny, nz = resolution
    voxel_size = 1.0
    L = nx * voxel_size
    H = ny * voxel_size
    W = nz * voxel_size
    
    # 1. Create Mesh
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([L, H, W])],
        [nx, ny, nz],
        cell_type=mesh.CellType.hexahedron
    )
    
    # 2. Function Space
    import basix.ufl
    element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(domain.geometry.dim,))
    V = fem.functionspace(domain, element)
    
    # 3. Boundary Conditions
    fdim = domain.topology.dim - 1
    
    # BC 1: Left Face (x=0) -> ux = 0
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    
    left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
    left_dofs = fem.locate_dofs_topological(V.sub(0), fdim, left_facets)
    bc_left = fem.dirichletbc(PETSc.ScalarType(0), left_dofs, V.sub(0))
    
    # BC 2: Bottom Right Edge (x=L, y=0) -> uy = 0
    # Since we are in 3D, "Edge" is a line along Z at x=L, y=0
    def support_boundary(x):
        return np.logical_and(
            np.isclose(x[0], L),
            np.isclose(x[1], 0.0)
        )
    
    # Locate DOFs geometrically for the edge/line
    # Must collapse subspace for geometrical location
    V_y, map_y = V.sub(1).collapse()
    support_dofs_y_local = fem.locate_dofs_geometrical(V_y, support_boundary)
    support_dofs_y = map_y[support_dofs_y_local]
    bc_support_y = fem.dirichletbc(PETSc.ScalarType(0), support_dofs_y, V.sub(1))
    
    # BC 3: Fix Z at support to prevent rigid body motion
    V_z, map_z = V.sub(2).collapse()
    support_dofs_z_local = fem.locate_dofs_geometrical(V_z, support_boundary)
    support_dofs_z = map_z[support_dofs_z_local]
    bc_support_z = fem.dirichletbc(PETSc.ScalarType(0), support_dofs_z, V.sub(2))
    
    bcs = [bc_left, bc_support_y, bc_support_z]
    
    # 4. Variational Problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    V_mat = fem.functionspace(domain, ("DG", 0))
    material_field = fem.Function(V_mat)
    material_field.x.array[:] = 1.0
    
    if props is None:
        props = PhysicalProperties()
        
    E_0 = props.E
    E_min = 1e-6
    rho = material_field
    E = E_min + (E_0 - E_min) * rho
    nu = props.nu
    
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
        
    def sigma(u):
        return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2.0 * mu * epsilon(u)
        
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L_form = ufl.dot(fem.Constant(domain, np.zeros(3, dtype=PETSc.ScalarType)), v) * ufl.dx
    
    solver_options = {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-6,
        "ksp_max_it": 2000
    }
    
    problem = dolfinx.fem.petsc.LinearProblem(a, L_form, bcs=bcs, petsc_options=solver_options, petsc_options_prefix="mbb")
    
    u_sol = fem.Function(V)
    
    # Dummy dof map (not used by new SIMP but required by dataclass)
    dof_map = np.arange(nx * ny * nz, dtype=np.int32)
    
    return FEMContext(
        domain=domain,
        V=V,
        bc=bcs[0], # Pass one BC for type checking, but problem has all
        problem=problem,
        u_sol=u_sol,
        dof_map=dof_map,
        material_field=material_field
    )

def run_mbb_test():
    print("=== Starting MBB Beam Validation Test ===")
    
    # Parameters
    resolution = (90, 30, 6) # Increased resolution for truss details
    vol_frac = 0.4
    r_min = 1.2 # Reduced filter radius for sharper features
    penal = 3.0
    
    # Load Config: Top Left (x=0, y=H)
    # Applying line load across Z
    load_config = {
        'x': 0,
        'y': resolution[1] - 1, # Top
        'z_start': 0,
        'z_end': resolution[2]
    }
    
    config = SIMPConfig(
        vol_frac=vol_frac,
        penal=penal,
        r_min=r_min,
        max_iter=60, # Increased iterations for slower continuation
        change_tol=0.01,
        move_limit=0.2,
        adaptive_penal=True, # Enable delayed continuation
        load_config=load_config
    )
    
    print(f"Config: Res={resolution}, Vol={vol_frac}, r_min={r_min}, p={penal}")
    
    # Initialize Context
    props = PhysicalProperties()
    ctx = initialize_mbb_context(resolution, props)
    print("FEM Context Initialized.")
    
    # Run Optimization
    history = run_simp_optimization_3d(ctx, props, config, resolution)
    print(f"Optimization completed in {len(history)} steps.")
    
    # Analyze Results
    final_step = history[-1]
    final_rho = final_step['density_map']
    compliance = final_step['compliance']
    
    print(f"Final Compliance: {compliance:.4f}")
    
    # Plotting
    plot_results(history, final_rho)
    
    # ASCII Visualization
    print_ascii_topology(final_rho)
    
    # Export to Frontend Mock
    export_to_mock_json(history, config, resolution)
    
    return final_rho

def export_to_mock_json(history, config, resolution):
    import json
    
    output_path = "alphabuilder/web/src/data/mock_episode_mbb.json"
    print(f"Exporting mock data to {output_path}...")
    
    frames = []
    for step_data in history:
        # Density: (nx, ny, nz) -> Frontend expects (z, y, x)? 
        # Let's check mock_episode.json. It seems to be [z][y][x] or [x][y][z].
        # In extract_mock_episode.py: tensor.flatten().tolist()
        # But in the json file viewed: "density": [[[...]]] 3D array.
        # Let's assume [x][y][z] based on how we index in python.
        # Actually, standard JSON serialization of numpy array is row-major (C-style).
        # If shape is (60, 20, 4), tolist() gives 60 lists of 20 lists of 4 items.
        # Let's just use .tolist().
        
        rho = step_data['density_map']
        
        # Mock Policy (Empty for SIMP)
        policy = {
            "add": np.zeros_like(rho).tolist(),
            "remove": np.zeros_like(rho).tolist()
        }
        
        frame = {
            "step": step_data['step'],
            "phase": "SIMP",
            "density": rho.tolist(),
            "fitness": 1.0 / (step_data['compliance'] + 1e-6), # Mock fitness
            "compliance": step_data['compliance'],
            "vol_frac": float(np.mean(rho)),
            "policy": policy
        }
        frames.append(frame)
        
    mock_data = {
        "episode_id": "mbb_validation_test",
        "load_config": config.load_config if config.load_config else {},
        "frames": frames
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(mock_data, f)
    
    print("Mock data exported successfully.")

def print_ascii_topology(rho, slice_idx=None):
    if slice_idx is None:
        slice_idx = rho.shape[2] // 2
    
    print(f"\nASCII Topology (Mid-plane z={slice_idx}):")
    # rho is (nx, ny, nz). We want to print y rows (top to bottom) and x columns (left to right).
    
    nx, ny, nz = rho.shape
    slice_data = rho[:, :, slice_idx]
    
    print("+" + "-" * nx + "+")
    for y in range(ny - 1, -1, -1):
        row_str = "|"
        for x in range(nx):
            val = slice_data[x, y]
            if val > 0.8:
                row_str += "#"
            elif val > 0.2:
                row_str += "."
            else:
                row_str += " "
        row_str += "|"
        print(row_str)
    print("+" + "-" * nx + "+")

def plot_results(history, final_rho):
    # 1. Convergence
    compliance_hist = [step['compliance'] for step in history]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(compliance_hist, marker='o')
    plt.title("Convergence (Compliance)")
    plt.xlabel("Iteration")
    plt.ylabel("Compliance")
    plt.grid(True)
    
    # 2. Histogram
    plt.subplot(1, 2, 2)
    plt.hist(final_rho.flatten(), bins=50, color='gray')
    plt.title("Density Histogram")
    plt.xlabel("Density")
    plt.ylabel("Count")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("mbb_validation_metrics.png")
    print("Metrics plot saved to mbb_validation_metrics.png")
    
    # 3. Slice Visualization (Mid-plane)
    mid_z = final_rho.shape[2] // 2
    plt.figure(figsize=(10, 4))
    plt.imshow(final_rho[:, :, mid_z].T, cmap='gray_r', origin='lower')
    plt.title(f"Mid-plane Slice (z={mid_z})")
    plt.colorbar(label='Density')
    plt.savefig("mbb_topology.png")
    print("Topology plot saved to mbb_topology.png")

if __name__ == "__main__":
    run_mbb_test()
```

### 3.3. Definição do Modelo Físico (`alphabuilder/src/core/physics_model.py`)
Define as estruturas de dados e a inicialização do contexto FEM.

```python
import numpy as np
import dolfinx
from dolfinx import mesh, fem
from mpi4py import MPI
from dataclasses import dataclass
import ufl
from petsc4py import PETSc
import dolfinx.fem.petsc

@dataclass
class PhysicalProperties:
    E: float = 1.0  # Young's Modulus (Normalized)
    nu: float = 0.3 # Poisson's Ratio
    rho: float = 1.0 # Density
    disp_limit: float = 100.0 # Loose limit for initial training
    penalty_epsilon: float = 0.1
    penalty_alpha: float = 10.0

@dataclass
class FEMContext:
    domain: mesh.Mesh
    V: fem.FunctionSpace
    bc: fem.DirichletBC
    problem: dolfinx.fem.petsc.LinearProblem
    u_sol: fem.Function
    dof_map: np.ndarray # Map from Grid Voxel to FEM Cell
    material_field: fem.Function # DG0 field for density

@dataclass
class SimulationResult:
    fitness: float
    max_displacement: float
    compliance: float
    valid: bool
    displacement_array: np.ndarray

def initialize_cantilever_context(resolution=(64, 32, 32), props: PhysicalProperties = None) -> FEMContext:
    """
    Initialize a 3D FEM context for a Cantilever Beam.
    Resolution: (L, H, W) -> (x, y, z)
    Physical Size: Derived from resolution (Cubic Voxels)
    """
    nx, ny, nz = resolution
    
    # Adaptive Physical Domain (Step 1 of Engineering Order)
    # Ensure cubic voxels by deriving physical size from resolution.
    voxel_size = 1.0
    L = nx * voxel_size
    H = ny * voxel_size
    W = nz * voxel_size
    
    # 1. Create 3D Hexahedral Mesh
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([L, H, W])],
        [nx, ny, nz],
        cell_type=mesh.CellType.hexahedron
    )
    
    # 2. Function Space (Vector Element for Displacement)
    import basix.ufl
    element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(domain.geometry.dim,))
    V = fem.functionspace(domain, element)
    
    # 3. Boundary Conditions (Fix Left Face: x=0)
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
        
    fdim = domain.topology.dim - 1
    left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
    
    # Fix all components (0, 1, 2) -> (x, y, z)
    u_zero = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
    bc = fem.dirichletbc(u_zero, fem.locate_dofs_topological(V, fdim, left_facets), V)
    
    # 4. Variational Problem (Linear Elasticity)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Material Field (Density 0 or 1)
    V_mat = fem.functionspace(domain, ("DG", 0))
    material_field = fem.Function(V_mat)
    material_field.x.array[:] = 1.0 # Start full
    
    if props is None:
        props = PhysicalProperties()
        
    E_0 = props.E
    E_min = 1e-6 # Fixed to 1e-6 as per Engineering Order
    rho = material_field
    E = E_min + (E_0 - E_min) * rho 
    nu = props.nu
    
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
        
    def sigma(u):
        return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2.0 * mu * epsilon(u)
        
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(fem.Constant(domain, np.zeros(3, dtype=PETSc.ScalarType)), v) * ufl.dx
    
    # 5. Linear Problem Setup
    solver_options = {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-6,
        "ksp_atol": 1e-10,
        "ksp_max_it": 2000,
    }
    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options=solver_options, petsc_options_prefix="cantilever")
    
    # 6. DOF Map (Voxel -> Cell Index)
    dof_map = np.arange(nx * ny * nz, dtype=np.int32)
    
    u_sol = fem.Function(V)
    
    return FEMContext(
        domain=domain,
        V=V,
        bc=bc,
        problem=problem,
        u_sol=u_sol,
        dof_map=dof_map,
        material_field=material_field
    )
```
