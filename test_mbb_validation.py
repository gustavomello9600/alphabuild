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
    
    output_path = "alphabuilder/web/public/data/mock_episode_mbb.json"
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
