"""
MBB Beam Validation using FEniTop Core
"""
import os
import numpy as np
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt
from dolfinx.mesh import create_box, CellType
import ufl
import json

# Import FEniTop core
from alphabuilder.src.logic.fenitop.topopt import topopt

def run_mbb_test():
    comm = MPI.COMM_WORLD
    
    # 1. Define Mesh (Half MBB)
    # Resolution: 90x30x6 (matches previous test)
    # Physical Size: 3.0 x 1.0 x 0.2 (matches previous test)
    L, H, W = 3.0, 1.0, 0.2
    nx, ny, nz = 90, 30, 6
    
    # Element sizes for tolerance
    dx = L / nx
    dy = H / ny
    dz = W / nz
    
    mesh = create_box(comm, [[0, 0, 0], [L, H, W]], [nx, ny, nz], CellType.hexahedron)
    
    if comm.rank == 0:
        mesh_serial = create_box(MPI.COMM_SELF, [[0, 0, 0], [L, H, W]], [nx, ny, nz], CellType.hexahedron)
    else:
        mesh_serial = None

    # 2. Define Boundary Conditions for Half-MBB
    # Standard MBB Half:
    # - Left (X=0): Symmetry (u_x = 0 on entire face)
    # - Bottom Right corner region: Roller support (u_y = 0, u_z = 0)
    # - Load: Top Left (X=0, Y=H): Force down (F_y = -1)
    
    # 3. Define Boundary Conditions using new 'bcs' list format
    # Format: (value, location_func, subspace_index)
    
    def symmetry_bc(x):
        """Symmetry plane at X=0 - entire left face"""
        return np.isclose(x[0], 0.0)
        
    def support_bc(x):
        """Roller support at bottom right corner region.
        Using a small region instead of exact line for robust BC application.
        """
        # Support region: X near L, Y near 0 (small strip at bottom right)
        return (x[0] > L - 2*dx) & (x[1] < 2*dy)
        
    # Load Area: Top Left Corner (distributed over a small patch for stability)
    def load_area(x):
        return np.isclose(x[1], H) & (x[0] < 3*dx)

    fem = {
        "mesh": mesh,
        "mesh_serial": mesh_serial,
        "young's modulus": 1.0,
        "poisson's ratio": 0.3,
        "bcs": [
            (0.0, symmetry_bc, 0), # Fix ux=0 at X=0 (Symmetry)
            (0.0, support_bc, 1),  # Fix uy=0 at support (Roller Support)
            (0.0, support_bc, 2)   # Fix uz=0 at support (Prevent Rigid Body Z-motion)
        ],
        "traction_bcs": [
            [(0, -1.0, 0), load_area]
        ],
        "body_force": (0, 0, 0),
        "quadrature_degree": 2,
        "petsc_options": {
            "ksp_type": "cg",
            "pc_type": "gamg",
            "ksp_rtol": 1e-6
        },
    }
    
    # Define solid zones to ensure connectivity
    # Solid zone near load application and support region
    def solid_zone_func(x):
        """Force material near load (top-left) and support (bottom-right)"""
        # Near load application point (top-left corner)
        near_load = (x[0] < 3*dx) & (x[1] > H - 3*dy)
        # Near support region (bottom-right corner) 
        near_support = (x[0] > L - 3*dx) & (x[1] < 3*dy)
        return near_load | near_support
    
    opt = {
        "max_iter": 150, # More iterations for convergence
        "opt_tol": 1e-4,
        "vol_frac": 0.4,
        "solid_zone": solid_zone_func,  # Force material at critical regions
        "void_zone": lambda x: np.full(x.shape[1], False),
        "penalty": 3.0, # Standard SIMP penalty
        "epsilon": 1e-6, # Standard void stiffness
        "filter_radius": 0.1, # ~3 element radii for good filtering
        "beta_interval": 50, # Heaviside continuation
        "beta_max": 128,
        "use_oc": True, # Use OC optimizer
        "move": 0.02, # Conservative move limit
        "opt_compliance": True,
    }
    
    # Pre-calculate DOF to Grid mapping on Rank 0
    grid_mapper = None
    if comm.rank == 0:
        import basix.ufl
        from dolfinx.fem import functionspace
        
        # Create CG1 space on serial mesh to match rho_phys_field
        element = basix.ufl.element("Lagrange", mesh_serial.topology.cell_name(), 1)
        V_serial = functionspace(mesh_serial, element)
        
        # Get coordinates of all DOFs
        # coords shape: (num_dofs, 3)
        coords = V_serial.tabulate_dof_coordinates()
        
        # Map coordinates to grid indices
        # We assume the mesh is regular box, using dx, dy, dz defined earlier
        
        # Use simple rounding to get integer indices
        # Note: coords might have small floating point errors
        x_idx = np.rint(coords[:, 0] / dx).astype(int)
        y_idx = np.rint(coords[:, 1] / dy).astype(int)
        z_idx = np.rint(coords[:, 2] / dz).astype(int)
        
        # Clip to ensure bounds (just in case)
        x_idx = np.clip(x_idx, 0, nx)
        y_idx = np.clip(y_idx, 0, ny)
        z_idx = np.clip(z_idx, 0, nz)
        
        grid_mapper = (x_idx, y_idx, z_idx)
        
        print(f"Initialized Grid Mapper. Total DOFs: {len(coords)}")

    history = []
    def record_step(data):
        # Data is on rank 0
        if comm.rank == 0:
            rho_flat = data['density']
            
            # Use coordinate mapping to reconstruct 3D grid
            # We create a nodal grid (nx+1, ny+1, nz+1) first
            rho_nodal = np.zeros((nx+1, ny+1, nz+1))
            
            # Assign values
            # rho_flat corresponds to V_serial DOFs
            if len(rho_flat) != len(grid_mapper[0]):
                 print(f"WARNING: Size mismatch in record_step. Flat: {len(rho_flat)}, Mapper: {len(grid_mapper[0])}")
                 # Fallback to direct reshape if sizes don't match (unlikely if setup is correct)
                 rho_3d = rho_flat # Keep flat to avoid crash, but visualization will fail
            else:
                rho_nodal[grid_mapper[0], grid_mapper[1], grid_mapper[2]] = rho_flat
                
                # Downsample to Voxel Grid (nx, ny, nz)
                # Take average of corners or just one corner?
                # Slicing is simplest: rho_voxel = rho_nodal[:-1, :-1, :-1]
                # This corresponds to the value at the bottom-left-front node of the voxel.
                rho_3d = rho_nodal[:-1, :-1, :-1]
            
            # Update data with 3D density
            data['density_map'] = rho_3d
            
            history.append(data)
            print(f"Step {data['iter']}: C={data['compliance']:.4f} V={data['vol_frac']:.4f}")

    print("Starting MBB Optimization with FEniTop Core...")
    topopt(fem, opt, callback=record_step)
    
    # Define load config for export (matches BCs)
    # Load at Top Left: x=0, y=H (index ny)
    load_config = {
        "x": 0,
        "y": ny, # Top face index
        "z_start": 0,
        "z_end": nz
    }
    
    if comm.rank == 0:
        print(f"Optimization finished. Steps: {len(history)}")
        
        final_rho = history[-1]['density_map']
        
        # ASCII Visualization
        print_ascii_topology(final_rho)
        
        # Plotting
        plot_results(history, final_rho)
        
        # Export
        export_to_mock_json(history, load_config)

def export_to_mock_json(history, load_config):
    output_path = "alphabuilder/web/public/data/mock_episode_mbb.json"
    print(f"Exporting mock data to {output_path}...")
    
    frames = []
    for step_data in history:
        rho = step_data['density_map']
        
        # Mock Policy (Empty for SIMP)
        policy = {
            "add": np.zeros_like(rho).tolist(),
            "remove": np.zeros_like(rho).tolist()
        }
        
        frame = {
            "step": step_data['iter'],
            "phase": "SIMP",
            "density": rho.tolist(),
            "fitness": 1.0 / (step_data['compliance'] + 1e-6), # Mock fitness
            "compliance": step_data['compliance'],
            "vol_frac": step_data['vol_frac'],
            "policy": policy
        }
        frames.append(frame)
        
    data = {
        "episode_id": "mbb_fenitop",
        "load_config": load_config,
        "frames": frames
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f)
    print(f"Exported to {output_path}")

def print_ascii_topology(rho, slice_idx=None):
    if slice_idx is None:
        slice_idx = rho.shape[2] // 2
    
    print(f"\nASCII Topology (Mid-plane z={slice_idx}):")
    # rho is (nx, ny, nz). We want to print y rows (top to bottom) and x columns (left to right).
    
    nx, ny, nz = rho.shape
    try:
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
    except Exception as e:
        print(f"Could not print ASCII topology: {e}")

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
    try:
        plt.hist(final_rho.flatten(), bins=50, color='gray')
        plt.title("Density Histogram")
        plt.xlabel("Density")
        plt.ylabel("Count")
        plt.grid(True)
    except:
        pass
    
    plt.tight_layout()
    plt.savefig("mbb_fenitop_metrics.png")
    print("Metrics plot saved to mbb_fenitop_metrics.png")
    
    # 3. Slice Visualization (Mid-plane)
    try:
        mid_z = final_rho.shape[2] // 2
        plt.figure(figsize=(10, 4))
        plt.imshow(final_rho[:, :, mid_z].T, cmap='gray_r', origin='lower')
        plt.title(f"Mid-plane Slice (z={mid_z})")
        plt.colorbar(label='Density')
        plt.savefig("mbb_fenitop_topology.png")
        print("Topology plot saved to mbb_fenitop_topology.png")
    except:
        pass

if __name__ == "__main__":
    run_mbb_test()
