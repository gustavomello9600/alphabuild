"""
Authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Chao Wang (chaow4@illinois.edu)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Sponsors:
- U.S. National Science Foundation (NSF) EAGER Award CMMI-2127134
- U.S. Defense Advanced Research Projects Agency (DARPA) Young Faculty Award
  (N660012314013)
- NSF CAREER Award CMMI-2047692
- NSF Award CMMI-2245251

Reference:
- Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation
  for 2D and 3D topology optimization supporting parallel computing.
  Struct Multidisc Optim 67, 140 (2024).
  https://doi.org/10.1007/s00158-024-03818-7
"""

import time

import numpy as np
from mpi4py import MPI

from .fem import form_fem
from .parameterize import DensityFilter, Heaviside
from .sensitivity import Sensitivity
from .optimize import optimality_criteria, mma_optimizer
from .utility import Communicator, Plotter, save_xdmf


def topopt(fem, opt, initial_density=None, callback=None):
    """Main function for topology optimization."""

    # Initialization
    comm = MPI.COMM_WORLD
    linear_problem, u_field, lambda_field, rho_field, rho_phys_field = form_fem(fem, opt)
    density_filter = DensityFilter(comm, rho_field, rho_phys_field,
                                   opt["filter_radius"], fem["petsc_options"])
    heaviside = Heaviside(rho_phys_field)
    sens_problem = Sensitivity(comm, opt, linear_problem, u_field, lambda_field, rho_phys_field)
    S_comm = Communicator(rho_phys_field.function_space, fem["mesh_serial"])
    # if comm.rank == 0:
    #     plotter = Plotter(fem["mesh_serial"])
    num_consts = 1 if opt["opt_compliance"] else 2
    # Use local size from PETSc vector (excludes ghost DOFs)
    num_elems = rho_field.x.petsc_vec.getLocalSize()
    if not opt["use_oc"]:
        rho_old1, rho_old2 = np.zeros(num_elems), np.zeros(num_elems)
        low, upp = None, None

    # Apply passive zones
    # centers shape: (3, num_elems) - coordinates of DG0 DOF centers (element centroids)
    centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems].T
    solid, void = opt["solid_zone"](centers), opt["void_zone"](centers)
    
    # Initialization Strategy
    if initial_density is not None:
        initial_density = np.asarray(initial_density)
        
        # Check if initial_density is a 3D voxel grid that needs mapping
        if initial_density.ndim == 3:
            # Map 3D voxel grid to 1D DG0 array using element centroid coordinates
            # centers[0,:] = x coords, centers[1,:] = y coords, centers[2,:] = z coords
            nx, ny, nz = initial_density.shape
            
            # Get voxel indices from centroid coordinates
            # Element centroids are at (i+0.5, j+0.5, k+0.5) for voxel (i,j,k)
            # So we use floor to get the voxel index
            x_idx = np.floor(centers[0, :]).astype(int)
            y_idx = np.floor(centers[1, :]).astype(int)
            z_idx = np.floor(centers[2, :]).astype(int)
            
            # Clip to valid range
            x_idx = np.clip(x_idx, 0, nx - 1)
            y_idx = np.clip(y_idx, 0, ny - 1)
            z_idx = np.clip(z_idx, 0, nz - 1)
            
            # Sample initial density at element centroids
            rho_ini = initial_density[x_idx, y_idx, z_idx].astype(np.float64)
        else:
            # Already 1D, use directly
            rho_ini = initial_density.flatten().astype(np.float64)
            
            if rho_ini.size != num_elems:
                raise ValueError(f"Initial density size {rho_ini.size} does not match num_elems {num_elems}")
    else:
        # Default uniform initialization
        rho_ini = np.full(num_elems, opt["vol_frac"])
        
    rho_ini[solid], rho_ini[void] = 0.995, 0.005
    rho_field.x.petsc_vec.array[:] = rho_ini
    rho_min, rho_max = np.zeros(num_elems), np.ones(num_elems)
    rho_min[solid], rho_max[void] = 0.99, 0.01

    # Start topology optimization
    opt_iter, beta, change = 0, 1, 2*opt["opt_tol"]
    
    # History container
    history = []
    
    while opt_iter < opt["max_iter"] and change > opt["opt_tol"]:
        opt_start_time = time.perf_counter()
        opt_iter += 1

        # Density filter and Heaviside projection
        density_filter.forward()
        if opt_iter % opt["beta_interval"] == 0 and beta < opt["beta_max"]:
            beta *= 2
            change = opt["opt_tol"] * 2
        heaviside.forward(beta)

        # Solve FEM
        linear_problem.solve_fem()

        # Compute function values and sensitivities
        [C_value, V_value, U_value], sensitivities = sens_problem.evaluate()
        heaviside.backward(sensitivities)
        [dCdrho, dVdrho, dUdrho] = density_filter.backward(sensitivities)
        if opt["opt_compliance"]:
            g_vec = np.array([V_value-opt["vol_frac"]])
            dJdrho, dgdrho = dCdrho, np.vstack([dVdrho])
        else:
            g_vec = np.array([V_value-opt["vol_frac"], C_value-opt["compliance_bound"]])
            dJdrho, dgdrho = dUdrho, np.vstack([dVdrho, dCdrho])

        # Update the design variables
        rho_values = rho_field.x.petsc_vec.array.copy()
        if opt["opt_compliance"] and opt["use_oc"]:
            rho_new, change = optimality_criteria(
                rho_values, rho_min, rho_max, g_vec, dJdrho, dgdrho[0], opt["move"])
        else:
            rho_new, change, low, upp = mma_optimizer(
                num_consts, num_elems, opt_iter, rho_values, rho_min, rho_max,
                rho_old1, rho_old2, dJdrho, g_vec, dgdrho, low, upp, opt["move"])
            rho_old2 = rho_old1.copy()
            rho_old1 = rho_values.copy()
        rho_field.x.petsc_vec.array[:] = rho_new.copy()

        # Output the histories
        opt_time = time.perf_counter() - opt_start_time
        if comm.rank == 0:
            print(f"opt_iter: {opt_iter}, opt_time: {opt_time:.3g} (s), "
                  f"beta: {beta}, C: {C_value:.3f}, V: {V_value:.3f}, "
                  f"U: {U_value:.3f}, change: {change:.3f}", flush=True)
            
            # Callback / History Recording
        # NOTE: gather is a collective MPI operation - ALL processes must call it
            if callback:
            # Gather physical density for saving (collective operation)
                rho_phys_global = S_comm.gather(rho_phys_field)
                
            # Only rank 0 executes the callback
            if comm.rank == 0 and rho_phys_global is not None:
                    step_data = {
                        "iter": opt_iter,
                        "compliance": C_value,
                        "vol_frac": V_value,
                        "change": change,
                    "density": rho_phys_global,
                        "beta": beta
                    }
                    callback(step_data)

    # Final Save/Plot
    values = S_comm.gather(rho_phys_field)
    # if comm.rank == 0:
    #     plotter.plot(values)
    # save_xdmf(fem["mesh"], rho_phys_field) # Disable XDMF for speed
    
    return history
