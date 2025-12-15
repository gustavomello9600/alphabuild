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

import numpy as np
import ufl
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.fem import (functionspace, Function, Constant,
                         dirichletbc, locate_dofs_topological)
import basix.ufl

from .utility import create_mechanism_vectors
from .utility import LinearProblem


def form_fem(fem, opt):
    """Form an FEA problem."""
    # Function spaces and functions
    mesh = fem["mesh"]
    
    # Vector Element for Displacement (CG1)
    # V = VectorFunctionSpace(mesh, ("CG", 1)) -> Deprecated
    v_element = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim,))
    V = functionspace(mesh, v_element)
    
    # Scalar Element for Density (DG0)
    # S0 = FunctionSpace(mesh, ("DG", 0)) -> Deprecated
    s0_element = basix.ufl.element("DG", mesh.topology.cell_name(), 0)
    S0 = functionspace(mesh, s0_element)
    
    # Scalar Element for Physical Density (CG1) - for filtering
    # S = FunctionSpace(mesh, ("CG", 1)) -> Deprecated
    s_element = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1)
    S = functionspace(mesh, s_element)
    
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    u_field = Function(V)  # Displacement field
    lambda_field = Function(V)  # Adjoint variable field
    rho_field = Function(S0)  # Density field
    rho_phys_field = Function(S)  # Physical density field

    # Material interpolation
    E0, nu = fem["young's modulus"], fem["poisson's ratio"]
    # Use Constant for penalty to allow updates (continuation)
    p_const = Constant(mesh, float(opt["penalty"]))
    opt["penalty_const"] = p_const # Store for access in callback
    eps = opt["epsilon"]
    E = (eps + (1-eps)*rho_phys_field**p_const) * E0
    _lambda, mu = E*nu/(1+nu)/(1-2*nu), E/(2*(1+nu))  # Lame constants

    # Kinematics
    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):  # 3D or plane strain
        return 2*mu*epsilon(u) + _lambda*ufl.tr(epsilon(u))*ufl.Identity(len(u))

    # Boundary conditions
    dim = mesh.topology.dim
    fdim = dim - 1
    
    bcs = []
    # Check if 'bcs' key exists (New flexible format)
    if "bcs" in fem:
        for value, location_func, subspace_idx in fem["bcs"]:
            # Locate facets
            facets = locate_entities_boundary(mesh, fdim, location_func)
            dofs = locate_dofs_topological(V, fdim, facets)
            
            # Apply BC
            if subspace_idx is None:
                # Apply to all components
                bc_val = Constant(mesh, np.array(value, dtype=float))
                bcs.append(dirichletbc(bc_val, dofs, V))
            else:
                # Apply to specific component (subspace)
                # We need to collapse the subspace to get the correct V.sub(i)
                V_sub, _ = V.sub(subspace_idx).collapse()
                dofs_sub = locate_dofs_topological((V.sub(subspace_idx), V_sub), fdim, facets)
                
                # locate_dofs_topological returns a list [dofs_in_V] when passed a tuple
                dofs_array = dofs_sub[0]
                
                # Handle value: if it's a scalar, use it directly. If it's a vector/list, pick component.
                if np.isscalar(value) or isinstance(value, float) or isinstance(value, int):
                    scalar_val = value
                else:
                    scalar_val = value[subspace_idx]
                    
                bc_const = Constant(mesh, float(scalar_val)) # Ensure float type
                bcs.append(dirichletbc(bc_const, dofs_array, V.sub(subspace_idx)))

    # Fallback to old 'disp_bc' if 'bcs' not present (Legacy FEniTop support)
    elif "disp_bc" in fem:
        disp_facets = locate_entities_boundary(mesh, fdim, fem["disp_bc"])
        bcs.append(dirichletbc(Constant(mesh, np.full(dim, 0.0)),
                         locate_dofs_topological(V, fdim, disp_facets), V))
    
    # Rename local var to match what is used later (although we pass list to LinearProblem)
    # bc = ... (removed single bc var)

    tractions, facets, markers = [], [], []
    for marker, (traction, traction_bc) in enumerate(fem["traction_bcs"]):
        tractions.append(Constant(mesh, np.array(traction, dtype=float)))
        current_facets = locate_entities_boundary(mesh, fdim, traction_bc)
        facets.extend(current_facets)
        markers.extend([marker,]*len(current_facets))
    facets = np.array(facets, dtype=np.int32)
    markers = np.array(markers, dtype=np.int32)
    _, unique_indices = np.unique(facets, return_index=True)
    facets, markers = facets[unique_indices], markers[unique_indices]
    sorted_indices = np.argsort(facets)
    facet_tags = meshtags(mesh, fdim, facets[sorted_indices], markers[sorted_indices])

    metadata = {"quadrature_degree": fem["quadrature_degree"]}
    dx = ufl.Measure("dx", metadata=metadata)
    ds = ufl.Measure("ds", domain=mesh, metadata=metadata, subdomain_data=facet_tags)
    if callable(fem["body_force"]):
        b = Function(V)
        b.interpolate(fem["body_force"])
    else:
        b = Constant(mesh, np.array(fem["body_force"], dtype=float))

    # Establish the equilibrium and adjoint equations
    lhs = ufl.inner(sigma(u), epsilon(v))*dx
    rhs = ufl.dot(b, v)*dx
    for marker, t in enumerate(tractions):
        rhs += ufl.dot(t, v)*ds(marker)
    if opt["opt_compliance"]:
        spring_vec = opt["l_vec"] = None
    else:
        spring_vec, opt["l_vec"] = create_mechanism_vectors(
            V, opt["in_spring"], opt["out_spring"])
    linear_problem = LinearProblem(u_field, lambda_field, lhs, rhs, opt["l_vec"],
                                   spring_vec, bcs, fem["petsc_options"])

    # Define optimization-related variables
    opt["f_int"] = ufl.inner(sigma(u_field), epsilon(v))*dx
    opt["compliance"] = ufl.inner(sigma(u_field), epsilon(u_field))*dx
    opt["volume"] = rho_phys_field*dx
    opt["total_volume"] = Constant(mesh, 1.0)*dx

    return linear_problem, u_field, lambda_field, rho_field, rho_phys_field
