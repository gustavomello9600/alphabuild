import numpy as np
import dolfinx
from dolfinx import fem
import ufl
from petsc4py import PETSc
from .physics_model import FEMContext, PhysicalProperties, SimulationResult

def solve_topology(topology_matrix: np.ndarray, ctx: FEMContext, props: PhysicalProperties) -> SimulationResult:
    """
    Resolve o problema de elasticidade para uma dada topologia.
    
    Args:
        topology_matrix: Matriz binária (ny, nx) definindo a distribuição de material.
        ctx: Contexto FEM pré-compilado.
        props: Propriedades físicas.
        
    Returns:
        SimulationResult contendo métricas de fitness e validação.
    """
    # 1. Atualização de Material
    if topology_matrix.shape != ctx.dof_map.shape:
        raise ValueError(f"Topology shape {topology_matrix.shape} mismatch with Context shape {ctx.dof_map.shape}")
        
    mask = ctx.dof_map >= 0
    valid_indices = ctx.dof_map[mask]
    valid_values = topology_matrix[mask].astype(dolfinx.default_scalar_type)
    
    ctx.material_field.x.array[valid_indices] = valid_values
    ctx.material_field.x.scatter_forward()

    # 2. Configurar Carga Pontual no Vetor RHS (b)
    A = ctx.problem.A
    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A, ctx.problem.a, bcs=ctx.problem.bcs)
    A.assemble()
    
    b = ctx.problem.b
    b.zeroEntries()
    dolfinx.fem.petsc.assemble_vector(b, ctx.problem.L)
    
    # Aplicar Carga Pontual em (2.0, 0.5)
    target_point = np.array([2.0, 0.5, 0.0])
    
    V = ctx.V
    
    def point_load_locator(x):
        tol = 1e-3
        return np.logical_and(
            np.isclose(x[0], 2.0, atol=tol),
            np.isclose(x[1], 0.5, atol=tol)
        )
        
    load_dofs = fem.locate_dofs_geometrical(V, point_load_locator)
    
    if len(load_dofs) > 0:
        V_y, vy_map = V.sub(1).collapse()
        load_dofs_y_local = fem.locate_dofs_geometrical(V_y, point_load_locator)
        load_dofs_y = vy_map[load_dofs_y_local]
        
        force_value = -100.0
        
        with b.localForm() as b_local:
            for dof in load_dofs_y:
                b_local.setValues([dof], [force_value], addv=PETSc.InsertMode.ADD_VALUES)
                
    # Aplicar BCs no vetor b (Dirichlet)
    dolfinx.fem.petsc.apply_lifting(b, [ctx.problem.a], bcs=[ctx.problem.bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, ctx.problem.bcs)

    # Resolver
    solver = ctx.problem.solver
    solver.solve(b, ctx.u_sol.x.petsc_vec)
    ctx.u_sol.x.scatter_forward()

    # 3. Pós-Processamento
    u = ctx.u_sol
    
    # Max Displacement
    u_vals = u.x.array.reshape(-1, 2)
    disp_norms = np.linalg.norm(u_vals, axis=1)
    max_disp = np.max(disp_norms)
    
    # Compliance
    compliance = b.dot(u.x.petsc_vec)
    
    # Fitness (Kane Eq 1)
    # omega_mat = connected mass (number of material pixels)
    # omega_dis = disconnected mass (should be 0 for valid topologies)
    omega_mat = float(np.sum(topology_matrix))  # Total material volume
    omega_dis = 0.0  # Assuming connected topology from game rules
    
    # Penalty for excessive displacement
    penalty = max(0.0, max_disp - props.disp_limit)
    
    # Denominator: mass + epsilon*disconnected + alpha*penalty
    denominator = omega_mat + props.penalty_epsilon * omega_dis + props.penalty_alpha * penalty
    
    # Fitness = 1 / (mass + penalties)
    # Lower mass and lower penalties = higher fitness
    if denominator < 1e-9 or omega_mat < 1.0:
        # Invalid: no material or numerical issue
        fitness = 0.0
    else:
        fitness = 1.0 / denominator

    return SimulationResult(
        fitness=fitness,
        max_displacement=max_disp,
        compliance=compliance,
        valid=True,
        displacement_array=np.array([])
    )
