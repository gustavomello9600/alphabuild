import numpy as np
import dolfinx
from dolfinx import mesh, fem
from mpi4py import MPI
from dataclasses import dataclass, field
from typing import Tuple, Any, Optional
import ufl

@dataclass(frozen=True)
class PhysicalProperties:
    """Constantes Físicas e Hiperparâmetros de Penalidade."""
    E_solid: float = 1.0          # Módulo de Young Base (Adimensionalizado)
    E_void: float = 1e-6          # Material "Ar" (suave para evitar singularidade)
    nu: float = 0.3               # Poisson
    penalty_alpha: float = 0.5    # Fator de penalidade (Eq. 1 Kane)
    penalty_epsilon: float = 0.05 # Penalidade secundária
    disp_limit: float = 2.0       # Restrição do projeto

@dataclass(frozen=True)
class FEMContext:
    """
    Objeto container que guarda os objetos compilados do FEniCSx.
    Isso é gerado uma vez e passado repetidamente para a função 'solve'.
    """
    mesh: dolfinx.mesh.Mesh
    V: dolfinx.fem.FunctionSpace        # Espaço de Deslocamento (Contínuo)
    D: dolfinx.fem.FunctionSpace        # Espaço de Material (Descontínuo/DG0)
    u_sol: dolfinx.fem.Function         # Placeholder da solução
    material_field: dolfinx.fem.Function # Coeficiente atualizável
    problem: Any                        # LinearProblem pré-configurado
    dof_map: np.ndarray                 # Mapeamento (Row, Col) -> Index do Material

@dataclass(frozen=True)
class SimulationResult:
    """Output da Simulação."""
    fitness: float
    max_displacement: float
    compliance: float
    valid: bool
    # Opcional: Campo de deslocamento apenas se for necessário plotar, 
    # para economizar memória em treino massivo.
    displacement_array: np.ndarray = field(default_factory=lambda: np.array([]))

def initialize_cantilever_context(resolution: Tuple[int, int], props: PhysicalProperties) -> FEMContext:
    """
    Inicializa o contexto FEM para uma viga em balanço (Cantilever).
    
    Args:
        resolution: Tupla (nx, ny) definindo a resolução da malha.
        props: Propriedades físicas e parâmetros de penalidade.
        
    Returns:
        FEMContext configurado e pronto para uso.
    """
    # 1. Malha
    # Dimensões 2.0 x 1.0
    L, H = 2.0, 1.0
    nx, ny = resolution
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (L, H)),
        n=(nx, ny),
        cell_type=mesh.CellType.quadrilateral,
    )

    # 2. Espaços de Função
    # V: Deslocamento (Vetorial, Lagrange grau 1)
    # Em dolfinx mais recente, VectorFunctionSpace foi removido.
    # Usamos functionspace com shape vetorial.
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))
    # D: Material (Escalar, Descontínuo grau 0 - constante por elemento)
    D = fem.functionspace(msh, ("DG", 0))

    # 3. Condições de Contorno (BCs)
    # Fixar lado esquerdo (x = 0)
    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    fdim = msh.topology.dim - 1
    left_facets = mesh.locate_entities_boundary(msh, fdim, left_boundary)
    
    # u = 0 na fronteira esquerda
    u_zero = np.array([0.0, 0.0], dtype=dolfinx.default_scalar_type)
    bc = fem.dirichletbc(u_zero, fem.locate_dofs_topological(V, fdim, left_facets), V)

    # 4. Forma Variacional
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Campo de material (densidade)
    material_field = fem.Function(D)
    # Inicializa com 1.0 (sólido) por padrão
    material_field.x.array[:] = 1.0 

    # Interpolação do Módulo de Young: E(rho) = E_void + (E_solid - E_void) * rho
    E = props.E_void + (props.E_solid - props.E_void) * material_field
    nu = props.nu
    
    # Modelo Constitutivo (Elasticidade Linear Plana)
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u))

    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx

    # Linear form (zero, carga será aplicada no solver)
    f_zero = dolfinx.fem.Constant(msh, dolfinx.default_scalar_type((0, 0)))
    L_form = ufl.inner(f_zero, v) * ufl.dx

    # Configuração do Solver
    from dolfinx.fem.petsc import LinearProblem
    petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    
    problem = LinearProblem(a, L_form, bcs=[bc], petsc_options=petsc_options, petsc_options_prefix="alphabuilder_")

    # 5. Mapeamento de Índices (Grid -> DoF)
    dof_coords = D.tabulate_dof_coordinates()[:, :2]
    num_dofs = D.dofmap.index_map.size_local
    dof_map = np.zeros((ny, nx), dtype=np.int32) - 1
    
    tol = 1e-4
    dx = L / nx
    dy = H / ny
    
    for i in range(num_dofs):
        x, y = dof_coords[i]
        col = int((x + tol) / dx)
        row = ny - 1 - int((y + tol) / dy)
        
        if 0 <= row < ny and 0 <= col < nx:
            dof_map[row, col] = i
            
    return FEMContext(
        mesh=msh,
        V=V,
        D=D,
        u_sol=problem.u,
        material_field=material_field,
        problem=problem,
        dof_map=dof_map
    )
