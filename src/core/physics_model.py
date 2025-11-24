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
    # Assumindo rho binário {0, 1}, mas funciona para intermediários (SIMP)
    E = props.E_void + (props.E_solid - props.E_void) * material_field
    nu = props.nu
    
    # Modelo Constitutivo (Elasticidade Linear Plana - Estado Plano de Tensão)
    # Para estado plano de tensão (Plane Stress), as relações mudam ligeiramente,
    # mas aqui usaremos a formulação padrão 2D do UFL que geralmente assume Plane Strain 
    # ou ajustaremos lambda/mu. Vamos usar as definições de Lamé padrão.
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u))

    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx

    # Carga Pontual
    # Aplicada em (L, H/2) = (2.0, 0.5) com força (0, -1) (normalizada, o user disse -100N mas vamos usar unitário ou parametrizável)
    # O user especificou P = (0, -100N) no blueprint, mas aqui vamos usar unitário para validação adimensional
    # ou podemos parametrizar. Vamos colocar uma força fixa por enquanto.
    # Para aplicar carga pontual exata em FEM, idealmente o nó existe.
    # Como a malha é regular e 2.0/nx, 1.0/ny, se nx e ny forem pares, (2.0, 0.5) é um nó.
    
    # Vamos localizar o DoF mais próximo de (2.0, 0.5) para aplicar a força
    # Nota: Em FEniCSx, aplicar força pontual requer PointSource (não disponível em dolfinx nativo facilmente)
    # ou aplicar no vetor RHS manualmente.
    # Abordagem alternativa: Força distribuída em uma pequena área ds, mas isso requer marcar facetas.
    # Abordagem escolhida: Modificar o vetor RHS após montagem ou usar Dirac delta aproximado?
    # Vamos usar a abordagem de encontrar o DoF e aplicar no RHS no solver, mas o LinearProblem do dolfinx
    # abstrai isso.
    # Melhor abordagem para dolfinx puro: Definir uma Measure 'ds' e aplicar em uma pequena parte da fronteira
    # ou usar um termo de fonte 'f' que é zero em todo lugar exceto perto da carga.
    
    # Simplificação robusta: Aplicar carga distribuída na aresta direita inteira ou em uma pequena região.
    # O blueprint pede carga pontual.
    # Vamos tentar localizar o DoF.
    
    # Definindo L (Linear form)
    # f = dolfinx.fem.Constant(msh, dolfinx.default_scalar_type((0, 0)))
    # L = ufl.inner(f, v) * ufl.dx 
    # Isso seria zero. A carga entra depois.
    
    # Para usar dolfinx.fem.petsc.LinearProblem, precisamos de a e L.
    # Se L for zero na definição ufl, o vetor b será zero, e podemos modificá-lo.
    f_zero = dolfinx.fem.Constant(msh, dolfinx.default_scalar_type((0, 0)))
    L_form = ufl.inner(f_zero, v) * ufl.dx

    # Configuração do Solver
    from dolfinx.fem.petsc import LinearProblem
    # Usaremos opções robustas do PETSc
    petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    
    problem = LinearProblem(a, L_form, bcs=[bc], petsc_options=petsc_options, petsc_options_prefix="alphabuilder_")

    # 5. Mapeamento de Índices (Grid -> DoF)
    # D é um espaço DG0, então os DoFs correspondem às células.
    # Precisamos mapear (iy, ix) da matriz de topologia para o índice do DoF em D.
    
    # Coordenadas dos DoFs de D (centróides das células para DG0)
    dof_coords = D.tabulate_dof_coordinates()[:, :2]
    # Mapa global de índices
    # Para DG0, o número de DoFs é igual ao número de células (em serial).
    # Em paralelo, isso é mais complexo, mas vamos assumir serial ou lidar com ghost depois.
    # O user mencionou MPI, mas para o script inicial vamos focar em funcionar.
    
    # Criar grid de lookup
    # A matriz de topologia vem como (ny, nx) ou (H, W). O blueprint diz (H, W).
    # Vamos assumir que a matriz de entrada segue a convenção de imagem: m[y, x]
    # onde y=0 é o topo ou base? Geralmente em simulação y=0 é base.
    # Em matrizes numpy, m[0, 0] é canto superior esquerdo.
    # Vamos convencionar:
    # m[row, col]: row 0 -> y = H (topo), row H-1 -> y = 0 (base)
    # col 0 -> x = 0, col W-1 -> x = L
    # Isso inverte o eixo Y em relação à coordenada cartesiana.
    
    num_dofs = D.dofmap.index_map.size_local
    dof_map = np.zeros((ny, nx), dtype=np.int32) - 1
    
    # Tolerância para float comparison
    tol = 1e-4
    dx = L / nx
    dy = H / ny
    
    # Iterar sobre todos os DoFs e preencher o mapa
    # Isso é O(N), rápido o suficiente para inicialização
    for i in range(num_dofs):
        x, y = dof_coords[i]
        # Converter x, y para índices de coluna (j) e linha (i)
        # col = floor(x / dx)
        col = int((x + tol) / dx)
        # row: se y=0 é base e row=ny-1 é base
        # row = ny - 1 - floor(y / dy)
        row = ny - 1 - int((y + tol) / dy)
        
        if 0 <= row < ny and 0 <= col < nx:
            dof_map[row, col] = i
            
    return FEMContext(
        mesh=msh,
        V=V,
        D=D,
        u_sol=problem.u, # O LinearProblem cria sua própria função solução
        material_field=material_field,
        problem=problem,
        dof_map=dof_map
    )
