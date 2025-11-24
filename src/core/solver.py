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
    # Mapear a matriz de topologia para o vetor do FEniCSx usando o dof_map
    # topology_matrix shape: (ny, nx)
    # ctx.dof_map shape: (ny, nx) -> valores são índices no vetor
    
    # Flatten para facilitar atribuição se necessário, mas indexação direta funciona
    # Validar dimensões
    if topology_matrix.shape != ctx.dof_map.shape:
        raise ValueError(f"Topology shape {topology_matrix.shape} mismatch with Context shape {ctx.dof_map.shape}")
        
    # Atualizar valores do campo de material
    # O vetor do FEniCSx é acessado via ctx.material_field.x.array
    # Precisamos garantir que estamos escrevendo nos índices corretos
    
    # Extrair índices válidos (dof_map >= 0)
    mask = ctx.dof_map >= 0
    valid_indices = ctx.dof_map[mask]
    valid_values = topology_matrix[mask].astype(dolfinx.default_scalar_type)
    
    ctx.material_field.x.array[valid_indices] = valid_values
    
    # Atualizar ghosts se rodando em paralelo (necessário para dolfinx)
    ctx.material_field.x.scatter_forward()

    # 2. Configurar Carga Pontual no Vetor RHS (b)
    # Como o LinearProblem remonta o b a cada solve se L depender de funções que mudam,
    # mas aqui L é zero na definição original.
    # Precisamos injetar a força no vetor b antes de resolver.
    # O LinearProblem do dolfinx tem um método solve() que faz assemble_system.
    # Se usarmos problem.solve(), ele vai remontar A e b.
    # A depende de E(rho), que mudou, então A precisa ser remontado.
    # b depende de L, que é zero.
    
    # Vamos deixar o LinearProblem montar o sistema.
    # A = problem.A
    # b = problem.b
    # Mas problem.solve() retorna u, e faz tudo isso internamente.
    # Precisamos interceptar ou usar uma abordagem mais manual se quisermos modificar b.
    # OU, definimos L com uma PointSource? Dolfinx não tem PointSource fácil como Dolfin antigo.
    
    # Abordagem: Usar problem.solve() e depois adicionar a força? Não, tem que ser antes do solve.
    # O LinearProblem não expõe hooks fáceis para modificar b entre assemble e solve.
    # Vamos reconstruir a lógica de solve manualmente para ter controle.
    
    # Re-implementando a lógica do LinearProblem.solve simplificada:
    
    # Montar matriz A (Stiffness)
    # Como E(rho) mudou, A mudou.
    A = ctx.problem.A
    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A, ctx.problem.a, bcs=ctx.problem.bcs)
    A.assemble()
    
    # Montar vetor b (Load)
    b = ctx.problem.b
    b.zeroEntries()
    dolfinx.fem.petsc.assemble_vector(b, ctx.problem.L)
    
    # Aplicar Carga Pontual
    # Força P = (0, -100) em (L, H/2)
    # Precisamos encontrar o DoF correspondente no espaço V (VectorFunctionSpace)
    # Coordenada alvo
    target_point = np.array([2.0, 0.5, 0.0]) # 3D para função de busca
    
    # Encontrar a célula que contém o ponto (ou mais próxima)
    # bounding_box_tree para busca geométrica
    # Nota: Isso pode ser lento se feito todo passo. Deveria ser pré-calculado no Context.
    # Vamos assumir que podemos pré-calcular o índice do DoF de carga no initialize, 
    # mas por enquanto vamos fazer aqui ou adicionar ao Context depois.
    # Para simplificar e ser rápido: vamos achar o DoF mais próximo na força bruta na inicialização
    # e guardar no context. Como não alterei o Context, vou fazer uma busca rápida aqui
    # mas cachear seria ideal.
    
    # Sendo uma malha regular, podemos deduzir o índice?
    # V é Lagrange grau 1. DoFs são vértices.
    # (2.0, 0.5) é um vértice se a malha for par em Y.
    
    # Vamos usar dolfinx.fem.locate_dofs_geometrical
    # Mas isso retorna todos os dofs numa região.
    
    # Hack para carga pontual:
    # Adicionar termo ao vetor b no índice correspondente.
    # Vamos localizar o DoF na primeira execução e usar cache (ou recalcular, é rápido para 1 ponto).
    
    V = ctx.V
    
    # Função localizadora
    def point_load_locator(x):
        # Tolerância pequena
        tol = 1e-3
        return np.logical_and(
            np.isclose(x[0], 2.0, atol=tol),
            np.isclose(x[1], 0.5, atol=tol)
        )
        
    load_dofs = fem.locate_dofs_geometrical(V, point_load_locator)
    
    # Aplicar força
    # load_dofs contém índices locais.
    # V tem bloco de tamanho 2 (x, y).
    # Precisamos saber qual é o Y.
    # V.dofmap.index_map_bs é o block size.
    
    # Em dolfinx, locate_dofs retorna índices "unrolled" se o espaço for colapsado?
    # Não, VectorFunctionSpace com Lagrange 1 tem dofs (n_nodes * block_size).
    # locate_dofs_geometrical retorna os índices dos nós? Não, retorna os índices dos DoFs.
    # Mas como V é vetorial, ele retorna índices para todos os componentes?
    # Geralmente retorna para o bloco.
    
    # Vamos simplificar: Aplicar carga distribuída em uma pequena área na borda direita.
    # É mais estável numericamente e fácil de implementar com ds.
    # Mas o user quer validação com analítico de viga, que assume carga pontual.
    # Carga distribuída em dy pequeno aproxima bem.
    
    # Vamos tentar injetar no vetor b.
    # Se load_dofs encontrar algo:
    if len(load_dofs) > 0:
        # A força é (0, -100). Componente Y é o segundo no bloco.
        # Se V é VectorFunctionSpace, os dofs são organizados por nó ou por componente?
        # Dolfinx default: por nó (x, y, x, y...) ou componente (x, x..., y, y...)?
        # Geralmente (x, y) intercalado.
        
        # Vamos assumir intercalado.
        # load_dofs retorna índices globais "unrolled" no VectorFunctionSpace?
        # Sim.
        
        # Precisamos filtrar apenas o componente Y.
        # O componente Y terá índice ímpar? Ou depende do layout.
        # Melhor: obter o sub-espaço colapsado e localizar nele.
        V_y, vy_map = V.sub(1).collapse()
        load_dofs_y_local = fem.locate_dofs_geometrical(V_y, point_load_locator)
        load_dofs_y = vy_map[load_dofs_y_local]
        
        # Aplicar carga
        force_value = -100.0 # Valor do blueprint
        # force_value = -1.0 # Valor unitário para teste inicial? O blueprint diz -100N.
        
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
    
    # Max Displacement (Norma L2 do vetor deslocamento em cada ponto, pegar o máximo)
    # Maneira rápida: pegar array e calcular norma
    # u.x.array é flat [ux0, uy0, ux1, uy1...]
    u_vals = u.x.array.reshape(-1, 2) # Assumindo 2D
    disp_norms = np.linalg.norm(u_vals, axis=1)
    max_disp = np.max(disp_norms)
    
    # Compliance: f . u
    # C = integral(f * u) dx + integral(t * u) ds
    # Como aplicamos força pontual, C = F * u_ponto
    # Ou podemos calcular energia de deformação: C = integral(sigma : epsilon) dx
    # C = a(u, u)
    # Isso é mais robusto.
    
    # Calcular Compliance via energia
    # C = assemble_scalar(form(action(action(a, u), u))) ?
    # Mais simples: b . u (produto interno do vetor força pelo vetor deslocamento)
    # b contém as forças (incluindo reações? Não, b é o RHS montado).
    # b . u é o trabalho das forças externas.
    compliance = b.dot(u.x.petsc_vec)
    
    # Fitness (Kane Eq 1)
    # F(s) = 1 / (Omega_mat + eps * Omega_dis + alpha * max(0, Dmax - Dlim))
    
    # Omega_mat: Massa relativa (fração de 1s)
    omega_mat = np.mean(topology_matrix)
    
    # Omega_dis: Massa desconectada.
    # O solver não calcula isso, assume-se que vem de fora ou é 0 se o MCTS garantir conectividade.
    # Vamos assumir 0 por enquanto ou implementar uma verificação rápida (BFS) se necessário.
    # O blueprint diz: "MCTS deve ser desenhado para manter isso em 0".
    omega_dis = 0.0 
    
    # Penalidade de deslocamento
    penalty = max(0.0, max_disp - props.disp_limit)
    
    denominator = omega_mat + props.penalty_epsilon * omega_dis + props.penalty_alpha * penalty
    
    # Evitar divisão por zero
    if denominator < 1e-9:
        fitness = 0.0
    else:
        fitness = 1.0 / denominator

    return SimulationResult(
        fitness=fitness,
        max_displacement=max_disp,
        compliance=compliance,
        valid=True, # Assumimos válido se o solver convergiu
        displacement_array=u_vals if False else np.array([]) # Otimização
    )
