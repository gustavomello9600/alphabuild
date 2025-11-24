# 🤖 MISSÃO: AGENTE 01 (PHYSICS_CORE)

**Função:** Especialista em Simulação Física (FEM) com FEniCSx
**Paradigma:** Funcional Prático (Stateless)
**Ambiente:** Linux (Arch) via Miniforge/Conda
**Stack:** Python 3.10, FEniCSx (dolfinx), UFL, MPI, NumPy.

---

## 1. CONTEXTO E AMBIENTE
Você é o motor de física do projeto **AlphaBuilder**. Sua missão é substituir a realidade por uma simulação numérica de alta fidelidade.
Você utilizará a biblioteca **FEniCSx** (dolfinx).

**⚠️ Instrução Crítica de Ambiente:**
Não tente instalar pacotes via `pacman` ou `pip` global. Assuma que você está rodando dentro de um ambiente Conda configurado assim:
```bash
# Setup esperado (Não execute, apenas assuma que existe)
mamba install -c conda-forge fenics-dolfinx mpich pyvista matplotlib scipy
```

**Diretriz de Performance (JIT):**
O FEniCSx compila formas variacionais (código C++) em tempo de execução. Isso é lento.
*   **Proibido:** Recompilar o problema (`dolfinx.fem.form(...)`) dentro do loop de otimização.
*   **Obrigatório:** Compilar a forma variacional **uma única vez** no início. Dentro do loop, você apenas atualiza os **Coeficientes** da função de material (`material_function.x.array[:] = new_values`).

---

## 2. ESTRUTURAS DE DADOS (INTERFACE)
Utilize `dataclasses` imutáveis para garantir pureza nas funções.

```python
from dataclasses import dataclass, field
from typing import Tuple, Any, Callable
import numpy as np
import dolfinx

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
```

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
```

---

## 3. TAREFAS DE IMPLEMENTAÇÃO

### Tarefa A: Setup do Contexto (Compilação Única)
Crie uma função que inicializa a malha, define o problema variacional e retorna o `FEMContext`.

*   **Função:** `initialize_cantilever_context(resolution: Tuple[int, int], props: PhysicalProperties) -> FEMContext`
*   **Passos Técnicos:**
    1.  **Malha:** Use `dolfinx.mesh.create_rectangle` com MPI COMM_WORLD. Dimensões $2.0 \times 1.0$.
    2.  **Espaços:**
        *   `V`: VectorFunctionSpace (Lagrange, grau 1) $\to$ para deslocamentos $\mathbf{u}$.
        *   `D`: FunctionSpace (Discontinuous Lagrange "DG", grau 0) $\to$ para densidade $\rho$.
    3.  **Condições de Contorno (BCs):**
        *   Localize os nós da esquerda ($x \approx 0$).
        *   Crie o objeto `dirichletbc` fixando $\mathbf{u}=0$.
    4.  **Forma Variacional (UFL):**
        *   Defina `rho = dolfinx.fem.Function(D)`.
        *   Defina o Módulo de Young Interpolado: $E(\rho) = E_{void} + (E_{solid} - E_{void}) \times \rho$.
        *   Escreva a equação da elasticidade linear padrão usando $\sigma(\mathbf{u}, E(\rho))$.
        *   Defina a carga pontual usando `dolfinx.fem.Constant` ou uma medida `ds` marcada, mas para simplificar carga pontual em nó específico, pode-se usar aproximação de força distribuída em um elemento pequeno ou encontrar o grau de liberdade (DoF) correspondente e aplicar força direta no vetor RHS (Lado Direito). *Recomendação:* Use `ufl.SpatialCoordinate` para aplicar uma carga de superfície (`ds`) muito concentrada na aresta direita se achar mais estável, ou localize o DoF mais próximo de $(2.0, 0.5)$.
    5.  **Mapeamento de Índices:**
        *   **Crítico:** A matriz NumPy `[row, col]` não mapeia linearmente para os DoFs do espaço `D` (material).
        *   Você deve usar `D.tabulate_dof_coordinates()` para criar um mapa de lookup `numpy_to_dof_map`.
        *   Este mapa permite saber: "O pixel em `matrix[i, j]` corresponde ao índice `k` no vetor de material do FEniCSx".

### Tarefa B: Solver Rápido (Função Pura com Efeito Lateral Controlado)
Crie a função que será chamada milhares de vezes.

*   **Função:** `solve_topology(topology_matrix: np.ndarray, ctx: FEMContext, props: PhysicalProperties) -> SimulationResult`
*   **Lógica:**
    1.  **Atualização de Material:**
        *   Use o `ctx.dof_map` para copiar os valores de `topology_matrix` (0 ou 1) para o vetor subjacente `ctx.material_field.x.array[:]`.
        *   **Não redefina o problema.** Apenas atualize o vetor.
    2.  **Resolução:**
        *   Chame `ctx.problem.solve()`.
    3.  **Pós-Processamento:**
        *   Calcule $D_{max} = \max \|\mathbf{u}\|_{L2}$.
        *   Calcule Compliance $\int \mathbf{f} \cdot \mathbf{u} dx$ (útil para debug).
        *   Aplique a **Equação 1** (Fitness Kane & Schoenauer).

---

## 4. DICAS TÉCNICAS "FEniCSx EXPERT"

### Mapeamento de Coordenadas (O Grande Desafio)
O FEniCSx pode reordenar a malha para otimização de cache. Não assuma ordem.
Faça algo assim na inicialização:
```python
# Pseudo-código para Task A
coordinates = D.tabulate_dof_coordinates()[:, :2] # x, y de cada célula
dof_indices = np.arange(D.dofmap.index_map.size_local)

# Crie uma matriz de indices que corresponda ao grid (H, W)
# Arredonde coordenadas para evitar erro de float
x_grid = np.round(coordinates[:, 0], 3)
y_grid = np.round(coordinates[:, 1], 3)

# Logica para preencher um mapa que diz: map[i, j] = dof_index
```

### Evitando Crash do Solver
Se a topologia for totalmente desconexa (ilhas de material flutuando), a matriz $K$ pode se tornar singular mesmo com $E_{void} > 0$ se $E_{void}$ for muito pequeno.
*   *Dica:* Use um Solver Iterativo (ex: 'cg' com precondicionador 'amg') ou um Solver Direto robusto ('mumps' ou 'superlu_dist' se disponível via PETSc). Para 2D pequeno, o solver padrão `scipy.sparse.linalg.spsolve` (se converter para scipy) ou o solver padrão do PETSc LU funcionam bem. Configure o `LinearProblem` para usar **PETSc options** `{"ksp_type": "preonly", "pc_type": "lu"}` para garantia absoluta de robustez em problemas pequenos.

---

## 5. VALIDAÇÃO

Seu script deve conter um `main` que:
1.  Inicialize o contexto $20 \times 10$.
2.  Preencha uma topologia cheia (tudo 1).
3.  Resolva e compare o $D_{max}$ com a teoria $PL^3/3EI$.
4.  Preencha uma topologia vazia (tudo 0).
5.  Resolva e verifique se $D_{max}$ é aprox $1/E_{void}$ vezes maior (comportamento linear esperado).

**Output Obrigatório:** Um print limpo: `[Validation] Analytical: X.XXX | FEM: Y.YYY | Error: Z.ZZ%`.