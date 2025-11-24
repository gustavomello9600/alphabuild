# 📂 ALPHABUILDER: PROTOCOLO MESTRE DO PROJETO
**Versão:** 1.1 - Blueprint Arquitetural
**Status:** Fase de Inicialização
**Referência Bibliográfica:** Kane, C., & Schoenauer, M. (1996). *Topological Optimum Design using Genetic Algorithms*.

---

## 1. VISÃO ESTRATÉGICA E ESCOPO

### 1.1. O Conceito
O **AlphaBuilder** é um sistema de Otimização Topológica (TO) de próxima geração que substitui as heurísticas tradicionais (como SIMP ou BESO) e a otimização estocástica pura (Algoritmos Genéticos) por uma abordagem de **Aprendizado por Reforço Baseado em Modelo (Model-Based RL)**.

Inspirado na arquitetura do *AlphaZero*, o sistema não recebe "conhecimento prévio" de engenharia estrutural. Ele aprende a construir estruturas eficientes através de um processo iterativo de auto-aperfeiçoamento, onde uma Rede Neural (Vision Transformer) aprende a intuir a física, guiando uma busca em árvore (MCTS) para resolver problemas de conformidade mecânica.

### 1.2. Ambições e Scalability (O "Grande Plano")
Embora o foco inicial seja um benchmark 2D, a arquitetura deve ser agnóstica à dimensionalidade e ao solver físico.
*   **Modularidade Funcional:** O motor de física não deve ser um objeto monolítico com estado interno. Ele deve ser tratado como uma **Pure Function** (ou quase pura) que recebe topologia + condições e retorna métricas. Isso permite trocar o *backend* (SfePy $\to$ FEniCSx) apenas passando uma função diferente para o pipeline de treinamento (`solver_fn`), sem refatorar classes.
*   **Expansão 3D:** A representação de dados (Tensores) e a arquitetura da Rede Neural (Transformers) facilitam a transição de *Pixels (2D)* para *Voxels (3D)*. O código deve operar sobre tensores genéricos, evitando hard-coding de loops indexados.
*   **Generalização:** O sistema deve aprender uma "intuição física" mapeando *Estado $\to$ Valor* de forma agnóstica à geometria específica.

---

## 2. ARQUITETURA DE SISTEMA: A TRÍADE

O sistema é composto por três módulos independentes que se comunicam através de interfaces de dados estritas.

### 2.1. O Explorador (Agent: Alpha_Architect)
*   **Algoritmo:** Monte Carlo Tree Search (MCTS) modificado.
*   **Função:** Navegar no espaço combinatório de topologias (2^N estados possíveis).
*   **Diferencial:** Utiliza uma política construtiva. Em vez de começar cheio e remover (top-down) ou aleatório, ele constrói a estrutura conectada passo-a-passo, garantindo viabilidade topológica intrínseca.

### 2.2. O Oráculo (Agent: Neural_Vision)
*   **Algoritmo:** Vision Transformer (ViT).
*   **Função:** Aproximador Universal de Função de Valor ($V(s) \approx \text{Fitness}$).
*   **Justificativa:** Problemas de elasticidade são globais (Princípio de Saint-Venant). Uma alteração local afeta o campo de tensão globalmente. O mecanismo de *Self-Attention* dos Transformers captura essas dependências de longo alcance melhor que CNNs tradicionais.
*   **Input:** Tensor de Estado.
*   **Output:** Escalar de qualidade (Fitness prevista).

### 2.3. O Juiz (Agent: Physics_Core)
*   **Algoritmo:** Método dos Elementos Finitos (FEM).
*   **Solver Inicial:** *SfePy* (Simple Finite Elements in Python).
*   **Função:** Fornecer o "Ground Truth". É o gargalo computacional do sistema, acionado apenas quando necessário para validar designs finais ou gerar dados de treino (Replay Buffer).

---

## 3. ESPECIFICAÇÕES DO AMBIENTE (BENCHMARK KANE 1996)

Para a Milestone 1, replicaremos estritamente as condições do paper para validação científica.

### 3.1. Domínio Físico
*   **Geometria:** Placa Retangular "Cantilever".
*   **Razão de Aspecto:** $2:1$ (Largura $L=2.0$, Altura $H=1.0$).
*   **Malha de Discretização:** Quadrangular Regular.
    *   Resoluções alvo: $32 \times 16$ (Debug), $64 \times 32$ (Validação Padrão), $80 \times 40$ (Alta Resolução).
*   **Condições de Contorno (BCs):**
    *   **Fixed ($\Gamma_u$):** $x = 0$ (Aresta esquerda inteira). $u_x=0, u_y=0$.
    *   **Load ($\Gamma_t$):** $x = L, y = H/2$ (Meio da aresta direita). Força pontual $P = (0, -100N)$.

### 3.2. Modelo de Material (Pseudo-SIMP)
Para evitar remalhagem (remeshing) custosa, usamos uma Grid Euleriana Fixa.
*   A matriz de rigidez global $K$ mantém dimensão constante.
*   **Material Sólido (1):** $E = E_{base}$ (ex: 210 GPa), $\nu = 0.3$.
*   **Material Vazio (0):** $E = 10^{-6} \times E_{base}$. (Material "fantasma" suave para evitar singularidade numérica sem afetar a física significativamente).

### 3.3. A Lei (Função de Custo)
A métrica de sucesso é definida pela **Equação 1** de Kane & Schoenauer. Todos os agentes devem otimizar para esta métrica específica:

$$ \mathcal{F}(s) = \frac{1}{ \Omega_{mat} + \epsilon \cdot \Omega_{dis} + \alpha \cdot \max(0, D_{max}(s) - D_{lim}) } $$

*   **$\Omega_{mat}$ (Massa Conectada):** Peso da estrutura útil.
*   **$\Omega_{dis}$ (Massa Desconectada):** Peso de "ilhas" flutuantes de material. (Nota: O MCTS deve ser desenhado para manter isso em 0, mas o FEM deve penalizar se ocorrer).
*   **$D_{max}$:** Deslocamento nodal máximo absoluto encontrado na malha.
*   **$D_{lim}$:** Restrição de projeto (Deslocamento máximo tolerado).
*   **$\alpha, \epsilon$:** Fatores de penalidade (Hiperparâmetros cruciais).

---

## 4. PROTOCOLOS DE DADOS (INTERFACES)

A integridade do sistema depende de formatos de dados rígidos.

### 4.1. O Tensor de Estado Universal
Qualquer representação de um design deve ser passada entre agentes como um tensor NumPy `ndarray`.

**Shape:** `(Height, Width, Channels)`
*   **2D (Atual):** `(H, W, 3)`
*   **3D (Futuro):** `(D, H, W, 3)` - O código deve ser preparado para esta expansão.

**Canais:**
1.  **Ch 0 - Topologia ($\rho$):** Matriz Binária. `0` (Vazio), `1` (Material). Futuramente pode suportar valores contínuos $[0, 1]$ se migrarmos para otimização baseada em densidade.
2.  **Ch 1 - Condições de Dirichlet ($\delta$):** Máscara de Suportes. `1` onde o nó é fixo, `0` caso contrário. Isso permite que a Rede Neural "veja" onde a estrutura deve se apoiar.
3.  **Ch 2 - Condições de Neumann ($F$):** Mapa de Cargas. Magnitude normalizada da força na célula. Permite generalização para múltiplos casos de carga.

### 4.2. Interface do Solver (Modularidade via High-Order Functions)
Em vez de instanciar classes de solvers, o sistema deve operar passando funções de resolução como argumentos.

**Assinatura de Tipo (Type Alias):**
```python
from typing import Callable, TypedDict, Any
import numpy as np

# Definição da estrutura de retorno
class SimulationResult(TypedDict):
    fitness: float
    max_displacement: float
    compliance: float
    valid: bool
    metadata: dict[str, Any]

# O Solver é qualquer função que cumpra esta assinatura
SolverFn = Callable[[np.ndarray, np.ndarray, np.ndarray], SimulationResult]
# args: (topology_matrix, supports_mask, loads_map) -> result
```

Isso garante que, quando mudarmos do SfePy para FEniCSx, basta criar uma nova função que respeite a assinatura `SolverFn` e passá-la para o orquestrador.

---

## 5. DINÂMICA DE EXECUÇÃO: O PIPELINE BIFÁSICO

A execução de um episódio de otimização deve obedecer estritamente a duas fases sequenciais. O Agente **Alpha_Architect** é responsável por gerenciar a transição de estado.

### 5.1. Fase 1: Fechamento Topológico (Topology Closing)
*   **Objetivo Único:** Estabelecer a conectividade mínima entre todos os nós de Carga ($\Gamma_t$) e os nós de Suporte ($\Gamma_u$).
*   **Restrição de Recurso:** A execução do Solver FEM (Agente 01) é **ESTRITAMENTE PROIBIDA** nesta fase.
*   **Heurística de Navegação:** O Agente deve utilizar uma busca gulosa ou A* baseada na Distância Euclidiana para minimizar o custo de conexão.
*   **Ações Permitidas:** Apenas `ADD` (Adicionar material) na fronteira de crescimento.
*   **Critério de Transição:** Ocorre no instante $t$ em que `is_connected(state) == True`.

### 5.2. Fase 2: Refinamento Topológico (Topology Refinement)
*   **Objetivo Único:** Minimizar a Função de Custo $\mathcal{F}$ (Eq. 1).
*   **Restrição de Recurso:** Toda nova topologia válida gerada deve ser submetida ao Solver FEM.
*   **Persistência:** O resultado da simulação deve ser gravado sincronicamente no banco de dados de treino.
*   **Ações Permitidas:**
    *   `ADD`: Em qualquer célula da fronteira externa.
    *   `REMOVE`: Em qualquer célula de material, **EXCETO** se a célula for um **Ponto de Articulação** (cuja remoção desconectaria a carga do suporte).
*   **Orçamento:** O episódio encerra após $N_{max}$ passos na Fase 2 ou por estagnação (delta de fitness $< 10^{-4}$ por 20 passos).

### 5.3. Pipeline de Dados (Data Lake)
O sistema deve manter um banco de dados relacional para *Replay Experience*.
*   **Tecnologia:** SQLite (Arquivo local `training_data.db`).
*   **Imutabilidade:** Registros inseridos nunca devem ser alterados.
*   **Esquema Obrigatório:**
    *   `episode_id` (UUID)
    *   `step` (Integer)
    *   `phase` (Enum: GROWTH/REFINEMENT)
    *   `state_blob` (Binary/Pickle do Tensor NumPy)
    *   `fitness_score` (Float)
    *   `valid_fem` (Boolean)

---

## 6. ROADMAP DE DESENVOLVIMENTO

### Milestone 1: A Replicação (Foco Atual)
*   **Objetivo:** Gerar a estrutura da Fig. 7-a (Kane & Schoenauer) em 2D.
*   **Solver:** SfePy.
*   **Tech:** MCTS básico + ViT supervisionado.

### Milestone 2: Generalização e Multi-Loading
*   **Objetivo:** Resolver o problema da "Bicicleta" (Fig. 11 do PDF), que envolve 3 casos de carga simultâneos.
*   **Desafio:** O tensor de entrada precisará empilhar canais de carga ou a ViT precisará entender cargas variáveis. O Solver FEM deverá rodar 3 sub-casos e agregar a fitness.

### Milestone 3: A Fronteira 3D
*   **Objetivo:** Otimizar um cubo engastado (Cantilever 3D).
*   **Desafio:** Explosão combinatória de estados. Substituição do ViT 2D por **Video ViT (VViT)** ou **3D-CNN**. Substituição do SfePy por **FEniCSx** (paralelizado via MPI) devido ao custo computacional da matriz de rigidez 3D.

---

## 7. STACK TECNOLÓGICA OBRIGATÓRIA

### 7.1. Gestão de Ambiente e Dependências
Devido à complexidade de combinar bibliotecas científicas compiladas (FEniCSx/PETSc) com Frameworks de IA (TensorFlow/Keras) em Linux (especialmente Arch), o uso de ambientes **Conda** é mandatório.

*   **Gerenciador:** `Miniforge` ou `Mambaforge` (evite Anaconda padrão por questões de licença/bloatware).
*   **Canal Principal:** `conda-forge` (prioridade máxima para evitar conflitos de binários C++).
*   **Versão Python Alvo:** **3.10** (Ponto ideal de estabilidade entre `dolfinx` atual e `tensorflow`).

**Comando de Reprodução de Ambiente:**
```bash
mamba create -n alphabuilder python=3.10
mamba activate alphabuilder
mamba install -c conda-forge fenics-dolfinx mpich pyvista matplotlib scipy numpy pandas tensorflow
# Nota: Se houver conflito com tensorflow no conda-forge, instale-o via pip dentro do env conda.
```

---

## 8. DIRETRIZES DE CÓDIGO E ESTILO
*   **Paradigma:** **Programação Funcional Prática**.
    *   Prefira funções puras e isoladas a classes com métodos que mutam `self`.
    *   Use `dataclasses(frozen=True)` ou `NamedTuple` para DTOs (Data Transfer Objects).
    *   Evite estado global (`global` variables são proibidas).
    *   Use *Type Hinting* estrito (`mypy`) para assinaturas de função.
*   **Composição:** Construa pipelines de processamento de dados (ex: `train_step = update_weights(calculate_loss(predict(batch)))`).
*   **Documentação:** Docstrings no formato Google Style.
*   **Reprodutibilidade:** *Seeds* devem ser passadas como argumentos para funções estocásticas, não configuradas globalmente dentro delas.
*   **Logging:** Todo experimento deve retornar logs estruturados (Dicts) que são agregados pelo orquestrador, em vez de escrever em arquivos dispersos durante a execução.