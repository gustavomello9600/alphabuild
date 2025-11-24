# 🤖 MISSÃO: AGENTE 02 (ALPHA_ARCHITECT)

**Função:** Motor de Busca, Lógica de Grafos e Gerenciamento de Dados.
**Paradigma:** Funcional Puro (Pure Functional Programming).
**Stack:** Python 3.10+, NumPy, SQLite, SciPy (Sparse Graph).

---

## 1. CONTEXTO E RESPONSABILIDADE
Você é o kernel lógico do **AlphaBuilder**. Sua função é deterministicamente transformar o estado do tabuleiro e orquestrar a execução do episódio bifásico.

**Diretrizes de Implementação:**
1.  **Imutabilidade:** Objetos de estado nunca são modificados. Funções de transição recebem `State` e retornam `NewState`. Isso é mandatório para permitir que o MCTS seja paralelizado futuramente sem *Race Conditions*.
2.  **Eficiência de Grafos:** A verificação de conectividade e pontos de articulação é o gargalo da Fase 2. Utilize algoritmos otimizados (ex: `scipy.sparse.csgraph` ou implementações NumPy vetorizadas).
3.  **Persistência Síncrona:** Não use buffers em memória para dados de treino. Escreva no SQLite a cada passo avaliado pelo FEM para evitar perda de dados em caso de crash do solver.

---

## 2. TIPAGEM E ESTRUTURAS DE DADOS

Implemente exatamente estas estruturas imutáveis.

```python
from typing import NamedTuple, Tuple, FrozenSet, Literal
import numpy as np

# Tipos Primitivos
Coord = Tuple[int, int]
ActionType = Literal['ADD', 'REMOVE']
PhaseType = Literal['GROWTH', 'REFINEMENT']

class GameAction(NamedTuple):
    """Ação atômica do jogo."""
    type: ActionType
    coord: Coord

class DesignState(NamedTuple):
    """
    Estado completo e imutável do sistema.
    Usa NamedTuple para ser hashable e leve.
    """
    grid: np.ndarray              # Matriz Binária (Read-only)
    supports: Tuple[Coord, ...]   # Coordenadas fixas
    loads: Tuple[Coord, ...]      # Coordenadas de carga
    phase: PhaseType              # Fase atual
    
    # Cache de Grafos (Metadados derivados)
    is_connected: bool            # Conectividade Global
    volume: int                   # Contagem de material
    perimeter: FrozenSet[Coord]   # Fronteira de expansão válida

    def __hash__(self):
        return hash(self.grid.tobytes())

class SimulationRecord(NamedTuple):
    """DTO para persistência no DB."""
    episode_id: str
    step: int
    state_bytes: bytes
    fitness: float
    is_valid: bool
```

---

## 3. IMPLEMENTAÇÃO FUNCIONAL (CORE)

### 3.1. Módulo de Topologia (Pure Functions)
Implemente funções estatizadas para análise de grid.

*   **`build_adjacency_graph(grid: np.ndarray) -> Any`**
    *   Constrói uma representação de grafo esparso (Matriz de Adjacência ou Lista) considerando conectividade-4 (Von Neumann).
    *   Deve ser executado apenas quando necessário.

*   **`check_global_connectivity(grid: np.ndarray, sources: Tuple[Coord], targets: Tuple[Coord]) -> bool`**
    *   Execute BFS ou componentes conexos.
    *   Retorna `True` se existir caminho entre `sources` e `targets`.

*   **`get_articulation_points(grid: np.ndarray) -> FrozenSet[Coord]`**
    *   Implemente o algoritmo de Tarjan ou Hopcroft-Tarjan para encontrar pontos de articulação em grafos não-direcionados.
    *   **Otimização:** Na Fase 2, esta função deve ser chamada apenas se a ação escolhida for `REMOVE`.

### 3.2. Módulo de Regras (Game Logic)
Implemente a função de transição de estado.

*   **`get_legal_actions(state: DesignState) -> Tuple[GameAction, ...]`**
    *   **Caso GROWTH:** Retorne `ADD` para todas as coordenadas em `state.perimeter`. Ordene heuristicamente (distância à carga) para acelerar convergência.
    *   **Caso REFINEMENT:**
        *   `ADD`: Todas as coordenadas em `state.perimeter`.
        *   `REMOVE`: Todas as coordenadas com material (`grid[r,c] == 1`), **EXCETO** as que estão em `get_articulation_points(grid)` e as que são Suportes/Cargas fixas.

*   **`apply_action(state: DesignState, action: GameAction) -> DesignState`**
    *   Cria cópia da grid: `new_grid = state.grid.copy()`.
    *   Aplica mutação na cópia.
    *   Recalcula `perimeter` incrementalmente (Adicionar vizinhos do novo bloco ou remover vizinhos do bloco deletado).
    *   Recalcula `is_connected` (Se ação for ADD e estado anterior era desconectado, checar. Se REMOVE, assumir True pois legal_actions garante).
    *   Determina nova `phase`.
    *   Retorna novo `DesignState`.

### 3.3. Módulo de Persistência (I/O)
Implemente gerenciamento de SQLite com Context Managers.

*   **`initialize_db(db_path: str = "training_data.db")`**
    *   Crie a tabela se não existir. Use modo WAL (`PRAGMA journal_mode=WAL;`) para performance de escrita concorrente.
*   **`persist_record(db_path: str, record: SimulationRecord)`**
    *   Insira o registro. Commit imediato.

---

## 4. LOOP DE EXECUÇÃO (ORQUESTRAÇÃO)

Implemente a função principal que controla o fluxo.

```python
def run_episode(
    episode_id: str,
    initial_config: dict,
    solver_fn: Callable[[np.ndarray], float],
    max_steps: int = 200
) -> None:
    """
    Executa um episódio completo (Growth -> Refinement).
    Não retorna valor, seu efeito colateral é popular o DB.
    """
    # 1. Setup Inicial
    state = create_initial_state(initial_config)
    
    # 2. Loop de Passos
    for step in range(max_steps):
        
        # 3. Seleção de Ação (Policy)
        legal_actions = get_legal_actions(state)
        
        if state.phase == 'GROWTH':
            # Heurística Determinística: Escolhe ação que minimiza dist(Carga)
            action = select_heuristic_action(legal_actions, state.loads)
        else:
            # MCTS / Random (Milestone 1)
            # Na Milestone 1, use seleção aleatória uniforme entre as legais
            action = select_random_action(legal_actions)

        # 4. Transição
        next_state = apply_action(state, action)
        
        # 5. Avaliação (Apenas Refinement)
        if next_state.phase == 'REFINEMENT':
            # A chamada ao solver é bloqueante
            result = solver_fn(next_state.grid) 
            
            # Persistência Obrigatória
            record = SimulationRecord(
                episode_id=episode_id,
                step=step,
                state_bytes=next_state.grid.tobytes(),
                fitness=result.fitness,
                is_valid=result.valid
            )
            persist_record("training_data.db", record)
        
        # 6. Avanço
        state = next_state
```

---

## 5. REQUISITOS DE VALIDAÇÃO

Ao final do script, inclua um bloco `__main__` que execute:
1.  **Teste de Articulação:** Crie manualmente um grid com uma "ponte" de 1 pixel de largura. Chame `get_legal_actions` e assevere (`assert`) que a ação de remover o pixel da ponte **não** está na lista.
2.  **Teste de Integridade de Dados:** Execute 10 passos da fase `REFINEMENT` usando um `solver_fn` mock (que retorna fitness aleatório). Consulte o banco SQLite e assevere que existem exatamente 10 registros novos.
3.  **Teste de Imutabilidade:** Assevere que `hash(state_t0) != hash(state_t1)` após uma ação.
