# 🤖 MISSÃO: AGENTE 02 (ALPHA_ARCHITECT)

**Função:** Motor de Busca (MCTS), Lógica de Jogo Bifásica e Gerenciamento de Estado.
**Paradigma:** Funcional Puro (Pure Functional Programming).
**Stack:** Python 3.10+, NumPy, SQLite, SciPy (Sparse Graph).

---

## 1. CONTEXTO E RESPONSABILIDADE
Você é o kernel lógico do **AlphaBuilder v1.1**.
Sua missão é orquestrar o episódio bifásico:
1.  **Fase 1 (Pathfinder):** Garantir conectividade entre Cargas e Suportes (Search for Function).
2.  **Fase 2 (Sculptor):** Otimizar a forma para reduzir volume mantendo a rigidez (Search for Form).

---

## 2. TIPAGEM E ESTRUTURAS DE DADOS

```python
from typing import NamedTuple, Tuple, FrozenSet, Literal, Optional
import numpy as np

# Tipos Primitivos
Coord = Tuple[int, int, int] # (z, y, x) - Agora 3D nativo
ActionType = Literal['ADD', 'REMOVE']
PhaseType = Literal['PATHFINDING', 'REFINEMENT']

class GameAction(NamedTuple):
    """Ação atômica do jogo."""
    type: ActionType
    coord: Coord
    prob_score: float # Score vindo da Policy Network (Swin-UNETR)

class DesignState(NamedTuple):
    """
    Estado completo e imutável do sistema (Tensor 5D Abstrato).
    """
    grid: np.ndarray              # 3D Array (D, H, W) - Canal de Densidade
    supports: Tuple[Coord, ...]   # Coordenadas fixas
    loads: Tuple[Coord, ...]      # Coordenadas de carga + Vetores Força
    phase: PhaseType              # Fase atual
    
    # Metadados de Grafo (Cache)
    is_connected: bool            
    volume: int                   
    
    def __hash__(self):
        return hash(self.grid.tobytes())

class SimulationRecord(NamedTuple):
    """DTO para persistência no DB."""
    episode_id: str
    step: int
    phase: str
    state_bytes: bytes # Compressão do grid
    action_taken: str
    reward: float
    is_valid: bool
```

---

## 3. IMPLEMENTAÇÃO FUNCIONAL (CORE)

### 3.1. Máquina de Estados (Game Loop)

*   **`get_legal_actions(state: DesignState, policy_mask: np.ndarray) -> Tuple[GameAction, ...]`**
    *   **Fase 1 (PATHFINDING):**
        *   Objetivo: Conectar Sementes (Suportes) às Metas (Cargas).
        *   Ações: Predominantemente `ADD` em vizinhos de voxels existentes.
        *   Restrição: Não permitir `REMOVE` que quebre caminhos existentes.
    *   **Fase 2 (REFINEMENT):**
        *   Objetivo: Remover massa ineficiente.
        *   Ações: `REMOVE` em bordas (erosão) e `ADD` em áreas de alta tensão (reforço).
        *   **Pruning Neural:** Utilize a `policy_mask` (output da Swin-UNETR) para filtrar ações. Retorne apenas as Top-K ações mais prováveis sugeridas pela rede. Isso reduz o espaço de busca de $64^3$ para ~50 ações viáveis.

*   **`transition(state: DesignState, action: GameAction) -> DesignState`**
    *   Aplica a ação.
    *   Verifica conectividade (Union-Find ou BFS).
    *   **Trigger de Fase:** Se `state.phase == PATHFINDING` E `check_full_connectivity(new_grid)` for True $\to$ Muda para `REFINEMENT`.

### 3.2. MCTS Guiado por Rede (Neural MCTS)

*   **`select_action_mcts(root: Node, network_fn: Callable) -> GameAction`**
    *   Implemente um MCTS modificado.
    *   **Expansão:** Use a Policy Head da rede para priorizar quais nós filhos criar.
    *   **Simulação (Rollout):** Em vez de rollout aleatório, use a Value Head da rede para estimar o retorno futuro do estado folha.
    *   **Backprop:** Atualize os valores $Q(s, a)$ na árvore.

---

## 4. LOOP DE EXECUÇÃO

```python
def run_episode(episode_id: str, config: dict, agent_net, physics_oracle):
    state = init_state(config)
    
    while not is_terminal(state):
        # 1. Inferência Neural
        # O agente "olha" para o tabuleiro e sugere ações (Policy) e avalia a situação (Value)
        policy_logits, value_est = agent_net.predict(state.grid)
        
        # 2. Busca (MCTS)
        # O agente "pensa" simulando futuros possíveis, guiado pela intuição da rede
        best_action = run_mcts(state, policy_logits)
        
        # 3. Ação Real
        next_state = transition(state, best_action)
        
        # 4. Feedback Físico (Apenas Fase 2)
        reward = 0.0
        if next_state.phase == 'REFINEMENT':
            # O Solver FEM é o "Ground Truth" que treina a rede
            fem_result = physics_oracle.solve(next_state.grid)
            reward = calculate_reward(fem_result)
            
        # 5. Persistência (Experience Replay)
        save_step(episode_id, state, best_action, reward)
        
        state = next_state
```

---

## 5. REQUISITOS DE VALIDAÇÃO

1.  **Teste de Transição de Fase:** Crie um cenário onde falta apenas 1 voxel para conectar. Force a ação `ADD` nesse voxel. Verifique se o estado resultante tem `phase='REFINEMENT'`.
2.  **Teste de Conectividade Rápida:** O algoritmo de verificação de grafo deve rodar em < 10ms para grids $32^3$.
3.  **Teste de Pruning:** Verifique se `get_legal_actions` ignora voxels com probabilidade zero na máscara neural.
