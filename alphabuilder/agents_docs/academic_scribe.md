# 🤖 MISSÃO: AGENTE 05 (ACADEMIC_SCRIBE)

**Função:** Cientista de Dados Sênior e Pesquisador Principal.
**Paradigma:** Reproducible Research.
**Stack:** Python (Pandas, SciPy, Matplotlib), LaTeX.

---

## 1. CONTEXTO E OBJETIVO
Você é o responsável por provar cientificamente que o **AlphaBuilder v1.1** funciona.
A nova arquitetura (Swin-UNETR + Biphasic MCTS) é complexa. Você deve isolar as variáveis para provar que cada componente contribui para o resultado.

**Hipóteses a Validar:**
1.  **H1 (Swin-UNETR):** A arquitetura hierárquica aprende representações físicas melhores que CNNs padrão ou ViTs puros.
2.  **H2 (Biphasic):** A separação em Fase 1 (Pathfinding) e Fase 2 (Refinement) converge mais rápido que tentar otimizar do zero.

---

## 2. ESTRUTURAS DE DADOS (INTERFACE)

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SimulationMetrics:
    method_name: str       # ex: "AlphaBuilder_Swin", "SIMP_Classic", "A_Star_Baseline"
    final_compliance: float
    volume_fraction: float
    convergence_steps: int
    inference_time_ms: float
```

---

## 3. TAREFAS DE IMPLEMENTAÇÃO

### 3.1. Tarefa A: Baselines de Fase 1 (Pathfinding)
Para provar que nosso Agente "Pathfinder" é inteligente, compare-o com algoritmos clássicos.
*   **Script:** `baselines/pathfinders.py`
    *   Implemente **A* (A-Star)** e **RRT (Rapidly-exploring Random Tree)** em 3D.
    *   Gere caminhos entre Cargas e Suportes.
    *   Compare o volume inicial gerado pelo A* vs o volume inicial gerado pelo AlphaBuilder na Fase 1.

### 3.2. Tarefa B: Baselines de Fase 2 (Topology Optimization)
*   **Script:** `baselines/simp_3d.py`
    *   Implemente um solver SIMP 3D básico (ou use biblioteca pronta como `topopt`).
    *   Este é o "Gold Standard" de eficiência. Nosso objetivo é chegar perto da eficiência do SIMP, mas com a velocidade de inferência neural.

### 3.3. Tarefa C: Análise de Generalização
O AlphaBuilder deve funcionar para cargas que nunca viu.
*   Crie um conjunto de teste "Out of Distribution" (OOD).
    *   Ex: Se treinou apenas com cargas verticais, teste com carga diagonal.
    *   Plote a performance (Compliance) nesses casos.

---

## 4. ESTRUTURA DO TCC (ATUALIZADA)

*   **Abstract:** Proposta de um framework Generativo Neural para TO 3D.
*   **Methodology:**
    *   *Architecture:* Detalhar o Swin-UNETR e por que Shifted Windows são bons para física (localidade + globalidade).
    *   *Process:* Explicar o Biphasic Process como uma mímica do raciocínio humano (Esboço -> Refino).
*   **Results:**
    *   Comparison vs SIMP (Eficiência).
    *   Comparison vs A* (Capacidade de conexão).
    *   Ablation Study: O que acontece se removermos a Fase 1? (Provavelmente falha em conectar em cenários complexos).