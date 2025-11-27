# AlphaBuilder: Especificações de Interface da Aplicação (Deep Dive)

**Status:** Detailed Spec
**Versão:** 1.1
**Referência Técnica:** `alphabuilder_v1_1_spec.md`, `src/logic/runner.py`, `src/neural/model_arch.py`

Este documento traduz a arquitetura "Physics-Aware Swin-UNETR" e o fluxo de jogo bifásico em especificações de interface precisas.

---

## 1. Layout Principal (App Shell)

A estrutura global deve refletir a natureza "Cyberpunk Industrial" do Design System.

### 1.1. Sidebar de Navegação (Esquerda)
*   **Estilo:** `Glassmorphism` (Blur 20px, Bg `Matter Grey` @ 80%).
*   **Largura:** 64px (ícones) expandindo para 240px.
*   **Navegação:**
    *   `Dashboard`: Visão geral dos experimentos.
    *   `Lab`: O Workspace de Engenharia (Rota principal).
    *   `Data Lake`: Visualizador do banco de dados `training_data.db` (Replay de episódios).
    *   `Neural Net`: Status do treinamento e métricas da Swin-UNETR.

### 1.2. Header (Topo)
*   **Contexto:** Exibe o `Episode ID` atual (UUID) e o `Step Count`.
*   **Status do Engine:** Indicador de conexão com o backend Python (WebSocket/API).
    *   🟢 *Online (GPU A100)*
    *   🔴 *Offline*

---

## 2. Workspace de Engenharia (The Lab)

Interface focada na manipulação do Tensor 5D `(5, D, H, W)`.

### 2.1. Canvas 3D (Voxel Grid)
*   **Renderização:** Instanced Mesh de Cubos (Voxels).
*   **Visualização de Canais (Layers):**
    *   **Densidade ($\rho$):** Voxels Sólidos (`Matter Grey`).
    *   **Suportes (Mask):** Voxels Fixos (`Support Cyan`, Canal 1).
    *   **Cargas ($F$):** Vetores de Força (`Load Magenta`, Canais 2-4).
*   **Interação:**
    *   *Hover:* Mostra coordenadas `(x, y, z)` e valor de probabilidade da IA.
    *   *Click:* Adiciona/Remove voxel (se modo manual).

### 2.2. Toolbar de Definição de Problema (Input)
Permite desenhar as Condições de Contorno (BCs) no Grid vazio.
*   **Ferramentas:**
    *   `Draw Support`: Pinta voxels no Canal 1 (Dirichlet).
    *   `Draw Load`: Define vetores de força nos Canais 2-4 (Neumann).
        *   *Input:* Magnitude (N) e Direção (X, Y, Z).
    *   `Eraser`: Limpa voxels.
*   **Configuração do Grid:**
    *   Dropdown de Resolução: `32x16x16` (Debug), `64x32x32` (Padrão), `128x64x64` (High-Res).

### 2.3. Painel de Controle de Simulação (Right Sidebar)
Gerencia o loop `run_episode_v1_1`.

*   **Controle de Fase (Game State):**
    *   Indicador de Fase: **GROWTH** (Pathfinding) $\to$ **REFINEMENT** (Optimization).
    *   Botão `[▶ START OPTIMIZATION]`: Inicia o MCTS.
    *   Botão `[❚❚ PAUSE]`: Interrompe o loop.
    *   Botão `[⏭ STEP]`: Avança um passo do MCTS (Debug mode).

*   **Métricas em Tempo Real (Solver Feedback):**
    *   **Volume Fraction:** $\% Vol$.
    *   **Compliance:** $C$ (Energia de Deformação - Minimizando).
    *   **Max Displacement:** $U_{max}$ (Restrição).
    *   *Gráfico Sparkline:* Evolução da Compliance por Step.

---

## 3. Neural HUD (Visualização da IA)

Uma camada de sobreposição ("Heads-Up Display") que revela o "pensamento" da rede neural.

### 3.1. Policy Head Visualization ($\pi$)
*   **Heatmap 3D:** Renderiza uma nuvem de pontos translúcida sobre o grid.
    *   🔴 **Vermelho:** Alta probabilidade de `ADD` (Canal 0 da saída da Policy).
    *   🔵 **Azul:** Alta probabilidade de `REMOVE` (Canal 1 da saída da Policy).
*   **Objetivo:** Permitir que o engenheiro veja onde a IA *quer* colocar material antes de ela agir.

### 3.2. Value Head Monitor ($v$)
*   **Confidence Graph:** Gráfico de linha estilo EKG.
    *   Eixo Y: Probabilidade de Sucesso (0.0 a 1.0) ou Estimativa de Reward.
    *   *Insight:* Se a linha cair subitamente, a IA percebeu que cometeu um erro estrutural (ex: desconectou a carga).

### 3.3. MCTS Tree Explorer (Ghosting)
*   **Conceito:** Mostrar os "ramos" explorados pelo MCTS que foram descartados.
*   **Visual:** "Fantasmas" de voxels amarelos que aparecem e somem rapidamente ao redor da estrutura principal, indicando as simulações mentais do agente.

---

## 4. Integração Técnica (Data Binding)

Como o Frontend se conecta ao Backend (`runner.py`).

*   **Estado:** O Frontend recebe o objeto `GameState` serializado (JSON/Binary) a cada passo.
*   **Ações:** O Frontend envia comandos `START`, `PAUSE`, `RESET` para o controlador do episódio.
*   **Sincronia:**
    *   O `runner.py` roda em uma thread separada ou processo (via WebSocket).
    *   O Frontend é apenas um "Espelho" do estado atual do Python.
