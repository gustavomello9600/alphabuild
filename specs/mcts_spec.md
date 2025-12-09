# AlphaBuilder v3.1: Especificação Técnica do MCTS

**Componente:** Monte Carlo Tree Search (Engine de Inferência)
**Papel:** Policy Improvement Operator & Batch Builder
**Input:** Estado $S_t$ (Grid + Metadata)
**Output:** Micro-Batch de Ações $\vec{a} = \{a_1, a_2, ..., a_k\}$

---

## 1. Estrutura de Dados e Nós

A árvore deve ser leve, armazenando apenas o necessário para a estatística UCB. Diferente do xadrez, não armazenamos o estado completo do grid em cada nó (seria inviável em memória), apenas o hash ou o delta da ação.

### 1.1. Classe `MCTSNode`
Cada nó representa um estado alcançado após uma sequência de edições unitárias.

*   `visit_count (N)`: Inteiro. Quantas vezes passamos por aqui.
*   `value_sum (W)`: Float. Soma dos valores $V$ retornados pela rede.
*   `mean_value (Q)`: Float ($W/N$). Qualidade média esperada da sub-árvore (Range $[-1, 1]$).
*   `prior (P)`: Float. Probabilidade inicial dada pela Policy Head da rede.
*   `parent`: Referência ao nó pai.
*   `action_to_parent`: A coordenada $(c, z, y, x)$ que levou a este nó.
*   `children`: Dicionário `{action: MCTSNode}`.
*   `is_expanded`: Booleano.

---

## 2. O Ciclo de Simulação (Kernel)

Para cada passo real do jogo, o MCTS executa **80 Simulações**.
Cada simulação segue rigorosamente 4 fases:

### Fase 1: Seleção (Selection)
Partindo da Raiz ($S_0$), descemos a árvore escolhendo o filho que maximiza a fórmula **PUCT (Predictor + Upper Confidence Bound applied to Trees)** até encontrar uma folha não expandida.

$$ \text{PUCT}(s, a) = Q(s, a) + C_{puct} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} $$

*   **$Q(s, a)$:** Exploração (Qualidade média da física).
*   **$P(s, a)$:** Exploração (Intuição da rede).
*   **$N(s, a)$:** Contagem de visitas.
*   **$C_{puct}$:** Constante de exploração (Recomendado: **1.25** a **1.5**).

> **Regra de Ouro:** A descida na árvore considera ações **Unitárias**. Não simulamos batches dentro da árvore. Cada aresta é 1 voxel.

### Fase 2: Expansão e Mascaramento (Expansion)
Ao chegar em um nó folha $S_L$:
1.  Verificar se é terminal (Physics Fail ou Max Steps). Se sim, $V = -1$ ou $0$.
2.  Se não, gerar as **Ações Válidas (Legal Moves)**:
    *   *Adição:* `Dilate(Grid) - Grid` (Apenas fronteira).
    *   *Remoção:* `Grid` (Apenas material existente).
    *   *Top-K Pruning:* Dentre as válidas, considerar apenas as ações que a rede neural (na etapa seguinte) der probabilidade não-nula ou, para otimização, pré-filtrar.

### Fase 3: Avaliação Neural (Evaluation)
O estado $S_L$ é enviado para a Rede Neural (SimpleBackbone ou Swin-UNETR).
*   **Input:** Tensor 7-Canais do estado $S_L$.
*   **Outputs:**
    *   `Logits_Policy` ($L$): Mapa denso de probabilidades.
    *   `Value` ($V$): Escalar $[-1, 1]$.

**Pós-Processamento da Policy:**
1.  Aplicar **Máscara de Legal Moves** nos Logits: $L_{masked} = L + (1-\text{Mask}) \cdot (-\infty)$.
2.  Calcular Softmax: $P = \text{Softmax}(L_{masked})$.
3.  Armazenar as probabilidades $P$ nos filhos do nó $S_L$.

### Fase 4: Backup (Backpropagation)
Propagar o valor $V$ de volta para a Raiz.
Para cada nó $i$ no caminho:
*   $N_i \leftarrow N_i + 1$
*   $W_i \leftarrow W_i + V$
*   $Q_i \leftarrow W_i / N_i$

> **Nota sobre Inversão de Perspectiva:** Como este é um jogo "single player" cooperativo (não há adversário), não invertemos o sinal do valor $V$ a cada nível. Todos os nós buscam maximizar o Score.

---

## 3. Decisão e Construção do Micro-Batch

Após as 80 simulações, a árvore na Raiz está populada com estatísticas ricas. O momento de decisão transforma essas estatísticas em ação física.

### 3.1. Estratégia de Seleção: "Top-K Greedy"
Não amostramos estocasticamente (exceto nas primeiras jogadas do Warm-up para diversidade). Usamos a contagem de visitas ($N$) como proxy de confiança.

1.  Listar todas as ações filhas da Raiz.
2.  Ordenar decrescentemente por $N$.
3.  Selecionar as **Top-8 Ações**.

$$ \text{Batch} = \{ a \in \text{Children}(Root) \mid \text{Rank}(N(a)) \le 8 \} $$

### 3.2. Tratamento de Conflitos e Exceções
*   **Fim de Jogo (Passividade):** Se o nó raiz tiver menos de 8 filhos válidos (ex: tabuleiro cheio/vazio e sem opções), ou se o nó mais visitado for "NO-OP", o batch é reduzido ou o jogo encerra.
*   **Redundância:** Em teoria, a ação 1 e a ação 2 poderiam ser conflitantes (ex: adicionar e remover o mesmo voxel). No AlphaBuilder v3.1, como separamos canais de Add/Rem e a máscara é rígida, isso é impossível. O batch é sempre geometricamente consistente.

---

## 4. Otimização de Implementação (Performance)

Para garantir os tempos estimados (40s na T4), o MCTS deve ser implementado com eficiência.

### 4.1. Cache de Rede Neural
*   Muitos caminhos na árvore se repetem ou estados são muito similares.
*   Manter um **LRU Cache** (Least Recently Used) para as inferências da rede neural. Se o MCTS encontrar um estado que já avaliou recentemente, usa o cache.

### 4.2. Batching de Inferência (Opcional Avançado)
Se estiver rodando múltiplos episódios em paralelo (ex: 8 workers):
1.  Os workers descem a árvore (Selection) em paralelo na CPU.
2.  Quando todos travam na fase de *Evaluation*, o despachante coleta os 8 estados.
3.  Monta um Tensor `(8, 7, D, H, W)`.
4.  Executa a inferência na GPU **uma única vez**.
5.  Distribui os resultados de volta para os workers.
*Isso é crucial para o Swin-UNETR.*

---

## 5. Algoritmo Resumido (Pseudocódigo)

```python
class AlphaBuilderMCTS:
    def __init__(self, network, simulations=80, batch_size=8):
        self.net = network
        self.sims = simulations
        self.k = batch_size
        self.c_puct = 1.25

    def get_action_batch(self, root_state):
        root = MCTSNode(state=root_state)
        
        # 1. Expandir raiz (primeira inferência)
        policy, value = self.net.predict(root_state)
        valid_moves = self.get_legal_moves(root_state)
        root.expand(policy, valid_moves)
        
        # 2. Loop de Simulações
        for _ in range(self.sims):
            node = root
            path = [node]
            
            # Selection
            while node.is_expanded:
                action, node = node.select_child(self.c_puct)
                path.append(node)
                
            # Expansion & Evaluation
            leaf_state = self.apply_path_to_state(root_state, path)
            if not self.is_terminal(leaf_state):
                policy, value = self.net.predict(leaf_state)
                valid_moves = self.get_legal_moves(leaf_state)
                node.expand(policy, valid_moves)
            else:
                value = self.get_terminal_reward(leaf_state)
                
            # Backup
            for n in reversed(path):
                n.update(value)
                
        # 3. Decision (Micro-Batch)
        # Ordena filhos por visitas (N)
        sorted_actions = sorted(root.children.items(), key=lambda x: x[1].visit_count, reverse=True)
        
        # Pega os Top-8
        micro_batch = [action for action, node in sorted_actions[:self.k]]
        
        return micro_batch, root.visit_count_distribution
```

---

## 6. Critérios de Sucesso do MCTS

O MCTS estará funcionando corretamente se observarmos:
1.  **Convergência de Seleção:** Nas primeiras 10 simulações, a exploração é alta. Nas últimas 10, o MCTS insiste repetidamente nos mesmos nós (Top-8).
2.  **Refutação da Rede:** Em ~5% a 10% dos casos, a ação mais visitada pelo MCTS **não** deve ser a ação com maior probabilidade inicial da rede ($P$). Isso prova que o *Value Head* corrigiu a *Policy Head* via simulação.
3.  **Estabilidade Física:** O Micro-Batch gerado raramente causa "Game Over" imediato por colapso físico, pois o Value Head penalizou ramos instáveis durante a busca.

---

## 7. Função de Recompensa

Para garantir consistência matemática entre a rede neural (treinada no Warm-up) e o MCTS (Self-Play), a função de recompensa espelha a normalização do treino.

### 7.1. Constantes de Normalização (Hard-Coded)

```python
MU_SCORE = -6.65      # Média do Score "Cru" no dataset
SIGMA_SCORE = 2.0     # Desvio Padrão para suavizar a Tanh
ALPHA_VOL = 12.0      # Peso do Volume (20% mais importante que compliance logarítmico)
EPSILON = 1e-9        # Estabilidade numérica
```

### 7.2. Score Físico Cru ($S_{raw}$)

$$ S_{raw} = -\ln(C + \epsilon) - \alpha \cdot V_{frac} $$

*   **$C$ (Compliance):** Energia de deformação (Joules). Quanto menor, melhor.
*   **$V_{frac}$ (Volume Fraction):** Razão voxels/total ($0.0$ a $1.0$). Quanto menor, melhor.

### 7.3. Recompensa Normalizada ($R$)

$$ R = \tanh\left( \frac{S_{raw} - \mu}{\sigma} \right) $$

**Casos Especiais:**
*   **Colapso/Desconexão:** $R = -1.0$ (Penalidade Máxima)
*   **Estrutura Média:** $R \approx 0.0$ (quando $S_{raw} \approx \mu$)
*   **Estrutura Excelente:** $R \rightarrow +1.0$

### 7.4. Estratégia por Fase

#### Fase 1: Growth (Construção)
Física inválida (estrutura desconectada). Não podemos calcular Compliance.

*   **Passo Normal:** $R = 0$ (ou penalidade de vida $-0.01$). O valor vem de $V_{net}(s)$.
*   **Conexão Bem-Sucedida:** $R = +0.5$. (Bônus por conectar suporte à carga).
*   **Falha (Max Steps):** $R = -1.0$.

#### Fase 2: Refinement (Otimização)
Física válida. MCTS compara predição com realidade física.

*   **Intra-Árvore (Simulação):** Usa $V_{net}(s)$ (FEM é caro demais).
*   **Estado Terminal Óbvio:** Checar conectividade (rápido). Se desconectado, $R = -1.0$.
*   **Após Micro-Batch:** Rodar FEM, calcular $R = \text{calculate\_reward}(...)$.

---

## 8. Configuração por Fase

Os parâmetros do MCTS variam conforme a fase do episódio:

| Parâmetro | Fase 1 (Growth) | Fase 2 (Refinement) |
|-----------|-----------------|---------------------|
| **Simulações** | 320 | 80 |
| **Micro-Batch** | 32 ações | 8 ações |
| **c_puct** | 1.5 (mais exploração) | 1.25 (mais exploração) |
| **Legal Moves** | Apenas ADD (fronteira) | ADD + REMOVE |
| **Terminal Check** | Conectividade | Conectividade + Volume |

### 8.1. Transição de Fase

A transição de Fase 1 → Fase 2 ocorre quando:
1.  A estrutura conecta o suporte (X=0) à região de aplicação de carga
2.  Verificado via `check_structure_connectivity(density, load_config)`

### 8.2. Condições de Término do Episódio

*   **Sucesso:** Volume frac < target E compliance estável por N passos
*   **Falha:** Max steps atingido (600) OU estrutura desconectada na Fase 2
*   **Colapso:** Deslocamento máximo > limite (FEM retornou erro)
