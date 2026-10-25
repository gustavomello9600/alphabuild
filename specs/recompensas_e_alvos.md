# Especificação de Recompensas e Alvos de Treinamento

## Visão Geral
Este documento define detalhadamente como a recompensa é calculada durante a exploração do MCTS (Self-Play) e como os alvos (targets) serão armazenados e reconstruídos para o treinamento supervisionado das redes neurais (Policy Head e Value Head).

---

## 1. Guia de Exploração do MCTS (Search Guidance)

Durante a busca MCTS, cada nó $s$ na árvore representa um estado de densidade da estrutura. Para guiar a expansão da árvore em direção a estruturas válidas, conectadas e otimizadas, o valor atribuído a cada nó, $V_{node}(s)$, combina a intuição da rede neural com correções analíticas imediatas.

### 1.1 Fórmula de Valor do Nó
O valor utilizado pelo algoritmo MCTS (PUCT) para selecionar e expandir nós é dado por:

$$ V_{node}(s) = V_{net}(s) + B_{conn}(s) - P_{ilhas}(s) $$

#### Componentes:

1.  **$V_{net}(s)$**: A saída bruta da Value Head da rede neural para o estado $s$.
    *   **Intervalo**: $[-1.0, 1.0]$.
    *   **Significado**: A estimativa "aprendida" da qualidade futura da estrutura. Em estados iniciais, funciona como um "instinto"; em estados finais, deve aproximar o Score FEM.

2.  **$B_{conn}(s)$**: Bônus de Conectividade Analítico (**Apenas Fase 1**).
    *   **Objetivo**: Incentivar a estrutura a crescer a partir dos suportes fixos até cobrir as regiões de carga.
    *   **Fórmula (Fase 1)**:
        Seja $n_{conn}$ o número de voxels de carga conectados ao suporte e $n_{total}$ o total de voxels onde a carga é aplicada.
        $$ f_c = \frac{n_{conn}}{n_{total}} $$
        $$ B_{conn}(s) = 0.5 \cdot f_c + \delta(f_c = 1.0) \cdot 0.5 $$
        *   **Progresso**: Se $f_c < 1.0$, o bônus escala linearmente até $0.5$.
        *   **Conclusão**: Se todas as cargas estão conectadas ($f_c=1.0$), adiciona-se $0.5$ extra, totalizando $1.0$.
    *   **Fase 2**: Nesta fase, **$B_{conn}(s) = 0$**. A conectividade é um pré-requisito rígido para estar na Fase 2. Se a estrutura perder a conectividade (qualquer voxel de carga desconectado de todos os suportes), aplica-se imediatamente a regra de **Falha Catastrófica** (Valor -1.0).

3.  **$P_{ilhas}(s)$**: Penalidade por Material Desconectado e Ilhas Múltiplas.
    *   **Objetivo**: Desencorajar a criação de voxels flutuantes e múltiplas ilhas desconexas, favorecendo uma topologia única e limpa.
    *   **Cálculo (Implementação Atual em `reward.py`)**:
        $$ P_{ilhas}(s) = \min(P_{max}, k_{ilha} \cdot (N_{ilhas} - 1) + k_{vol} \cdot \frac{Voxels_{soltos}}{Voxels_{total}}) $$
        *   **Parâmetros Atuais**:
            *   $k_{ilha} = 0.02$ (Custo por ilha extra)
            *   $k_{vol} = 0.1$ (Custo por fração de volume solto)
            *   $P_{max} = 0.3$ (Teto da penalidade)
    *   **Aplicação**: Esta penalidade é subtraída diretamente do $V_{net}$, reduzindo a atratividade de nós que geram "sujeira" no grid.

### 1.2 Regra de Falha Catastrófica (Terminal)
Se o estado violar as condições mínimas de integridade estrutural na Fase 2, a busca é interrompida para aquele ramo.

$$ V_{node}(s) = -1.0 $$

**Condições de Falha**:
*   **Perda de Conectividade de Carga**: Se qualquer voxel de carga deixar de estar conectado a pelo menos um suporte (a "Ilha Principal").
*   **Perda de Suporte**: Se a ilha principal não contiver mais os voxels de suporte necessários.
*   **Colapso Físico**: Embora a física (FEM) não seja calculada a cada nó do MCTS por custo, a perda de conectividade lógica é um proxy determinístico para colapso (matriz de rigidez singular).

---

## 2. Diferenças Críticas Entre Fases

### 2.1 Fase 1: Crescimento (Growth)
*   **Objetivo**: Construir topologia inicial.
*   **Ações em Lote (Micro-Batches)**:
    *   O agente pode selecionar **múltiplas ações simultaneamente**.
    *   **Configuração**: `batch_size = 32`.
    *   **Mecanismo**: O MCTS explora a árvore e identifica as ações mais promissoras (maior contagem de visitas). As Top-K ações compatíveis são aplicadas de uma vez só ao grid real para avançar o estado.
    *   **Justificativa**: Em uma resolução 64x32x8, adicionar 1 voxel por vez seria muito lento para preencher o volume inicial. O batching acelera a formação da estrutura grosseira.
*   **Avaliação**:
    *   Não há cálculo de FEM.
    *   A orientação vem do **Bônus de Conectividade** e da penalidade de ilhas.

### 2.2 Fase 2: Refinamento (Refinement)
*   **Transição**: Automática ao atingir $f_c=1.0$ (100% conexão).
*   **Objetivo**: Otimização topológica (remover material ineficiente, adicionar reforços).
*   **Ações Precisas**:
    *   **Configuração**: `batch_size = 8` (nominal), mas tendendo a passos unitários ou pequenos batches para precisão.
    *   O foco muda de "preenchimento rápido" para "decisão crítica".
*   **Avaliação com Física (FEM)**:
    *   A cada passo real do episódio (não nos nós do MCTS, mas no estado consolidado), executa-se o solver FEniCSx.
    *   **Isolamento da Ilha Principal**: Antes da simulação, o algoritmo (`evaluate_main_island`) identifica o maior componente conectado aos suportes e **descarta** voxels soltos. O FEM é rodado apenas na estrutura limpa.
    *   **Target de Valor**: O score FEM calculado serve de base para o alvo de treinamento da Value Head.

---

## 3. Alvos de Treinamento e Fórmulas de Score

### 3.1 Score Físico (FEM Score)
Utilizado para avaliar o estado na Fase 2 e compor o alvo da Value Head.

$$ S_{raw} = -\ln(\text{Compliance} + \epsilon) - \alpha \cdot \text{VolFrac} $$
$$ S_{FEM} = \tanh\left(\frac{S_{raw} - \mu}{\sigma}\right) $$

**Parâmetros Atuais (`harvest/config.py` e `reward.py`)**:
*   $\alpha = 12.0$: Peso da fração de volume. Calibrado para que o volume tenha impacto significativo (aprox. 20% mais que compliance).
*   $\mu = -6.65$: Média estimada de $S_{raw}$. Centraliza a distribuição para o $\tanh$.
*   $\sigma = 2.0$: Desvio padrão estimado. Controla a inclinação da curva de ativação.
*   $\epsilon = 1e-9$: Estabilidade numérica.

### 3.2 Alvo da Value Head ($z_{target}$)
Para treinar a rede a prever o valor correto, reconstruímos o alvo usando uma mistura temporal ($\lambda$-target simplificado).

$$ z_{target}(t) = \lambda_t \cdot S_{avaliado}(t) + (1 - \lambda_t) \cdot R_{final} $$

*   $\lambda_t = t / T_{max}$: Confiança crescente na física local conforme o episódio avança.
*   $S_{avaliado}(t)$:
    *   **Fase 2**: Valor de $S_{FEM}$ calculado para o passo $t$.
    *   **Fase 1**: Nesta fase, como a física não é avaliada e a estrutura está incompleta, o alvo de valor deve olhar exclusivamente para o futuro. Portanto, define-se que para $t$ na Fase 1, o alvo é simplesmente a Recompensa Final: $z_{target}(t) = R_{final}$. (Efetivamente tratando $\lambda_t = 0$ nesta fase).
*   $R_{final}$: Resultado final do episódio (Compliance Final Normalizada ou -1.0 em caso de falha).

---
