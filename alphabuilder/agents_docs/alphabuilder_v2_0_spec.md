# AlphaBuilder v2.0: Especificação de Sistema e Estratégia de Treinamento

**Status:** APROVADO PARA IMPLEMENTAÇÃO
**Versão:** 2.0.0 (Bézier Blueprint & Unconstrained MCTS)
**Contexto:** Otimização Topológica Generativa via AlphaZero
**Target Hardware:** NVIDIA A100/H100 (Treino), T4 (Inferência)

---

## 0. Resumo Executivo (Destaques da v2.0)
Esta versão introduz mudanças estruturais fundamentais para permitir visão de longo prazo e liberdade criativa ao agente:

1.  **Generator v2 (Paramétrico):** Substituição do A* por **Curvas de Bézier com Seção Retangular**. Isso cria *priors* estruturais mais naturais (semelhantes a vigas/ossos) e evita singularidades numéricas.
2.  **Estratégia Blueprint & Oracle:**
    *   **Fase 1:** A rede aprende a alucinar a estrutura completa (Blueprint) de uma vez, mas o MCTS constrói passo-a-passo.
    *   **Value Oracle:** A Value Head aprende a prever a *performance final pós-otimização* desde o estado vazio, guiando a construção inicial para formas otimizáveis.
3.  **Unconstrained MCTS:** O MCTS não tem mais travas de fase ("só pode adicionar" ou "só pode remover"). Ele é livre para adicionar ou remover voxels a qualquer momento, permitindo correção de erros e reforço estrutural dinâmico.
4.  **Normalização Robusta:** Uso de **Instance Normalization** (devido a batches 3D pequenos) e Policy Head de 2 canais independentes.

---

## 1. Pipeline de Geração de Dados (The Data Factory v2)

O objetivo é gerar um dataset que ensine "Design Generativo" (Fase 1) e "Refinamento Cirúrgico" (Fase 2).

### 1.1. Configuração do Cenário (Cantilever Paramétrico)
*   **Domínio:** Grid `64 (X) x 32 (Y) x 8 (Z)`.
*   **Engaste (Suporte):** Face $X=0$ completa (travada).
*   **Carga (P):**
    *   Posição aleatória: $x \in [32, 63]$, $y \in [0, 31]$.
    *   Distribuição: Linear ao longo de $Z \in [z, z+3]$ na direção $-Y$.
    *   *Nota:* A carga não é pontual, é uma "linha de carga" para evitar singularidade.

### 1.2. Geração da Geometria Base (Ground Truth Fase 1)
O algoritmo deve gerar estruturas procedurais válidas:

1.  **Traçado:** Sortear $N \in [1, 3]$ curvas de **Bézier Quadráticas** conectando a parede $X=0$ à linha de carga.
    *   *Ruído:* Adicionar perturbação estocástica (ruído gaussiano) aos pontos de controle intermediários para variar a curvatura.
2.  **Voxelização Retangular Variável:**
    *   Ao longo da curva $t \in [0, 1]$, a seção transversal é um retângulo $W(t) \times H(t)$.
    *   **Interpolação Linear:**
        *   Base ($t=0$): $W_{base} \in [4, 8]$, $H_{base} \in [4, 16]$.
        *   Ponta ($t=1$): $W_{tip} \in [2, 4]$, $H_{tip} \in [2, 4]$.
    *   O volume resultante dessa voxelização é a máscara `V_constructed`.

### 1.3. Otimização Física (Ground Truth Fase 2)
1.  Utilizar `V_constructed` como densidade inicial.
2.  Rodar Solver SIMP até convergência.
3.  Calcular **Score Final ($S_{final}$)** baseado na Compliance Final e Volume Final.
    *   Este $S_{final}$ será usado como *Target Value* para **todos** os registros deste episódio (Retropropagação do Oráculo).

---

## 2. Especificação do Dataset (Schema & Slicing)

O registro no banco de dados deve capturar a evolução temporal.

### 2.1. Registros Tipo A - Fase 1 (Slicing de Crescimento)
Para cada episódio, gerar **exatamente 50 registros** simulando o crescimento da estrutura.
*   **Método:** Calcular distância geodésica dos voxels de `V_constructed` em relação à base. Fatiar em 50 passos cumulativos ($2\%, 4\%, \dots, 100\%$).
*   **Input State:** A fatia parcial da estrutura.
*   **Target Policy (Canal Add):** A máscara completa `V_constructed` (A rede deve aprender a completar a imagem).
*   **Target Policy (Canal Remove):** Zeros (Máscara vazia).
*   **Target Value:** $S_{final}$ (O Score do SIMP futuro).

### 2.2. Registros Tipo B - Fase 2 (Histórico SIMP)
Para cada episódio, extrair frames do histórico do SIMP.
*   **Input State:** Densidade intermediária do SIMP.
*   **Target Policy (Canal Remove):** Diferença temporal ($State_t - State_{t+k}$).
*   **Target Policy (Canal Add):** Zeros (exceto se o SIMP adicionar material, o que é raro mas deve ser capturado).
*   **Target Value:** $S_{final}$.

---

## 3. Arquitetura Neural e Treinamento

### 3.1. Modelo (Swin-UNETR Modificado)
*   **Input (5 Canais):** $\rho$, Mask, $F_x, F_y, F_z$.
*   **Backbone:** Swin Transformer 3D.
*   **Normalization Layer:** **Instance Normalization (IN)**.
    *   *Motivo:* Batch Normalization é instável com os pequenos batch sizes (ex: 2 ou 4) típicos de treino 3D. IN é o padrão para segmentação volumétrica.
*   **Policy Head (2 Canais):**
    *   Canal 0: Logits de Adição.
    *   Canal 1: Logits de Remoção.
*   **Value Head (1 Escalar):** Tanh $[-1, 1]$.

### 3.2. Physics-Aware Data Augmentation
O `DataLoader` deve aplicar transformações geométricas aleatórias (Rotação 90°, Flip) em tempo de execução.
*   **Regra de Ouro:** Ao rotacionar o grid geométrico, os canais de vetores de força ($F_x, F_y, F_z$) devem sofrer a rotação vetorial correspondente.

### 3.3. Loss Function & Targets
*   **Value Normalization:** Converter Compliance (que varia exponencialmente) para espaço linear.
    $$ V_{target} = \tanh\left( \frac{-\log(Compliance) - \alpha \cdot Volume - \mu}{\sigma} \right) $$
    *(Onde $\mu, \sigma$ são médias/std do dataset)*.
*   **Policy Loss Masking:**
    *   Se o registro é Tipo A (Fase 1), a Loss só incide sobre o Canal Add (Canal Remove é ignorado).
    *   Se o registro é Tipo B (Fase 2), a Loss incide sobre ambos (mas predominantemente Remove).

---

## 4. Dinâmica de Jogo e MCTS (Inferência)

O AlphaBuilder opera em um ciclo contínuo sem travas rígidas de ação.

### 4.1. Regras de Transição
*   **Estado 1 (Desconectado):** O objetivo é conectar a carga a **pelo menos um** suporte.
*   **Estado 2 (Conectado):** O objetivo é minimizar volume sujeito a $U_{max} \le Limit$.

### 4.2. Execução Unconstrained (Liberdade Total)
O MCTS recebe os 2 canais da Policy Head.
*   Ele seleciona as $Top-K$ ações mais promissoras combinadas (pode ser "Adicionar em X" e "Remover em Y" na mesma lista de candidatos).
*   **Decisão:** O MCTS escolhe baseado no Score PUCT ($Q + U$).
    *   Se adicionar um voxel aumenta a probabilidade de conexão (Fase 1), o Value Head sobe, e a ação é escolhida.
    *   Se remover um voxel economiza peso sem violar $U_{max}$ (Fase 2), o Value Head sobe, e a ação é escolhida.
    *   Se remover um voxel quebra a peça, o Value Head cai (-1.0), e a ação é evitada.

### 4.3. Micro-Batches
Para acelerar a inferência sem perder precisão:
*   O MCTS não executa 1 voxel por vez.
*   **Tamanho do Batch:** **8 Voxels**.
*   O agente compromete-se com as 8 ações mais prováveis de uma vez, roda a verificação física/conectividade, e então reavalia.

### 4.4. Função de Recompensa (O Juiz)
*   **Fase 1:** Penalidade fixa por passo (-0.01) para incentivar velocidade.
*   **Fase 2:** Recompensa proporcional à redução de volume.
*   **Hard Constraint:** Se $U_{max} > Limit \implies R = -1.0$ (Game Over).

---

## 5. Requisitos Funcionais Recapitulados

*   **RF-01 (Conexão Relaxada):** Conexão a 1 suporte basta para iniciar Fase 2.
*   **RF-02 (Augmentation Físico):** Rotação solidária de vetores de força.
*   **RF-03 (Blueprint Learning):** Treino Fase 1 via segmentação completa (Image Completion), não passo-a-passo.
*   **RF-04 (Oracle Value):** Value Head treinado sempre com o Score Final do episódio.
*   **RF-05 (Robustez de Batch):** Uso de Instance Norm na rede.
