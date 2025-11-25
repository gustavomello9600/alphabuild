# Contexto Crucial: Arquitetura Neural e Generalização

> [!IMPORTANT]
> Este documento define diretrizes **invioláveis** para a arquitetura da rede neural do AlphaBuilder. Qualquer alteração futura deve respeitar estes princípios para garantir a generalização física e a robustez do modelo.

## 1. O Output da Rede é Físico, não "Fitness"
A rede neural **NÃO DEVE** prever uma pontuação de "fitness" composta (ex: $w_1 \cdot C + w_2 \cdot V$).
*   **Motivo:** Pesos de penalidade mudam conforme o projeto. Se a rede aprender uma função composta, ela se torna inútil quando o usuário muda a prioridade (ex: priorizar leveza sobre rigidez).
*   **Diretriz:** O output deve ser **Max Displacement** (Deslocamento Máximo Absoluto) ou **Compliance**.
*   **Implementação:**
    ```python
    # Correto
    outputs = layers.Dense(1, activation='softplus', name="max_displacement_output")
    
    # Incorreto
    # outputs = layers.Dense(1, name="fitness_score")
    ```

## 2. Generalização via Positional Embeddings
Redes neurais em física devem generalizar para diferentes resoluções e não apenas memorizar posições absolutas de pixels.
*   **Proibido:** `layers.Embedding` (Learned Embeddings). Eles fixam o tamanho da sequência e não extrapolam.
*   **Obrigatório:** **Sinusoidal Positional Embeddings** (Fixos). Eles codificam a posição relativa de forma contínua, permitindo que o modelo entenda "distância" melhor que "índice".

## 3. Agregação Robusta (CLS Token vs. GAP)
Para prever falhas estruturais (como o deslocamento máximo), a média global é perigosa.
*   **Problema do GAP (Global Average Pooling):** Se 99% da estrutura está rígida mas 1% está colapsando, a média esconde o colapso.
*   **Solução:** Usar **CLS Token** (padrão ViT) ou **Global Max Pooling**.
    *   O CLS Token permite que a rede aprenda *como* agregar a informação de toda a estrutura em um único vetor latente, focando nas áreas críticas (atenção) em vez de fazer uma média cega.

## 4. Rotação e Aumentação de Dados
*   **Atenção:** Não rotacionar cegamente as entradas. A gravidade ($g$) e as Cargas ($F$) são vetores.
*   **Regra:** Se rotacionar a geometria (grid), **DEVE-SE** rotacionar também os vetores de condição de contorno nos canais de input. Caso contrário, a rede aprenderá uma física impossível.
