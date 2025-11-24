# 🤖 MISSÃO: AGENTE 03 (NEURAL_VISION)

**Função:** Arquiteto de Deep Learning (Vision Transformers).
**Paradigma:** Funcional Prático (Keras Functional API).
**Stack:** Python 3.10+, TensorFlow 2.x, NumPy.

---

## 1. CONTEXTO E OBJETIVO
Sua responsabilidade é escrever o código do módulo de Inteligência Artificial do **AlphaBuilder**. Este módulo será importado e utilizado pelo motor de estratégia (escrito por outro agente) para estimar a qualidade de designs estruturais.

**A Estratégia de Unificação Volumétrica:**
Você deve implementar uma arquitetura **3D Vision Transformer**.
*   O código não deve tratar problemas 2D e 3D de formas distintas.
*   O input para a rede neural será sempre um **Volume Euclidiano** $(D, H, W, C)$.
*   Problemas 2D (como a viga do paper) são tratados através da **extrusão** da malha 2D ao longo do eixo Z (profundidade). A "espessura" da peça define quantas fatias de voxels serão preenchidas no tensor de entrada.

---

## 2. ESTRUTURAS DE DADOS (INTERFACE)

Defina estas estruturas utilizando `dataclasses` imutáveis. Elas servirão como contrato de dados para quem importar seu módulo.

```python
from dataclasses import dataclass
import tensorflow as tf
import numpy as np

# Constantes do Espaço Canônico de Entrada
# O modelo sempre espera este shape fixo.
MAX_DEPTH = 16   # Espessura máxima em voxels
MAX_HEIGHT = 64
MAX_WIDTH = 128
CHANNELS = 3     # (1: Material, 2: Suportes, 3: Cargas)

@dataclass(frozen=True)
class VolumetricInput:
    """
    Container imutável para o tensor de entrada.
    Garante que o tensor esteja no formato (Batch, D, H, W, C).
    """
    tensor: tf.Tensor

@dataclass(frozen=True)
class TrainingBatch:
    """
    Par (Input, Target) para o loop de treinamento.
    """
    inputs: VolumetricInput
    targets: tf.Tensor  # Shape: (Batch, 1) -> Fitness Real
```

---

## 3. TAREFAS DE IMPLEMENTAÇÃO

Você deve produzir um arquivo (ou conjunto de arquivos) contendo as seguintes **Funções Puras** e **Construtores de Modelo**.

### 3.1. Pré-processamento: Extrusão e Tensorização
Quem usar seu código enviará matrizes NumPy 2D brutas e um inteiro de espessura. Você deve transformar isso no formato que a rede aceita.

*   **`prepare_volumetric_batch(grids_2d: list[np.ndarray], thicknesses: list[int]) -> VolumetricInput`**
    *   **Função Pura.**
    *   **Lógica de Extrusão:** Para cada grid $G$ de dimensão $(H, W)$ e espessura $T$:
        1.  Crie um volume $V$ de zeros com shape $(MAX\_DEPTH, MAX\_HEIGHT, MAX\_WIDTH, 3)$.
        2.  Repita a grid $G$ nas primeiras $T$ fatias do eixo de profundidade ($z=0$ até $z=T-1$).
        3.  Faça o mesmo para os canais de Suporte e Carga (assumindo que permeiam a espessura).
    *   **Normalização:** Garante `dtype=float32`.
    *   **Retorno:** Objeto `VolumetricInput` contendo o tensor em batch.

### 3.2. Arquitetura: 3D Vision Transformer
Implemente o construtor do modelo usando `tf.keras`.

*   **`build_3d_vit(patch_size: tuple[int, int, int] = (2, 8, 8)) -> tf.keras.Model`**
    *   **Entrada:** `(MAX_DEPTH, MAX_HEIGHT, MAX_WIDTH, 3)`.
    *   **Patching Volumétrico:** Utilize `Conv3D` com stride igual ao tamanho do kernel para criar os embeddings lineares dos patches cúbicos.
    *   **Positional Embeddings:** Implemente uma camada customizada ou use `Embedding` somado, para que o modelo entenda coordenadas $(z, y, x)$. *Isso é crucial para diferenciar uma camada superficial de uma interna.*
    *   **Transformer Block:** Implemente a sequência padrão (Norm -> Attention -> Norm -> MLP). Use conexões residuais.
    *   **Head:** Global Average Pooling 3D seguido de MLP denso para regressão escalar (1 saída).

### 3.3. API de Treinamento e Inferência
Exponha funções que abstraiam a complexidade do TensorFlow.

*   **`train_step(model, batch: TrainingBatch, optimizer, loss_fn) -> dict`**
    *   Decore com `@tf.function`.
    *   Executa um passo de gradiente descendente.
    *   Retorna dicionário de métricas (Loss, MAE).

*   **`predict_fitness(model, grids: list[np.ndarray], thicknesses: list[int]) -> np.ndarray`**
    *   **Esta é a função que o código do MCTS chamará.**
    *   Recebe dados brutos.
    *   Chama internamente `prepare_volumetric_batch`.
    *   Executa `model(input, training=False)`.
    *   Retorna array NumPy com os valores previstos.

---

## 4. REQUISITOS TÉCNICOS

1.  **Agnosticismo de Chamada:** Seu código não deve importar nada do módulo de Física ou do MCTS. Ele deve ser totalmente independente, dependendo apenas de NumPy e TensorFlow.
2.  **Tratamento de Padding:** O ViT deve ser robusto a voxels vazios. Como usamos *Zero Padding* para preencher o cubo até `MAX_DEPTH`, certifique-se de que o mecanismo de atenção ou a normalização não sejam desestabilizados por muitos zeros. O uso de `LayerNormalization` geralmente resolve isso bem.
3.  **Persistência do Modelo:** Inclua funções simples `save_model(model, path)` e `load_model(path)` usando o formato `.keras` nativo.

---

## 5. VALIDAÇÃO (Smoke Test)

No final do seu script (bloco `if __name__ == "__main__":`), escreva um teste de integração interna:

1.  **Mock Data:** Crie uma lista com 2 matrizes aleatórias $(64, 128)$. Defina espessuras `[1, 5]`.
2.  **Pipeline Check:** Chame `prepare_volumetric_batch` e verifique (`assert`) se o tensor resultante tem shape `(2, 16, 64, 128, 3)`.
3.  **Model Build:** Instancie o modelo com `build_3d_vit()`.
4.  **Forward Pass:** Passe o tensor pelo modelo e verifique se o output tem shape `(2, 1)`.
5.  **Output:** Imprima "Neural Module Ready. Input Shape: [...] Output: [...]".