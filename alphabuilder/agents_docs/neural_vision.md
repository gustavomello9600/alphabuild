# 🤖 MISSÃO: AGENTE 03 (NEURAL_VISION)

**Função:** Arquiteto de Deep Learning (Swin-UNETR & Physics-Aware AI).
**Paradigma:** Funcional Prático (PyTorch / MONAI).
**Stack:** Python 3.10+, PyTorch, MONAI (Medical Open Network for AI), Einops.

---

## 1. CONTEXTO E OBJETIVO
Sua responsabilidade é implementar o "Cérebro" do **AlphaBuilder v1.1**.
Você deve abandonar a abordagem ViT pura e implementar uma arquitetura **Swin-UNETR (Swin Transformer U-Net)**.

**Por que Swin-UNETR?**
O problema de otimização topológica exige duas competências simultâneas:
1.  **Visão Global (Encoder Swin):** Entender o fluxo de carga macroscópico (ex: "Isso é uma viga em balanço, preciso reforçar a base").
2.  **Precisão Local (Decoder U-Net):** Decidir exatamente qual voxel da borda remover para alisar a estrutura sem desconectar.

---

## 2. ESTRUTURAS DE DADOS (INTERFACE)

Defina estas estruturas utilizando `dataclasses` imutáveis.

```python
from dataclasses import dataclass
import torch
from typing import Tuple

# Constantes do Espaço Canônico de Entrada
# Tensor 5D: (Batch, Channels, Depth, Height, Width)
INPUT_SHAPE = (5, 16, 64, 64) # Exemplo, ajustável via config
CHANNELS = 5 

# Canal 0: Densidade (0=Ar, 1=Material)
# Canal 1: Máscara de Suporte (1=Fixo)
# Canal 2: Força X (Normalizada)
# Canal 3: Força Y (Normalizada)
# Canal 4: Força Z (Normalizada)

@dataclass(frozen=True)
class VolumetricInput:
    """
    Container imutável para o tensor de entrada.
    Shape esperado: (Batch, 5, D, H, W)
    """
    tensor: torch.Tensor

@dataclass(frozen=True)
class ModelOutput:
    """
    Saída Dual-Head da Rede.
    """
    policy_logits: torch.Tensor  # Shape: (Batch, 2, D, H, W) -> [Add_Score, Remove_Score]
    value_pred: torch.Tensor     # Shape: (Batch, 1) -> Probabilidade de Sucesso / Compliance Estimado
```

---

## 3. TAREFAS DE IMPLEMENTAÇÃO

### 3.1. Arquitetura: Physics-Aware Swin-UNETR
Implemente o modelo utilizando `monai.networks.nets.SwinUNETR` como base ou implemente do zero se precisar de customização fina nos embeddings de força.

*   **`build_swin_unetr(input_shape: Tuple[int, ...]) -> torch.nn.Module`**
    *   **Encoder (Swin Transformer):**
        *   Utiliza *Shifted Windows* para capturar dependências de longo alcance com complexidade linear.
        *   Extrai features em 4 escalas hierárquicas.
    *   **Bottleneck:**
        *   Representação latente compacta da física global do problema.
    *   **Decoder (U-Net style):**
        *   Reconstrói a resolução espacial usando Deconvoluções (Transpose Conv).
        *   **Skip Connections:** Concatena features do Encoder para recuperar detalhes geométricos perdidos.
    *   **Heads (Saídas):**
        1.  **Policy Head ($1 \times 1 \times 1$ Conv):** Produz 2 canais de saída (Logits para Ação ADD e Ação REMOVE) com a mesma resolução espacial do input.
        2.  **Value Head (MLP no Bottleneck):** Global Average Pooling sobre o bottleneck -> MLP -> Escalar.

### 3.2. Pré-processamento de Forças
A rede deve ser invariante à magnitude absoluta das forças, mas sensível à direção e proporção.

*   **`normalize_forces(force_tensor: torch.Tensor) -> torch.Tensor`**
    *   Normaliza os canais de força (2, 3, 4) para o intervalo $[-1, 1]$ ou $[0, 1]$ baseando-se na força máxima presente no grid.
    *   Isso garante que uma carga de 100N e uma de 1000N gerem a "mesma" topologia relativa se o material for linear elástico.

### 3.3. API de Inferência
Exponha uma função simples para o MCTS.

*   **`predict_action_value(model, grid_tensor: torch.Tensor) -> ModelOutput`**
    *   Recebe o grid bruto.
    *   Executa o forward pass.
    *   Aplica `softmax` na Policy Head (opcional, dependendo de como o MCTS consome).
    *   Retorna `ModelOutput`.

---

## 4. REQUISITOS TÉCNICOS

1.  **Framework:** Migração para **PyTorch** é recomendada dada a disponibilidade de implementações Swin-UNETR robustas (MONAI). Se preferir TensorFlow, terá que implementar Swin 3D do zero.
2.  **Eficiência 3D:** Utilize operações `Conv3d` e `Attention` otimizadas. O grid $64^3$ é pesado.
3.  **Mixed Precision:** O modelo deve suportar treinamento em `float16` (AMP) para caber na memória da GPU A100/T4.

---

## 5. VALIDAÇÃO (Smoke Test)

No `if __name__ == "__main__":`:
1.  Instancie o modelo `SwinUNETR` com input channels=5.
2.  Crie um tensor aleatório `(1, 5, 32, 32, 32)`.
3.  Faça um forward pass.
4.  Verifique se `policy_logits.shape == (1, 2, 32, 32, 32)`.
5.  Verifique se `value_pred.shape == (1, 1)`.
6.  Imprima: "Swin-UNETR Architecture Ready."