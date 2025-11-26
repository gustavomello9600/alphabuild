# Contexto Crucial: Arquitetura Neural e Generalização

> [!IMPORTANT]
> Este documento define diretrizes **invioláveis** para a arquitetura da rede neural do AlphaBuilder v1.1.

## 1. Arquitetura Hierárquica (Swin-UNETR)
A escolha do **Swin-UNETR** não é estética, é funcional.
*   **Localidade:** Diferente do ViT Global (que mistura tudo com tudo), o Swin usa janelas locais. Isso é crucial para física, pois a tensão em um ponto depende primariamente de seus vizinhos imediatos (equações diferenciais parciais são locais).
*   **Hierarquia:** A rede deve entender estruturas finas (resolução alta) e fluxos de carga (resolução baixa). O design U-Net com Skip Connections é obrigatório para não perder a precisão geométrica na reconstrução.

## 2. Input 5D e Normalização
*   **Canais de Força:** Os canais de força ($F_x, F_y, F_z$) **DEVEM** ser normalizados. A rede não deve ver valores como "1000N", mas sim "1.0" (Carga Máxima Relativa).
*   **Invariância de Escala:** O modelo deve ser capaz de operar em grids maiores que o treino via *Sliding Window Inference* (nativo do Swin). Não use camadas `Dense` (Fully Connected) que fixem o tamanho do input global. Use apenas Convoluções e Atenção.

## 3. Output Dual-Head
A rede não prevê apenas "Fitness". Ela prevê **Ação** e **Valor**.
*   **Policy Head:** Deve ter a mesma resolução espacial do input. Cada voxel tem um score de "Quero ser Material" vs "Quero ser Ar".
*   **Value Head:** Deve agregar a informação global para dizer "Essa estrutura vai aguentar".

## 4. Ground Truth via FEM
Nunca treine a rede com recompensas heurísticas inventadas. A recompensa **TEM** que vir do Solver FEM (Compliance Real). Se a rede aprender uma física errada, o projeto falha.
