# Decisão Técnica: Estratégia de Data Augmentation

**Projeto:** AlphaBuilder v3.1  
**Data:** 2025-12-05  
**Status:** Aguardando decisão do time especialista

---

## Contexto

O AlphaBuilder treina uma rede neural para otimização topológica 3D. Os dados de treinamento consistem em:
- **122.358 samples** (estados de otimização + políticas alvo + valores)
- Cada sample: tensor 3D de shape `(7, 64, 32, 8)` (~600KB)
- Banco de dados total: ~800MB

### Tipos de Augmentation Disponíveis

**1. Simetrias Físicas (preservam física):**
- `flip_y`: Espelhamento no eixo Y (inverte Fy)
- `rotate_90_z`: Rotação 90° no plano XY (rotaciona vetores Fx, Fy)
- Combinações: 4 rotações × 2 flips = **8 variantes** por sample

**2. Negative Sampling (gera exemplos de falha):**
- `erosion_attack`: Remove material, ensina a reparar (~100% para estados finais)
- `load_multiplier`: Aumenta força, simula sobrecarga (~5%)
- `sabotage`: Remove perto do suporte (~5%)
- `saboteur`: Remove cubo aleatório (~10%)

---

## Questão para Decisão

**Como aplicar as augmentations durante o treinamento?**

---

## Opção A: Augmentation Estocástica (Implementação Atual)

### Descrição
Cada sample é transformado **uma vez** por epoch com probabilidade:
- 50% de chance de flip_y
- 50% de chance de rotate_90 (se D=H)
- 5-10% de chance de negative sampling

### Fluxo por Epoch
```
Sample #0 → [sorteio] → 1 variante → GPU
Sample #1 → [sorteio] → 1 variante → GPU
...
Total: 122.358 samples/epoch
```

### Características
| Aspecto | Valor |
|---------|-------|
| Samples por epoch | 122.358 |
| Epochs para 30M samples | ~245 epochs |
| Memória/batch | Baixa (1 sample = 1 variante) |
| Variabilidade | Alta entre epochs |
| Regularização | Implícita (nunca vê mesma variante 2x) |

### Tempo de Treino Estimado (30 epochs)
- **Local (Iris Xe):** ~27 horas
- **Kaggle (T4 x2):** ~3 horas

---

## Opção B: Augmentation Multiplicativa

### Descrição
Cada sample gera **todas as variantes** em cada epoch:
- 8 variantes de simetria (4 rotações × 2 flips)
- +1 variante com negative sampling (quando aplicável)
- Total: ~9-10x multiplicação

### Fluxo por Epoch
```
Sample #0 → [gera 8 simetrias + negatives] → 9 variantes → buffer → GPU
Sample #1 → [gera 8 simetrias + negatives] → 9 variantes → buffer → GPU
...
Total: 122.358 × ~10 = ~1.2M samples/epoch
```

### Características
| Aspecto | Valor |
|---------|-------|
| Samples por epoch | ~1.200.000 |
| Epochs para 30M samples | ~25 epochs |
| Memória/batch | Alta (precisa bufferizar variantes) |
| Variabilidade | Determinística dentro do epoch |
| Cobertura | Garante exposição a todas as simetrias |

### Tempo de Treino Estimado (3 epochs equivalentes)
- **Local (Iris Xe):** ~27 horas (mesmo tempo, menos epochs)
- **Kaggle (T4 x2):** ~3 horas (mesmo tempo, menos epochs)

---

## Opção C: Híbrida

### Descrição
Gera **todas as simetrias** (8x) mas aplica negative sampling **estocasticamente** nas variantes.

### Fluxo por Epoch
```
Sample #0 → 8 simetrias → [5-10% negative em cada] → 8 variantes → GPU
...
Total: 122.358 × 8 = ~980K samples/epoch
```

### Características
- Garante cobertura de simetrias
- Mantém variabilidade no negative sampling
- Multiplicação previsível (8x)

---

## Comparação

| Critério | Opção A (Estocástica) | Opção B (Multiplicativa) | Opção C (Híbrida) |
|----------|----------------------|-------------------------|-------------------|
| Implementação | ✅ Já implementada | ⚠️ Requer mudanças | ⚠️ Requer mudanças |
| Memória GPU | ✅ Baixa | ❌ Alta | ⚠️ Média |
| Cobertura simetrias | ⚠️ Probabilística | ✅ Garantida | ✅ Garantida |
| Regularização | ✅ Forte | ⚠️ Fraca | ⚠️ Média |
| Reprodutibilidade | ❌ Baixa (aleatório) | ✅ Alta | ⚠️ Parcial |
| Overfitting risk | ✅ Baixo | ⚠️ Maior | ⚠️ Médio |

---

## Considerações Adicionais

### 1. Natureza do Problema
- Otimização topológica tem **simetrias físicas reais** (estrutura rotacionada é igualmente válida)
- O modelo DEVE ser invariante a essas simetrias
- Negative sampling ensina o que **NÃO** fazer (estruturas que falham)

### 2. Tamanho do Dataset
- 122K samples é relativamente pequeno para deep learning
- Multiplicar por 10x daria ~1.2M, ainda moderado
- GPT-3 treinou em ~300B tokens; ImageNet tem ~14M imagens

### 3. Arquitetura da Rede
- **SimpleBackbone:** 745K params - pode overfit facilmente
- **Swin-UNETR:** ~20M params - precisa mais dados

### 4. Literatura Relevante
- AlphaFold usa augmentation estocástica com múltiplas simetrias
- AlphaGo usa augmentation multiplicativa (todas rotações do tabuleiro)
- CNNs para imagens geralmente usam estocástica

---

## Perguntas para o Time Especialista

1. **Invariância a simetrias é crítica?** Se sim, Opção B ou C garantem que o modelo veja todas as orientações.

2. **Risco de overfitting?** Com 745K params e 122K samples, a Opção A oferece mais regularização.

3. **Tempo de treino é limitante?** Todas as opções têm tempo similar; a diferença está em epochs vs samples/epoch.

4. **Reprodutibilidade é importante?** Opção B é determinística; A e C têm componentes aleatórios.

---

## Recomendação Preliminar

**Para SimpleBackbone:** Opção A (estocástica) - menor risco de overfitting

**Para Swin-UNETR:** Opção C (híbrida) - mais dados para modelo maior, garante simetrias

---

## Decisão Solicitada

Por favor indiquem:
- [ ] Opção A: Manter implementação atual (estocástica)
- [ ] Opção B: Implementar multiplicativa (todas variantes)
- [ ] Opção C: Implementar híbrida (simetrias × negative estocástico)
- [ ] Outra abordagem: _____________________

**Justificativa:**

_____________________________________
