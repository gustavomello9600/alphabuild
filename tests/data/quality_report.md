# 📊 Relatório de Qualidade de Dados de Treino

**Gerado em:** 2025-12-02 01:53:14
**Banco de Dados:** `episodios_de_testes_de_integracao.db`

---

## 1. Resumo Geral

| Métrica | Valor |
|---------|-------|
| Total de Registros | **106** |
| Episódios Únicos | **2** |
| Registros GROWTH (Fase 1) | 49 |
| Registros REFINEMENT (Fase 2) | 57 |

## 2. Análise por Episódio

| Episode ID | Records | GROWTH | REFINEMENT | Avg Fitness | Min | Max |
|------------|---------|--------|------------|-------------|-----|-----|
| `2b9f5ead...` | 80 | 49 | 31 | -0.9647 | -0.9919 | -0.9385 |
| `2c322f9b...` | 26 | 0 | 26 | -0.9597 | -0.9836 | -0.9461 |

## 3. Análise de Tensores

### 3.1 Dimensões dos Tensores

| Tensor | Shape | Dtype |
|--------|-------|-------|
| State | `(7, 32, 16, 4)` | `float32` |
| Policy | `(2, 32, 16, 4)` | `float32` |

### 3.2 Análise dos Canais do State (7 canais v3.1)

| Canal | Nome | Min | Max | Mean | Non-zero % |
|-------|------|-----|-----|------|------------|
| 0 | Density (ρ) | 0.0000 | 1.0000 | 0.0215 | 2.15% |
| 1 | Mask X (ux) | 0.0000 | 1.0000 | 0.0312 | 3.12% |
| 2 | Mask Y (uy) | 0.0000 | 1.0000 | 0.0312 | 3.12% |
| 3 | Mask Z (uz) | 0.0000 | 0.0000 | 0.0000 | 0.00% |
| 4 | Force X (Fx) | 0.0000 | 0.0000 | 0.0000 | 0.00% |
| 5 | Force Y (Fy) | -1.0000 | 0.0000 | -0.0044 | 0.44% |
| 6 | Force Z (Fz) | 0.0000 | 0.0000 | 0.0000 | 0.00% |

### 3.3 Análise dos Canais da Policy (2 canais)

| Canal | Nome | Min | Max | Mean | Non-zero % |
|-------|------|-----|-----|------|------------|
| 0 | Add (Adição) | 0.0000 | 1.0000 | 0.3291 | 32.91% |
| 1 | Remove (Remoção) | 0.0000 | 0.0000 | 0.0000 | 0.00% |

## 4. Análise de Fitness Score (Value Target)

### 4.1 Estatísticas Gerais

| Estatística | Valor |
|-------------|-------|
| Mínimo | -0.991854 |
| Máximo | -0.938510 |
| Média | -0.963461 |
| Mediana | -0.957936 |
| Desvio Padrão | 0.010996 |
| Percentil 25 | -0.972978 |
| Percentil 75 | -0.957936 |

### 4.2 Distribuição por Faixa

| Faixa | Contagem | Percentual |
|-------|----------|------------|
| [-1.1, -0.5) | 106 | 100.0% |
| [-0.5, 0.0) | 0 | 0.0% |
| [0.0, 0.5) | 0 | 0.0% |
| [0.5, 1.1) | 0 | 0.0% |

### 4.3 Fitness por Fase

| Fase | Count | Avg | Min | Max |
|------|-------|-----|-----|-----|
| GROWTH | 49 | -0.9579 | -0.9579 | -0.9579 |
| REFINEMENT | 57 | -0.9682 | -0.9919 | -0.9385 |

## 5. Análise de Policy Targets

### 5.1 Balanceamento de Classes

| Métrica | Canal ADD | Canal REMOVE |
|---------|-----------|--------------|
| Média % Positivos | 21.08% | 14.69% |
| Mediana % Positivos | 24.10% | 8.86% |
| Max % Positivos | 55.08% | 88.87% |

### 5.2 Recomendação de pos_weight (BCEWithLogitsLoss)

| Canal | pos_weight Recomendado |
|-------|------------------------|
| ADD | **3.7** |
| REMOVE | **5.8** |

### 5.3 Balanceamento por Fase

**GROWTH (Fase 1):** ADD médio = 14.58%
**REFINEMENT (Fase 2):** ADD médio = 26.67%

## 6. Análise de Metadados

### 6.1 Boundary Conditions (BC Types)

| BC Type | Contagem | Percentual |
|---------|----------|------------|
| RAIL_XY | 80 | 75.5% |
| FULL_CLAMP | 26 | 24.5% |

### 6.2 Estratégias de Geração

| Estratégia | Contagem | Percentual |
|------------|----------|------------|
| unknown | 106 | 100.0% |

## 7. Amostras Aleatórias

| # | Episode | Step | Phase | Fitness | BC Type |
|---|---------|------|-------|---------|---------|
| 1 | `2c322f9b...` | 24 | REFINEMENT | -0.9471 | FULL_CLAMP |
| 2 | `2b9f5ead...` | 76 | REFINEMENT | -0.9890 | RAIL_XY |
| 3 | `2b9f5ead...` | 70 | REFINEMENT | -0.9763 | RAIL_XY |
| 4 | `2b9f5ead...` | 62 | REFINEMENT | -0.9820 | RAIL_XY |
| 5 | `2b9f5ead...` | 71 | REFINEMENT | -0.9755 | RAIL_XY |
| 6 | `2b9f5ead...` | 72 | REFINEMENT | -0.9746 | RAIL_XY |
| 7 | `2b9f5ead...` | 53 | REFINEMENT | -0.9615 | RAIL_XY |
| 8 | `2b9f5ead...` | 59 | REFINEMENT | -0.9841 | RAIL_XY |
| 9 | `2b9f5ead...` | 21 | GROWTH | -0.9579 | RAIL_XY |
| 10 | `2b9f5ead...` | 34 | GROWTH | -0.9579 | RAIL_XY |

## 8. Validação de Conformidade v3.1

| Check | Status |
|-------|--------|
| State tem 7 canais | ✅ PASS |
| Policy tem 2 canais | ✅ PASS |
| Fase GROWTH presente | ✅ PASS |
| Fase REFINEMENT presente | ✅ PASS |
| Fitness score em [-1, 1] | ✅ PASS |
| Metadados bc_type presentes | ✅ PASS |
| Policy dtype é float32 | ✅ PASS |

### ✅ Todos os checks de conformidade v3.1 passaram!

---

*Relatório gerado automaticamente pelo teste de integração AlphaBuilder v3.1*