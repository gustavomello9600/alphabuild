# 📊 Relatório de Qualidade de Dados de Treino

**Gerado em:** 2025-12-03 06:07:05
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
| `1d18456f...` | 80 | 49 | 31 | -0.8615 | -0.9532 | -0.8244 |
| `ff7d1232...` | 26 | 0 | 26 | -0.9041 | -0.9492 | -0.8743 |

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
| Mínimo | -0.953204 |
| Máximo | -0.824384 |
| Média | -0.871944 |
| Mediana | -0.882072 |
| Desvio Padrão | 0.046378 |
| Percentil 25 | -0.914150 |
| Percentil 75 | -0.824384 |

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
| GROWTH | 49 | -0.8244 | -0.8244 | -0.8244 |
| REFINEMENT | 57 | -0.9128 | -0.9532 | -0.8743 |

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
| 1 | `1d18456f...` | 30 | GROWTH | -0.8244 | RAIL_XY |
| 2 | `1d18456f...` | 8 | GROWTH | -0.8244 | RAIL_XY |
| 3 | `1d18456f...` | 25 | GROWTH | -0.8244 | RAIL_XY |
| 4 | `1d18456f...` | 55 | REFINEMENT | -0.9207 | RAIL_XY |
| 5 | `1d18456f...` | 40 | GROWTH | -0.8244 | RAIL_XY |
| 6 | `1d18456f...` | 42 | GROWTH | -0.8244 | RAIL_XY |
| 7 | `1d18456f...` | 60 | REFINEMENT | -0.9354 | RAIL_XY |
| 8 | `1d18456f...` | 10 | GROWTH | -0.8244 | RAIL_XY |
| 9 | `1d18456f...` | 61 | REFINEMENT | -0.9332 | RAIL_XY |
| 10 | `ff7d1232...` | 25 | REFINEMENT | -0.8743 | FULL_CLAMP |

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