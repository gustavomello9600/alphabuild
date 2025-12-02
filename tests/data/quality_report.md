# 📊 Relatório de Qualidade de Dados de Treino

**Gerado em:** 2025-12-02 01:46:03
**Banco de Dados:** `episodios_de_testes_de_integracao.db`

---

## 1. Resumo Geral

| Métrica | Valor |
|---------|-------|
| Total de Registros | **50** |
| Episódios Únicos | **1** |
| Registros GROWTH (Fase 1) | 49 |
| Registros REFINEMENT (Fase 2) | 1 |

## 2. Análise por Episódio

| Episode ID | Records | GROWTH | REFINEMENT | Avg Fitness | Min | Max |
|------------|---------|--------|------------|-------------|-----|-----|
| `ec5d6d12...` | 50 | 49 | 1 | -0.9720 | -0.9727 | -0.9385 |

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
| Mínimo | -0.972712 |
| Máximo | -0.938510 |
| Média | -0.972028 |
| Mediana | -0.972712 |
| Desvio Padrão | 0.004788 |
| Percentil 25 | -0.972712 |
| Percentil 75 | -0.972712 |

### 4.2 Distribuição por Faixa

| Faixa | Contagem | Percentual |
|-------|----------|------------|
| [-1.1, -0.5) | 50 | 100.0% |
| [-0.5, 0.0) | 0 | 0.0% |
| [0.0, 0.5) | 0 | 0.0% |
| [0.5, 1.1) | 0 | 0.0% |

### 4.3 Fitness por Fase

| Fase | Count | Avg | Min | Max |
|------|-------|-----|-----|-----|
| GROWTH | 49 | -0.9727 | -0.9727 | -0.9727 |
| REFINEMENT | 1 | -0.9385 | -0.9385 | -0.9385 |

## 5. Análise de Policy Targets

### 5.1 Balanceamento de Classes

| Métrica | Canal ADD | Canal REMOVE |
|---------|-----------|--------------|
| Média % Positivos | 14.29% | 1.21% |
| Mediana % Positivos | 13.43% | 0.00% |
| Max % Positivos | 32.91% | 60.55% |

### 5.2 Recomendação de pos_weight (BCEWithLogitsLoss)

| Canal | pos_weight Recomendado |
|-------|------------------------|
| ADD | **6.0** |
| REMOVE | **15.0** |

### 5.3 Balanceamento por Fase

**GROWTH (Fase 1):** ADD médio = 14.58%
**REFINEMENT (Fase 2):** ADD médio = 0.00%

## 6. Análise de Metadados

### 6.1 Boundary Conditions (BC Types)

| BC Type | Contagem | Percentual |
|---------|----------|------------|
| RAIL_XY | 50 | 100.0% |

### 6.2 Estratégias de Geração

| Estratégia | Contagem | Percentual |
|------------|----------|------------|
| unknown | 50 | 100.0% |

## 7. Amostras Aleatórias

| # | Episode | Step | Phase | Fitness | BC Type |
|---|---------|------|-------|---------|---------|
| 1 | `ec5d6d12...` | 28 | GROWTH | -0.9727 | RAIL_XY |
| 2 | `ec5d6d12...` | 27 | GROWTH | -0.9727 | RAIL_XY |
| 3 | `ec5d6d12...` | 33 | GROWTH | -0.9727 | RAIL_XY |
| 4 | `ec5d6d12...` | 10 | GROWTH | -0.9727 | RAIL_XY |
| 5 | `ec5d6d12...` | 22 | GROWTH | -0.9727 | RAIL_XY |
| 6 | `ec5d6d12...` | 34 | GROWTH | -0.9727 | RAIL_XY |
| 7 | `ec5d6d12...` | 3 | GROWTH | -0.9727 | RAIL_XY |
| 8 | `ec5d6d12...` | 26 | GROWTH | -0.9727 | RAIL_XY |
| 9 | `ec5d6d12...` | 46 | GROWTH | -0.9727 | RAIL_XY |
| 10 | `ec5d6d12...` | 31 | GROWTH | -0.9727 | RAIL_XY |

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