# 📊 Relatório de Qualidade de Dados de Treino

**Gerado em:** 2025-12-04 02:51:47
**Banco de Dados:** `episodios_de_testes_de_integracao.db`

---

## 1. Resumo Geral

| Métrica | Valor |
|---------|-------|
| Total de Registros | **0** |
| Episódios Únicos | **0** |
| Registros GROWTH (Fase 1) | 0 |
| Registros REFINEMENT (Fase 2) | 0 |

## 2. Análise por Episódio

| Episode ID | Records | GROWTH | REFINEMENT | Avg Fitness | Min | Max |
|------------|---------|--------|------------|-------------|-----|-----|

## 3. Análise de Tensores

## 4. Análise de Fitness Score (Value Target)

### 4.1 Estatísticas Gerais

| Estatística | Valor |
|-------------|-------|
| Mínimo | -0.953204 |
| Máximo | -0.824384 |
| Média | -0.872022 |
| Mediana | -0.877649 |
| Desvio Padrão | 0.044749 |
| Percentil 25 | -0.912865 |
| Percentil 75 | -0.824384 |

### 4.2 Distribuição por Faixa

| Faixa | Contagem | Percentual |
|-------|----------|------------|
| [-1.1, -0.5) | 114 | 100.0% |
| [-0.5, 0.0) | 0 | 0.0% |
| [0.0, 0.5) | 0 | 0.0% |
| [0.5, 1.1) | 0 | 0.0% |

### 4.3 Fitness por Fase

| Fase | Count | Avg | Min | Max |
|------|-------|-----|-----|-----|

## 5. Análise de Policy Targets

### 5.1 Balanceamento de Classes

| Métrica | Canal ADD | Canal REMOVE |
|---------|-----------|--------------|
| Média % Positivos | 8.36% | 5.39% |
| Mediana % Positivos | 4.91% | 0.81% |
| Max % Positivos | 32.91% | 34.86% |

### 5.2 Recomendação de pos_weight (BCEWithLogitsLoss)

| Canal | pos_weight Recomendado |
|-------|------------------------|
| ADD | **11.0** |
| REMOVE | **15.0** |

### 5.3 Balanceamento por Fase

**GROWTH (Fase 1):** ADD médio = 14.58%
**REFINEMENT (Fase 2):** ADD médio = 3.68%

## 6. Análise de Metadados

### 6.1 Boundary Conditions (BC Types)

| BC Type | Contagem | Percentual |
|---------|----------|------------|

### 6.2 Estratégias de Geração

| Estratégia | Contagem | Percentual |
|------------|----------|------------|

## 7. Amostras Aleatórias

| # | Episode | Step | Phase | Fitness | BC Type |
|---|---------|------|-------|---------|---------|

## 8. Validação de Conformidade v3.1

| Check | Status |
|-------|--------|
| Fase GROWTH presente | ❌ FAIL |
| Fase REFINEMENT presente | ❌ FAIL |
| Fitness score em [-1, 1] | ✅ PASS |
| Metadados bc_type presentes | ❌ FAIL |

### ⚠️ Alguns checks falharam. Verificar implementação.

---

*Relatório gerado automaticamente pelo teste de integração AlphaBuilder v3.1*