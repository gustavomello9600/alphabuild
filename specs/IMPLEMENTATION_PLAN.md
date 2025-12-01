# Plano de Implementação v3.1

## Blocos Incrementais (TDD)

Cada bloco leva o teste de integração a um novo estado de "passing".

---

### Bloco 1: Geração de Dados ✅ test_00 → test_02
**Objetivo:** Gerar episódios Bezier e Full Domain no DB de testes

**Tarefas:**
- [ ] Adaptar `data_generation.py` para chamar run_data_harvest
- [ ] Adicionar suporte a DB path customizado em run_data_harvest
- [ ] Manter 5 canais temporariamente (migrar para 7 no Bloco 2)

**Critério:** `pytest -x` passa até test_02

---

### Bloco 2: Schema v3.1 (7 canais) ✅ test_03 → test_06  
**Objetivo:** Validar schema e tensores com 7 canais

**Tarefas:**
- [ ] Atualizar `tensor_utils.py` para 7 canais
- [ ] Atualizar `run_data_harvest.py` para gerar 7 canais
- [ ] Adicionar metadados `is_final_step` e `is_connected`
- [ ] Atualizar `storage.py` se necessário

**Critério:** `pytest -x` passa até test_06

---

### Bloco 3: Augmentations ✅ test_07 → test_12
**Objetivo:** Implementar todas as transformações de dados

**Tarefas:**
- [ ] Criar `alphabuilder/src/neural/augmentation.py`
- [ ] Implementar `rotate_90_z(state, policy)`
- [ ] Implementar `flip_y(state, policy)`
- [ ] Implementar `erosion_attack(state, policy, value)`
- [ ] Implementar `load_multiplier(state, policy, value, k)`
- [ ] Implementar `sabotage(state, policy, value)`
- [ ] Implementar `saboteur(state, policy, value)`

**⚡ Requisito de Performance (Kaggle):**
- Operações vetorizadas (NumPy/PyTorch broadcasting)
- Kernels de erosão/dilatação pré-computados
- Target: < 5ms por sample (200+ samples/s)
- Sem I/O durante augmentation

**Critério:** `pytest -x` passa até test_12

---

### Bloco 4: Training Loop ✅ test_13
**Objetivo:** Treinar uma epoch com dados augmentados

**Tarefas:**
- [ ] Criar `dataset_v31.py` com augmentation on-the-fly
- [ ] Criar `model_v31.py` (7 canais, dynamic padding, InstanceNorm)
- [ ] Criar `trainer_v31.py` com weighted loss

**Critério:** `pytest -x` passa até test_13

---

### Bloco 5: Inferência ✅ test_14 → test_15
**Objetivo:** Validar inferência em Fase 1 e Fase 2

**Tarefas:**
- [ ] Garantir model.eval() funciona
- [ ] Validar shapes de saída
- [ ] Validar range de value [-1, 1]

**Critério:** `pytest -x` passa TODOS os 16 testes 🎉

---

## Status Atual

| Bloco | Testes | Status |
|-------|--------|--------|
| 1 | test_00 → test_02 | ✅ COMPLETO |
| 2 | test_03 → test_06 | ✅ COMPLETO |
| 3 | test_07 → test_12 | ✅ COMPLETO |
| 4 | test_13 | ✅ COMPLETO |
| 5 | test_14 → test_15 | ✅ COMPLETO |

**🎉 TODOS OS 16 TESTES PASSARAM!** (14.82s)
