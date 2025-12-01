# Plano de Implementação v3.1

## Validação Final: Teste de Integração

O desenvolvimento segue abordagem TDD (Test-Driven Development). O teste de integração é o norte que guia a implementação.

---

## Checklist de Implementação

### Fase 0: Infraestrutura de Testes
- [ ] Criar estrutura `tests/` com `conftest.py`
- [ ] Criar `test_integration_v31.py` (teste mestre)
- [ ] Configurar pytest para parar na primeira falha (`-x`)

### Fase 1: Storage v3.1 (7 canais)
- [ ] Atualizar `storage.py` para schema v3.1
- [ ] Tensor de entrada: 7 canais (densidade + 3 masks + 3 forças)
- [ ] Metadados: `is_final_step`, `is_connected`, `phase`
- [ ] Função de inicialização do DB de testes

### Fase 2: Data Harvest v3.1
- [ ] Atualizar `run_data_harvest.py` para gerar 7 canais
- [ ] Separar masks X/Y/Z para suporte
- [ ] Garantir metadados `is_final_step` no último frame
- [ ] Garantir metadados `is_connected` via connected components

### Fase 3: Dataset v3.1
- [ ] Criar `dataset_v31.py` com PyTorch Dataset
- [ ] Implementar carregamento de 7 canais
- [ ] Implementar detecção de `is_final_step` e `is_connected`

### Fase 4: Data Augmentation
- [ ] Rotação 90° (eixo Z) com inversão de forças
- [ ] Flip com inversão de forças
- [ ] Erosion Attack (para `is_final_step`)
- [ ] Load Multiplier (para `is_connected`, 5% chance)
- [ ] Sabotage (5% chance)
- [ ] Saboteur (10% chance)

### Fase 5: Modelo Neural v3.1
- [ ] Atualizar modelo para 7 canais de entrada
- [ ] Implementar Dynamic Padding
- [ ] Policy Head: (B, 2, D, H, W)
- [ ] Value Head: (B, 1) com Tanh
- [ ] InstanceNorm3d

### Fase 6: Training Loop v3.1
- [ ] Weighted Value Loss (w_neg = 5.0)
- [ ] Policy Loss com masking por fase
- [ ] Integração com DataLoader augmentado

---

## Ordem de Execução do Teste de Integração

```
1. Limpar DB de testes
2. Gerar 1 episódio BEZIER → salvar no DB
3. Gerar 1 episódio FULL_DOMAIN → salvar no DB
4. Verificar dados no DB (asserts de schema)
5. Carregar Dataset
6. Testar cada augmentation individualmente
7. Treinar 1 epoch (dados augmentados + originais)
8. Inferência em 1 ponto Fase 1
9. Inferência em 1 ponto Fase 2
10. Verificar outputs válidos
```

---

## Estrutura de Arquivos

```
tests/
├── conftest.py                 # Fixtures compartilhadas
├── test_integration_v31.py     # Teste mestre
└── data/
    └── episodios_de_testes_de_integracao.db

alphabuilder/src/
├── logic/
│   └── storage.py              # Atualizado para v3.1
├── neural/
│   ├── model_v31.py            # Modelo com 7 canais
│   ├── dataset_v31.py          # Dataset com augmentation
│   ├── augmentation.py         # Transformações
│   └── trainer_v31.py          # Loop de treino
└── core/
    └── tensor_utils.py         # Atualizado para 7 canais
```

