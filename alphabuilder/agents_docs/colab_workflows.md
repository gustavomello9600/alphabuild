# AlphaBuilder Cloud Workflows (Google Colab)

Este documento define os 3 fluxos principais para execução no Google Colab, aproveitando a biblioteca `fem-on-colab` para simulação física acelerada.

## Setup Inicial (Comum a todos)

Antes de rodar qualquer fluxo, execute o setup do ambiente no Colab:

```python
!git clone https://github.com/gustavomello9600/alphabuild.git
%cd alphabuild
!python alphabuilder/src/neural/colab_entrypoint.py
```

---

## Fluxo 1: Geração de Dados Iniciais (Data Harvest)

**Objetivo:** Gerar um dataset robusto de exemplos "clássicos" (SIMP e Heurísticos) para ensinar a rede neural o básico da física e conectividade.

*   **Script:** `run_data_harvest.py`
*   **Estratégia:** `balanced` (Mistura SIMP, Heurística e Aleatório).
*   **Output:** `data/training_data.db`

**Comando Colab:**
```python
# Gera 1000 episódios de treino
!python run_data_harvest.py \
  --episodes 1000 \
  --steps 50 \
  --resolution 64x32x32 \
  --strategy balanced \
  --db-path data/warmup_data.db
```

---

## Fluxo 2: Warm-up da Rede Neural

**Objetivo:** Treinar a rede Swin-UNETR (do zero) usando os dados gerados no Fluxo 1. Isso cria a "intuição" inicial.

*   **Script:** `alphabuilder/src/neural/train_v1.py`
*   **Input:** `data/warmup_data.db`
*   **Output:** `checkpoints/swin_unetr_warmup.pt`

**Comando Colab:**
```python
# Treina por 50 épocas
!python alphabuilder/src/neural/train_v1.py \
  --db-path data/warmup_data.db \
  --epochs 50 \
  --batch-size 32
```
*(Nota: O script `train_v1.py` salvará o modelo automaticamente. Certifique-se de baixar o `.pt` ou salvar no Google Drive).*

---

## Fluxo 3: Self-Play (Refinamento)

**Objetivo:** Usar a rede treinada (Fluxo 2) para jogar novos episódios. O MCTS usa a rede para guiar a busca, gerando dados de alta qualidade (estratégias que a rede "descobriu" ou refinou).

*   **Script:** `run_self_play.py`
*   **Input:** `checkpoints/swin_unetr_warmup.pt` (Modelo treinado)
*   **Output:** `data/self_play_data.db` (Novos dados para re-treino futuro)

**Comando Colab:**
```python
# Roda 100 episódios guiados pela rede
!python run_self_play.py \
  --episodes 100 \
  --checkpoint checkpoints/swin_unetr_warmup.pt \
  --resolution 64x32x32 \
  --steps 50 \
  --db-path data/self_play_data.db
```

---

## Ciclo Virtuoso (Next Steps)

Para continuar melhorando ("se tornar ainda melhor"):
1.  Junte `warmup_data.db` e `self_play_data.db`.
2.  Re-treine a rede (Fluxo 2) com esse dataset combinado.
3.  Rode mais Self-Play (Fluxo 3) com a nova rede.
4.  Repita.
