# AlphaBuilder Cloud Workflows (Google Colab)

Este documento define os 3 fluxos principais para execução no Google Colab, garantindo persistência de dados no Google Drive.

## Setup Inicial (Comum a todos)

**1. Montar o Google Drive:**
Execute este bloco primeiro para acessar sua pasta `AlphaBuilder`.

```python
from google.colab import drive
drive.mount('/content/drive')

# Cria a pasta se não existir
!mkdir -p "/content/drive/MyDrive/AlphaBuilder/data"
!mkdir -p "/content/drive/MyDrive/AlphaBuilder/checkpoints"
```

**2. Instalar Dependências:**
Clone o repo (ou puxe atualizações) e instale as libs.

```python
%cd /content
!rm -rf alphabuild
!git clone https://github.com/gustavomello9600/alphabuild.git
%cd alphabuild
!python alphabuilder/src/neural/colab_entrypoint.py
```

---

## Fluxo 1: Geração de Dados Iniciais (Data Harvest)

**Objetivo:** Gerar dados de treino e salvar diretamente no Drive.

*   **Script:** `run_data_harvest.py`
*   **Output:** `/content/drive/MyDrive/AlphaBuilder/data/warmup_data.db`

**Comando Colab:**
```python
!python run_data_harvest.py \
  --episodes 1000 \
  --resolution 64x32x8 \
  --strategy simp \
  --db-path "/content/drive/MyDrive/AlphaBuilder/data/warmup_data.db"
```

---

## Fluxo 2: Warm-up da Rede Neural

**Objetivo:** Treinar a rede lendo do Drive e salvando o checkpoint no Drive.

*   **Script:** `alphabuilder/src/neural/train_v1.py`
*   **Input:** `/content/drive/MyDrive/AlphaBuilder/data/warmup_data.db`
*   **Checkpoint:** `/content/drive/MyDrive/AlphaBuilder/checkpoints/swin_unetr_warmup.pt`

**Comando Colab:**
```python
!python alphabuilder/src/neural/train_v1.py \
  --db-path "/content/drive/MyDrive/AlphaBuilder/data/warmup_data.db" \
  --epochs 50 \
  --batch-size 32 \
  --checkpoint-path "/content/drive/MyDrive/AlphaBuilder/checkpoints/swin_unetr_warmup.pt"
```

---

## Fluxo 3: Self-Play (Refinamento)

**Objetivo:** Usar o modelo do Drive para gerar novos dados no Drive.

*   **Script:** `run_self_play.py`
*   **Input:** `/content/drive/MyDrive/AlphaBuilder/checkpoints/swin_unetr_warmup.pt`
*   **Output:** `/content/drive/MyDrive/AlphaBuilder/data/self_play_data.db`

**Comando Colab:**
```python
!python run_self_play.py \
  --episodes 100 \
  --checkpoint "/content/drive/MyDrive/AlphaBuilder/checkpoints/swin_unetr_warmup.pt" \
  --resolution 64x32x32 \
  --steps 50 \
  --db-path "/content/drive/MyDrive/AlphaBuilder/data/self_play_data.db"
```

---

## Ciclo Virtuoso (Next Steps)

Para continuar melhorando ("se tornar ainda melhor"):
1.  Junte `warmup_data.db` e `self_play_data.db`.
2.  Re-treine a rede (Fluxo 2) com esse dataset combinado.
3.  Rode mais Self-Play (Fluxo 3) com a nova rede.
4.  Repita.
