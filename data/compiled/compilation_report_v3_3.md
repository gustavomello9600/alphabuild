# Relatório de Compilação do Dataset AlphaBuilder v3.3

- **Banco compilado:** `warmup_v3_3_kaggle.db` (`2557.30 MB`, `0.639 MB/episódio`)
- **Shards processadas:** `400` mescladas | `0` episódios duplicados ignorados | `0` quarentenadas
- **Tempo de compilação:** `2881.97 s` | **Integridade SQLite:** `ok` | **Erros de auditoria:** `0`

## Cobertura e Conectividade
- **Episódios:** `4000` (`4000/4000 = 100.0%` conectados)
- **Registros:** `477050` (`GROWTH`: `161500`, `REFINEMENT`: `315550`)
- **Proveniência por worker:** `{"katastopoulos-wsl": 1600, "gustavo-oracle-server": 250, "idcons-smart-vps": 420, "archgustavo-acer": 240, "katastopoulos-wsl-2": 360, "google-colab-1": 280, "google-colab-2": 280, "google-colab-3": 280, "core-ultra5-wsl": 290}`

## Separação Física e Distribuições
- **`FULL_CLAMP` compliance:** `{"n": 2759, "min": 46.654, "p10": 120.5002, "p25": 208.6341, "p50": 425.9481, "p75": 928.1273, "p90": 1622.5082, "max": 5550.7998, "mean": 681.7799, "std": 680.5088}`
- **`RAIL_XY` compliance:** `{"n": 1241, "min": 46.9783, "p10": 113.7158, "p25": 211.3867, "p50": 416.15, "p75": 960.5745, "p90": 1752.783, "max": 5338.1028, "mean": 718.6394, "std": 753.2742}`
- **Phase 2 `log10(C)`:** `{"n": 315550, "min": 1.6689, "p10": 2.2794, "p25": 2.6035, "p50": 3.0209, "p75": 3.4689, "p90": 3.8642, "max": 4.6519, "mean": 3.0453, "std": 0.5934}`
- **Value `GROWTH`:** `{"n": 161500, "min": 0.1155, "p10": 0.2706, "p25": 0.3312, "p50": 0.4152, "p75": 0.507, "p90": 0.5967, "max": 0.8307, "mean": 0.4249, "std": 0.1237}`
- **Value `REFINEMENT`:** `{"n": 315550, "min": -0.6031, "p10": 0.2041, "p25": 0.3349, "p50": 0.4667, "p75": 0.5872, "p90": 0.6841, "max": 0.8391, "mean": 0.4524, "std": 0.1834}`

## Split Estratificado & Teste Smoke (`TopologyDatasetV31`)
- **Split (`42`, val=0.1):** `{"total_episodes": 4000, "train_episodes": 3600, "val_episodes": 400}`
- **Estratos:** `{"BEZIER:FULL_CLAMP": {"total": 2227, "train": 2004, "val": 223}, "BEZIER:RAIL_XY": {"total": 1003, "train": 903, "val": 100}, "FULL_DOMAIN:FULL_CLAMP": {"total": 532, "train": 479, "val": 53}, "FULL_DOMAIN:RAIL_XY": {"total": 238, "train": 214, "val": 24}}`
- **Smoke `TopologyDatasetV31`:** `{"skipped": false, "dataset_len": 477050, "checked_indices": [0, 238525, 477049], "sample_shapes": [[[7, 64, 32, 8], [2, 64, 32, 8]], [[7, 64, 32, 8], [2, 64, 32, 8]], [[7, 64, 32, 8], [2, 64, 32, 8]]], "elapsed_sec": 340.073}`
