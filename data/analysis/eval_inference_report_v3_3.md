# Avaliação Real de Inferência + MCTS (v3.3), Análise de Padrões Estruturais e Estimativa de Custo/Viabilidade (`Swin-UNETR`)

**Data:** 2026-11-24  
**Checkpoint avaliado neste ciclo:** [`checkpoints/simple_v3_3_smoke_ep2.pt`](file:///c:/Users/admin/Projetos/alphabuild/checkpoints/simple_v3_3_smoke_ep2.pt) (`SimpleBackbone`, `4.433.132` parâmetros, smoke test de **2 épocas**, `val_loss = 0.5953`, dispositivo `xpu` = `Intel(R) Graphics [0x7dd1]`)  
**Banco SQLite inspecionável (Studio v4):** [`data/eval_inference_v3_3.db`](file:///c:/Users/admin/Projetos/alphabuild/data/eval_inference_v3_3.db) (`105,57 MB`, `12` episódios = `6 ep2__*` + `6 simp__*`, `762` passos, `PRAGMA integrity_check = ok`, zero blobs pickle)  
**Script reprodutível:** [`scripts/evaluate_inference_scenarios.py`](file:///c:/Users/admin/Projetos/alphabuild/scripts/evaluate_inference_scenarios.py)  
**Dados brutos:** [`data/analysis/eval_ep2_results.json`](file:///c:/Users/admin/Projetos/alphabuild/data/analysis/eval_ep2_results.json)

> [!NOTE]
> O treino completo Stage 3 (`SimpleBackbone`, 30 épocas no Kaggle `2 × Tesla T4`) ainda está em execução no momento deste relatório (~04:33 BRT, término previsto ~05:13 BRT). Ao finalizar, o modelo de 30 épocas será avaliado com `--skip-simp` no **mesmo** banco [`data/eval_inference_v3_3.db`](file:///c:/Users/admin/Projetos/alphabuild/data/eval_inference_v3_3.db) com prefixo `ep30__*` para comparação direta lado a lado (`ep2` × `ep30` × `simp`).

---

## 1. Resumo Executivo

1. **Conectividade robusta na Fase 1 (`6/6 = 100%`):** usando o loop canônico (`AlphaBuilderMCTS` com `160` simulações, `select_pv_branches` profundidade até 4, PBRS na Fase 1 e FEniCSx real na Fase 2), a rede de apenas **2 épocas** conectou apoio (`x = 0`) à região de carga em **todos os 6 cenários** (`18` a `60` passos), terminando com **`1` ilha principal e `0` voxels soltos** em 100% dos casos e reduzindo a compliance em **`9,7%` a `34,8%`** durante os 30 passos de Fase 2.
2. **Padrão geométrico atual — viga compacta e baixa utilização de altura ($I_z$):** ao contrário do FEniTop SIMP, que empurra banzos para as bordas superior/inferior (`inertia_y_norm = 0,341–0,518`) e escava diagonais vazadas (`hollow_ratio_xy = 0,40–0,63`), a política atual com 2 épocas converge para uma **barra sólida quase maciça** perto da linha neutra (`inertia_y_norm = 0,012–0,052`, **10× a 42× menor** que o SIMP; `hollow_ratio_xy = 0,00–0,16`) e exibe **alargamento invertido na ponta** (`y_span_tip > y_span_root`) e **overshoot em X** (`12,3%–16,4%`) quando a carga não está no extremo `x = 63`.
3. **Por que a Fase 2 apenas engrossa a barra (`99,93% ADD`) em vez de esculpir treliça:** em `1.440` ações executadas na Fase 2 (`6 × 240`), houve **`1.439` adições (`ADD`)** e apenas **`1` remoção (`REMOVE`)**. A causa medida está na **calibração otimista da Value Head** de 2 épocas: ela prevê `+0,38` a `+0,61` enquanto o `fem_reward` real FEniCSx é negativo/baixo (`MAE = 0,42–0,70`, viés médio de **`+0,595`**), levando o MCTS a evitar remoções.
4. **Blueprint Head vs Trajetória MCTS:** no passo 0 (domínio vazio), a Blueprint Head já prevê envoltória significativamente mais alta (`inertia_y_norm = 0,057–0,110`) do que a estrutura realmente construída pelo MCTS (`0,012–0,052`), demonstrando que a cabeça auxiliar já capta parte da altura estrutural antes mesmo de a política explorá-la.
5. **Arquitetura completa (`Swin-UNETR`, 6,83M parâmetros):** medimos no hardware local que ela é **`3,66×` mais lenta no passo de treino (`BS=8`)** que o `SimpleBackbone` (e **`~6,4×`** considerando `BS=16`). Um pré-treino completo de 30 épocas no `warmup_v3_3_kaggle.db` requer **`~96–136 GPU-horas` de Tesla T4** (~`48–68h` em `2 × T4`, inviável num único commit de 12h no Kaggle sem fatiamento em múltiplos resumes). Na `Intel(R) Graphics [0x7dd1]` local, ela é **viável para Raio-X Neural** (`169 ms`), mas **pesada para MCTS ao vivo de 160 sims** (`~17–20 s/passo`, exigindo reduzir para `40–64` sims ou otimizar via OpenVINO).

---

## 2. Benchmark Quantitativo por Cenário (`ep2` MCTS vs `FEniTop SIMP`)

Parâmetros idênticos em todos os cenários: malha `64 × 32 × 8` (`16.384` voxels), `E = 100,0`, `nu = 0,25`, `P1 = 160 sims / action_batch = 32`, `P2 = 160 sims / action_batch = 8`, teto de `60` passos Fase 1 e `30` passos Fase 2; SIMP executado com `vol_frac = 0,15`, `max_iter = 60`, `r_min = 1,5`, `FULL_DOMAIN`.

### 2.1 Comparação de Performance Física (`Compliance`, `Volume` e `Reward v3.3`)

| Cenário (`BC`, carga) | EP2 Passos (`P1+P2`) | EP2 Motivo Fim | EP2 `Vol` | EP2 `C_P2_ini → C_final` (Δ%) | EP2 `Score v3.3` | SIMP Contínuo `C_cont` (`V`, `Reward`) | SIMP Binarizado `rho>0,5` `C_bin` (`conn?`) |
|---|---|---|---|---|---|---|---|
| **S1** (`FULL_CLAMP`, `x=63, y=16`) | `41 + 30` | `CONVERGENCE` | `9,46%` | `75.281 → 62.228` (**-17,3%**) | `-0,164` | `596,4` (`12,83%`, `+0,458`) | `21.083` (`False`) |
| **S2** (`FULL_CLAMP`, `x=63, y=6`) | `60 + 30` | `MAX_PHASE2_STEPS` | `13,18%` | `26.606 → 17.347` (**-34,8%**) | `-0,063` | `600,4` (`13,04%`, `+0,452`) | `48.847` (`False`) |
| **S3** (`FULL_CLAMP`, `x=44, y=24`) | `30 + 30` | `MAX_PHASE2_STEPS` | `7,32%` | `61.312 → 55.015` (**-10,3%**) | `-0,092` | `249,6` (`13,84%`, `+0,565`) | `25.127` (`False`) |
| **S4** (`RAIL_XY`, `x=58, y=16`) | `34 + 30` | `MAX_PHASE2_STEPS` | `8,11%` | `95.658 → 85.583` (**-10,5%**) | `-0,179` | `789,1` (`13,14%`, `+0,408`) | `254.506` (`False`) |
| **S5** (`RAIL_XY`, `x=52, y=11, z=[1,3]`) | `33 + 30` | `MAX_PHASE2_STEPS` | `7,91%` | `51.728 → 46.687` (**-9,7%**) | `-0,082` | `420,5` (`13,70%`, `+0,489`) | `70.139` (`False`) |
| **S6** (`FULL_CLAMP`, `x=36, y=16`) | `18 + 30` | `MAX_PHASE2_STEPS` | `4,98%` | `42.207 → 36.020` (**-14,7%**) | **`+0,031`** | `102,0` (`14,52%`, `+0,684`) | **`6.801` (`True`)** |

> [!IMPORTANT]
> **Observação metodológica sobre o SIMP binarizado em `max_iter = 60`:** o `optimization.py` utiliza `beta_interval = 30`. Com 60 iterações, o fator de projeção Heaviside termina em **`beta = 4,0`** (quantil `p85 ≈ 0,35–0,47`), com barras inclinadas finas ainda em densidade cinzenta intermediária. Por isso, ao binarizar ingenuamente com `rho > 0,5`, o SIMP perde parte da ilha na patch de carga em S1–S5 (`C_bin = 21k–254k`, perdendo para `ep2` em S2, S4 e S5), enquanto no campo contínuo (`beta = 4,0`, usado no harvest longo até `beta = 128`) o SIMP é **`28,9×` (S2) a `353,0×` (S6) mais rígido** que o `ep2`. No cenário S6 (vão curto `x=36`), o SIMP já fecha conexão binarizada completa e supera o `ep2` em `5,30×` até nos voxels discretos.

### 2.2 Métricas Geométricas (`Forma Aprendida` vs `SIMP`)

| Cenário | `y_span` (`raiz / meio / ponta`) EP2 vs SIMP | `inertia_y_norm` EP2 vs SIMP | `hollow_ratio_xy` EP2 vs SIMP | `overshoot_frac` (`x > lx+2`) EP2 vs SIMP | `z_centroid` EP2 vs SIMP |
|---|---|---|---|---|---|
| **S1** (`x=63, y=16, z=[3,5]`) | `6 / 8 / 12` vs `32 / 32 / 28` | `0,025` vs `0,508` (**-20×**) | `0,00` vs `0,63` | `0,0%` vs `0,0%` | `3,41` vs `5,79` |
| **S2** (`x=63, y=6, z=[3,5]`) | `8 / 7 / 17` vs `32 / 32 / 26` | `0,052` vs `0,518` (**-10×**) | `0,09` vs `0,58` | `0,0%` vs `0,0%` | `2,97` vs `5,79` |
| **S3** (`x=44, y=24, z=[3,5]`) | `5 / 6 / 11` vs `32 / 28 / 19` | `0,044` vs `0,380` (**-8,6×**) | `0,07` vs `0,41` | **`12,4%`** vs `0,0%` | `2,84` vs `5,26` |
| **S4** (`x=58, y=16, z=[3,5]`) | `7 / 7 / 7` vs `32 / 32 / 24` | `0,012` vs `0,494` (**-42×**) | `0,01` vs `0,57` | **`6,8%`** vs `0,0%` | `3,46` vs `5,71` |
| **S5** (`x=52, y=11, z=[1,3]`) | `7 / 6 / 11` vs `32 / 32 / 24` | `0,042` vs `0,431` (**-10×**) | `0,16` vs `0,40` | **`12,3%`** vs `0,0%` | `3,05` vs **`0,99`** |
| **S6** (`x=36, y=16, z=[3,5]`) | `5 / 5 / 7` vs `32 / 26 / 16` | `0,016` vs `0,341` (**-21×**) | `0,00` vs `0,42` | **`16,4%`** vs `0,0%` | `2,90` vs `4,25` |

---

## 3. Análise Aprofundada dos Padrões Aprendidos + Algoritmo de Jogo

### 3.1 Dinâmica de Crescimento (Fase 1) e Disciplina Espacial
- **Capacidade de ponte 3D em todos os regimes:** a rede não colapsou nem estagnou em ilhas soltas em nenhum cenário. Em S1, S3, S4, S5 e S6, o crescimento partiu de `x = 0` alinhado à altura/profundidade da carga e fechou o caminho em **`18` a `41` passos**. No caso mais assimétrico e distante (`S2`, `x=63, y=6`), precisou de exatos `60` passos (`V = 11,7%` na conexão) para espalhar até a borda inferior.
- **Vazamento pós-carga (*overshoot* em X):** nos cenários em que `lx < 63` (`S3 x=44`, `S5 x=52`, `S6 x=36`), a estrutura MCTS chegou até **`x_max = 63`** (`overshoot_frac` entre `12,3%` e `16,4%`), adicionando voxels em zona de tensão nula à direita da carga. Isso revela que, com 2 épocas, o MCTS durante a Fase 2 ainda escolhe adições na ponta livre onde a rede neural atribui logit residual razoável, porque a Value Head não penaliza adequadamente o desperdício de massa inútil na extremidade livre.
- **Insensibilidade parcial à condição 3D `RAIL_XY` em `Z` (`S5`):** no cenário `S5` (`RAIL_XY`, carga excêntrica em `z = [1, 3]`), o SIMP concentra o material do lado carregado (`z_centroid = 0,99`, ocupando apenas `z = 0..4`), pois o suporte `RAIL_XY` permite deslizamento livre em `Z` e qualquer massa simétrica em `Z` distante gera torção/flexão lateral desnecessária. Já o `ep2` distribuiu material quase simetricamente no eixo `Z` (`z_centroid = 3,05`, `z = 0..7`), mostrando que em 2 épocas o backbone ainda trata a dimensão `Z` de forma quase homogênea.

### 3.2 Geometria Estrutural na Fase 2: Viga Compacta vs Treliça Otimizada
A inspeção combinada de `inertia_y_norm`, `y_span` e `hollow_ratio_xy` explica mecanicamente o gap de rigidez:
1. **Déficit de momento de inércia ($I_z \propto \sum y_i^2$):** a rigidez à flexão de um engaste cresce quadraticamente com a distância ao eixo neutro. O SIMP explora 100% da altura disponível (`y = 0` e `y = 31`, `inertia_y_norm ≈ 0,34–0,52`). O `ep2` mantém os voxels concentrados dentro de uma faixa de ~5–12 voxels ao redor do eixo (`inertia_y_norm ≈ 0,012–0,052`), custando um fator de **10× a 40×** em rigidez antes mesmo de contar vazios internos.
2. **Alargamento invertido (`y_span_tip > y_span_root`):** mecanicamente, o momento fletor máximo $M(x) = F \cdot (L - x)$ ocorre no engaste (`x = 0`), exigindo seção alta na raiz (`y_span_root = 32`) e estreita na ponta (`y_span_tip = 16–28`), exatamente como o SIMP faz. No `ep2`, ocorreu o oposto em S1, S2, S3, S5 e S6 (`y_span_root = 5–8` < `y_span_tip = 7–17`): ao conectar na ponta perto da carga na Fase 1/início da Fase 2, o MCTS empilhou material junto à carga (`x ≈ lx`) enquanto deixou o engaste fino.
3. **Ausência de escavação (`hollow_ratio_xy` próximo de 0):** o `hollow_ratio_xy` (fração de vazios internos dentro da envoltória lateral X-Y) foi `0,00` em S1 e S6, `0,01` em S4 e no máximo `0,16` em S5 (contra `0,40–0,63` no SIMP). Isso bate diretamente com a distribuição de ações na Fase 2:

| Cenário | Refutação P1 | Refutação P2 | Top-8 Conc. P1 / P2 | Prof. Média Árvore | Fase 2 `ADD` / `REMOVE` | Value Head `MAE` (P2) | Viés `V_net - R_FEM` | Correlação `r(V_net, R_FEM)` |
|---|---|---|---|---|---|---|---|---|
| **S1** | `73,2%` | `100,0%` | `60,6%` / `66,5%` | `18,5` | `239` / `1` | `0,632` | `+0,632` | `+0,478` |
| **S2** | `83,3%` | `100,0%` | `67,4%` / `73,5%` | `17,1` | `240` / `0` | `0,422` | `+0,422` | `+0,360` |
| **S3** | `83,3%` | `96,7%` | `54,3%` / `67,3%` | `19,2` | `240` / `0` | `0,685` | `+0,685` | `+0,759` |
| **S4** | `91,2%` | `96,7%` | `61,1%` / `67,5%` | `17,6` | `240` / `0` | `0,703` | `+0,703` | **`-0,786`** |
| **S5** | `93,9%` | `100,0%` | `59,5%` / `65,7%` | `18,8` | `240` / `0` | `0,597` | `+0,597` | `+0,621` |
| **S6** | `94,4%` | `90,0%` | `53,9%` / `61,2%` | `19,5` | `240` / `0` | `0,535` | `+0,531` | `-0,101` |

### 3.3 Diagnóstico do Algoritmo (`MCTS` + `Value Head` + `Blueprint Head`)
- **MCTS está atuando forte (`refutation = 73%–100%`):** em quase 100% dos passos, o MCTS (`160 sims`, profundidade ~18, concentração Top-8 entre `54%` e `73%`) corrige o `argmax` bruto da Policy Head graças ao PBRS na Fase 1 e à penalidade de desconexão/ilhas na Fase 2.
- **Gargalo central na Fase 2 — viés positivo da Value Head (`+0,42` a `+0,70`):** no passo 0 (domínio vazio) a rede já prevê `V_net = +0,28` a `+0,45`; durante a Fase 2, para barras com `fem_reward = -0,18..-0,06` (`C ≈ 46k–86k`), a rede continua prevendo `+0,38` a `+0,61`. Sem penalização realista da compliance dentro das folhas da árvore, o MCTS nunca prefere `REMOVE` (`1` ação em `1.440`), pois `REMOVE` arrisca cortar caminho ou reduzir o `V_net` otimista das vizinhanças sólidas.
  - *Nota encorajadora:* os logs do treino Stage 3 (30 épocas) já mostraram queda da loss da Value Head de `0,0041` (época 2 val) para `0,0006` (época 24 train), o que deve reduzir drasticamente esse viés no teste `ep30`. Ainda assim, a remoção ativa de material interno em estados fora-da-distribuição do warmup será consolidada principalmente na **Etapa 6 (Self-Play MCTS com recompensa FEM real)**.
- **Blueprint Head já vê mais altura que a Policy (`step 0`):** na inspeção do tensor `blueprint` no passo 0 (antes de qualquer voxel sólido):

| Cenário | `bp_mean` / `bp_max` | Voxels `bp>0,5` | `bp_y_std` vs `EP2_y_std` | `bp_inertia_y` vs `EP2_inertia_y` | `BP_Dice(t=0,5)` | `BP_Dice(t=0,2)` | `MCTS_final_Dice` |
|---|---|---|---|---|---|---|---|
| **S1** | `0,267` / `0,977` | `3.500` | `4,32` vs `2,53` | `0,073` vs `0,025` (**2,9×**) | `0,110` | `0,110` | `0,065` |
| **S2** | `0,266` / `0,985` | `3.339` | `4,87` vs `3,66` | `0,093` vs `0,052` (**1,8×**) | `0,059` | `0,117` | `0,061` |
| **S3** | `0,218` / `0,993` | `2.780` | `5,31` vs `3,36` | `0,110` vs `0,044` (**2,5×**) | `0,143` | `0,266` | `0,077` |
| **S4** | `0,246` / `0,974` | `3.127` | `4,27` vs `1,73` | `0,071` vs `0,012` (**5,9×**) | `0,064` | `0,140` | `0,049` |
| **S5** | `0,217` / `0,991` | `2.488` | `3,81` vs `3,27` | `0,057` vs `0,042` (**1,4×**) | `0,096` | `0,150` | `0,083` |
| **S6** | `0,183` / `0,993` | `2.145` | `4,69` vs `2,05` | `0,086` vs `0,016` (**5,4×**) | `0,170` | `0,278` | `0,096` |

Embora no domínio completamente vazio (`step = 0`) o `Blueprint Dice` contra o SIMP de 60 iterações fique em `0,06–0,17` (`0,11–0,28` em `t = 0,2`) — contrastando com `0,8479` na validação agregada de treino onde o estado já contém grande parte da trajetória construída — a Blueprint Head já codifica uma envoltória **1,4× a 5,9× mais alta em momento de inércia** do que a viga estreita produzida pelo rollout guloso/MCTS da época 2.

---

## 4. Estimativa de Custo de Horas de GPU — Pré-Treino Completo (`Swin-UNETR`)

Todos os cálculos abaixo combinam os **tempos reais observados** no Kaggle (`2 × Tesla T4` 16 GB, `429.345` amostras de treino + `47.705` de validação, preset `kaggle_t4_dual_simple` vs `kaggle_t4_dual_swin` em [`train_config.py`](file:///c:/Users/admin/Projetos/alphabuild/alphabuilder/src/neural/train_config.py#L102-L124)) e os **micro-benchmarks reais** medidos na GPU local (`Intel(R) Graphics [0x7dd1]`, PyTorch `2.5.1+xpu`, MONAI `1.5.1`):

### 4.1 Razão Empírica Medida (`SimpleBackbone` vs `Swin-UNETR`)

- **Parâmetros:** `4.433.132` (`SimpleBackbone`) vs `6.831.758` (`Swin-UNETR`, **`1,54×`**).
- **Volume espacial após padding interno ([`model.py`](file:///c:/Users/admin/Projetos/alphabuild/alphabuilder/src/neural/model.py#L258)):**
  - `SimpleBackbone` (`divisor = 8`): `(64, 32, 8)` = **`16.384` voxels** (`0%` padding).
  - `Swin-UNETR` (`divisor = 32` para 4 estágios Swin): `(64, 32, 32)` = **`65.536` voxels** (**`4,00×` o custo espacial** — `75%` do tensor em Z é zero-padding!).
- **Passo de treino medido (`BF16 AMP`, forward + 3 losses + backward + AdamW no `xpu`):**
  - `SimpleBackbone` `BS = 8`: `541,6 ms` (`14,8 samples/s`)
  - `SimpleBackbone` `BS = 16`: `625,5 ms` (`25,6 samples/s`)
  - `Swin-UNETR` `BS = 8` (preset máximo para caber em T4 16 GB): **`1.984,7 ms`** (`4,0 samples/s`) → **`3,66×` mais lento no mesmo `BS = 8`** e **`6,40×` mais lento comparado ao `SimpleBackbone` em `BS = 16`**!
- **Passos DDP por época no Kaggle `2 × T4`:** `Simple` usa `BS = 64/GPU` (`128` global = `3.354 it/época`); `Swin` usa `BS = 8/GPU` (`16` global = **`26.834 it/época`**, `8×` mais sincronizações all-reduce/época).

### 4.2 Tabela de Custo de Pré-Treino (`477.050` registros completos)

Base de referência observada no Kaggle `2 × Tesla T4` para o `SimpleBackbone` (`kaggle_t4_dual_simple`): **`21,0 min/época`** (`~340 samples/s`), totalizando **`~10,5h` wall-clock (`~21 GPU-h`) para 30 épocas** e **`~14,0h` wall-clock (`~28 GPU-h`) para 40 épocas**.

Aplicando um fator conservador-realista de **`4,5×` a `6,5×`** por amostra (entre o fator iso-batch `3,66×` e o fator com saturação/all-reduce `~6,4×` observado no micro-benchmark):

| Configuração / Hardware | Tempo por Época | 20 Épocas (Wall / GPU-h) | 30 Épocas (Wall / GPU-h) | 40 Épocas (Wall / GPU-h) | Observação Operacional |
|---|---|---|---|---|---|
| **`SimpleBackbone`** · Kaggle `2 × T4` (medido) | **`21,0 min`** | `7,0 h` / `14 GPU-h` | **`10,5 h` / `21 GPU-h`** | `14,0 h` / `28 GPU-h` | Cabe em 1 sessão Kaggle (30 épocas) |
| **`Swin-UNETR`** · Kaggle `2 × T4` (`BS=8/GPU`, est.) | **`95 – 135 min`** (`~1,6–2,25 h`) | `32 – 45 h` / **`64 – 90 GPU-h`** | **`48 – 68 h` / `96 – 136 GPU-h`** | `63 – 90 h` / **`126 – 180 GPU-h`** | Excede cota semanal de 30h; exige **5–7 resumes** ao longo de **2–4 semanas** |
| **`Swin-UNETR`** · Nuvem `1 × RTX 4090` 24GB (est. ~3–4× sobre 2×T4, `BS=32`) | `~25 – 38 min` | `~8,5 – 12,5 h` GPU-h | **`~12,5 – 19 h` GPU-h** | `~17 – 25 h` GPU-h | Custo estimado spot (~US\$0,40/h): **~US\$5–US\$8** (30 épocas) |
| **`Swin-UNETR`** · Nuvem `1 × A100` 40/80GB (est. ~4–5× sobre 2×T4, `BS=48–64`) | `~20 – 30 min` | `~6,5 – 10 h` GPU-h | **`~10 – 15 h` GPU-h** | `~13 – 20 h` GPU-h | Custo estimado spot (~US\$1,40/h): **~US\$14–US\$21** (30 épocas) |

> [!TIP]
> **Como reduzir o custo do `Swin-UNETR` em ~2×–3× sem perder capacidade:**
> 1. **Amostragem por época ou 15–20 épocas com `cosine` schedule curto:** como `warmup_v3_3_kaggle.db` tem `477k` frames com alta correlação temporal entre passos adjacentes (além de `16x` simetrias), treinar com stride 2 (`~238k` frames/época por `20–25` épocas) cai para **`~32–45 GPU-h` de T4** (~1 semana Kaggle + 1 resume).
> 2. **Reduzir `divisor` de Z de `32` para `16`** se adaptar os estágios Swin para patch inicial `(2, 2, 1)` ou 3 níveis em Z (elimina metade do padding `8 → 32`, economizando ~`1,8×` VRAM e FLOPs).

---

## 5. Viabilidade para Inferência em Tempo Real no Hardware Atual

Hardware medido (`katastopoulos` via WSL2 `archlinux`): **`Intel(R) Graphics [0x7dd1]`** (`xpu`, PyTorch `2.5.1+xpu` + IPEX) e CPU do host (NPU `Intel(R) AI Boost` existe no Windows mas não está exposta ao WSL2 e `openvino` ainda não está instalado nos ambientes Python atuais).

### 5.1 Latências Medidas (`BS=1` Raio-X e `BS=8` MCTS Batch)

| Arquitetura | `xpu` `BS=1` (FP32 / BF16) | `xpu` `BS=8` (FP32 / BF16) | `cpu` `BS=1` (FP32) | Raio-X Neural Interativo (`120 ms` debounce) | MCTS `160 sims` (`20 × BS=8`) só Rede | MCTS `160 sims` Passo Completo (F1 / F2+FEM) |
|---|---|---|---|---|---|---|
| **`SimpleBackbone`** (`4,43M`, `64×32×8`) | **`70,8 ms`** / `79,0 ms` (`~14 FPS`) | `218,0 ms` / **`183,7 ms`** | `183,7 ms` | **Viável em tempo real** (`< 120 ms`) | **`~3,7 s/passo`** | **`~3,7 s`** / **`~6,1 s`** (medido) |
| **`Swin-UNETR`** (`6,83M`, pad `64×32×32`) | **`169,3 ms`** / `192,8 ms` (`~5,9 FPS`) | `865,6 ms` / **`836,7 ms`** | `270,4 ms` | **Viável** (~5–6 Hz; fluido ajustando debounce para ~200 ms) | **`~16,7 s/passo`** | **`~17–18 s`** / **`~19–20 s`** (est.) |

### 5.2 Veredito Prático para o Hardware Atual
1. **Raio-X Neural ao pintar voxels (`BS=1`):** **Sim, ambos são viáveis.** O `SimpleBackbone` roda a ~`71 ms` (`14 FPS`, folgado dentro dos `120 ms` do Studio); o `Swin-UNETR` roda a ~`169 ms` (`~6 FPS`) — há uma latência perceptível de ~`0,17 s` ao arrastar o pincel, perfeitamente usável para projeto interativo no Studio.
2. **MCTS ao Vivo com `160` simulações no Studio:**
   - No **`SimpleBackbone`**, **`~3,7–6,1 s/passo`** é confortável para acompanhar o episódio ao vivo (`~5–6 min` para `60–70` passos) e pausar/intervir.
   - No **`Swin-UNETR`**, **`~17–20 s/passo`** (`~21–24 min` por episódio completo de `70` passos) **não é tempo real interativo** com `160` simulações. Porém, reduzindo no Studio para **`40–64` simulações** (`5–8` batches de `BS=8`), o tempo cai para **`~4,5–7,5 s/passo`**, tornando o MCTS interativo perfeitamente utilizável no seu notebook.
3. **Otimizações locais recomendadas:**
   - Foi observado nos benchmarks o aviso `Aten Op fallback from XPU to CPU` em [`model.py#L210`](file:///c:/Users/admin/Projetos/alphabuild/alphabuilder/src/neural/model.py#L210) (`torch.linspace` no CoordConv rodando em CPU a cada forward). Cachear a grade `cd, ch, cw, domain_mask` por shape/device evita cópias CPU→XPU repetidas e melhora especialmente `BS=1`.
   - Instalar `openvino` nativamente no Windows para exportar o `best_model.pt` (`FP16`/`INT8`) e usar GPU iGPU/NPU nativos deve reduzir ainda mais a latência de `BS=1`.
   - **Self-Play de treino em massa (Etapa 6):** não é recomendável gerar centenas de episódios completos no notebook (`~3 ep/h` no Swin ou `~11 ep/h` no Simple); o Self-Play deve rodar em GPU NVIDIA (Kaggle/Colab/Nuvem).

---

## 6. Como Inspecionar os Episódios no AlphaBuilder Studio v4

Todos os 12 episódios já estão persistidos em [`data/eval_inference_v3_3.db`](file:///c:/Users/admin/Projetos/alphabuild/data/eval_inference_v3_3.db) no formato nativo `selfplay` (`games` + `game_steps` com densidade, políticas esparsas, visitas MCTS, Q-values MCTS e campos de deslocamento FEM em todos os passos de Fase 2 e no SIMP):

- **Nomes dos jogos no banco:**
  - Rede + MCTS (2 épocas): `ep2__S1_cantilever_long_center` ... `ep2__S6_short_deep_clamp`
  - Referência FEniTop SIMP: `simp__S1_cantilever_long_center` ... `simp__S6_short_deep_clamp`
- **No app:** abra a gaveta **Biblioteca** (`L`), selecione o banco `eval_inference_v3_3.db` e use:
  - **Inspetor:** para percorrer passo a passo Fase 1 → transição → Fase 2 com as lentes *Estrutura*, *Policy*, *Visitas MCTS*, *Refutação* e *Deformação / Tensão*.
  - **Comparação:** para colocar lado a lado `ep2__S*` e `simp__S*` no mesmo cenário (e, após as `05:15 BRT`, também comparar `ep30__S*`).
