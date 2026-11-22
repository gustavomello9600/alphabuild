# Impacto das Novas Referências no AlphaBuilder: Arquitetura, Código e Originalidade

Este resumo traduz em linguagem direta como os artigos levantados na **Etapa 0** ([referencias_fichamento.md](file:///c:/Users/admin/Projetos/alphabuild/docs/tcc/referencias_fichamento.md)) moldaram as correções propostas na **Etapa 1** ([etapa1_veredito_politica_recompensa.md](file:///c:/Users/admin/Projetos/alphabuild/docs/tcc/etapa1_veredito_politica_recompensa.md)) e responde à pergunta central: **a ideia do AlphaBuilder continua interessante e única ou ficou obsoleta?**

---

## 1. Veredito Rápido: A Ideia Continua Única ou Ficou Obsoleta?

**Não ficou obsoleta — continua única, relevante e atual em 2026.** Porém, as referências mais novas (2024–2026) mudam **como devemos posicionar a tese do TCC** para que ela seja defensável diante de uma banca exigente.

### O que outros paradigmas já resolveram (onde não vale competir só por velocidade)
* **Modelos neurais *one-shot* e de fundação:** Trabalhos como o **NITO-3D** (`nobari2024towards`, treinado em 122 mil topologias 3D, *speedup* de ~2.000×) e o **OAT** (`nobari2025optimize`, NeurIPS 2025, 2,2 milhões de estruturas 2D em <1 s) mostraram que, se o objetivo for apenas gerar rapidamente uma viga linear-elástica já vista em grandes bancos de dados, modelos implícitos/difusão levam vantagem em latência pura.
* **O próprio SIMP paralelo moderno:** Como lembram `jia2024fenitop`, `ferrari2020new` e `woldseth2022use`, quando existe formulação diferenciável com gradientes adjuntos limpos, o SIMP em FEniCSx já resolve vigas 3D individuais em dezenas de iterações sem precisar treinar nada antes.

### Por que a combinação do AlphaBuilder continua sem equivalente direto até 2026
Mesmo com esse avanço, **a lacuna de busca deliberativa 3D voxelizada com rede neural continua aberta**. Os trabalhos mais próximos na linha de RL e MCTS cobrem apenas partes isoladas da ideia:

| Trabalho | Domínio | Representação / Ações | Busca / Algoritmo | O que o AlphaBuilder faz além |
| :--- | :---: | :--- | :--- | :--- |
| `gaymann2019deep` | Fluido 2D ($8\times 20$) | Células 2D, 1 ação/passo | MCTS + DNN (AlphaZero) | Leva para **3D elástico** (`64×32×8`), 1 jogador, **micro-batches** |
| `brown2022deep` | Contínuo 2D | Só **remoção** do domínio cheio | DDQN (*model-free*, sem árvore) | Parte do **vazio** com **Add/Remove** + **MCTS (PUCT)** |
| `shin2025numerical` | Acústica 2D | Add/Remove do vazio, 1/passo | DDQN (*model-free*, sem árvore) | Usa **Swin-UNETR + MCTS** e **micro-batches PV** em **3D** |
| `luo2022alphatruss`, `du2023automatic`, `garayalde2025mastering` | Treliças discretas 2D/3D | Barras/nós pré-definidos | MCTS (UCT puro, sem rede guia) | Opera no **contínuo voxelizado 3D** sem *ground structure* fixa |
| `karvanpour2025topology` | Contínuo 2D (CST) | Adição unitária sob gramática | MCTS (UCT puro, sem rede guia) | Mostra que o grupo do Politecnico di Milano está indo para o contínuo só agora (2025/26), ainda em 2D sem rede neural |
| `rochefortbeaudoin2025structural` (**SOgym**) | Contínuo 2D (MMC) | Primitivas geométricas contínuas | DreamerV3 (*model-based RL*) | Trabalha direto em **voxels 3D binários** com árvore PUCT explícita |

Em resumo: **ninguém juntou até hoje contínuo elástico 3D voxelizado + Swin-UNETR dual-head + PUCT/MCTS + Add/Remove do vazio + micro-batches PV + warm-up $\to$ self-play em FEniCSx**. O fato de grupos fortes como Garayalde/Karvanpour (2025) e Rochefort-Beaudoin et al. (2025) estarem convergindo agora para MCTS e *model-based RL* no contínuo confirma que a linha de pesquisa está viva e atual — e não superada.

---

## 2. Como as Novas Referências Guiaram as Correções no Código e na Arquitetura

Quase todas as falhas encontradas na auditoria da Etapa 1 ganharam explicação teórica exata — e solução concreta — a partir dos novos artigos:

### A. Política, Ações Legais e MCTS
* **`huang2022closer` (Invalid Action Masking):** Provou que mascarar ações inválidas só gera gradientes consistentes quando a perda é normalizada **após aplicar a máscara**. No código atual, `trainer.py` treina com `BCEWithLogitsLoss` em todo o volume (inclusive fora da fronteira, onde o MCTS nunca pode jogar — desperdiçando ~55% da probabilidade na Fase 1), e não bloqueia a remoção de apoios/cargas. **Correção:** migrar para **Cross-Entropy / KL-Divergence mascarada sobre $\mathcal{A}_{\text{legal}}(s)$** e ativar `get_protected_regions` na remoção.
* **`jacob2022modeling` ($\pi\text{KL}$):** Mostrou que pesos assimétricos ou calibração ruim no pré-treino distorcem os *logits* do *softmax* consumido pelo MCTS, enquanto *self-play* sem âncora diverge do suporte válido. **Correção:** eliminar `POS_WEIGHT_ADD=8.0` vs `POS_WEIGHT_REM=5.0` (que criava viés artificial de $1,6\times$ para `Add`), usar normalização conjunta na Fase 2 e considerar regularização KL à política de *warm-up* no início do *self-play*.
* **`oh2026efficient` e `jeon2026multiscale`:** Reforçam que espaços voxelizados sofrem explosão combinatória e validam o uso de seleção em lote / multi-escala (como o `select_pv_sequences`), mas expuseram que no código atual o corte `top_k_expansion` não renormaliza os *priors* antes do ruído de Dirichlet (fazendo o ruído dominar a rede por $2,56\times$) e usa lote grande demais (`32`) na Fase 2, exigindo redução para `4–8`.
* **`wu2019accelerating` (KataGo):** Demonstrou redução de ~50× no custo de aprendizado por *self-play* ao adicionar uma cabeça auxiliar espacial prevendo o estado final do tabuleiro (*ownership target*) para enriquecer os gradientes no *backbone* compartilhado. Inspirou diretamente a nova **`Blueprint Head` auxiliar** (1 canal prevendo a estrutura final $X_T$ tanto no pré-treino quanto nos episódios bem-sucedidos do *self-play*), dando sinal espacial denso ao "cérebro" da rede sem contaminar os *logits* do MCTS.

### B. Recompensa, Calibração Offline-to-Online e Busca
* **`nakamoto2023calql` (Cal-QL):** Explica o risco de *unlearning* imediato quando a Value Head é pré-treinada numa escala (*offline*) e avaliada noutra (*online*). Foi decisivo para exigir a remoção imediata da fórmula legada `v3.1 tanh` ainda ativa em `run_data_harvest.py`/`processing.py`, unificando o `fitness_score` do banco com a escala física real do `reward.py`.
* **`rochefortbeaudoin2025structural` e `yi2026reinforcement`:** Confirmam que escala logarítmica ($\log_{10} C$) é o caminho certo para *compliance*, mas alertam para *reward hacking* quando o peso do volume domina o da rigidez: na `v3.2` atual, reduzir $10\%$ de volume compensa piorar a *compliance* em quase **$18\times$**, e todos os exemplos válidos ficam comprimidos entre `0,53` e `0,71`. **Correção:** recalibrar os coeficientes num piloto de 50–100 episódios para cobrir `[+0,15, +0,75]`.
* **`hayashi2020reinforcement`:** Inspirou trocar o corte seco do colapso (`max_displacement > U_lim -> -1.0`, invisível até quebrar) por uma **penalidade suave contínua de margem de deslocamento ($P_{\text{disp}}$)** a partir de $70\%$ do limite, dando gradiente de alerta à rede.
* **`muller2025improving` e `wang2020warmstart`:** Mostraram que somar o bônus absoluto `+B_conn - P_ilhas` direto sobre `V_net` no `hybrid_predictor` viola a invariância do *Potential-Based Reward Shaping* (PBRS), causa dupla contagem e gera uma queda brusca na transição Fase 1 $\to$ Fase 2. **Correção:** usar PBRS como diferença de potencial $\Phi(s') - \Phi(s)$ com decaimento gradual ($\alpha_{\text{guide}} \to 0$) do *warm-up* para o *self-play*.
* **`luo2022alphatruss`:** Alertou que a média simples `W/N` no MCTS pode deixar ramos com `-1.0` afogarem subárvores promissoras na Fase 1, motivando monitorar/testar backup híbrido *MixMax* na busca.
* **`garayalde2025mastering` e `du2023automatic`:** Provaram que os **primeiros passos na árvore** dominam a qualidade final alcançável — revelando quão grave era `min_volume_fraction = 0.05` travar os 32 primeiros passos da Fase 1 em profundidade 1 com `Q = -1.0`. **Correção:** zerar `min_volume_fraction = 0.0` na Fase 1.

### C. Arquitetura 3D, Normalização e Padding
* **`hatamizadeh2022swin` (Swin-UNETR):** Mostrou que o `SwinPolicyDecoder` do AlphaBuilder omitiu a *skip connection* em resolução cheia (`enc0`), dependendo de interpolação trilinear `D/2 -> D` que borra decisões de 1 voxel na fronteira. **Correção:** adicionar bloco residual `enc0` e *skip* na resolução original.
* **`nie2021topologygan`, `maze2023diffusion` e `lutheran2026diffusion`:** Explicaram por que redes 3D sofrem quando recebem condições de contorno e forças como tensores quase inteiros de zeros (`9` voxels ativos em `65.536`). Combinado à nossa auditoria, isso explicou por que `InstanceNorm3d(affine=False)` anulava a escala da força e distorcia a Value Head em `0,35` apenas por causa do padding. **Correção:** trocar por `GroupNorm(..., affine=True)` preservando escala dos canais físicos e dar ao `SimpleBackbone` campo receptivo $\ge 64$ + condicionamento global (*FiLM/SE* + *CoordConv*).
* **`zhao2026variable` e `nobari2024towards`:** Expuseram o custo do `_dynamic_pad(..., divisor=32)` sobre domínios finos como `64×32×8` (75% de zeros artificiais), motivando padronizar o padding entre treino/inferência agora e registrar tokenização dinâmica / campos implícitos como caminho futuro.

### D. Rigor Experimental e Baselines (Etapa 6)
* **`jia2024fenitop` e `baratta2023dolfinx`:** Preenchem a lacuna formal de citação do próprio solver FEniTop/DOLFINx já embutido no projeto e garantem um baseline 3D exato.
* **`sigmund2022benchmarking`, `woldseth2022use` e `ferrari2020new`:** Impõem as regras de ouro para a Etapa 6: só comparar estruturas **binárias 0–1**, medir **tempo de parede** (*wall-clock*) e **número de avaliações FEM ($Ku=f$)** em hardware equivalente, contabilizar **taxa de desconexão** e calcular o ***break-even*** incluindo o custo de gerar dados e treinar a rede.

---

## 3. Como Isso Muda a Perspectiva do TCC (Antes vs. Agora)

| Aspecto | Visão Anterior (Risco de Crítica da Banca) | Nova Perspectiva (Sólida e Atualizada para 2026) |
| :--- | :--- | :--- |
| **Promessa principal** | "Substituir o SIMP e bater métodos clássicos em velocidade/compliance geral." | **Prova de conceito de um agente deliberativo 3D (Swin-UNETR + MCTS) livre de gradientes adjuntos e nativamente binário (0–1)**, testado em benchmark rigoroso controlado pelo FEniTop. |
| **Comparação com ML *one-shot*** | Competir com CNNs antigas de 2018–2021 (`Banga`, `Rawat & Shen`, `Yu`). | Posicionar claramente a diferença frente a modelos de fundação *one-shot* de 2024–2025 (`NITO-3D`, `OAT`): o AlphaBuilder não busca 122 mil–2,2 milhões de amostras para memorizar elástico linear, mas sim **busca deliberativa com verificação física**, essencial onde *one-shot* desconecta ou falha fora da distribuição. |
| **Papel da Etapa 6 (Self-Play)** | Apenas "mostrar que a rede melhora com o tempo". | **Estudo quantitativo de escalabilidade e *break-even*** nos moldes de `woldseth2022use` e `sigmund2022benchmarking`: mesmo que o orçamento de hardware de um TCC não ultrapasse o SIMP em *wall-clock*, mapear *quantas simulações e horas de GPU* seriam necessárias para cada patamar é contribuição científica genuína. |
| **Horizonte de aplicação (Etapa 7)** | Foco restrito à viga em balanço linear. | Deixar claro que a viga linear no FEniTop é o **ambiente de validação controlada**; o valor estratégico dessa arquitetura aparece em problemas **não-diferenciáveis, com restrições discretas de manufatura, contato, flambagem ou múltiplos objetivos**, onde o SIMP adjunto e o *one-shot* sem busca enfrentam barreiras sérias. |

---

## 4. Conclusão Prática

1. **Vale totalmente seguir em frente:** o nicho **MCTS + Rede Neural 3D em contínuo voxelizado** continua aberto na literatura até 2026 e é altamente original para um TCC de Engenharia Civil / Computação.
2. **As referências blindam o trabalho:** incorporar os ajustes da Etapa 1 (máscara legal na loss, escala unificada sem `tanh`, `GroupNorm`, PBRS, `min_volume_fraction=0.0` e protocolo de `sigmund2022benchmarking`) transforma o projeto de "experimento que falhou no *warm-up*" em uma **engenharia de RL estrutural fundamentada no estado da arte**.
3. **Próximo passo recomendado:** validar os pontos do checkpoint da Etapa 1 e Etapa 2 para aplicarmos as correções no código antes de iniciar a geração distribuída dos novos dados.
