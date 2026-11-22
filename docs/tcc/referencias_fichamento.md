# Fichamento Bibliográfico Consolidado e Auditoria de Referências — AlphaBuilder (TCC)

Este documento consolida, deduplica e organiza por eixos temáticos todas as referências bibliográficas e fontes técnicas provenientes do levantamento prévio no Google Drive (**Relatório 0A**) e das buscas especializadas em **RL/MCTS** (**Relatório 0B**), **Deep Learning e Modelos Generativos 3D** (**Relatório 0C**), **Reward Shaping, Conectividade e Offline-to-Online RL** (**Relatório 0D**) e **Baselines Clássicos e Benchmarks 3D** (**Relatório 0E**).

Todas as chaves BibTeX abaixo correspondem exatamente às entradas válidas do arquivo [referencias.bib](file:///c:/Users/admin/Projetos/alphabuild/docs/tcc/referencias.bib).

---

## Sumário Quantitativo e Registro de Deduplicação (`0A`–`0E`)

| Eixo Temático | Pré-existentes (`0A`) | Enriquecidas / Corrigidas (`0A + 0B..0E`) | Novas Incorporadas (`0B..0E`) | Total de Entradas `.bib` |
| :--- | :---: | :---: | :---: | :---: |
| **1. Fundamentos de OT e Baselines 3D / FEniCSx / FEniTop** | 6 | 1 | 9 | **16** |
| **2. Deep Learning e Modelos Generativos 3D para TopOpt** | 21 | 3 | 10 | **34** |
| **3. Reinforcement Learning, MCTS e AlphaZero em Design Estrutural** | 8 | 2 | 17 | **27** |
| **4. Reward Shaping, Conectividade e Pré-treino + Self-Play** | 1 | 0 | 7 | **8** |
| **5. Referências Prévias do Projeto / Recursos Técnicos / Estado v5** | 26 | 0 | 0 | **26** |
| **TOTAL CONSOLIDADO** | **62** | **6** | **43** | **111** |

> **Contagem total de pré-existentes únicas de `0A`:** $62 + 6 = \mathbf{68}$ entradas únicas no `.bib` (além de 10 fragmentos autor/ano sem título, 1 menção a PyTorch e links terciários não acadêmicos auditados na Seção 5.3).
> **Contagem total de novas referências únicas (`0B`–`0E` + `wu2019accelerating`):** $\mathbf{43}$ entradas ($17$ de `0B`, $10$ de `0C`, $7$ de `0D` + `wu2019accelerating`, $9$ de `0E`), após remoção de interseções entre relatórios (`luo2022alphatruss`, `brown2022deep`, `hayashi2020reinforcement`, `rochefortbeaudoin2025structural`).

### Principais Deduplicações e Correções Realizadas

1. **`du2023automatic` (*AutoTruss*, IJCAI 2023):** Em `tcc_v5.txt` (L1061) e `tcc_draft_01.md` (L65), o artigo *Automatic Truss Design with Reinforcement Learning* estava atribuído erroneamente a `LI, Y. et al.`. O Relatório `0B` confirmou tratar-se de **Du, Weihua et al. (2023)** (tendo Yi Wu como último autor; DOI `10.24963/ijcai.2023/407`, `arXiv:2306.15182`).
2. **`hayashi2020reinforcement` (Hayashi & Ohsaki):** Em `tcc_v5.txt` (L1058) e `tcc_draft_01.md` (L63), o ano constava como `2021`. Os Relatórios `0B` e `0D` verificaram via DOI `10.3389/fbuil.2020.00059` que o ano correto é **2020**.
3. **`bhuiyan2026graph` vs. `Tabarraei & Bhuiyan (2025, ResearchGate)`:** Em `tcc_v5.txt` (L1073) e `tcc_draft_01.md` (L77), constava apenas uma referência incompleta ao ResearchGate (2025) com ordem invertida de autores (`Tabarraei, A.; Bhuiyan, M.`). O Relatório `0C` localizou o preprint oficial (`arXiv:2508.19169`) e a publicação definitiva em *Engineering with Computers* (2026, DOI `10.1007/s00366-026-02344-y`) como **Bhuiyan, Saquib Ahmad; Tabarraei, Alireza**.
4. **`yan2022deep` vs. item #6 de `pesquisa_referencias.md`:** O item #6 da tabela de `pesquisa_referencias.md` (`(2021) Deep learning driven real time topology optimisation based on initial stress learning`) não possuía autores na fonte `0A`. O Relatório `0B` identificou os autores completos, periódico e DOI (`10.1016/j.aei.2021.101472`, *Advanced Engineering Informatics*, Vol. 51, 2022).
5. **`rochefortbeaudoin2025structural` (*SOgym*):** Deduplicado entre `0E` (que citava o preprint `arXiv:2407.07288` de 2024) e `0B`/`0D` (que trouxeram a publicação em *Engineering Applications of Artificial Intelligence*, Vol. 161, 2025, DOI `10.1016/j.engappai.2025.112273`).
6. **`hatamizadeh2022swin` e `bendsoe2003topology`:** Já presentes em `tcc_v5.txt` (`0A`), enriquecidos com lista completa de autores, DOI, ISBN e arXiv ID via `0C` e `0E`.
7. **`lutheran2025transformer` vs. `lutheran2026diffusion`:** Mantidos como entradas distintas por apresentarem títulos e IDs diferentes (`arXiv:2509.05800` de `0A` vs. *Structural and Multidisciplinary Optimization*, 2026 / `arXiv:2605.02158` de `0C`), recomendando-se conferência do preprint `2509.05800` antes da versão final do texto.
8. **`wang2024micrometer`:** Resolvida a divergência de ano entre `tcc_v5.txt` (`2024`) e `tcc_draft_01.md` (`2024` na lista e `2025` no texto) mantendo `2024` (`arXiv:2410.05281`).

---

## Eixo 1: Fundamentos de Otimização Topológica e Baselines 3D / FEniCSx / FEniTop

### `bendsoe2003topology`
- **Citação Completa:** BENDSØE, M. P.; SIGMUND, O. *Topology Optimization: Theory, Methods, and Applications*. Berlin: Springer Berlin Heidelberg, 2003. ISBN 978-3-540-42992-0. DOI: `10.1007/978-3-662-05086-6`. *(Origem: Pré-existente `0A` + Enriquecida `0E`)*
- **Resumo Técnico Objetivo:** Obra fundacional que formaliza a otimização topológica do contínuo pelo método SIMP (*Solid Isotropic Material with Penalization*), modelando rigidez pela lei de potência $E_e = \rho_e^p E_0$ ($p \ge 3$), derivando sensibilidades adjuntas para minimização de *compliance* sob restrição de volume e sistematizando filtros contra instabilidades de tabuleiro (*checkerboard*) e dependência de malha.
- **Pontos Fortes / Limitações:** Rigor matemático completo e padrão universal de formulação; entretanto, a abordagem contínua baseada em gradientes requer filtros/projeções para eliminar densidades cinzentas e tende a convergir deterministicamente para um único ótimo local por execução.
- **Comparação Direta com o AlphaBuilder:** Fornece a formulação elástica linear e a penalização SIMP ($p=3{,}0$) usadas tanto no solver de referência FEniTop quanto no avaliador FEM da Fase 2 (`reward.py` / `evaluator.py`), diferindo do AlphaBuilder na representação (densidades contínuas vs. ações binárias discretas 0–1 em voxels 3D).
- **Onde Citar no TCC:** Cap. 1.1 (O problema), Cap. 2.1 (Fundamentação teórica — SIMP) e Cap. 4.1 (Problema de referência); corrigir a ausência de citação inline no corpo de `tcc_v5.txt`.

### `sigmund200199line`
- **Citação Completa:** SIGMUND, O. A 99 line topology optimization code written in Matlab. *Structural and Multidisciplinary Optimization*, v. 21, n. 2, p. 120–127, 2001. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Apresenta a implementação didática clássica do método SIMP em 2D usando Critérios de Otimalidade (OC), filtragem de sensibilidade independente de malha e formulação compacta de minimização de *compliance* em viga MBB.
- **Pontos Fortes / Limitações:** Referência pedagógica consolidada para validação de pipelines SIMP 2D; restrito a geometrias cartesianas 2D simples e solver direto pequeno.
- **Comparação Direta com o AlphaBuilder:** Inspira os testes preliminares de sanidade em 2D do projeto (como o baseline SIMP 2D `64x32` da Seção 5.2 de `tcc_v5.txt`), antes da escalada para domínios 3D paralelos via FEniTop/DOLFINx.
- **Onde Citar no TCC:** Cap. 2.1 (SIMP) e Cap. 5.2 (Baseline SIMP 2D); atualmente listado em `tcc_v5.txt` sem citação inline.

### `sigmund2013topology`
- **Citação Completa:** SIGMUND, O.; MAUTE, K. Topology optimization approaches. *Structural and Multidisciplinary Optimization*, v. 48, p. 1031–1055, 2013. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Revisão comparativa das grandes famílias de otimização topológica (métodos baseados em densidade/SIMP, derivadas topológicas, *Level-Set*, *phase-field*, ESO/BESO e métodos sem gradiente), analisando custo computacional, tratabilidade de restrições e escalabilidade.
- **Pontos Fortes / Limitações:** Síntese crítica autoritativa comparando paradigmas contínuos e discretos; anterior à onda de modelos de Deep Learning e RL profundo pós-2018.
- **Comparação Direta com o AlphaBuilder:** Fundamenta a discussão do porquê o paradigma sequencial discreto (Add/Remove em grade) e métodos baseados em aprendizado enfrentam o trade-off de custo de avaliações FEM frente ao método adjunto do SIMP.
- **Onde Citar no TCC:** Cap. 1.1, Cap. 2.1 e Cap. 2.7 (já citado em L61, L103 e L183 de `tcc_v5.txt`).

### `deaton2014survey`
- **Citação Completa:** DEATON, J. D.; GRANDHI, R. V. A survey of structural and multidisciplinary continuum topology optimization: post 2000. *Structural and Multidisciplinary Optimization*, 2014. *(Origem: Pré-existente `0A` — `ref_alphabuilder.md`)*
- **Resumo Técnico Objetivo:** Revisão abrangente dos avanços em otimização topológica estrutural e multidisciplinar no contínuo entre 2000 e 2013, cobrindo formulações multifísicas, manufaturabilidade e métodos numéricos de grande porte.
- **Pontos Fortes / Limitações:** Contextualiza extensões práticas além da elasticidade linear pura; não cobre métodos neurais ou aprendizado por reforço.
- **Comparação Direta com o AlphaBuilder:** Contextualiza a evolução histórica entre o SIMP clássico (Bendsøe & Sigmund, 2003) e os desafios contemporâneos de conectividade, manufatura e escalabilidade 3D atacados pelo AlphaBuilder.
- **Onde Citar no TCC:** Cap. 2.1 (SIMP e evolução pós-2000) ou Cap. 2.6 (Conectividade e manufaturabilidade).

### `kane1996topological`
- **Citação Completa:** KANE, C.; SCHOENAUER, M. Topological optimum design using genetic algorithms. *Control and Cybernetics*, v. 25, n. 5, p. 1059–1088, 1996. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Trabalho pioneiro em otimização topológica discreta via Algoritmos Genéticos (AGs), operando diretamente em representações binárias 0–1 sem necessidade de cálculo de sensibilidades analíticas.
- **Pontos Fortes / Limitações:** Dispensa gradientes contínuos e opera no espaço discreto nativo; sofre explosão combinatória e alto número de chamadas FEM em grades finas, além de desconexões frequentes sob cruzamento/mutação ingênuos.
- **Comparação Direta com o AlphaBuilder:** Representa a linhagem direta do trabalho anterior de iniciação científica do autor (`mello2021relatorio`), contrastando busca evolutiva sem memória aprendida com a busca deliberativa guiada por rede neural (MCTS + Swin-UNETR) do AlphaBuilder.
- **Onde Citar no TCC:** Cap. 1.3 (Origem PIBIC), Cap. 2.2 (Métodos sem gradiente e AGs) e Cap. 2.7 (já citado em L77, L108 e L184 de `tcc_v5.txt`).

### `zienkiewicz2013finite`
- **Citação Completa:** ZIENKIEWICZ, O. C.; TAYLOR, R. L.; ZHU, J. Z. *The Finite Element Method: Its Basis and Fundamentals*. Butterworth-Heinemann, 2013. *(Origem: Pré-existente `0A` — `ref_alphabuilder.md`)*
- **Resumo Técnico Objetivo:** Texto de referência clássica sobre a formulação variacional de Galerkin, discretização por elementos finitos, funções de forma, condições de contorno de Dirichlet/Neumann e montagem do sistema de equilíbrio $Ku = f$.
- **Pontos Fortes / Limitações:** Base teórica completa para o acoplamento com solvers FEM; obra geral de elementos finitos sem foco específico em otimização topológica.
- **Comparação Direta com o AlphaBuilder:** Embasamento teórico para a formulação fraca UFL em elasticidade linear 3D implementada em `fenitop/fem.py` e no avaliador de recompensas em `evaluator.py`.
- **Onde Citar no TCC:** Cap. 2.1 ou Cap. 3.1 / Cap. 4.1 (Discretização FEM em elementos hexaédricos).

### `scroggs2022construction`
- **Citação Completa:** SCROGGS, M. W. et al. Construction of arbitrary order finite element degree-of-freedom maps on polygonal and polyhedral cell meshes. *ACM Transactions on Mathematical Software*, v. 48, n. 2, 2022. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Formaliza a construção de mapas de graus de liberdade (DoFs) de ordem arbitrária em malhas poligonais e poliédricas (incluindo hexaedros), sustentando a arquitetura moderna do FEniCSx/DOLFINx.
- **Pontos Fortes / Limitações:** Garante suporte matemático rigoroso e desempenho paralelo para malhas 3D hexaédricas; referência altamente específica de infraestrutura matemática do FEniCSx.
- **Comparação Direta com o AlphaBuilder:** Fundamenta o mapeamento entre a grade voxelizada 3D `(64, 32, 8)` da rede neural (`tensor_utils.py`) e os espaços de funções `DG0` (densidade de projeto) e `CG1` (deslocamento e densidade física) na malha hexaédrica `create_box(..., CellType.hexahedron)` em `harvest/optimization.py`.
- **Onde Citar no TCC:** Cap. 3.6 (Infraestrutura FEniCSx) ou Cap. 4.1–4.2 (Mapeamento grade-malha); corrigir ausência de citação inline em `tcc_v5.txt`.

### `liu2014efficient`
- **Citação Completa:** LIU, K.; TOVAR, A. An efficient 3D topology optimization code written in Matlab. *Structural and Multidisciplinary Optimization*, v. 50, n. 6, p. 1175–1196, 2014. DOI: `10.1007/s00158-014-1107-x`. *(Origem: Nova `0E`)*
- **Resumo Técnico Objetivo:** Introduz o código compacto `top3d.m` (169 linhas) em elementos hexaédricos de 8 nós para otimização topológica 3D com SIMP, filtro espacial e atualização via OC, estabelecendo a parametrização canônica da viga em balanço 3D (`nelx, nely, nelz, volfrac, penal, rmin`).
- **Pontos Fortes / Limitações:** Código-referência amplamente replicado para benchmarks 3D em grades cartesianas; limitado em escala por solvers diretos em MATLAB quando o número de elementos cresce.
- **Comparação Direta com o AlphaBuilder:** Consolida a parametrização da viga em balanço 3D em grade hexaédrica adotada pelo AlphaBuilder (ex.: `64 x 32 x 8` com $p=3{,}0$), servindo de referência histórica de baseline antes da migração para MPI/PETSc via FEniTop.
- **Onde Citar no TCC:** Cap. 2.1 (SIMP em 3D) e Cap. 4.1 (Padronização do problema de viga em balanço 3D).

### `ferrari2020new`
- **Citação Completa:** FERRARI, F.; SIGMUND, O. A new generation 99 line Matlab code for compliance topology optimization and its extension to 3D. *Structural and Multidisciplinary Optimization*, v. 62, p. 2211–2228, 2020. DOI: `10.1007/s00158-020-02629-w`. `arXiv:2005.05436`. *(Origem: Nova `0E`)*
- **Resumo Técnico Objetivo:** Propõe `top99neo` (2D) e `top3D125` (3D), otimizando montagem matricial, filtragem e atalhos de OC, e demonstrando em malhas $48\times24\times24$ e $96\times48\times48$ que até **96% do tempo total de CPU** em 3D ocorre na resolução do sistema linear $Ku=f$, alcançando speedup de ~1,8–1,9× com PCG pré-condicionado por multigrid (MGCG).
- **Pontos Fortes / Limitações:** Prova quantitativamente onde está o gargalo computacional da OT 3D e fornece código de referência vetorizado; permanece um otimizador iterativo clássico baseado em gradiente.
- **Comparação Direta com o AlphaBuilder:** Justifica tecnicamente a escolha de `ksp_type="cg"` + `pc_type="gamg"` (multigrid algébrico no PETSc) em `harvest/optimization.py` e fundamenta a exigência da Etapa 6 de medir *wall-clock time* e número total de chamadas FEM ($Ku=f$), e não apenas "iterações".
- **Onde Citar no TCC:** Cap. 2.1, Cap. 4.7 (Métricas) e Cap. 6.3 (Comparação de custo computacional entre caminhos).

### `huang2007convergent`
- **Citação Completa:** HUANG, X.; XIE, Y. M. Convergent and mesh-independent solutions for the bi-directional evolutionary structural optimization method. *Finite Elements in Analysis and Design*, v. 43, n. 14, p. 1039–1049, 2007. DOI: `10.1016/j.finel.2007.06.006`. *(Origem: Nova `0E`)*
- **Resumo Técnico Objetivo:** Formaliza o método BESO (*Bi-directional Evolutionary Structural Optimization*), que adiciona e remove elementos discretos (0/1) com base em números de sensibilidade, resolvendo instabilidades, efeito tabuleiro e dependência de malha via filtro espacial e média histórica entre iterações.
- **Pontos Fortes / Limitações:** Produz soluções discretas 0–1 nítidas sem pós-binarização e permite adição e remoção simultânea; ainda depende de sensibilidades adjuntas do FEM a cada passo.
- **Comparação Direta com o AlphaBuilder:** Oferece o análogo clássico mais próximo à formulação de espaço de ações bidirecional **Add/Remove** em voxels binários do AlphaBuilder, contrastando com o AlphaBuilder pelo uso de política aprendida e busca MCTS em vez de atualização determinística por sensibilidade local.
- **Onde Citar no TCC:** Cap. 2.1 / Cap. 2.2 (Métodos discretos vs. contínuos) e Cap. 2.3 (Motivação para Add/Remove no MDP).

### `huang2010evolutionary`
- **Citação Completa:** HUANG, X.; XIE, Y. M. *Evolutionary Topology Optimization of Continuum Structures: Methods and Applications*. Wiley, 2010. ISBN 978-0-470-74653-0. *(Origem: Nova `0E`)*
- **Resumo Técnico Objetivo:** Monografia abrangente que consolida o método BESO para estruturas contínuas 2D e 3D, cobrindo problemas de rigidez, frequência natural, não linearidade e extensões práticas de engenharia. *(Complementado na literatura por implementações compactas Python 3D como Zuo & Xie, 2015, *Advances in Engineering Software*, DOI `10.1016/j.advengsoft.2015.02.006`).*
- **Pontos Fortes / Limitações:** Referência de livro-texto completa para métodos evolucionários discretos 0–1; menos flexível que MMA/FEniTop para múltiplos tipos de restrições acopladas.
- **Comparação Direta com o AlphaBuilder:** Ajuda a posicionar a escolha do AlphaBuilder por topologias binárias diretas (0/1) em voxels 3D frente aos dois grandes paradigmas clássicos: densidade contínua (SIMP/FEniTop) e evolucionário discreto (BESO).
- **Onde Citar no TCC:** Cap. 2.1 / Cap. 2.2 (Taxonomia de métodos clássicos 3D).

### `allaire2004structural`
- **Citação Completa:** ALLAIRE, G.; JOUVE, F.; TOADER, A.-M. Structural optimization using sensitivity analysis and a level-set method. *Journal of Computational Physics*, v. 194, p. 363–393, 2004. DOI: `10.1016/j.jcp.2003.09.032`. *(Origem: Nova `0E`)*
- **Resumo Técnico Objetivo:** Acopla derivadas de forma via método adjunto à equação de Hamilton–Jacobi de propagação de frente (*Level-Set*) sobre malha Euleriana fixa em 2D e 3D (incluindo viga em balanço 3D), permitindo fusão/abertura de furos com fronteiras nítidas sem densidades intermediárias. *(Contemporâneo a Wang, Wang & Guo, 2003, *CMAME*, DOI `10.1016/S0045-7825(02)00559-5`).*
- **Pontos Fortes / Limitações:** Fronteiras geométricas suaves e livres de *greyness*; implementação numérica mais complexa (reinicialização da função $\phi$) e sensibilidade à inicialização de furos.
- **Comparação Direta com o AlphaBuilder:** Posiciona a representação por fronteira implícita (*Level-Set*) em oposição à parametrização por densidade contínua (FEniTop) e por ocupação discreta de voxels (AlphaBuilder).
- **Onde Citar no TCC:** Cap. 2.1 (Panorama das abordagens clássicas de OT 3D).

### `baratta2023dolfinx`
- **Citação Completa:** BARATTA, I. A.; DEAN, J. P.; DOKKEN, J. S.; HABERA, M.; HALE, J. S.; RICHARDSON, C. N.; ROGNES, M. E.; SCROGGS, M. W.; SIME, N.; WELLS, G. N. *DOLFINx: The next generation FEniCS problem solving environment*. Zenodo, 2023. DOI: `10.5281/zenodo.10447666`. *(Origem: Nova `0E`)*
- **Resumo Técnico Objetivo:** Apresenta a plataforma DOLFINx (núcleo do ecossistema FEniCSx), reconstruída em C++/Python com suporte nativo a computação paralela via MPI, geração automática de código via UFL e resolução distribuída de álgebra linear com PETSc/`petsc4py`.
- **Pontos Fortes / Limitações:** Alto desempenho paralelo e expressividade variacional em UFL; exige compatibilidade estrita de versões entre FEniCSx, PETSc e MPI (como evidenciado pelo erro MPI código 59 registrado no projeto).
- **Comparação Direta com o AlphaBuilder:** É o motor FEM efetivo usado tanto na geração de dados via FEniTop (`harvest/optimization.py`) quanto na avaliação física da Fase 2 (`evaluator.py`).
- **Onde Citar no TCC:** Cap. 3.1 / Cap. 3.6 (Infraestrutura de simulação) e Cap. 3.8 (Discussão sobre containerização e MPI código 59).

### `jia2024fenitop`
- **Citação Completa:** JIA, Y.; WANG, C.; ZHANG, X. S. FEniTop: a simple FEniCSx implementation for 2D and 3D topology optimization supporting parallel computing. *Structural and Multidisciplinary Optimization*, v. 67, p. 140, 2024. DOI: `10.1007/s00158-024-03818-7`. *(Origem: Nova `0E` — supre ausência crítica apontada por `0A`)*
- **Resumo Técnico Objetivo:** Implementa otimização topológica 2D/3D em FEniCSx com diferenciação automática UFL, filtro PDE de Helmholtz, projeção suave de Heaviside com continuação $\beta$, e otimizadores paralelos OC e MMA (Svanberg, 1987, DOI `10.1002/nme.1620240207`) sobre PETSc/MPI, incluindo benchmark 3D em viga em balanço (`beam_3d.py`).
- **Pontos Fortes / Limitações:** Código modular, limpo e paralelo em Python/C++ com rigor matemático SIMP; requer dezenas a centenas de iterações FEM por nova instância geométrica.
- **Comparação Direta com o AlphaBuilder:** É a **base direta do solver incorporado em `alphabuilder/src/logic/fenitop/`** (`fem.py`, `parameterize.py`, `sensitivity.py`, `optimize.py`, `topopt.py`) e também o **baseline clássico exato** para a Etapa 6 (`harvest/optimization.py`), garantindo paridade perfeita de malha hexaédrica, propriedades ($E=100$, $\nu=0{,}25$) e solver (`cg` + `gamg`).
- **Onde Citar no TCC:** Cap. 1, Cap. 3.2, Cap. 4.1–4.2, Cap. 5.2–5.3 e Cap. 6.3 (substitui a lacuna formal nas 9 citações de "FEniTop" no corpo de `tcc_v5.txt`).

### `woldseth2022use`
- **Citação Completa:** WOLDSETH, R. V.; AAGE, N.; BÆRENTZEN, J. A.; SIGMUND, O. On the use of artificial neural networks in topology optimisation. *Structural and Multidisciplinary Optimization*, v. 65, p. 294, 2022. DOI: `10.1007/s00158-022-03347-1`. `arXiv:2208.02563`. *(Origem: Nova `0E`)*
- **Resumo Técnico Objetivo:** Revisão crítica das abordagens de redes neurais em OT, propondo métricas rigorosas como *break-even score* (quantas consultas amortizam coleta de dados + treino frente ao SIMP) e *generativity score* (generalização fora da distribuição), e alertando que erros voxel-a-voxel geram desconexões catastróficas e explosão de *compliance* após binarização 0–1.
- **Pontos Fortes / Limitações:** Estabelece padrões experimentais indispensáveis contra afirmações exageradas de *speedup* em ML-TopOpt; foca majoritariamente em arquiteturas supervisionadas/generativas *one-shot* (menos em busca sequencial via RL/MCTS).
- **Comparação Direta com o AlphaBuilder:** Valida cientificamente a necessidade de penalizar ilhas e desconexões (`reward.py`), testar generalização em novas instâncias e contabilizar o custo *offline* (1.069–2.000 episódios + treino Swin-UNETR) no cálculo de *break-even* frente ao FEniTop.
- **Onde Citar no TCC:** Cap. 2.6 (Conectividade), Cap. 4.7 (Métricas), Cap. 6.3–6.4 (Ameaças à validade e comparação justa) e Cap. 7 (Critérios de comparação da Etapa 6).

### `sigmund2022benchmarking`
- **Citação Completa:** SIGMUND, O. On benchmarking and good scientific practise in topology optimization. *Structural and Multidisciplinary Optimization*, v. 65, p. 315, 2022. DOI: `10.1007/s00158-022-03427-2`. *(Origem: Nova `0E`)*
- **Resumo Técnico Objetivo:** Define diretrizes metodológicas para comparação justa entre métodos (SIMP/MMA vs. heurísticas, ML e RL): mesma malha, mesmas condições de contorno, paridade de fração de volume, medição de *greyness* ($M_{nd}$) e avaliação pós-binarização 0–1 acompanhada de tempo de parede (*wall-clock time*) em hardware equivalente.
- **Pontos Fortes / Limitações:** Padrão-ouro metodológico aceito pela comunidade de engenharia estrutural; recomendações conceituais aplicáveis a qualquer protocolo experimental.
- **Comparação Direta com o AlphaBuilder:** Proíbe comparar soluções contínuas cinzentas do FEniTop sem projeção binária contra as estruturas nativamente 0–1 do AlphaBuilder e exige que o orçamento igualado na Tabela 17 de `tcc_v5.txt` seja medido em tempo de relógio total e chamadas FEM no mesmo hardware.
- **Onde Citar no TCC:** Cap. 4.1, Cap. 4.7, Cap. 6.3 (Tabela 17) e Cap. 6.4 (Ameaças à validade).

---

## Eixo 2: Deep Learning e Modelos Generativos 3D para TopOpt

### 2A. Arquiteturas Base de Transformers, ViT e Swin-UNETR

### `vaswani2017attention`
- **Citação Completa:** VASWANI, A.; SHAZEER, N.; PARMAR, N. et al. Attention is All you Need. In: *NeurIPS 2017*, 2017. *(Origem: Pré-existente `0A` — `ref_alphabuilder.md`)*
- **Resumo Técnico Objetivo:** Introduz a arquitetura *Transformer* baseada integralmente em mecanismos de autoatenção (*Multi-Head Self-Attention*) e codificações posicionais, eliminando recorrência na modelagem de dependências de longo alcance.
- **Pontos Fortes / Limitações:** Base conceitual de toda a família ViT/Swin; complexidade quadrática no número de tokens quando aplicada sem janelamento hierárquico em grades 3D densas.
- **Comparação Direta com o AlphaBuilder:** Fundamenta indiretamente os blocos de autoatenção presentes no `SwinEncoder` do `AlphaBuilderV31`.
- **Onde Citar no TCC:** Cap. 2.5 (ViTs como estimador de valor — introdução breve ao mecanismo de atenção).

### `dosovitskiy2021image`
- **Citação Completa:** DOSOVITSKIY, A.; BEYER, L.; KOLESNIKOV, A. et al. An image is worth 16x16 words: transformers for image recognition at scale. In: *International Conference on Learning Representations (ICLR)*, 2021. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Propõe o *Vision Transformer* (ViT), dividindo imagens em *patches* lineares tratados como tokens por um Transformer padrão para capturar contexto espacial global.
- **Pontos Fortes / Limitações:** Captura interações globais em uma única camada; carece de hierarquia multi-escala e exige grande volume de dados ou atenção janelada para predições densas voxel-a-voxel em 3D.
- **Comparação Direta com o AlphaBuilder:** Foi o ponto de partida conceitual no DIP inicial do AlphaBuilder (ViT apenas para Value Head em Keras), posteriormente substituído em produção pelo `SwinUNETR` 3D com decoder U-Net denso no PyTorch/MONAI (`model.py`).
- **Onde Citar no TCC:** Cap. 2.5 e Cap. 3.2 (Evolução do projeto do ViT inicial para Swin-UNETR; corrigir ausência de citação inline em `tcc_v5.txt`).

### `liu2021swin`
- **Citação Completa:** LIU, Z. et al. Swin Transformer: hierarchical vision transformer using shifted windows. In: *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Introduz o *Swin Transformer*, construindo representações hierárquicas multi-escala com complexidade linear no tamanho da entrada por meio de autoatenção em janelas locais e deslocamento alternado entre camadas (*shifted windows*, W-MSA/SW-MSA).
- **Pontos Fortes / Limitações:** Viabiliza Transformers em predição densa de alta resolução; impõe restrições de divisibilidade espacial pelo tamanho dos patches e estágios hierárquicos.
- **Comparação Direta com o AlphaBuilder:** Base direta do mecanismo de janelas deslocadas `window_size=(7,7,7)` e redução hierárquica de `SwinEncoder` em `model.py`.
- **Onde Citar no TCC:** Cap. 2.5 e Cap. 3.2 (Justificativa do backbone Swin frente a ViT plano e CNN pura; corrigir ausência de citação inline em `tcc_v5.txt`).

### `hatamizadeh2022swin`
- **Citação Completa:** HATAMIZADEH, A.; NATH, V.; TANG, Y.; YANG, D.; ROTH, H. R.; XU, D. Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. In: *Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (BrainLes 2021)*, LNCS, Springer, 2022. DOI: `10.1007/978-3-031-08999-2_22`. `arXiv:2201.01266`. *(Origem: Pré-existente `0A` + Enriquecida `0C`)*
- **Resumo Técnico Objetivo:** Acopla um encoder 3D Swin Transformer de 4 estágios (patches $2\times2\times2$, janela $7\times7\times7$, redução total $32\times$) a um decoder U-Net residual convolucional via *skip connections* e `InstanceNorm3d` em 5 escalas para segmentação volumétrica 3D na biblioteca MONAI.
- **Pontos Fortes / Limitações:** Combina modelagem de longo alcance do Swin com precisão de borda local das convoluções 3D e *skips*; exige entrada divisível por 32 e consome elevada VRAM em 3D.
- **Comparação Direta com o AlphaBuilder:** É literalmente a base estrutural do `SwinEncoder` e dos `UnetrBasicBlock` de `AlphaBuilderV31` (`model.py`), adaptado de 4 canais MRI (`feature_size=48`) para **7 canais mecânicos** com `feature_size=24` (`bottleneck_dim=384`), `_dynamic_pad(..., divisor=32)` e bifurcação para `value_head` e `SwinPolicyDecoder` (que aplica interpolação trilinear final $D/2\to D$).
- **Onde Citar no TCC:** Cap. 2.5, Cap. 3.2 e Cap. 4.5 (Arquitetura de produção Swin-UNETR; corrigir ausência crítica de citação inline em `tcc_v5.txt`).

### `zhang2023channel`
- **Citação Completa:** ZHANG, H. et al. Channel Vision Transformers: An Image Is Worth 1x16x16 Words. *arXiv preprint arXiv:2309.16108v4*, 2023. *(Origem: Pré-existente `0A` — `ref_alphabuilder.md`)*
- **Resumo Técnico Objetivo:** Propõe atenção conjunta entre canais e patches espaciais para melhorar a interação entre múltiplos canais heterogêneos em Vision Transformers.
- **Pontos Fortes / Limitações:** Útil quando canais carregam informações fisicamente distintas; formulado originalmente para classificação/imagens 2D.
- **Comparação Direta com o AlphaBuilder:** Relaciona-se à codificação de 7 canais heterogêneos (`density`, `mask_x/y/z`, `force_x/y/z`) em `tensor_utils.py`, onde o modelo precisa relacionar o canal de densidade aos canais esparsos de contorno.
- **Onde Citar no TCC:** Cap. 2.5 ou Cap. 4.3 (Discussão sobre interação entre canais físicos heterogêneos).

### `wang2024micrometer`
- **Citação Completa:** WANG, Y. et al. Micrometer: Micromechanics Transformer for Predicting Mechanical Responses of Heterogeneous Materials. *arXiv preprint arXiv:2410.05281*, 2024. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Emprega arquitetura Transformer para capturar interações mecânicas de longo alcance e prever respostas de tensão/deformação em materiais heterogêneos.
- **Pontos Fortes / Limitações:** Demonstra eficácia de Transformers na mecânica computacional frente a CNNs de campo receptivo limitado; foca em predição direta de campos micromecânicos e não em otimização topológica sequencial.
- **Comparação Direta com o AlphaBuilder:** Reforça a hipótese central do Cap. 2.5 de que atenção multi-escala modela melhor o acoplamento mecânico global entre apoios e cargas do que convoluções locais puras (`SimpleBackbone`).
- **Onde Citar no TCC:** Cap. 2.5 e Cap. 2.7 (já citado em L172 de `tcc_v5.txt`; corrigir divergência 2024/2025 presente em `tcc_draft_01.md`).

### `wang2025cvit`
- **Citação Completa:** WANG, Z. et al. CViT: Continuous Vision Transformer for Operator Learning. *arXiv preprint arXiv:2405.13998*, 2025. *(Origem: Pré-existente `0A` — `tcc_draft_01.md`)*
- **Resumo Técnico Objetivo:** Estende Vision Transformers para aprendizado contínuo de operadores em EDP/mecânica contínua independente de discretização fixa.
- **Pontos Fortes / Limitações:** Alivia dependência rígida de grade fixa em operadores físicos; focado em solução de EDPs (operator learning) e não em tomada de decisão discreta.
- **Comparação Direta com o AlphaBuilder:** Contrasta com o `_dynamic_pad` em grade discreta de `AlphaBuilderV31`, apontando caminhos para representação contínua em versões futuras.
- **Onde Citar no TCC:** Cap. 2.5 ou Cap. 2.7 (estava em `tcc_draft_01.md` L81 e foi omitido em `tcc_v5.txt` — decidir entre reincorporar ou descartar formalmente).

### `lutheran2025transformer`
- **Citação Completa:** LUTHERAN, A. et al. Transformer-based Topology Optimization. *arXiv preprint arXiv:2509.05800*, 2025. *(Origem: Pré-existente `0A` — `tcc_v5.txt` / `tcc_draft_01.md`)*
- **Resumo Técnico Objetivo:** Explora arquiteturas baseadas em Transformers aplicadas diretamente à síntese e aceleração de otimização topológica estrutural.
- **Pontos Fortes / Limitações:** Aplicação direta de Transformers em OT; metadados limitados em `0A` (preprint arXiv a ser conferido contra a publicação de periódico de 2026 do mesmo autor principal, `lutheran2026diffusion`).
- **Comparação Direta com o AlphaBuilder:** Alinha-se ao uso de backbone Transformer em OT, embora em predição direta/generativa sem busca deliberativa MCTS passo a passo.
- **Onde Citar no TCC:** Cap. 2.7 (já citado em L172 de `tcc_v5.txt`; conferir relação exata com `lutheran2026diffusion`).

### `nagayama2025predicting`
- **Citação Completa:** NAGAYAMA, T.; SASAKI, H. Predicting torque characteristics of synchronous reluctance motors using Swin Transformer. *COMPEL: The International Journal for Computation and Mathematics in Electrical and Electronic Engineering*, 2025. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Aplica o *Swin Transformer* como surrogate para estimar desempenho físico (torque) a partir de geometrias em engenharia eletromecânica.
- **Pontos Fortes / Limitações:** Confirma capacidade do Swin Transformer como estimador de propriedades físicas a partir de distribuição espacial de materiais; domínio 2D eletromagnético e não mecânico 3D.
- **Comparação Direta com o AlphaBuilder:** Serve de paralelo análogo para a `value_head` acoplada ao `SwinEncoder` no AlphaBuilder, estimando qualidade física global a partir da geometria espacial.
- **Onde Citar no TCC:** Cap. 2.7 (já citado em L172 de `tcc_v5.txt`).

### 2B. Deep Learning 3D, Modelos Generativos e Modelos de Fundação para TopOpt (`0C` + Identificações `0B`/`0A`)

### `behzadi2021realtime`
- **Citação Completa:** BEHZADI, M.; ILIEȘ, H. T. Real-Time Topology Optimization in 3D via Deep Transfer Learning. *Computer-Aided Design*, v. 135, p. 103014, 2021. DOI: `10.1016/j.cad.2021.103014`. `arXiv:2102.07657`. *(Origem: Nova `0C`)*
- **Resumo Técnico Objetivo:** Usa CNN 3D encoder-decoder alimentada por tensores de 5 canais (densidade, BCs e cargas) e *Transfer Learning* de baixa resolução para alta resolução 3D (~95% de acurácia binária média e refino SIMP opcional para membros de 1–2 voxels).
- **Pontos Fortes / Limitações:** Corta drasticamente o custo de geração de datasets 3D caros; sofre degradação em barras finas de 1–2 voxels na predição pura e depende de transferência entre resoluções fixas.
- **Comparação Direta com o AlphaBuilder:** Enquanto Behzadi & Ilieș usam 5 canais *one-shot*, o AlphaBuilder desdobra explicitamente os 3 DOFs de suporte e 3 componentes de força nos **7 canais** de `tensor_utils.py` e aceita resoluções arbitrárias via `_dynamic_pad`; a ideia de pré-treino *coarse-to-fine* poderia baratear a geração em `run_data_harvest.py`.
- **Onde Citar no TCC:** Cap. 2.7 (Trabalhos relacionados 3D) e Cap. 7 (Alternativas para geração de dados e resolução maior).

### `zheng2021generating`
- **Citação Completa:** ZHENG, S.; HE, Z.; LIU, H. Generating three-dimensional structural topologies via a U-Net convolutional neural network. *Thin-Walled Structures*, v. 159, p. 107263, 2021. DOI: `10.1016/j.tws.2020.107263`. *(Origem: Nova `0C`)*
- **Resumo Técnico Objetivo:** Aplica uma 3D U-Net com *skip connections* para inferência instantânea de topologias 3D a partir de tensores de carga e Dirichlet, introduzindo invólucro com *zero-padding* adaptativo para acomodar domínios de dimensões variadas.
- **Pontos Fortes / Limitações:** Predição direta rápida e flexibilidade de tamanho por *zero-padding*; sofre crescimento cúbico de memória, perda de generalização OOD e borramento de bordas.
- **Comparação Direta com o AlphaBuilder:** Paralelo direto do `SimpleBackbone` (`SimpleEncoder` + `SimplePolicyDecoder` em `model.py`), mas o AlphaBuilder substitui o invólucro fixo pelo `_dynamic_pad` sob demanda, adiciona a `value_head` para busca sequencial e adota o `SwinEncoder` para superar o campo receptivo estritamente local das convoluções 3D.
- **Onde Citar no TCC:** Cap. 2.5, Cap. 2.7 e Cap. 3.2 (Comparação entre `SimpleBackbone` e `SwinUNETR`).

### `xiang2025two`
- **Citação Completa:** XIANG, C.; CHEN, A.; LI, H.; CHANG, H. Two stage multiobjective topology optimization method via SwinUnet with enhanced generalization. *Scientific Reports*, 2025. DOI: `10.1038/s41598-025-92793-0`. *(Origem: Nova `0C`)*
- **Resumo Técnico Objetivo:** Emprega um backbone SwinUnet 2D com tensores de entrada adaptativos no Estágio 1 para generalizar a domínios retangulares e em L, acoplado a um Estágio 2 físico multiobjetivo (compliance, tensão e fração de volume).
- **Pontos Fortes / Limitações:** Valida empiricamente atenção por janelas deslocadas em OT com boa generalização de BCs; restrito a 2D e dependente de pós-refinamento físico.
- **Comparação Direta com o AlphaBuilder:** Corrobora a superioridade de atenção hierárquica em janelas (Swin) na modelagem de caminhos de carga, mas o AlphaBuilder já opera diretamente em **3D** com decoder híbrido CNN residual (`SwinUNETR`), mais adequado a gradientes espaciais 3D do que *patch expanding* puro.
- **Onde Citar no TCC:** Cap. 2.5 e Cap. 2.7 (Justificativa do uso de arquitetura da família Swin em OT).

### `zhao2026variable`
- **Citação Completa:** ZHAO, Z.; SUN, B. Variable-Resolution Topology Optimization Based on Dynamic Vision Transformer. *ASME Journal of Mechanical Design*, v. 148, n. 12, 2026. DOI: `10.1115/1.4072316`. *(Origem: Nova `0C`)*
- **Resumo Técnico Objetivo:** Framework DT-VRTO (2D e 3D) que inicia em malha grossa e escala progressivamente a resolução com super-resolução guiada por um *Dynamic Vision Transformer* adaptando tokens online para atuar como surrogate e cortar ~53,1% das iterações FEA em 2D.
- **Pontos Fortes / Limitações:** Adapta tokens dinamicamente a múltiplas resoluções sem desperdício de padding; ainda exige loop iterativo acoplado ao solver físico.
- **Comparação Direta com o AlphaBuilder:** Evidencia uma fraqueza concreta do `_dynamic_pad(..., divisor=32)` do `SwinUNETR` em volumes anisotrópicos como `(64, 32, 8)`, onde o eixo `W=8` é preenchido com zeros até `32` (quadruplicando o volume computacional antes do encoder), sugerindo tokenização dinâmica como melhoria futura.
- **Onde Citar no TCC:** Cap. 3.2 / Cap. 6.4 (Limitações do `dynamic_pad` em grades finas) e Cap. 7 / Cap. 8 (Trabalhos futuros).

### `maze2023diffusion`
- **Citação Completa:** MAZÉ, F.; AHMED, F. Diffusion Models Beat GANs on Topology Optimization. In: *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-23)*, v. 37, n. 8, p. 9108–9116, 2023. DOI: `10.1609/aaai.v37i8.26093`. `arXiv:2208.09591`. *(Origem: Nova `0C`)*
- **Resumo Técnico Objetivo:** Propõe o **TopoDiff**, um modelo de difusão condicional (DDPM) sobre grade 2D $64\times64$ guiado por regressores substitutos de *compliance* e detecção de material flutuante, condicionado a campos FEA iniciais (tensão de von Mises, energia de deformação, volume, BCs). Reduziu erro físico em $8\times$ e amostras inviáveis em $11\times$ frente a cGANs.
- **Pontos Fortes / Limitações:** Excelente viabilidade física e controle explícito contra desconexões; restrito a 2D no artigo original, lento por exigir dezenas de passos de denoising e dependente de FEA inicial.
- **Comparação Direta com o AlphaBuilder:** Mostra que fornecer campos físicos densos nos canais de entrada facilita muito a generalização em comparação aos canais `1–6` esparsos de `tensor_utils.py`, mas difusão 3D em espaço de voxels seria proibitivamente lenta para servir de oráculo dentro dos 160 *rollouts* de MCTS do AlphaBuilder.
- **Onde Citar no TCC:** Cap. 2.6 (Conectividade), Cap. 2.7 (Modelos generativos) e Cap. 4.3 (Discussão sobre canais de entrada esparsos vs. campos físicos densos).

### `lutheran2026diffusion`
- **Citação Completa:** LUTHERAN, A.; DAS, S.; TABARRAEI, A. Diffusion Transformers with Hybrid Conditioning for Structural Optimization. *Structural and Multidisciplinary Optimization*, v. 69, p. 231, 2026. DOI: `10.1007/s00158-026-04429-0`. `arXiv:2605.02158`. *(Origem: Nova `0C`)*
- **Resumo Técnico Objetivo:** Substitui a U-Net de difusão por um *Diffusion Transformer* (DiT) em 2D (30.000 amostras SIMP) com condicionamento híbrido: campos espaciais de von Mises/energia de deformação concatenados nos *tokens* e escalares globais (posição da carga, forças, fração de volume) injetados via *Adaptive Layer Normalization* (AdaLN).
- **Pontos Fortes / Limitações:** Evita esparsidade extrema nos tensores espaciais ao separar condicionamento local de parâmetros escalares globais; validado apenas em 2D com denoising iterativo.
- **Comparação Direta com o AlphaBuilder:** Aponta alternativa arquitetural superior ao empilhamento de mapas esparsos nos canais `4–6` do `AlphaBuilderV31`: parâmetros globais (ex.: fração de volume alvo) seriam mais bem injetados via `AdaLN` no `SwinEncoder` do que replicados em volumes 3D inteiros.
- **Onde Citar no TCC:** Cap. 2.7 e Cap. 4.3 / Cap. 6.4 (Arquitetura de condicionamento de entrada).

### `nie2021topologygan`
- **Citação Completa:** NIE, Z.; LIN, T.; JIANG, H.; KARA, L. B. TopologyGAN: Topology Optimization Using Generative Adversarial Networks Based on Physical Fields Over the Initial Domain. *ASME Journal of Mechanical Design*, v. 143, n. 3, p. 031715, 2021. DOI: `10.1115/1.4049533`. `arXiv:2003.04685`. *(Origem: Nova `0C`)*
- **Resumo Técnico Objetivo:** cGAN 2D baseada em gerador U-SE-ResNet (com blocos *Squeeze-and-Excitation* de atenção por canal) que recebe campos físicos iniciais (densidade de energia de deformação, tensões e deslocamentos), reduzindo MSE em ~3× e MAE em 2,5× em BCs não vistas frente a cGANs guiadas apenas por máscaras esparsas.
- **Pontos Fortes / Limitações:** Prova o ganho de generalização via campos físicos iniciais e reponderação de canais (*SE blocks*); herda instabilidade de GANs, desconexões e gargalo cúbico $\mathcal{O}(N^3)$ em 3D.
- **Comparação Direta com o AlphaBuilder:** O AlphaBuilder supera a estaticidade do domínio inicial ao alimentar a densidade corrente $\rho$ (canal 0) passo a passo, mas ainda codifica BCs e cargas de forma esparsa nos canais `1–6` e carece de atenção explícita por canal (*SE*) no `SimpleBackbone`.
- **Onde Citar no TCC:** Cap. 2.7 e Cap. 4.3 (Justificativa e limitações da codificação de entrada de 7 canais).

### `sridhara2026latent`
- **Citação Completa:** SRIDHARA, S.; DEODHARE, G. G.; SURESH, K. A latent space approach to multi-material topology optimization. *Structural and Multidisciplinary Optimization*, v. 69, n. 3, p. 54, 2026. DOI: `10.1007/s00158-025-04225-2`. *(Origem: Nova `0C`)*
- **Resumo Técnico Objetivo:** Usa um VAE para mapear propriedades discretas de até 20 materiais para um espaço latente contínuo diferenciável acoplado diretamente ao otimizador MMA em problemas 2D e 3D com mais de **1 milhão de graus de liberdade (DOFs)**.
- **Pontos Fortes / Limitações:** Demonstra escalabilidade acima de 1M DOFs em 3D mantendo garantia física pelo solver MMA/FEA; não elimina o loop iterativo FEA nem opera *one-shot*.
- **Comparação Direta com o AlphaBuilder:** Contrasta o material isotrópico único no canal 0 ($\rho\in[0,1]$) e *bottleneck* não-generativo (`96`/`384` canais) do AlphaBuilder com o uso de espaços latentes VAE para domínios 3D massivos ou multimateriais.
- **Onde Citar no TCC:** Cap. 2.7 (Escala 3D) e Cap. 8 (Extensões futuras multimateriais).

### `nobari2024towards`
- **Citação Completa:** NOBARI, A. H.; REGENWETTER, L.; AHMED, F. Towards Domain-Adaptive, Resolution-Free 3D Topology Optimization With Neural Implicit Fields. In: *ASME IDETC-CIE 2024*, Vol. 3A: 50th Design Automation Conference (DAC), Paper No. DETC2024-142135, 2024. DOI: `10.1115/DETC2024-142135`. *(Origem: Nova `0C`)*
- **Resumo Técnico Objetivo:** Introduz o **NITO-3D**, treinado em **122.000 topologias 3D**, representando geometrias como *Neural Implicit Fields* contínuos combinados ao codificador invariante a pontos de contorno **BPOM** (*Boundary Point Order-Invariant MLP*). Atinge *speedup* de **2.000×** com erro de 0,3% de *compliance*, ou <0,1% com 10 passos de ajuste fino (15× mais rápido que SIMP).
- **Pontos Fortes / Limitações:** Totalmente independente de grade cartesiana, resolução e razão de aspecto, com generalização *zero-shot* a BCs; não é uma política construtiva passo a passo.
- **Comparação Direta com o AlphaBuilder:** Representa a principal contraposição arquitetural ao `AlphaBuilderV31`: enquanto o AlphaBuilder depende de grade voxelizada regular e `_dynamic_pad(..., divisor=32)` (além de canais esparsos `1–6`), NITO-3D resolve arbitrariedade geométrica por consulta implícita contínua e `BPOM`.
- **Onde Citar no TCC:** Cap. 2.7 (Estado da arte 3D), Cap. 3.2 / Cap. 6.4 (Limitações da discretização cartesiana e do padding).

### `bhuiyan2026graph`
- **Citação Completa:** BHUIYAN, S. A.; TABARRAEI, A. Graph neural network-based topology optimization for efficient support structure design in additive manufacturing. *Engineering with Computers*, v. 42, p. 103, 2026. DOI: `10.1007/s00366-026-02344-y`. `arXiv:2508.19169`. *(Origem: Pré-existente `0A` como `Tabarraei & Bhuiyan, ResearchGate 2025` + Atualizada `0C`)*
- **Resumo Técnico Objetivo:** Converte malhas FEA 2D e 3D em grafos e aplica GNN com convoluções espectrais de Chebyshev e *Fourier features* em pipeline totalmente diferenciável (GNN + filtro de *overhang* + FEA) para gerar estruturas autossustentadas em manufatura aditiva sem derivação manual de gradientes.
- **Pontos Fortes / Limitações:** Opera nativamente sobre malhas não estruturadas curvas sem voxelização ou *padding* e impõe restrições físicas exatas; requer otimização iterativa por instância (não amortiza entre problemas distintos).
- **Comparação Direta com o AlphaBuilder:** Enquanto o AlphaBuilder amortiza o custo offline no SQLite (`1.069` episódios) para gerar uma política reutilizável em grade retangular `(64,32,8)`, a GNN de Bhuiyan & Tabarraei opera em qualquer geometria de malha mas otimiza instância por instância acoplada ao FEA.
- **Onde Citar no TCC:** Cap. 2.6 e Cap. 2.7 (substitui a entrada incompleta "ResearchGate, 2025" em L183 e L1073 de `tcc_v5.txt`).

### `nobari2025optimize`
- **Citação Completa:** NOBARI, A. H.; REGENWETTER, L.; PICARD, C.; HAN, L.; AHMED, F. Optimize Any Topology: A Foundation Model for Shape- and Resolution-Free Structural Topology Optimization. *arXiv preprint arXiv:2510.23667*, 2025. DOI: `10.48550/arXiv.2510.23667`. *(Aceito no NeurIPS 2025 — OAT)*. *(Origem: Nova `0C`)*
- **Resumo Técnico Objetivo:** Modelo de fundação (**OAT**) treinado sobre **2,2 milhões de estruturas** (corpus OpenTO, $64^2$–$256^2$, razões de aspecto até 10:1) combinando *Neural Field Autoencoder* (NFAE), difusão latente condicional (LDM) e decoder implícito, alcançando inferência em **<1 s** e redução de até 90% na *compliance* média frente a baselines anteriores.
- **Pontos Fortes / Limitações:** Referência máxima de generalização *zero-shot* em formas e resoluções arbitrárias; validado em 2D e dependente de escala de dados/infraestrutura massiva inacessível localmente.
- **Comparação Direta com o AlphaBuilder:** Contrasta modelos de fundação *one-shot* de grande escala com o enfoque enxuto de aprendizado sequencial + busca (`SwinUNETR` `feature_size=24` + MCTS + FEniTop) viável em hardware acadêmico (GPU T4 / 1.069–2.000 episódios).
- **Onde Citar no TCC:** Cap. 2.7 (Fronteira de modelos de fundação) e Cap. 6.4 (Limitações de escala computacional).

### `yan2022deep`
- **Citação Completa:** YAN, J.; ZHANG, Q.; XU, Q.; FAN, Z.; LI, H.; SUN, W.; WANG, G. Deep learning driven real time topology optimisation based on initial stress learning. *Advanced Engineering Informatics*, v. 51, p. 101472, 2022. DOI: `10.1016/j.aei.2021.101472`. *(Origem: Pré-existente `0A` sem autores em `pesquisa_referencias.md` #6 + Identificada `0B`)*
- **Resumo Técnico Objetivo:** Utiliza aprendizado profundo supervisionado baseado em CNN alimentada pela matriz/campos de tensão iniciais para predizer topologias ótimas em tempo real sem iterações online.
- **Pontos Fortes / Limitações:** Demonstra valor informacional dos campos iniciais de tensão para aceleração; abordagem supervisionada pura sem agente RL nem verificação sequencial em árvore.
- **Comparação Direta com o AlphaBuilder:** Contrasta predição supervisionada direta por tensões iniciais com a arquitetura iterativa de estado corrente (densidade + BCs) + MCTS do AlphaBuilder.
- **Onde Citar no TCC:** Cap. 2.7 (Trabalhos relacionados de DL supervisionado em OT).

### 2C. Corpus Prévio de DL/ML em TopOpt (`0A` — `ref_alphabuilder.md` e `pesquisa_referencias.md` com autoria identificada)

> *Nota metodológica:* As entradas abaixo pertencem ao levantamento prévio `0A` e tiveram apenas os metadados explicitamente verificados mantidos no `.bib` (periódico e DOI devem ser confirmados pelo autor caso sejam incluídas na versão final do texto).

### `shin2023review`
- **Citação Completa:** SHIN, S. et al. Topology optimization via machine learning and deep learning: a review. 2023. *(Origem: Pré-existente `0A` — `pesquisa_referencias.md`)*
- **Resumo Técnico Objetivo:** Revisão abrangente da literatura de Machine Learning e Deep Learning aplicada à otimização topológica estrutural e multifísica, cobrindo abordagens diretas, surrogates, GANs e aplicações em nanofotônica/manufatura.
- **Pontos Fortes / Limitações:** Mapeia gargalos clássicos (custo de dataset, escalabilidade 3D, conectividade e restrições de manufatura); metadados bibliográficos completos não registrados em `0A`.
- **Comparação Direta com o AlphaBuilder:** Ajuda a contextualizar o estado da arte da revisão bibliográfica até 2023 antes das contribuições recentes de 2024–2026 (`0B`–`0E`).
- **Onde Citar no TCC:** Cap. 2.7 (Contextualização de revisões recentes de DL em OT).

### `banga20183d`
- **Citação Completa:** BANGA, S. et al. 3D topology optimization using convolutional neural networks. 2018. *(Origem: Pré-existente `0A` — `pesquisa_referencias.md`)*
- **Resumo Técnico Objetivo:** Trabalho precursor aplicando redes neurais convolucionais 3D (encoder-decoder) para prever rapidamente estruturas 3D a partir de estágios intermediários do SIMP.
- **Pontos Fortes / Limitações:** Um dos primeiros estudos a demonstrar predição neural diretamente em volumes 3D; limitado em resolução e dependente de iterações iniciais do SIMP como entrada.
- **Comparação Direta com o AlphaBuilder:** Precursor da ideia de acelerar parte da trajetória SIMP com CNN 3D, enquanto o AlphaBuilder substitui a CNN simples por `SwinUNETR` e acopla política dual Add/Remove ao MCTS.
- **Onde Citar no TCC:** Cap. 2.7 (Precursores de DL 3D em OT).

### `rawat2018deep`
- **Citação Completa:** RAWAT, S.; SHEN, L. Deep learning for topology optimization. 2018. *(Origem: Pré-existente `0A` — `pesquisa_referencias.md`)*
- **Resumo Técnico Objetivo:** Explora redes generativas profundas para geração direta de layouts 2D otimizados condicionada a parâmetros de carga e volume.
- **Pontos Fortes / Limitações:** Demonstra viabilidade de síntese rápida; limitado a 2D e sujeito a desconexões ou violações de restrições físicas sem solver em loop.
- **Comparação Direta com o AlphaBuilder:** Ilustra a primeira geração de modelos generativos 2D sem garantia física em malha, superada pela verificação FEM e pelo MCTS do AlphaBuilder.
- **Onde Citar no TCC:** Cap. 2.7 (Precursores históricos de DL em OT).

### `rawat2019deep3d`
- **Citação Completa:** RAWAT, S.; SHEN, L. Deep learning for 3D topology optimization. 2019. *(Citado como Rawat & Shen, 2019a em `0A`)*. *(Origem: Pré-existente `0A` — `pesquisa_referencias.md`)*
- **Resumo Técnico Objetivo:** Estende a abordagem de redes generativas profundas (WGAN/CNN) dos autores para síntese direta de estruturas 3D.
- **Pontos Fortes / Limitações:** Trata o salto dimensional para 3D com síntese rápida; sofre das limitações de fidelidade mecânica e conectividade típicas de GANs 3D puras.
- **Comparação Direta com o AlphaBuilder:** Contrasta síntese *one-shot* em 3D sem verificação física com construção passo a passo guiada por conectividade e avaliação DOLFINx.
- **Onde Citar no TCC:** Cap. 2.7 (Precursores de DL generativo 3D).

### `yu2018deep`
- **Citação Completa:** YU, Y.; HUR, T.; JUNG, J.; KIM, Y. Deep learning for determining a near-optimal topological design without any iteration. *arXiv preprint arXiv:1801.05463*, 2018. *(Origem: Pré-existente `0A` — `ref_alphabuilder.md`)*
- **Resumo Técnico Objetivo:** Propõe pipeline CNN/cGAN em duas etapas (predição em baixa resolução + super-resolução) para gerar topologias quase ótimas 2D sem iterações online de otimização.
- **Pontos Fortes / Limitações:** Trabalho seminal em geração *iteration-free* e super-resolução; restrito a 2D e vulnerável a desconexões em BCs fora da amostra.
- **Comparação Direta com o AlphaBuilder:** Representa a filosofia *non-iterative one-shot* citada no guia inicial (`ref_alphabuilder.md`), da qual o AlphaBuilder se distancia ao manter planejamento sequencial deliberativo (MCTS) e FEM na Fase 2.
- **Onde Citar no TCC:** Cap. 2.7 (Abordagens *one-shot* vs. iterativas/MCTS).

### `yu2019topology`
- **Citação Completa:** YU, Y. et al. Topology optimization using deep learning and generative adversarial networks. 2019. *(Origem: Pré-existente `0A` — `pesquisa_referencias.md`)*
- **Resumo Técnico Objetivo:** Consolida a abordagem baseada em aprendizado profundo e redes adversárias (cGAN) para geração e refinamento de topologias estruturais 2D sem otimização iterativa online. *(Provável versão publicada relacionada a `yu2018deep`, mantida distinta por diferença de título em `0A`)*.
- **Pontos Fortes / Limitações:** Referência frequentemente citada em cGANs para OT; restrição 2D e fragilidade de conectividade.
- **Comparação Direta com o AlphaBuilder:** Mesmo contraste de `yu2018deep`: geração adversária 2D *one-shot* vs. política sequencial 3D + MCTS + FEM no AlphaBuilder.
- **Onde Citar no TCC:** Cap. 2.7 (Modelos generativos adversariais em OT).

### `zhang2019deep`
- **Citação Completa:** ZHANG, Y.; WANG, C.; CHEN, S. et al. A deep Convolutional Neural Network for topology optimization with strong generalization ability. *arXiv preprint arXiv:1901.07761*, 2019. *(Origem: Pré-existente `0A` — `ref_alphabuilder.md`)*
- **Resumo Técnico Objetivo:** Propõe uma CNN profunda que recebe campos físicos (componentes de deslocamento e deformação calculados na malha inicial) em vez de apenas máscaras geométricas, elevando a capacidade de generalização a novas condições de contorno.
- **Pontos Fortes / Limitações:** Antecipa o insight de *TopologyGAN* e *TopoDiff* sobre representação baseada em campos mecânicos; restrito a 2D.
- **Comparação Direta com o AlphaBuilder:** Já listado no guia inicial (`ref_alphabuilder.md`), reforça a discussão sobre a suficiência ou limitação dos 7 canais esparsos do AlphaBuilder para generalizar BCs inéditas.
- **Onde Citar no TCC:** Cap. 2.7 e Cap. 4.3 (Representação de entrada em OT neural).

### `zhang2019heat`
- **Citação Completa:** ZHANG, W. et al. Topology optimization using deep learning for heat conduction problems. 2019. *(Origem: Pré-existente `0A` — `pesquisa_referencias.md`)*
- **Resumo Técnico Objetivo:** Aplica redes neurais profundas à otimização topológica em problemas de condução de calor (dissipação térmica) com redução drástica do tempo de síntese.
- **Pontos Fortes / Limitações:** Mostra transferibilidade de DL-TopOpt para física térmica; fora do escopo de elasticidade linear estrutural 3D.
- **Comparação Direta com o AlphaBuilder:** Evidencia que o paradigma voxelizado com BCs em canais se aplica a múltiplas físicas, embora o AlphaBuilder seja especializado em elasticidade linear 3D.
- **Onde Citar no TCC:** Cap. 2.7 (Apenas como breve menção à amplitude de aplicações físicas de DL-TopOpt, se pertinente).

### `sosnovik2019fast`
- **Citação Completa:** SOSNOVIK, I.; OSELEDETS, I. Fast topology optimization with machine learning. 2019. *(Origem: Pré-existente `0A` — `pesquisa_referencias.md`)*
- **Resumo Técnico Objetivo:** Formula a aceleração do SIMP como tarefa de segmentação imagem-a-imagem via CNN/encoder-decoder, recebendo iterações iniciais do otimizador para prever a topologia binária final sem rodar todas as iterações até a convergência.
- **Pontos Fortes / Limitações:** Marco histórico que introduziu aceleração por segmentação de estados intermediários do SIMP; requer executar passos iniciais do SIMP e opera em 2D.
- **Comparação Direta com o AlphaBuilder:** Semelhante em espírito ao uso de trajetórias FEniTop como supervisão da Fase 2 no AlphaBuilder, porém substituindo a predição *one-shot* por política voxel-a-voxel iterativa no MCTS.
- **Onde Citar no TCC:** Cap. 2.7 (Aceleração neural de trajetórias SIMP).

### `hoyer2019robust`
- **Citação Completa:** HOYER, S. et al. Robust topology optimization using generative adversarial networks. 2019. *(Origem: Pré-existente `0A` — `pesquisa_referencias.md`)*
- **Resumo Técnico Objetivo:** Usa parametrização neural (reparametrização da densidade pela saída de uma rede) em conjunto com o solver físico diferenciável para melhorar a robustez e evitar mínimos locais fracos na otimização topológica.
- **Pontos Fortes / Limitações:** Mantém a física exata no loop via reparametrização diferenciável (em vez de substituir o FEA por completo); custo iterativo por instância.
- **Comparação Direta com o AlphaBuilder:** Contrasta otimização por gradiente em espaço latente neural por instância com o aprendizado amortizado de política + MCTS do AlphaBuilder.
- **Onde Citar no TCC:** Cap. 2.7 (Reparametrização neural e GANs em OT).

### `cang2019theory`
- **Citação Completa:** CANG, Z. et al. Theory-driven mechanism with adaptive sampling for topology optimization. 2019. *(Origem: Pré-existente `0A` — `pesquisa_referencias.md`)*
- **Resumo Técnico Objetivo:** Propõe um mecanismo guiado por teoria física com amostragem adaptativa ativa para treinar redes neurais em OT, reduzindo a necessidade de datasets massivos aleatórios.
- **Pontos Fortes / Limitações:** Ataca diretamente o custo de geração de amostras de treinamento; focado em predição direta 2D.
- **Comparação Direta com o AlphaBuilder:** Dialoga com a estratégia do AlphaBuilder de combinar geração procedural (80% Bézier 3D + 20% domínio completo) e *Negative Sampling* ativo na Fase 1/2 para enriquecer regiões críticas do espaço de estados.
- **Onde Citar no TCC:** Cap. 2.7 ou Cap. 4.2 (Estratégias de amostragem de dados de treino).

### `oh2019improving`
- **Citação Completa:** OH, S. et al. Improving topology optimization algorithms through AI. 2019. *(Origem: Pré-existente `0A` — `pesquisa_referencias.md`)*
- **Resumo Técnico Objetivo:** Integra modelos generativos profundos (GAN/VAE) à otimização topológica para gerar designs conceituais diversos em engenharia (como rodas) e inicializar ou orientar otimizadores clássicos.
- **Pontos Fortes / Limitações:** Explora geração de diversidade geométrica em engenharia mecânica; restrito a geometrias 2D/axissimétricas simples.
- **Comparação Direta com o AlphaBuilder:** Antecipa a combinação entre exploração generativa e otimização física, que o AlphaBuilder busca realizar via MCTS + FEniCSx em 3D.
- **Onde Citar no TCC:** Cap. 2.7 (Design generativo e diversidade em engenharia).

### `wang2021realtime`
- **Citação Completa:** WANG, Y. et al. Real-time topology optimization using deep learning. 2021. *(Origem: Pré-existente `0A` — `pesquisa_referencias.md`)*
- **Resumo Técnico Objetivo:** Aplica redes neurais profundas para predição de topologias estruturais em tempo real a partir de especificações de contorno e carga.
- **Pontos Fortes / Limitações:** Ênfase em baixa latência de inferência; detalhes completos de periódico/DOI ausentes em `0A`, devendo ser verificados antes da citação formal.
- **Comparação Direta com o AlphaBuilder:** Representa a linha de inferência instantânea sem busca, contrastada no AlphaBuilder com a busca deliberativa MCTS para assegurar viabilidade física em 3D.
- **Onde Citar no TCC:** Cap. 2.7 (opcional, somente após verificação bibliográfica completa).

---

## Eixo 3: Reinforcement Learning, MCTS e AlphaZero em Design Estrutural

### 3A. Fundamentos de MCTS, UCT/PUCT e Família AlphaGo / AlphaZero

### `kocsis2006bandit`
- **Citação Completa:** KOCSIS, L.; SZEPESVÁRI, C. Bandit based Monte-Carlo planning. In: *European Conference on Machine Learning (ECML)*, p. 282–293, 2006. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Propõe o algoritmo **UCT** (*Upper Confidence Bounds applied to Trees*), formalizando o equilíbrio entre exploração e aproveitamento (*exploitation*) em busca em árvore via critério UCB1 nos nós internos.
- **Pontos Fortes / Limitações:** Garantia assintótica de convergência para o ótimo em árvores finitas; em espaços combinatórios vastos (como voxels 3D) exige *priors* neurais e função valor para evitar *rollouts* aleatórios caros.
- **Comparação Direta com o AlphaBuilder:** Base teórica imediata do algoritmo PUCT implementado em `MCTSNode.puct_score` (`node.py`).
- **Onde Citar no TCC:** Cap. 2.4 (MCTS — evolução de UCT para PUCT; corrigir ausência de citação inline em `tcc_v5.txt`).

### `rosin2011multi`
- **Citação Completa:** ROSIN, C. D. Multi-armed bandits with episode context. *Annals of Mathematics and Artificial Intelligence*, v. 61, n. 3, p. 203–230, 2011. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Introduz o **PUCT** (*Predictor + Upper Confidence Bounds applied to Trees*), incorporando probabilidades a priori $P(s,a)$ geradas por um preditor externo na fórmula de seleção de bandidos contextuais.
- **Pontos Fortes / Limitações:** Reduz drasticamente o fator de ramificação efetivo ao concentrar simulações nas ações de maior *prior*; depende da calibração de $P(s,a)$ e da constante $c_{\text{puct}}$.
- **Comparação Direta com o AlphaBuilder:** Regra matemática exata adotada em `MCTSNode.puct_score` (`node.py`) com `c_puct=1.2` (Fase 1) e `0.8` (Fase 2) em `engine.py`, guiada pelo decoder de política da `AlphaBuilderV31`.
- **Onde Citar no TCC:** Cap. 2.4 e Cap. 4.6 (Fórmula PUCT do MCTS; corrigir ausência de citação inline em `tcc_v5.txt`).

### `browne2012survey`
- **Citação Completa:** BROWNE, C. B.; POWLEY, E.; WHITEHOUSE, D. et al. A survey of Monte Carlo tree search methods. *IEEE Transactions on Computational Intelligence and AI in Games*, v. 4, n. 1, p. 1–43, 2012. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Revisão clássica sistematizando as quatro etapas do MCTS (seleção, expansão, simulação/*rollout* e retropropagação), variantes de UCT, paralelização (*virtual loss*, *tree parallelization*) e domínios combinatórios.
- **Pontos Fortes / Limitações:** Referência didática padrão para engenharia de MCTS; anterior à integração com redes profundas estilo AlphaZero.
- **Comparação Direta com o AlphaBuilder:** Sustenta a descrição das quatro fases do MCTS e do uso de *virtual loss* para avaliação em micro-lotes (`batch_size=8`) em `engine.py`.
- **Onde Citar no TCC:** Cap. 2.4 (Ciclo de busca do MCTS; corrigir ausência de citação inline em `tcc_v5.txt`).

### `silver2016mastering`
- **Citação Completa:** SILVER, D. et al. Mastering the game of Go with deep neural networks and tree search. 2016. URL: `https://www.researchgate.net/publication/292074166_Mastering_the_game_of_Go_with_deep_neural_networks_and_tree_search`. *(Origem: Pré-existente `0A` — `ref_alphabuilder.md`)*
- **Resumo Técnico Objetivo:** Introduz o **AlphaGo**, demonstrando que a combinação de aprendizado supervisionado inicial em partidas de especialistas seguido de *Reinforcement Learning* via *self-play* e MCTS supera jogadores humanos profissionais em espaços combinatórios extremos.
- **Pontos Fortes / Limitações:** Valida o esquema em duas etapas (**pré-treino supervisionado + RL/MCTS**), exatamente análogo ao pipeline do AlphaBuilder; versão inicial ainda utilizava *rollouts* rápidos e redes separadas de política/valor.
- **Comparação Direta com o AlphaBuilder:** Enquanto o AlphaZero (2018) parte do zero absoluto, o AlphaBuilder segue conceitualmente a estratégia híbrida inaugurada no AlphaGo (2016): pré-treino supervisionado (*warm-up* com Bézier + FEniTop) seguido de *self-play* MCTS.
- **Onde Citar no TCC:** Cap. 2.4 / Cap. 2.5 (Transição de pré-treino supervisionado para MCTS).

### `silver2017mastering`
- **Citação Completa:** SILVER, D.; SCHRITTWIESER, J.; SIMONYAN, K. et al. Mastering the game of Go without human knowledge. *Nature*, v. 550, p. 354–359, 2017. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Apresenta o **AlphaGo Zero**, unificando política e valor em uma única arquitetura de rede neural residual de duas cabeças treinada exclusivamente por *self-play* com MCTS (PUCT), eliminando *rollouts* aleatórios e dados humanos.
- **Pontos Fortes / Limitações:** Arquitetura unificada *Policy + Value* diretamente transferível para busca em grade; custo de *cold-start* proibitivo quando avaliações terminais exigem solver FEM 3D caro.
- **Comparação Direta com o AlphaBuilder:** Inspira diretamente a arquitetura unificada dual-head de `AlphaBuilderV31` (encoder compartilhado + `value_head` + `policy_decoder`) e o uso da distribuição de visitas do MCTS como alvo de melhoria iterativa de política.
- **Onde Citar no TCC:** Cap. 2.4, Cap. 2.5 e Cap. 3.2 (corrigir ausência de citação inline em `tcc_v5.txt`).

### `silver2018general`
- **Citação Completa:** SILVER, D.; HUBERT, T.; SCHRITTWIESER, J. et al. A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 2018. *(Origem: Pré-existente `0A` — `ref_alphabuilder.md`, ausente na lista final de `tcc_v5.txt`)*
- **Resumo Técnico Objetivo:** Generaliza o algoritmo do AlphaGo Zero para **AlphaZero**, consolidando o framework geral de planejamento baseado em PUCT + rede neural dual (*Policy + Value*) aplicável a múltiplos domínios discretos sem conhecimento específico além das regras legais.
- **Pontos Fortes / Limitações:** Marco de generalidade do paradigma MCTS + Deep RL; assumido como jogo de soma zero de dois jogadores com transições rápidas, exigindo adaptações (como micro-batches e *warm-up*) em engenharia 3D com simulador FEM lento.
- **Comparação Direta com o AlphaBuilder:** Referência nominal central citada em L161 de `tcc_v5.txt` ("AlphaZero") que faltava na seção `Referências`, embasando a adaptação para problema de **um jogador** em grade 3D com `apply_action_mask` e *Principal Variations* em lote.
- **Onde Citar no TCC:** Cap. 1, Cap. 2.4 e Cap. 2.7 (adicionar entrada formal na bibliografia para a citação já existente em L161 de `tcc_v5.txt`).

### 3B. RL, MCTS e Construção Sequencial em Design Estrutural e Otimização Topológica (`0B` + `0A`)

### `bapst2019structured`
- **Citação Completa:** BAPST, V.; SANCHEZ-GONZALEZ, A.; DOERSCH, C.; STACHENFELD, K. L.; KOHLI, P.; BATTAGLIA, P. W.; HAMRICK, J. B. Structured agents for physical construction. In: *International Conference on Machine Learning (ICML 2019)*, 2019. DOI: `10.48550/arXiv.1904.03177`. `arXiv:1904.03177`. *(Origem: Nova `0B`)*
- **Resumo Técnico Objetivo:** Propõe agentes com *Graph Networks* (GNs) e ações centradas em objetos para construção física sequencial sob gravidade e corpos rígidos (tarefas *Silhouettes*, *Connecting* e *Covering*), mostrando que agentes *model-based* equipados com **MCTS** superam drasticamente RL *model-free* (DQN) em horizontes longos e generalizam *zero-shot* para cenas maiores.
- **Pontos Fortes / Limitações:** Prova empírica seminal de que MCTS + representações estruturadas vence RL *model-free* em construção física sob restrições de estabilidade; opera em blocos rígidos 2D e não em contínuos elásticos 3D.
- **Comparação Direta com o AlphaBuilder:** Reforça a escolha de MCTS na Fase 1 (*GROWTH*: conectar suportes e carga partindo do vazio), substituindo blocos rígidos/GNN por voxels 3D contínuos com `SwinUNETR`, ações duais Add/Remove e FEM elástico na Fase 2.
- **Onde Citar no TCC:** Cap. 2.3 (OT como construção sequencial) e Cap. 2.4 (Superioridade de MCTS em planejamento físico).

### `gaymann2019deep`
- **Citação Completa:** GAYMANN, A.; MONTOMOLI, F. Deep Neural Network and Monte Carlo Tree Search applied to Fluid-Structure Topology Optimization. *Scientific Reports*, 2019. DOI: `10.1038/s41598-019-51111-1`. *(Origem: Nova `0B`)*
- **Resumo Técnico Objetivo:** Primeiro trabalho a aplicar diretamente DNN + MCTS estilo AlphaZero à otimização topológica em grade euleriana (fluido-estrutura 2D até $8\times20\times1$), formulado como jogo de dois jogadores que modificam células fluido/sólido (ou escolhem *Pass*) avaliados por solver CFD incompressível ($\Delta P$).
- **Pontos Fortes / Limitações:** Precursor direto de MCTS + NN acoplado a simulador físico em grade sem gradientes adjuntos; limitado a grade 2D pequena ($8\times20\times1$), ação unitária e física fluida.
- **Comparação Direta com o AlphaBuilder:** É o antecedente conceitual mais próximo do AlphaBuilder em domínio de grade, do qual o AlphaBuilder avança ao passar de 2D ($160$ células) para **3D mecânico** (`64x32x8 = 16.384` voxels), usar formulação de 1 jogador com tensor físico de 7 canais, separar canais Add/Remove com máscara 6-conectada e adotar **micro-batches PV (até 32 ações/passo)** + **pré-treino híbrido**.
- **Onde Citar no TCC:** Cap. 1.2, Cap. 2.4 e Cap. 2.7 (Marco pioneiro de AlphaZero em OT baseada em grade).

### `sun2020generative`
- **Citação Completa:** SUN, H.; MA, L. Generative Design by Using Exploration Approaches of Reinforcement Learning in Density-Based Structural Topology Optimization. *Designs*, v. 4, n. 2, p. 10, 2020. DOI: `10.3390/designs4020010`. *(Origem: Nova `0B`)*
- **Resumo Técnico Objetivo:** Injeta quatro estratégias de exploração de RL/bandidos ($\varepsilon$-greedy, UCB, *Thompson Sampling* e *Information-Directed Sampling* — IDS) nas sensibilidades do SIMP em 2D e 3D (incluindo *Atmospheric Diving Suit*), gerando famílias diversas de soluções; o IDS falhou em convergir em 3D pelo alto custo.
- **Pontos Fortes / Limitações:** Demonstra geração de diversidade em 3D usando conceitos de UCB/RL com apenas 1–2 parâmetros; continua dependente de gradientes/sensibilidades do SIMP e não usa rede neural nem árvore MCTS.
- **Comparação Direta com o AlphaBuilder:** Enquanto Sun & Ma perturbam sensibilidades locais do SIMP, o AlphaBuilder aprende uma política espacial global (`SwinUNETR`) e executa busca explícita PUCT em árvore sobre estados parciais e finais sem depender de gradientes contínuos durante a inferência.
- **Onde Citar no TCC:** Cap. 2.7 (Abordagens híbridas de exploração RL em OT 3D).

### `sun2021structural`
- **Citação Completa:** SUN, H.; MA, L. Structural Topology Optimization by Combining BESO with Reinforcement Learning. *Journal of Harbin Institute of Technology (New Series)*, v. 28, n. 1, p. 85–96, 2021. DOI: `10.11916/j.issn.1005-9113.19078`. *(Origem: Nova `0B`)*
- **Resumo Técnico Objetivo:** Estende o trabalho de 2020 combinando o método evolucionário discreto BESO com RL Monte Carlo ($\varepsilon$-greedy) em 2D e 3D para obter múltiplas topologias distintas avaliadas por *Intersection over Union* (IoU).
- **Pontos Fortes / Limitações:** Acopla decisão discreta (BESO) a exploração estocástica com diversidade controlada; ainda guiado por sensibilidades elementares clássicas sem modelo neural generalista.
- **Comparação Direta com o AlphaBuilder:** Reforça o contraste entre adicionar ruído exploratório a heurísticas de sensibilidade (BESO+RL) e construir uma política/valor neural reutilizável sobre MCTS no AlphaBuilder.
- **Onde Citar no TCC:** Cap. 2.7 (Extensão discreta BESO + RL).

### `hayashi2020reinforcement`
- **Citação Completa:** HAYASHI, K.; OHSAKI, M. Reinforcement Learning and Graph Embedding for Binary Truss Topology Optimization Under Stress and Displacement Constraints. *Frontiers in Built Environment*, v. 6, p. 59, 2020. DOI: `10.3389/fbuil.2020.00059`. *(Origem: Pré-existente `0A` como ano 2021 + Corrigida/Enriquecida `0B`/`0D`)*
- **Resumo Técnico Objetivo:** Modela a poda binária de barras a partir de uma *ground structure* de treliça 2D como MDP usando *Graph Embedding* e Q-learning, adotando recompensa contínua baseada na margem de tensão $r = L_e (1 - \max_i \sigma_i / \sigma_{\text{allow}})$ além de penalidades por deslocamento e desconexão.
- **Pontos Fortes / Limitações:** Permite atravessar estados temporariamente inviáveis via sinal contínuo de margem física e generaliza entre geometrias de treliça; restrito a treliças 2D discretas, apenas remoção de barras e RL *model-free* sem MCTS.
- **Comparação Direta com o AlphaBuilder:** Espelha o uso de restrições de deslocamento/desconexão no loop FEM, mas evidencia que a penalidade abrupta `-1.0` do AlphaBuilder para `max_displacement > 128000.0` ou desconexão na Fase 2 (`reward.py`) é menos suave que sinais contínuos de margem física e que apenas remover barras limita a recuperação estrutural frente ao espaço Add/Remove.
- **Onde Citar no TCC:** Cap. 2.7 e Cap. 4.4 / Cap. 6.2 (já citado em L185 de `tcc_v5.txt`; corrigir o ano de 2021 para **2020** no texto e na bibliografia).

### `hayashi2021reinforcement3d`
- **Citação Completa:** HAYASHI, K.; OHSAKI, M. Reinforcement learning for optimal topology design of 3D trusses. 2021. DOI: `10.15126/900337`. *(Origem: Nova `0B`)*
- **Resumo Técnico Objetivo:** Estende o framework de *Graph Embedding* e Q-learning de Hayashi & Ohsaki (2020) para otimização topológica binária de treliças espaciais 3D.
- **Pontos Fortes / Limitações:** Valida transferência da formulação sequencial baseada em grafos para 3D em treliças; permanece restrito a barras discretas pré-definidas (*ground structure*) sem ações aditivas nem MCTS.
- **Comparação Direta com o AlphaBuilder:** Mostra a diferença entre escalar treliças de barras para 3D e escalar um domínio contínuo 3D voxelizado `(64, 32, 8)` sem *ground structure* fixa como no AlphaBuilder.
- **Onde Citar no TCC:** Cap. 2.7 (Referência complementar ao trabalho de 2020 na transição para 3D).

### `brown2022deep`
- **Citação Completa:** BROWN, N. K.; GARLAND, A. P.; FADEL, G. M.; LI, G. Deep reinforcement learning for engineering design through topology optimization of elementally discretized design domains. *Materials & Design*, v. 218, p. 110672, 2022. DOI: `10.1016/j.matdes.2022.110672`. *(Origem: Nova `0B`/`0D`)*
- **Resumo Técnico Objetivo:** Aplica *Double Q-learning* (DDQN) baseado em CNN elemento-a-elemento em domínios contínuos 2D sem datasets SIMP prévios, removendo elementos sólidos sequencialmente com filtro de ações ilegais (suportes/cargas/desconexões), recompensa multiobjetivo (*strain energy* + fração de volume) e refinamento progressivo de malha ($6\times6 \to 24\times24$).
- **Pontos Fortes / Limitações:** Opera diretamente sobre malhas contínuas FEM com máscara de ações ilegais e generalização multirresolução; restrito a 2D, puramente destrutivo (só remoção partindo do domínio cheio), sem MCTS e sujeito a padrões *checkerboard* / fragmentos sem controle explícito.
- **Comparação Direta com o AlphaBuilder:** Um dos paralelos operacionais mais próximos no contínuo discretizado por FEM. O AlphaBuilder avança de 2D para **3D**, substitui a remoção unidirecional pela política **Add/Remove a partir do vazio**, substitui DDQN *model-free* por **MCTS (PUCT) + micro-batches de 32 voxels**, e apoia a ideia de validação multirresolução no `_dynamic_pad`.
- **Onde Citar no TCC:** Cap. 2.3, Cap. 2.7, Cap. 4.4 e Cap. 4.6 (Referência essencial de DRL elemento-a-elemento em contínuo).

### `jang2022generative`
- **Citação Completa:** JANG, S.; YOO, S.; KANG, N. Generative Design by Reinforcement Learning: Enhancing the Diversity of Topology Optimization Designs. *Computer-Aided Design*, v. 146, p. 103225, 2022. DOI: `10.1016/j.cad.2022.103225`. `arXiv:2008.07119`. *(Origem: Nova `0B`)*
- **Resumo Técnico Objetivo:** Treina agente **PPO** com *surrogate* neural de SIMP/FEA e *data augmentation* simétrico para ajustar parâmetros de entrada de OT (cargas, BCs, fração de volume) visando gerar um portfólio diverso de rodas automotivas válidas.
- **Pontos Fortes / Limitações:** Resolve convergência única do SIMP e evita chamadas caras de FEM durante o treino via surrogate; não edita diretamente a topologia em grade e depende da fidelidade do surrogate em 2D.
- **Comparação Direta com o AlphaBuilder:** Enquanto Jang et al. ajustam hiperparâmetros em espaço de alto nível sobre um surrogate 2D, o AlphaBuilder edita diretamente voxels 3D via política densa `(2, D, H, W)`, avalia o estado real na Fase 2 com **FEniCSx** e usa aumento de dados com simetria física (`augmentation.py`).
- **Onde Citar no TCC:** Cap. 2.7 e Cap. 4.3 (*Data augmentation* com preservação de simetria física).

### `luo2022alphatruss`
- **Citação Completa:** LUO, R.; WANG, Y.; XIAO, W.; ZHAO, X. AlphaTruss: Monte Carlo Tree Search for Optimal Truss Layout Design. *Buildings*, v. 12, n. 5, p. 641, 2022. DOI: `10.3390/buildings12050641`. *(Origem: Nova `0B`/`0D`)*
- **Resumo Técnico Objetivo:** Modela otimização de treliças 2D como MDP em 2 estágios (Estágio I: MCTS UCT adiciona nós, barras e áreas; Estágio II: refino contínuo de coordenadas e seções) avaliado no OpenSees sob hierarquia condicional de recompensa ($-1$ instável, $0$ violação de tensão/deslocamento/flambagem, $\lambda/W$ válido) e discute backups híbridos *MixMax* contra diluição de recompensas esparsas.
- **Pontos Fortes / Limitações:** Supera AGs e Q-learning/DQN em múltiplas cargas e restrições severas; UCT puro sem rede neural (*policy/value*) e restrito a treliças 2D de barras.
- **Comparação Direta com o AlphaBuilder:** Valida a decomposição em 2 fases e a separação hierárquica entre inviabilidade (`-1.0`) e score contínuo viável (`calculate_fem_score`), porém o cálculo de média pura `W/N` no `MCTSNode` do AlphaBuilder corre o risco apontado por Luo et al. de subestimar ramos bons cercados por estados falhos `-1.0`.
- **Onde Citar no TCC:** Cap. 2.4, Cap. 2.7, Cap. 4.4 e Cap. 6.2 (Backup de valores e recompensas hierárquicas em MCTS estrutural).

### `du2023automatic`
- **Citação Completa:** DU, W.; ZHAO, J.; YU, C.; YAO, X.; SONG, Z.; WU, S.; LUO, R.; LIU, Z.; ZHAO, X.; WU, Y. Automatic Truss Design with Reinforcement Learning. In: *Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI-23)*, 2023. DOI: `10.24963/ijcai.2023/407`. `arXiv:2306.15182`. *(Origem: Pré-existente `0A` atribuída erroneamente a `LI, Y. et al.` + Corrigida/Enriquecida `0B`)*
- **Resumo Técnico Objetivo:** Propõe o **AutoTruss**, combinando MCTS (UCT) no Estágio 1 para encontrar layouts válidos diversos sob recompensa esparsa ($-1$ inválido estrutural, $0$ outro inválido, $\kappa/\text{Mass}(G)^2$ válido) e *Soft Actor-Critic* (SAC) contínuo no Estágio 2 para refinar coordenadas e áreas, atingindo 25,1% de melhoria nos casos 3D mais difíceis (ex.: *Cantilever Sundial*).
- **Pontos Fortes / Limitações:** Referência marcante na validação de MCTS + Deep RL em estruturas **3D**; separa os estágios em algoritmos distintos (UCT e SAC) e aplica-se a treliças discretas.
- **Comparação Direta com o AlphaBuilder:** Mesma filosofia de transição entre descoberta topológica (Fase 1) e refinamento de eficiência material (Fase 2), mas o AlphaBuilder unifica ambas em um único motor MCTS (PUCT) guiado pela `AlphaBuilderV31` em **contínuo 3D voxelizado** e pré-treinado em Bézier + FEniTop em vez de UCT cego.
- **Onde Citar no TCC:** Cap. 2.3 e Cap. 2.7 (substituir `LI, Y. et al., 2023` em L148, L185 e L1061 de `tcc_v5.txt` por **Du et al., 2023**).

### `ororbia2023discrete`
- **Citação Completa:** ORORBIA, A.; WARN, G. Discrete structural design synthesis: a hierarchical-inspired deep reinforcement learning approach. *Journal of Mechanical Design*, v. 146, n. 9, 2023. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Emprega aprendizado por reforço profundo com inspiração hierárquica para síntese discreta de estruturas, decompondo decisões construtivas sob restrições mecânicas.
- **Pontos Fortes / Limitações:** Trata síntese estrutural discreta hierarquicamente com DRL; restrito a configurações estruturais discretas/treliças sem busca MCTS voxelizada 3D.
- **Comparação Direta com o AlphaBuilder:** Dialoga com a motivação hierárquica de decompor a síntese do AlphaBuilder em Fase 1 (conectividade global) e Fase 2 (refinamento fino), bem como o uso de arquitetura multi-escala.
- **Onde Citar no TCC:** Cap. 2.3 e Cap. 2.7 (já citado em L148 e L185 de `tcc_v5.txt`).

### `padhy2024treetop`
- **Citação Completa:** PADHY, S. et al. TreeTOp: topology optimization using constructive solid geometry trees. *arXiv preprint arXiv:2409.02300*, 2024. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Formula a otimização topológica por meio de árvores de *Constructive Solid Geometry* (CSG), representando a topologia como composição hierárquica de primitivas geométricas editáveis.
- **Pontos Fortes / Limitações:** Garante geometrias CAD limpas e parametrizáveis sem pixelização; expressividade restrita ao vocabulário de primitivas CSG e regras da árvore.
- **Comparação Direta com o AlphaBuilder:** Contrasta representação paramétrica em árvores CSG com a grade voxelizada livre 3D `(64, 32, 8)` + busca MCTS do AlphaBuilder.
- **Onde Citar no TCC:** Cap. 2.3 e Cap. 2.7 (já citado em L148 e L185 de `tcc_v5.txt`).

### `rochefortbeaudoin2025structural`
- **Citação Completa:** ROCHEFORT-BEAUDOIN, T.; VADEAN, A.; AAGE, N.; ACHICHE, S. Structural design through reinforcement learning. *Engineering Applications of Artificial Intelligence*, v. 161, p. 112273, 2025. DOI: `10.1016/j.engappai.2025.112273`. `arXiv:2407.07288`. *(Origem: Nova `0B`/`0D`/`0E` — inclui preprint 2024)*
- **Resumo Técnico Objetivo:** Introduz o **SOgym**, ambiente open-source de RL para OT contínua acoplada a FEM com parametrização independente de malha (*Moving Morphable Components* — MMC), comparando PPO, SAC e **DreamerV3-100M** sob normalização logarítmica da *compliance* ($-\ln C$). O DreamerV3 com observação inspirada no *TopOpt game* alcançou **0% de taxa de desconexão** (contra falhas frequentes em modelos puramente supervisionados) e *compliance* dentro de 54% do SIMP.
- **Pontos Fortes / Limitações:** Benchmark maduro de *model-based RL* vs. *model-free RL* em OT contínua, com métricas explícitas de eficiência amostral FEM e taxa de desconexão; evita ações voxel-a-voxel limitando-se a primitivas MMC em 2D.
- **Comparação Direta com o AlphaBuilder:** Valida três pilares do AlphaBuilder: (i) RL *model-based* / planejamento supera *model-free*; (ii) escalonamento logarítmico (`log10` em `reward.py`) estabiliza recompensas de *compliance*; (iii) modelos puramente supervisionados (Etapa 1) não garantem 0% de desconexão sem verificação física/MCTS subsequente. Serve também de protocolo de métricas na Etapa 6 (taxa de desconexão, chamadas FEM e gap vs. SIMP).
- **Onde Citar no TCC:** Cap. 2.6, Cap. 2.7, Cap. 4.4, Cap. 4.7 e Cap. 6.3–6.4.

### `garayalde2025mastering`
- **Citação Completa:** GARAYALDE, G.; ROSAFALCO, L.; TORZONI, M.; CORIGLIANO, A. Mastering Truss Structure Optimization With Tree Search. *ASME Journal of Mechanical Design*, v. 147, n. 10, p. 101702, 2025. DOI: `10.1115/1.4068300`. `arXiv:2406.06145`. *(Origem: Nova `0B`)*
- **Resumo Técnico Objetivo:** Acopla gramática generativa de construção progressiva (garantindo transições estáveis passo a passo) a **MCTS (UCT)** avaliado via FEM (*compliance*, volume, deslocamentos e estabilidade), superando Q-learning, **DQN** e Algoritmos Genéticos em convergência e economia de chamadas FEM, e revelando que as **primeiras decisões na árvore** determinam fortemente o ótimo atingível.
- **Pontos Fortes / Limitações:** Demonstra empiricamente superioridade do MCTS sobre DQN e AGs e a criticidade das decisões iniciais; utiliza UCT puro sem rede neural em treliças com regras gramaticais manuais.
- **Comparação Direta com o AlphaBuilder:** Fundamenta diretamente por que o AlphaBuilder combina MCTS com **pré-treino supervisionado (Bézier 3D + FEniTop)** para orientar os passos iniciais da árvore, evitando busca cega onde erros iniciais comprometem toda a trajetória.
- **Onde Citar no TCC:** Cap. 2.2, Cap. 2.4, Cap. 2.7 e Cap. 4.6 (Justificativa da superioridade do MCTS sobre AGs/DQN e do papel das decisões iniciais).

### `shin2025numerical`
- **Citação Completa:** SHIN, M.; YOON, G. H. A numerical study of reinforcement learning for acoustic topology optimization. *Structural and Multidisciplinary Optimization*, 2025. DOI: `10.1007/s00158-025-04130-8`. *(Origem: Nova `0B`)*
- **Resumo Técnico Objetivo:** Aplica **Double DQN (DDQN)** a otimização topológica acústica 2D partindo de **domínio vazio**, permitindo ao agente **adicionar ou remover** elementos sólidos acoplados a solver FEM acústico com recompensa incremental $\Delta J$ sem dataset prévio.
- **Pontos Fortes / Limitações:** Formulação bidirecional *empty-to-structure* (Add/Remove) livre de gradientes adjuntos; limitado a 2D acústico, ações unitárias por vez e RL *model-free* sem MCTS.
- **Comparação Direta com o AlphaBuilder:** Paralelo operacional muito próximo quanto ao início do vazio + ações duais Add/Remove + FEM no loop; o AlphaBuilder amplia essa ideia para **3D elástico** com *lookahead* MCTS e micro-batches PV de até 32 voxels.
- **Onde Citar no TCC:** Cap. 2.3 e Cap. 2.7 (Formulações *empty-to-structure* Add/Remove com FEM).

### `keerthikumar2025reinforcement`
- **Citação Completa:** KEERTHI KUMAR, N.; MANASA, C. M.; PAVAN KUMAR, B. K.; BALI. Reinforcement learning-based topology optimization for generative designed lightweight structures. *MethodsX*, 2025. DOI: `10.1016/j.mex.2025.103539`. *(Origem: Nova `0B`)*
- **Resumo Técnico Objetivo:** Integra **PPO**, FEA (restrições de tensão e deslocamento) e suavização por *Signed Distance Field* (SDF) para exportação STL e impressão 3D, reportando até 40% de redução de peso em componente mecânico (cubo de roda).
- **Pontos Fortes / Limitações:** Conecta saída de RL-TopOpt a pós-processamento geométrico SDF/STL para manufatura aditiva; focado em caso paramétrico/específico com PPO *model-free*.
- **Comparação Direta com o AlphaBuilder:** Exemplifica a ponte entre saídas discretas de RL e suavização para manufatura (STL/SDF), complementando o visualizador 3D atual do AlphaBuilder caso haja futura exportação CAD.
- **Onde Citar no TCC:** Cap. 2.6 / Cap. 2.7 ou Cap. 8 (Manufaturabilidade e pós-processamento geométrico).

### `karvanpour2025topology`
- **Citação Completa:** KARVANPOUR. *Topology optimization via Monte Carlo tree search in linear-elastic continua*. Dissertação de Mestrado, Politecnico di Milano, 2025. *(2025/2026; orientada por A. Corigliano e M. Torzoni)*. *(Origem: Nova `0B`)*
- **Resumo Técnico Objetivo:** Estende a metodologia de MCTS + gramática generativa de Garayalde et al. (2025) de treliças para contínuos elásticos lineares 2D discretizados por elementos finitos CST, adicionando elementos um a um sob conectividade e simetria.
- **Pontos Fortes / Limitações:** Confirma a tendência atual (2025–2026) de levar MCTS diretamente ao contínuo elástico sob conectividade explícita; restrito a 2D CST sem rede neural guia (*policy/value*).
- **Comparação Direta com o AlphaBuilder:** Mostra que a própria linha de Garayalde et al. está migrando para o contínuo elástico — estágio em que o AlphaBuilder já introduz **3D**, **Swin-UNETR** e **micro-batches PV**.
- **Onde Citar no TCC:** Cap. 2.7 e Cap. 2.8 (Delimitação da lacuna de MCTS no contínuo 3D).

### `oh2026efficient`
- **Citação Completa:** OH, K. S.; KIM, Y. Y.; CHUNG, H.; OH, J. H. An efficient reinforcement learning action strategy for topology optimization: application to muffler design. *Structural and Multidisciplinary Optimization*, v. 69, n. 3, p. 51, 2026. DOI: `10.1007/s00158-025-04244-z`. *(Origem: Nova `0B`)*
- **Resumo Técnico Objetivo:** Enfrenta a explosão combinatória do espaço de ações em malhas finas prevendo **Q-values individuais por elemento** e selecionando-os em ordem decrescente até a fração de volume alvo, maximizando *Sound Transmission Loss* via FEM acústico.
- **Pontos Fortes / Limitações:** Escalabilidade tratável para malhas 2D finas evitando busca exponencial; seleção gulosa *one-shot* até volume fixo sem revisão por árvore nem remoção corretiva em trajetória.
- **Comparação Direta com o AlphaBuilder:** Ataca exatamente a mesma dor que motivou o `action_batch_size=32` e `top_k_expansion=32` (`select_pv_sequences`) sobre os mapas densos `(2, D, H, W)` do AlphaBuilder, que entretanto preserva correção iterativa via busca em árvore MCTS e ação dual Add/Remove.
- **Onde Citar no TCC:** Cap. 2.3, Cap. 2.7 e Cap. 4.6 (Mitigação da explosão combinatória no espaço de ações em grade).

### `jeon2026multiscale`
- **Citação Completa:** JEON; YOON; KIM. Multi-scale multi-resolution reinforcement learning topology optimization: application to elastic-wave metamaterial design. *Structural and Multidisciplinary Optimization*, v. 69, n. 3, p. 80, 2026. DOI: `10.1007/s00158-026-04268-z`. *(Origem: Nova `0B`)*
- **Resumo Técnico Objetivo:** Introduz um espaço de ações multiescala baseado em *wavelets* para reduzir o custo computacional de RL elemento-a-elemento em projeto de metamateriais de ondas elásticas.
- **Pontos Fortes / Limitações:** Reduz dimensão efetiva da busca por decomposição multiescala; aplicado a células unitárias periódicas 2D em metamateriais elásticos.
- **Comparação Direta com o AlphaBuilder:** Alternativa conceitual à extração de micro-batches PV e hierarquia multi-escala de janelas do `SwinUNETR` para domar o fator de ramificação em grades densas.
- **Onde Citar no TCC:** Cap. 2.7 ou Cap. 4.6 (Estratégias recentes de redução do espaço de ações em RL-TopOpt).

### `zheng2021vulnerability`
- **Citação Completa:** ZHENG, Y.; YAN, Z.; CHEN, K.; SUN, J.; XU, Y.; LIU, Y. Vulnerability Assessment of Deep Reinforcement Learning Models for Power System Topology Optimization. *IEEE Transactions on Smart Grid*, 2021. *(Origem: Nova `0B`)*
- **Resumo Técnico Objetivo:** Analisa vulnerabilidade e robustez de modelos de Deep RL aplicados à comutação topológica em redes elétricas de potência (grafos).
- **Pontos Fortes / Limitações:** Relevante para confiabilidade/robustez de políticas DRL sob perturbações; pertence ao domínio de sistemas elétricos de potência e não ao contínuo estrutural 3D.
- **Comparação Direta com o AlphaBuilder:** Compartilha o termo "Topology Optimization" com DRL discreto em grafos, mas sem relação com mecânica do contínuo 3D ou FEM; citado apenas para desambiguação terminológica da literatura ("Yan et al.").
- **Onde Citar no TCC:** Cap. 2.7 (Opcional — nota breve de desambiguação de domínios caso necessário).

### `dorfer2022power`
- **Citação Completa:** DORFER et al. Power Grid Congestion Management via Topology Optimization with AlphaZero. *arXiv preprint arXiv:2211.05612*, 2022. *(Origem: Nova `0B`)*
- **Resumo Técnico Objetivo:** Aplica arquitetura AlphaZero (MCTS + rede neural) à reconfiguração topológica de redes elétricas sob restrições operacionais de fluxo.
- **Pontos Fortes / Limitações:** Demonstra eficácia do AlphaZero em otimização topológica combinatória sujeita a física de redes; domínio discreto de grafos elétricos e não sólido contínuo 3D.
- **Comparação Direta com o AlphaBuilder:** Paralelo metodológico da adaptação do AlphaZero para otimização topológica de engenharia sob restrições físicas, em outro domínio físico.
- **Onde Citar no TCC:** Cap. 2.4 ou Cap. 2.7 (Nota breve sobre sucesso do AlphaZero em engenharia além de jogos).

---

## Eixo 4: Reward Shaping, Conectividade e Pré-treino + Self-Play

> *Nota de integração:* Além dos 7 artigos primários fichados abaixo, as evidências de `0D` apoiam-se diretamente em 4 trabalhos já fichados nos Eixos 1–3 (`luo2022alphatruss`, `brown2022deep`, `hayashi2020reinforcement`, `rochefortbeaudoin2025structural`) e nos protocolos de `woldseth2022use` e `sigmund2022benchmarking`.

### `behzadi2023taming`
- **Citação Completa:** BEHZADI, M. Taming Connectedness in Machine-Learning-based Topology Optimization with Connectivity Graphs. *Computer-Aided Design*, v. 160, 2023. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Ataca o problema crônico de ilhas desconectadas e quebras de caminho de carga em otimização topológica baseada em Machine Learning, incorporando grafos explícitos de conectividade ao pipeline de modelagem/avaliação.
- **Pontos Fortes / Limitações:** Abordagem direta para mitigar desconexão estrutural em predições neurais; focada em conectividade topológica em pipelines ML sem busca sequencial MCTS + RL.
- **Comparação Direta com o AlphaBuilder:** Fundamenta a necessidade das garantias explícitas de conectividade do AlphaBuilder: máscaras morfológicas 6-conectadas (`legal_moves.py`), *Negative Sampling* com sabotagem/erosão em `trainer.py` (`W_NEG = 5.0`), penalização de ilhas e checagem de cobertura de carga em `reward.py`.
- **Onde Citar no TCC:** Cap. 2.6 (Conectividade e manufaturabilidade; já citado em L177 de `tcc_v5.txt`) e Cap. 4.4.

### `muller2025improving`
- **Citação Completa:** MÜLLER, H.; KUDENKO, D. Improving the Effectiveness of Potential-Based Reward Shaping in Reinforcement Learning. *arXiv preprint arXiv:2502.01307*, 2025. *(AAMAS 2025 Extended Abstract / arXiv Preprint)*. *(Origem: Nova `0D`)*
- **Resumo Técnico Objetivo:** Analisa *Potential-Based Reward Shaping* (PBRS), demonstrando que sua eficácia em Deep RL depende da interação entre inicialização de $Q/V$ e recompensas externas, além de propor um deslocamento linear na função potencial que melhora eficiência amostral em recompensas esparsas preservando a invariância da política ótima.
- **Pontos Fortes / Limitações:** Fornece critério matemático rigoroso para desenhar termos auxiliares de *shaping* sem distorcer o ótimo; resultado geral de teoria de RL (requer tradução para a arquitetura específica do agente).
- **Comparação Direta com o AlphaBuilder:** Gera um **alerta teórico crítico sobre `hybrid_predictor`** (`runner.py`): na Fase 1, somar `net_val + connectivity_bonus - island_penalty` (com *clamp* em `[-1, 1]`) usa heurísticas absolutas de estado $B(s)-P(s)$ e **não** uma diferença potencial $\Phi(s')-\Phi(s)$. Isso quebra a invariância da política ótima e pode causar dupla contagem e saturação precoce em `+1.0`, mascarando diferenças de *compliance* entre estruturas conectadas.
- **Onde Citar no TCC:** Cap. 4.4 (Análise teórica de *reward shaping*), Cap. 6.4 (Ameaças à validade) e Cap. 7 (Refinamento da função de busca/recompensa).

### `huang2022closer`
- **Citação Completa:** HUANG, S.; ONTAÑÓN, S. A Closer Look at Invalid Action Masking in Policy Gradient Algorithms. In: *The International FLAIRS Conference Proceedings*, v. 35, 2022. DOI: `10.32473/flairs.v35i.130584`. `arXiv:2006.14171`. *(Origem: Nova `0D`)*
- **Resumo Técnico Objetivo:** Prova matematicamente que zerar probabilidades de ações inválidas atribuindo $-\infty$ aos seus *logits* antes do *softmax* produz gradientes de política válidos (zerando exatamente o gradiente nos *logits* ilegais) e escala de maneira muito superior à penalização por recompensa negativa quando a maior parte do espaço de ações é ilegal.
- **Pontos Fortes / Limitações:** Justificativa matemática exata para *action masking* em espaços combinatórios grandes; analisa principalmente algoritmos de *policy gradient*, exigindo cuidado na consistência entre treino supervisionado e inferência com máscara.
- **Comparação Direta com o AlphaBuilder:** Valida formalmente `apply_action_mask` e `masked_softmax` (`legal_moves.py`) no grid $64\times32\times8$ ($32.768$ ações totais com fronteira legal esparsa), mas expõe uma assimetria importante: em `trainer.py`, `policy_loss` aplica `BCEWithLogitsLoss` densa em todos os voxels sem mascarar voxels ilegais, além da função `get_protected_regions` não estar ainda acoplada para proibir remoção destrutiva de apoios/cargas (Item 3 da Tabela 19 de `tcc_v5.txt`).
- **Onde Citar no TCC:** Cap. 2.3, Cap. 4.5–4.6 (Mascaramento de ações legais) e Cap. 7 (Decisão registrada de tornar ilegal a desconexão da carga).

### `yi2026reinforcement`
- **Citação Completa:** YI, J.; ZHANG, X.; ZHANG, X.; XIA, W. Reinforcement Learning for Engineering Design Optimization: A Task-Oriented Critical Review. *Mathematics*, v. 14, n. 22, p. 4132, 2026. DOI: `10.3390/math14224132`. *(Origem: Nova `0D`)*
- **Resumo Técnico Objetivo:** Revisão crítica de 197 estudos mapeando design em engenharia para MDPs em RL, sistematizando escalarização multiobjetivo (rigidez/compliance vs. volume), *action masking* e patologias recorrentes como ilhas flutuantes, *checkerboard*, má calibração de pesos levando a ótimos degenerados e gargalo de chamadas FEM.
- **Pontos Fortes / Limitações:** Taxonomia atualizada e orientada a formulação de MDP/recompensas; artigo de revisão sem novos algoritmos próprios.
- **Comparação Direta com o AlphaBuilder:** Enquadra a auditoria quantitativa da fórmula v3.2 em `reward.py`: na relação entre `COMPLIANCE_SLOPE=0.16` ($\log_{10}$) e `VOLUME_SENSITIVITY=2.0`, reduzir $\Delta V = -0{,}10$ compensa um aumento de $10^{1{,}25} \approx 17{,}8\times$ na *compliance*, enquanto `VOLUME_BONUS_MAX=0.30` não é atingível para $V \ge 0$ (`max=0.20`). Alerta para risco de afinamento extremo até o corte abrupto `DISPLACEMENT_LIMIT_RAW=128000.0` (*reward hacking*).
- **Onde Citar no TCC:** Cap. 2.7, Cap. 4.4 (Calibração multiobjetivo da recompensa) e Cap. 6.2 / Cap. 7 (Recalibração de penalidades).

### `wang2020warmstart`
- **Citação Completa:** WANG, H.; PREUSS, M.; PLAAT, A. Warm-Start AlphaZero Self-Play Search Enhancements. In: *Parallel Problem Solving from Nature — PPSN XVI*, LNCS, v. 12270, p. 528–542, Springer, Cham, 2020. DOI: `10.1007/978-3-030-58115-2_37`. `arXiv:2004.12357`. *(Origem: Nova `0D`)*
- **Resumo Técnico Objetivo:** Estuda o *cold-start* no AlphaZero quando estimativas iniciais de política e valor são imprecisas, demonstrando que transições adaptativas e **gradualmente ponderadas** entre heurísticas clássicas (como RAVE/*rollout*) e avaliação neural superam tanto o AlphaZero puro quanto cortes binários abruptos.
- **Pontos Fortes / Limitações:** Evidência direta sobre suavização da transição heurística $\to$ rede neural em MCTS; avaliado originalmente em jogos de tabuleiro.
- **Comparação Direta com o AlphaBuilder:** Questiona o chaveamento booleano rígido `use_guided_value` em `runner.py` (`net_val + connectivity_bonus - island_penalty` vs. `net_val` puro) e reforça que o *warm-up* (Etapa 1) precisa entregar erro suficientemente baixo fora da trajetória ótima para não corromper o termo $Q(s,a)$ do PUCT no início do *self-play*.
- **Onde Citar no TCC:** Cap. 4.6 (Transição *warm-up* para *self-play*) e Cap. 6.1–6.2 / Cap. 7.

### `jacob2022modeling`
- **Citação Completa:** JACOB, A. P.; WU, D. J.; FARINA, G.; LERER, A.; HU, H.; BAKHTIN, A.; ANDREAS, J.; BROWN, N. Modeling Strong and Human-Like Gameplay with KL-Regularized Search. In: *Proceedings of the 39th International Conference on Machine Learning (ICML 2022)*, PMLR, v. 162, p. 9695–9728, 2022. `arXiv:2112.07544`. *(Origem: Nova `0D`)*
- **Resumo Técnico Objetivo:** Mostra que *Behavioral Cloning* (BC) puro acumula erros sequenciais, enquanto *self-play* não regularizado pode divergir para estratégias extremas/degeneradas fora do suporte; propõe o $\pi\text{KL}$, acoplando MCTS com regularização por divergência KL à política âncora aprendida por imitação.
- **Pontos Fortes / Limitações:** Formaliza a sinergia entre pré-treino supervisionado e busca em árvore MCTS sem perder suporte estrutural; focado em jogos estratégicos e requer escolha adequada da força de regularização KL.
- **Comparação Direta com o AlphaBuilder:** Justifica manter o *prior* da Etapa 1 (treinado em Bézier + FEniTop) como âncora contra desvios degenerados no MCTS de baixo orçamento (`160` simulações para até `32` ações em lote), mas alerta que `POS_WEIGHT_ADD=8.0` e `POS_WEIGHT_REM=5.0` em `trainer.py` deslocam os *logits* positivos ($\text{logit} + \ln w_{\text{pos}}$), alterando a entropia de `masked_softmax` que alimenta $P(s,a)$ no PUCT.
- **Onde Citar no TCC:** Cap. 4.5–4.6 (Calibração da política supervisionada para busca) e Cap. 6.4.

### `nakamoto2023calql`
- **Citação Completa:** NAKAMOTO, M.; ZHAI, Y.; SINGH, A.; SOBOL MARK, M.; MA, Y.; FINN, C.; KUMAR, A.; LEVINE, S. Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning. In: *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*, 2023. DOI: `10.48550/arXiv.2303.05479`. `arXiv:2303.05479`. *(Origem: Nova `0D`)*
- **Resumo Técnico Objetivo:** Explica o colapso inicial (*unlearning*) na transição *offline-to-online*: quando a função valor pré-treinada *offline* tem escala descalibrada em relação aos retornos *online*, gradientes iniciais sob *distribution shift* destroem a política pré-treinada. Propõe **Cal-QL** para manter $V/Q$ calibrado na escala real da política de referência.
- **Pontos Fortes / Limitações:** Explica mecanisticamente um dos maiores riscos do pipeline pré-treino + *self-play*; originalmente formulado para Q-learning *actor-critic* e não especificamente para PUCT.
- **Comparação Direta com o AlphaBuilder:** Fundamenta uma descoberta crítica da auditoria de `0D`: `harvest/processing.py` ainda define `compute_normalized_value` com a fórmula legada `tanh((-ln(C) - 12*vol + 6.65)/2)`, enquanto `harvest/optimization.py` e `selfplay/reward.py` usam `calculate_fem_score` v3.2 ($\log_{10}$). Qualquer divergência entre os alvos gravados no SQLite da Etapa 1 (`122.358` registros) e a escala online do `reward.py` — somada ao viés negativo de `W_NEG=5.0` em `weighted_value_loss` — induz descalibração severa no *self-play*.
- **Onde Citar no TCC:** Cap. 3.8 (Consistência de alvos), Cap. 4.4–4.5 (Calibração offline-to-online) e Cap. 6.1–6.2.

### `wu2019accelerating`
- **Citação Completa:** WU, D. J. Accelerating Self-Play Learning in Go. *arXiv preprint arXiv:1902.10565*, 2019. *(Origem: Incorporada na revisão da Etapa 1 — Precedente direto da `Blueprint Head` auxiliar)*
- **Resumo Técnico Objetivo:** Introduz o **KataGo**, demonstrando redução de cerca de **50×** no custo computacional de treinamento por *self-play* frente a implementações padrão estilo AlphaZero por meio de três inovações principais: **alvos auxiliares espaciais e escalares** (*ownership target* prevendo o estado final de cada interseção do tabuleiro + margem exata de pontos), *global pooling* nas camadas convolucionais e *playout cap randomization*. Os estudos de ablação comprovam que a supervisão espacial densa do desfecho final atua como regularizador poderoso e acelera drasticamente o aprendizado das representações internas para a política e o valor.
- **Pontos Fortes / Limitações:** Prova empírica robusta do ganho de eficiência amostral com cabeça auxiliar espacial de estado final em busca MCTS + *self-play*; originalmente validado em jogo de tabuleiro 2D determinístico (Go).
- **Comparação Direta com o AlphaBuilder:** Fornece o paralelo 1 para 1 para a **`Blueprint Head` auxiliar** de 1 canal `(B, 1, D, H, W)` do AlphaBuilder: assim como o *ownership target* prevê a posse final de cada casa a partir de um estado intermediário, a `Blueprint Head` prevê a estrutura sólida final $X_T$ (via `V_constructed` no *warm-up* e $X_T$ de episódios conectados de alta qualidade no *self-play*). Isso entrega milhares de sinais espaciais densos ao *backbone* compartilhado desde o passo `t = 0` sem contaminar os *logits* da `Policy Head` nem os backups do MCTS.
- **Onde Citar no TCC:** Cap. 2.4 (Evolução pós-AlphaZero — KataGo), Cap. 3.2, Cap. 4.5 (Arquitetura multi-head e `Blueprint Head` auxiliar) e Cap. 6 (Eficiência amostral no *self-play*).

---

## Eixo 5: Referências Prévias do Projeto / Estado do Texto `tcc_v5.txt`

### 5.1. Diagnóstico Consolidado do Estado Textual (`tcc_v5.txt` e Documentos do Drive — `0A`)

Com base na inspeção completa trazida pelo **Relatório 0A**, o rascunho mais recente (`tcc_v5.txt`, 1.322 linhas, refletindo o repositório em 10/12/2025, verificação de 12/10/2026 e auditoria de cópia de 25/10/2026) apresenta:
- **Consolidado e Medido:** Fundamentação e arquitetura/metodologia descritas (Cap. 1–4 e Anexos A–E); baseline SIMP 2D (`64x32`, VF 0,5, compliance `65,94` em 167 iterações / 17 s); convergência MBB 3D (`64x32x8`, compliance `1,524 -> 0,245` em 150 iterações / `576,6 s` em 4 processos MPI); banco de aquecimento com `1.069` episódios e `122.358` registros de 7 canais (80% Bézier / 20% domínio completo); treino de aquecimento apenas com o backbone simples (~745k parâmetros, 30 épocas); 4 episódios de *self-play* em dez/2025 (recompensas entre `-1,0` e `-0,333`, 2 falhas catastróficas por desconexão) + 5ª partida auditada (`131` passos, `-0,824`, compliance anômala $2{,}6\times10^{14}$ a $1{,}7\times10^{15}$).
- **Pendências Experimentais Principais (Cap. 5.8–5.9 e Tabela 19):** (1) corrigir falha de avaliação física no *self-play* (compliance $10^{14}$–$10^{15}$); (2) resolver erro paralelo MPI código `59` via containerização; (3) finalizar aumento de dados, tornar ilegal a remoção que desconecta a carga e recalibrar penalidade de ilhas; (4) ampliar coleta para $\ge 2.000$ episódios; (5) treinar efetivamente o **Swin-UNETR** (~20M parâmetros, nunca treinado até o momento); (6) executar *self-play* válido, preencher instrumentação de $Q$ por voxel (vazia nas 974 linhas) e comparar contra FEniTop em orçamento igualado (Tabela 17 ainda `"não medido"`).
- **Pendências Redacionais e Bibliográficas Críticas em `tcc_v5.txt`:**
  1. Preencher `Orientação: [a definir]` (L4), elaborar *Abstract* e *Sumário*, e atualizar Cap. 5, 6 e 8 após experimentos.
  2. Corrigir numeração embaralhada de **Tabelas** (lista enumera 19, texto vai até `Tabela 20`, pula `Tabela 2`, coloca `Tabela 11` antes de `Tabela 6` e `Tabela 9` antes de `Tabela 8`) e **Figuras** (ordem no texto: `9, 2, 4, 7, 1, 8, 21, 3, 10, 5, 17, 11, 12, 13, 6, 14, 16, 15, 18, 19, 24, 20, 22, 23, 25`), além da referência errada em `1.6` (L98) sobre "anexo de falhas" (está na Seção 3.8).
  3. Corrigir autoria de `du2023automatic` (`LI, Y. et al.` $\to$ `DU, W. et al.`), ano de `hayashi2020reinforcement` (`2021` $\to$ `2020`) e versão publicada de `bhuiyan2026graph` (`Tabarraei & Bhuiyan, ResearchGate 2025` $\to$ `Bhuiyan & Tabarraei, Engineering with Computers, 2026`).
  4. Citar no corpo os **10 artigos científicos** que constavam na lista `Referências` de `tcc_v5.txt` com zero ocorrências inline (`bendsoe2003topology`, `browne2012survey`, `dosovitskiy2021image`, `hatamizadeh2022swin`, `kocsis2006bandit`, `liu2021swin`, `rosin2011multi`, `scroggs2022construction`, `sigmund200199line`, `silver2017mastering`), e remover/mover para nota histórica os 2 itens legados sem uso (`keras2025vit`, `sfepy2025docs`).
  5. Incorporar entradas formais faltantes já supridas por esta consolidação (**FEniTop** $\to$ `jia2024fenitop`; **AlphaZero** $\to$ `silver2018general`) e completar a citação pendente do **PyTorch** (citado em L235 e L317).

### 5.2. Fichamento das Referências Prévias do Projeto, Documentação Técnica e Registros Legados (`0A`)

#### 5.2.1. Documentação Técnica e Fontes Internas (`tcc_v5.txt` e `ref_alphabuilder.md`)

##### `mello2021relatorio`
- **Citação Completa:** MELLO, G. A. S. *Relatório parcial de avaliação de bolsista: PIBIC/VOL, Engenharia Civil*. Aracaju: Universidade Federal de Sergipe, 2021. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Relatório de Iniciação Científica documentando o projeto antecessor de otimização topológica estrutural via Algoritmos Genéticos desenvolvido na UFS (2020–2021).
- **Pontos Fortes / Limitações:** Documenta a continuidade histórica da linha de pesquisa do autor; relatório interno não indexado externamente.
- **Comparação Direta com o AlphaBuilder:** Marco de origem do trabalho, motivando a superação dos Algoritmos Genéticos discretos pelo acoplamento MCTS + rede neural profunda 3D no AlphaBuilder.
- **Onde Citar no TCC:** Cap. 1.3 (Origem) e Cap. 2.2 (já citado em L76 e L110 de `tcc_v5.txt`).

##### `alphabuilder2025repo`
- **Citação Completa:** ALPHA BUILDER. *Repositório de código-fonte*. Desenvolvimento próprio, 2025. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Registro de referência para o repositório de código do projeto AlphaBuilder (backend Python/PyTorch/MONAI/FEniCSx, pipeline MPI e interface web FastAPI/React).
- **Pontos Fortes / Limitações:** Centraliza a rastreabilidade do software implementado; convém especificar URL/commit ou versão consolidada na entrega final.
- **Comparação Direta com o AlphaBuilder:** É o próprio artefato de software do trabalho.
- **Onde Citar no TCC:** Cap. 3 e Cap. 4.8 (Reprodutibilidade).

##### `abnt2018nbr6023`
- **Citação Completa:** ASSOCIAÇÃO BRASILEIRA DE NORMAS TÉCNICAS. *NBR 6023: informação e documentação, referências, elaboração*. Rio de Janeiro: ABNT, 2018. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Norma técnica brasileira para formatação de referências bibliográficas.
- **Pontos Fortes / Limitações:** Padrão formal exigido em trabalhos acadêmicos brasileiros; norma editorial e não técnica da área de OT/IA.
- **Comparação Direta com o AlphaBuilder:** Guia a formatação final da lista bibliográfica do TCC a partir de `referencias.bib`.
- **Onde Citar no TCC:** Elementos pós-textuais (metodologia de normalização / ficha catalográfica ou seção `Referências`).

##### `dokken2025fenicsx`
- **Citação Completa:** FENICS PROJECT; DOKKEN, J. S. *FEniCSx tutorial*. 2025. URL: `https://jsdokken.com/dolfinx-tutorial/`. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Documentação tutorial de referência prática para programação em DOLFINx/FEniCSx (definição de malhas, espaços funcionais, condições de contorno e solução de EDPs).
- **Pontos Fortes / Limitações:** Guia prático essencial de implementação; deve complementar (e não substituir) a citação científica formal `baratta2023dolfinx`.
- **Comparação Direta com o AlphaBuilder:** Utilizado na integração do solver FEM `evaluator.py` e no entendimento das rotinas `fenitop/`.
- **Onde Citar no TCC:** Cap. 3.1 / Cap. 3.6 (Recursos técnicos de implementação do solver).

##### `monai2025swinunetr`
- **Citação Completa:** PROJECT MONAI. *MONAI: Medical Open Network for AI — Documentação dos modelos Swin-UNETR*. 2025. URL: `https://monai.io`. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Documentação oficial da biblioteca MONAI contendo a API `monai.networks.nets.swin_unetr.SwinTransformer` e blocos `UnetrBasicBlock` utilizados na implementação.
- **Pontos Fortes / Limitações:** Referência de software exata para os módulos importados em `model.py`; deve acompanhar a citação científica `hatamizadeh2022swin`.
- **Comparação Direta com o AlphaBuilder:** Biblioteca que fornece os blocos base do `SwinEncoder` em `AlphaBuilderV31`.
- **Onde Citar no TCC:** Cap. 3.2 e Cap. 4.5 (Implementação da arquitetura em PyTorch/MONAI).

##### `bleyer2025comet`
- **Citação Completa:** BLEYER, J. *Computational Solid and Structural Mechanics with FEniCSx (comet-fenicsx)*. GitHub. URL: `https://github.com/bleyerj/comet-fenicsx/blob/main/index.md`. *(Origem: Pré-existente `0A` — `ref_alphabuilder.md`)*
- **Resumo Técnico Objetivo:** Repositório educacional de referência para formulações variacionais de mecânica dos sólidos e elasticidade em FEniCSx.
- **Pontos Fortes / Limitações:** Exemplos práticos validados de mecânica estrutural em FEniCSx; material online de apoio e não artigo revisado por pares.
- **Comparação Direta com o AlphaBuilder:** Recurso consultado durante a transição conceitual de SfePy para FEniCSx na fase do guia inicial.
- **Onde Citar no TCC:** Cap. 3.1 (Opcional, junto de `dokken2025fenicsx`, dando preferência acadêmica a `baratta2023dolfinx` e `jia2024fenitop`).

##### `keras2025vit`
- **Citação Completa:** KERAS TEAM. *Image classification with Vision Transformer*. 2025. URL: `https://keras.io/examples/vision/image_classification_with_vision_transformer/`. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Tutorial oficial de implementação de ViT em TensorFlow/Keras consultado no estágio inicial (DIP) do projeto.
- **Pontos Fortes / Limitações:** Obsoleto em relação ao código de produção atual (que usa PyTorch e MONAI).
- **Comparação Direta com o AlphaBuilder:** Documenta apenas a arquitetura planejada inicialmente em `ref_alphabuilder.md`, superada por `AlphaBuilderV31` em PyTorch.
- **Onde Citar no TCC:** Remover da seção principal de referências de `tcc_v5.txt` (ou manter explicitamente apenas em nota histórica da evolução de stack no Cap. 3).

##### `keras2025vit_small`
- **Citação Completa:** KERAS TEAM. *Train a Vision Transformer on small datasets*. URL: `https://keras.io/examples/vision/vit_small_ds/`. *(Origem: Pré-existente `0A` — `ref_alphabuilder.md`)*
- **Resumo Técnico Objetivo:** Tutorial de treinamento de ViT em conjuntos de dados reduzidos em Keras.
- **Pontos Fortes / Limitações:** Legado da fase inicial em Keras, sem uso no sistema atual.
- **Comparação Direta com o AlphaBuilder:** Motivou discussões iniciais de *warm-up* com poucos dados antes da migração para PyTorch/MONAI e geração MPI com 1.069+ episódios.
- **Onde Citar no TCC:** Não recomendado para citação formal no TCC; registrar apenas como histórico do DIP.

##### `sfepy2025docs`
- **Citação Completa:** SFEPY DEVELOPMENT TEAM. *SfePy: simple finite elements in Python*. 2025. URL: `https://sfepy.org`. *(Origem: Pré-existente `0A`)*
- **Resumo Técnico Objetivo:** Documentação do solver FEM Python SfePy cogitado inicialmente antes da adoção de FEniCSx/FEniTop.
- **Pontos Fortes / Limitações:** Legado sem uso no código atual (`0` ocorrências no corpo de `tcc_v5.txt`).
- **Comparação Direta com o AlphaBuilder:** Substituído por DOLFINx/FEniTop pela necessidade de paralelismo MPI e multigrid PETSc (`gamg`) em 3D.
- **Onde Citar no TCC:** Remover da bibliografia de `tcc_v5.txt` ou citar exclusivamente caso o Cap. 3 registre a decisão arquitetural SfePy $\to$ FEniCSx.

#### 5.2.2. Repositórios e Materiais Exploratórios do Guia Inicial (`ref_alphabuilder.md`) e `fontes.tsv`

##### `multiplayer2019alphazero`
- **Citação Completa:** *Multiplayer AlphaZero*. `arXiv:1910.13012`, 2019. URL: `https://ar5iv.labs.arxiv.org/html/1910.13012`. *(Origem: Pré-existente `0A` — `ref_alphabuilder.md`)*
- **Resumo Técnico Objetivo:** Preprint explorando extensões do AlphaZero fora do padrão clássico de 2 jogadores em soma zero.
- **Pontos Fortes / Limitações:** Útil para compreender adaptações de PUCT/valor além de jogos tradicionais; autores não listados em `0A`.
- **Comparação Direta com o AlphaBuilder:** Leitura exploratória inicial da etapa de adaptação do AlphaZero para otimização de 1 jogador.
- **Onde Citar no TCC:** Material de apoio histórico; se citado formalmente, verificar autores completos no arXiv.

##### `cogitontnu2019deeptactics`
- **Citação Completa:** COGITONTNU. *Deeptactics-AlphaZero*. GitHub / `arXiv:1911.08265`, 2019. URL: `https://github.com/CogitoNTNU/Deeptactics-AlphaZero`. *(Origem: Pré-existente `0A` — `ref_alphabuilder.md`)*
- **Resumo Técnico Objetivo:** Implementação aberta de AlphaZero aplicado a jogo tático em grade, estudada como referência prática de MCTS + rede neural.
- **Pontos Fortes / Limitações:** Referência prática de código MCTS no início do projeto; fora do domínio de engenharia estrutural.
- **Comparação Direta com o AlphaBuilder:** Inspirou padrões iniciais de implementação do motor MCTS antes da arquitetura `engine.py`/`node.py` com micro-batches PV.
- **Onde Citar no TCC:** Não essencial para citação formal (preferir `silver2018general` e `browne2012survey`).

##### `mdpi_review_statics`
- **Citação Completa:** *Topology Optimization: A Review for Structural Designs Under Statics Problems*. MDPI. URL: `https://www.mdpi.com/1996-1944/17/23/5970`. *(Origem: Pré-existente `0A` — `ref_alphabuilder.md`)*
- **Resumo Técnico Objetivo:** Artigo de revisão sobre otimização topológica aplicada a projetos estruturais sob carregamentos estáticos.
- **Pontos Fortes / Limitações:** Material de revisão listado em `ref_alphabuilder.md`, porém sem autores e ano gravados em `0A`.
- **Comparação Direta com o AlphaBuilder:** Leitura auxiliar de fundamentos estáticos, superada pelas revisões completas fichadas no Eixo 1 (`sigmund2013topology`, `deaton2014survey`).
- **Onde Citar no TCC:** Evitar citar sem recuperar autores/ano da página MDPI, priorizando as referências completas do Eixo 1.

##### `aguilarmadeira_99lines_python`
- **Citação Completa:** *Adaptation of Ole Sigmund's Topology Optimization Code with 99 Lines to Python*. URL: `https://web.tecnico.ulisboa.pt/aguilarmadeira/files/99lines_Python.pdf`. *(Origem: Pré-existente `0A` — `ref_alphabuilder.md`)*
- **Resumo Técnico Objetivo:** Adaptação didática do algoritmo de 99 linhas de Sigmund (2001) para Python.
- **Pontos Fortes / Limitações:** Apoio para prototipagem rápida em Python 2D; substituído por `jia2024fenitop` para produção 3D.
- **Comparação Direta com o AlphaBuilder:** Referência introdutória para o baseline SIMP 2D antes da integração completa do FEniTop.
- **Onde Citar no TCC:** Opcional no Cap. 5.2, preferindo `sigmund200199line` e `jia2024fenitop`.

##### `ferguson_topopt_github`
- **Citação Completa:** `zfergus/topopt`: *A Python Library for Topology Optimization*. GitHub. URL: `https://github.com/zfergus/topopt`. *(Origem: Pré-existente `0A` — `ref_alphabuilder.md`)*
- **Resumo Técnico Objetivo:** Biblioteca Python open-source para otimização topológica 2D/3D em grades regulares.
- **Pontos Fortes / Limitações:** Referência prática de código consultada na concepção do projeto; substituída em produção por FEniTop/FEniCSx.
- **Comparação Direta com o AlphaBuilder:** Ferramenta exploratória prévia antes da adoção de DOLFINx + PETSc.
- **Onde Citar no TCC:** Apenas se houver menção a alternativas de software avaliadas no Cap. 3.

##### `lagerweij_topopt_github`
- **Citação Completa:** `AJJLagerweij/topopt`: *Topology optimization*. GitHub. URL: `https://github.com/AJJLagerweij/topopt`. *(Origem: Pré-existente `0A` — `ref_alphabuilder.md`)*
- **Resumo Técnico Objetivo:** Repositório Python de implementação de otimização topológica consultado no guia inicial.
- **Pontos Fortes / Limitações:** Código didático de referência; não usado no sistema final.
- **Comparação Direta com o AlphaBuilder:** Material exploratório da etapa DIP superado pelo módulo `fenitop/`.
- **Onde Citar no TCC:** Apenas se houver histórico de levantamento de ferramentas no Cap. 3.

##### `patel_vit_examples`
- **Citação Completa:** `ashishpatel26/Vision-Transformer-Keras-Tensorflow-Pytorch-Examples`. GitHub. *(Origem: Pré-existente `0A` — `ref_alphabuilder.md`)*
- **Resumo Técnico Objetivo:** Repositório de exemplos de implementação de Vision Transformers em Keras, TensorFlow e PyTorch.
- **Pontos Fortes / Limitações:** Material informal de estudo de código na fase DIP.
- **Comparação Direta com o AlphaBuilder:** Superado pelo uso direto de PyTorch + MONAI (`SwinUNETR`).
- **Onde Citar no TCC:** Não citar como referência bibliográfica acadêmica.

##### `arxiv250409754_llm_structural`
- **Citação Completa:** *Integrating Large Language Models for Automated Structural Analysis*. `arXiv:2504.09754v1`, 2025. URL: `https://arxiv.org/html/2504.09754v1?hl=pt-BR`. *(Origem: Pré-existente `0A` — `fontes.tsv`)*
- **Resumo Técnico Objetivo:** Explora o uso de Grandes Modelos de Linguagem (LLMs) para automatização de fluxos de análise estrutural.
- **Pontos Fortes / Limitações:** Único registro salvo na planilha `fontes.tsv`; autores não constam na planilha e o tema (LLMs) diverge da arquitetura espacial voxel/MCTS do AlphaBuilder.
- **Comparação Direta com o AlphaBuilder:** Fora do núcleo metodológico do projeto (Swin-UNETR + MCTS + FEM 3D).
- **Onde Citar no TCC:** Não incorporar no corpo principal a menos que seja citado em trabalhos futuros (Cap. 8) sobre interfaces em linguagem natural.

#### 5.2.3. Registros Incompletos de `pesquisa_referencias.md` (Sem Autoria ou Com Título Provisório em `0A`)

> *Alerta de integridade:* Os 9 registros abaixo constam na tabela de `pesquisa_referencias.md` (`0A` §2.4), mas faltam autores ou título bibliográfico exato na fonte. Mantidos em `@misc` apenas para rastreabilidade completa do legado do Drive; **não devem ser citados no TCC sem recuperação prévia dos metadados completos (autores, periódico e DOI)**.

- **`pesquisa2024structural`** — *Structural topology optimization based on deep learning* (2024). Item #1 de `pesquisa_referencias.md` (autores e periódico ausentes em `0A`). Escopo: síntese de OT estrutural por DL. Relação: panorama genérico de DL-TopOpt, coberto com rigor em Eixo 2 por `xiang2025two`, `nobari2024towards` e `nobari2025optimize`. Citação no TCC: substituir pelas referências verificadas do Eixo 2.
- **`pesquisa2022realtime_stress`** — *Real-time stress-based topology optimization via deep learning* (2022). Item #2 de `pesquisa_referencias.md`. Escopo: inclusão de restrições de tensão em tempo real via DL. Relação: o AlphaBuilder avalia *compliance* e deslocamento máximo (`DISPLACEMENT_LIMIT_RAW`), deixando restrições locais de tensão como extensão futura. Citação no TCC: verificar autores antes de citar no Cap. 2.7/Cap. 8.
- **`pesquisa2022accelerated_3d`** — *Accelerated topology optimization design of 3D structures based on deep learning* (2022). Item #3 de `pesquisa_referencias.md`. Escopo: aceleração de OT 3D via DL. Relação: alinhado ao Eixo 2 3D, já coberto com dados completos por `behzadi2021realtime`, `zheng2021generating` e `nobari2024towards`. Citação no TCC: verificar autoria ou substituir por `behzadi2021realtime`/`nobari2024towards`.
- **`pesquisa2022mmc_paunet`** — *Real-time topology optimization based on deep learning for moving morphable components* (2022; PA-U-Net). Item #4 de `pesquisa_referencias.md`. Escopo: predição de componentes móveis (MMC) via U-Net. Relação: contrasta representação MMC (como em `rochefortbeaudoin2025structural`) com grade voxelizada do AlphaBuilder. Citação no TCC: verificar autores ou usar `rochefortbeaudoin2025structural`.
- **`pesquisa2021algorithmically`** — *Algorithmically-consistent deep learning frameworks for structural topology optimization* (2021; 3D SIMP-validated). Item #5 de `pesquisa_referencias.md`. Escopo: consistência algorítmica com validação SIMP 3D. Relação: alinhado ao objetivo de consistência física do AlphaBuilder com FEniTop. Citação no TCC: recuperar autores se necessário para o Cap. 2.7.
- **`pesquisa2021element_removal`** — *A deep learning approach for efficient topology optimization based on the element removal strategy* (2021; CNN + U-Net + RNN). Item #7 de `pesquisa_referencias.md`. Escopo: remoção sequencial recorrente de elementos via CNN/U-Net/RNN. Relação: paralelo sequencial destrutivo próximo a `brown2022deep`, mas sem política aditiva nem MCTS. Citação no TCC: verificar autores ou usar `brown2022deep`.
- **`pesquisa2020drl_compliance`** — *Topology optimization based on deep representation learning (DRL) for compliance and stress-constrained design* (2020). Item #8 de `pesquisa_referencias.md`. Escopo: aprendizado de representação para *compliance* e tensão. Relação: pré-cursor conceitual de compressão de estado, comparado ao encoder Swin/CNN do AlphaBuilder. Citação no TCC: verificar autores caso seja usado no Cap. 2.7.
- **`patel2012early`** — PATEL, N.; CHOI, S. *Early work on ML in TO* (2012). Item #18 de `pesquisa_referencias.md` (título descritivo genérico em `0A`, claramente não corresponde ao título original publicado). Relação: marco histórico inicial de ML em OT. Citação no TCC: **proibido citar sem verificar o título original do artigo**.
- **`nie2021complex`** — NIE, Z. et al. *Topology optimization with deep learning for complex constraints* (2021). Item #19 de `pesquisa_referencias.md`. Relação: provável descrição parafraseada de `nie2021topologygan` (*TopologyGAN*, ASME JMD 2021) ou trabalho paralelo do mesmo grupo. Citação no TCC: preferir a entrada verificada `nie2021topologygan`.

---

### 5.3. Inventário de Lacunas Bibliográficas e Fontes Terciárias Fora do `.bib`

Para transparência absoluta e cumprimento estrito da regra de integridade, registram-se abaixo os itens mencionados nos levantamentos `0A`–`0E` que **não puderam gerar entrada BibTeX completa** por ausência de título ou natureza não acadêmica:

1. **Menção Tecnológica Faltante no Corpo de `tcc_v5.txt`:**
   - **PyTorch** (citado em L235 e L317 de `tcc_v5.txt`): nenhum dos relatórios `0A`–`0E` trouxe a referência bibliográfica formal (autores/ano/URL oficial da biblioteca). Deve ser adicionada pelo autor ao fechar a lista de referências de implementação.
2. **Citações Órfãs / Fragmentos Apenas Autor-Ano em `0A` (sem título, periódico ou identificador em `0A`):**
   - `Liu et al. (2025)` — citação órfã em `tcc_draft_01.md` (L20) sobre limitações de convoluções em dependências de longo alcance na mecânica do contínuo.
   - `Guo et al. (2018)` — redução de dimensionalidade em condução de calor (`pesquisa_referencias.md` L33).
   - `Chandrasekhar e Suresh (2021c)` — FEA na função de perda / explicabilidade física (`pesquisa_referencias.md` L34, L50).
   - `Jiang e Fan (2019a, 2019b)` — aplicações em nanofotônica (`pesquisa_referencias.md` L38).
   - `Rade et al. (2020)` — custo computacional em OT 3D com 31.093 nós e 154.677 elementos (`pesquisa_referencias.md` L48).
   - `Bi et al. (2020)` — geração de datasets no supercomputador ORNL OLCF Summit (`pesquisa_referencias.md` L49).
   - `Sharpe e Seepersad (2019)` — restrições de manufatura em ML-based TO (`pesquisa_referencias.md` L51).
   - `Qiu et al. (2021a)` — ganho de $810\times$ em malha $20\times60$ (`pesquisa_referencias.md` L53).
   - `Zhuang et al. (2021b)` — interpretabilidade física em condução térmica não linear (`pesquisa_referencias.md` L59).
   - `Kim et al. (2022)` — manufatura aditiva (`pesquisa_referencias.md` L61).
3. **Referências Secundárias de `0E` Citadas Apenas com Autor, Ano e DOI (sem título impresso em `0E`):**
   - **Svanberg (1987)** — Método das Assíntotas Móveis (MMA), DOI `10.1002/nme.1620240207` (citado em `0E` no contexto de `jia2024fenitop`).
   - **Wang, Wang & Guo (2003)** — *Level-Set* contemporâneo em *CMAME*, DOI `10.1016/S0045-7825(02)00559-5` (citado em `0E` junto a `allaire2004structural`).
   - **Zuo & Xie (2015)** — Código Python 3D compacto para BESO em *Advances in Engineering Software*, DOI `10.1016/j.advengsoft.2015.02.006` (citado em `0E` junto a `huang2010evolutionary`).
4. **Materiais Terciários / Fóruns / Documentos Internos de `ref_alphabuilder.md` (não recomendados para `.bib` acadêmico):**
   - Verbetes da Wikipedia (*Monte Carlo tree search*, *Vision transformer*), guia do portal *Built In* sobre MCTS, documentação geral do TensorFlow/Keras, duas threads do Reddit `r/fea` (SfePy vs. FEniCS e FEniCS vs. códigos comerciais) e o documento interno **Documento de Início de Projeto (DIP)**, Seções 1–7.
