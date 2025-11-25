# Otimização Topológica Construtiva via Aprendizado por Reforço Híbrido e Vision Transformers

**Resumo:** Este trabalho investiga uma metodologia alternativa para Otimização Topológica (OT) que integra Aprendizado por Reforço (RL) baseado em busca (MCTS) com modelos de atenção visual (Vision Transformers). Diferente das abordagens baseadas em densidade (SIMP) ou redes convolucionais (CNNs), que tratam o design predominantemente como um problema de imagem, nossa abordagem explora a OT como um processo de decisão sequencial construtivo. O objetivo é avaliar se essa formulação pode mitigar desafios de conectividade estrutural e manufaturabilidade frequentemente observados em métodos anteriores.

---

## 1. Introdução

A Otimização Topológica (OT) consolidou-se como uma ferramenta relevante na engenharia, auxiliando na concepção de estruturas eficientes. O método padrão, *Solid Isotropic Material with Penalization* (SIMP), formula o problema como a distribuição de densidade material. Embora amplamente utilizado, o SIMP e seus derivados podem apresentar custos computacionais elevados para malhas de alta resolução (SIGMUND; MAUTE, 2013).

Recentemente, a integração de Inteligência Artificial (IA) tem sido explorada para reduzir esse custo. Tentativas iniciais com Redes Neurais Convolucionais (CNNs), como a U-Net, buscaram aprender o mapeamento entre condições de contorno e topologia. No entanto, a literatura aponta desafios nessa abordagem "baseada em imagem", como dificuldades em assegurar a conectividade estrutural e a precisão física em fronteiras complexas, além de limitações na generalização para domínios fora do conjunto de treinamento (WANG et al., 2024).

Este trabalho apresenta o **AlphaBuilder**, um framework experimental que propõe formular a OT como um **Processo de Decisão de Markov (MDP) Construtivo**. Inspirado por avanços em IA para jogos e geometria, o estudo avalia um sistema híbrido onde um agente de Aprendizado por Reforço (RL), auxiliado por uma busca em árvore de Monte Carlo (MCTS), constrói a estrutura sequencialmente. Para lidar com a esparsidade de recompensa, investiga-se o uso de um *Vision Transformer* (ViT) como um estimador de função de valor, visando prever o desempenho de estruturas parciais.

## 2. Revisão Bibliográfica e Estado da Arte

A evolução da OT assistida por IA pode ser analisada em três fases, conforme a literatura recente (2021-2025).

### 2.1. Limitações das CNNs e o Paradigma "Image-to-Image"
A primeira fase focou em arquiteturas Encoder-Decoder. Embora rápidas, estudos sugerem que operações de convolução podem ter limitações em capturar dependências de longo alcance essenciais na mecânica do contínuo (LIU et al., 2025). Além disso, a validação desses modelos requer cuidado para evitar o "Inverse Crime", garantindo que a avaliação física seja independente da geração de dados (TABARRAEI; BHUIYAN, 2025).

### 2.2. A Ascensão dos Transformers na Mecânica
A partir de 2023, observa-se um interesse crescente em *Vision Transformers* (ViTs). O mecanismo de *Self-Attention* oferece uma forma alternativa de modelar a conectividade global.
*   **Lutheran et al. (2025)** indicaram que ViTs podem gerar topologias com boa fidelidade física.
*   **Wang et al. (2025)** propuseram o *Micrometer*, alcançando resultados promissores na previsão de campos de tensão em materiais heterogêneos.
*   **Nagayama e Sasaki (2025)** aplicaram *Swin Transformers* para otimização de motores, relatando melhorias na previsão de torque comparado a baselines de CNN.

### 2.3. Otimização Topológica como Processo de Decisão Sequencial
Paralelamente, o paradigma de "construção sequencial" tem ganhado atenção.
*   **Ororbia e Warn (2023)** exploraram o *Hierarchical Deep Reinforcement Learning* (HDRL) com gramáticas de design.
*   **Li et al. (2023)** desenvolveram o framework *AutoTruss*, combinando MCTS para busca topológica com RL para refinamento.
*   **Padhy et al. (2024)** introduziram o *TreeTOp*, utilizando Árvores CSG para garantir volumes fechados.

## 3. Metodologia: O Framework AlphaBuilder

O AlphaBuilder propõe uma arquitetura que busca integrar essas tecnologias para endereçar lacunas identificadas.

### 3.1. Representação de Estado via Grafos Dinâmicos
O AlphaBuilder adota uma abordagem aditiva onde o estado $s_t$ é representado por um grafo $G_t = (V_t, E_t)$. Esta representação visa explorar a invariância de permutação e escala, buscando superar limitações associadas a malhas regulares fixas.

### 3.2. Busca Guiada por MCTS com Restrições Rígidas
Utilizamos *Monte Carlo Tree Search* (MCTS) para a navegação no espaço de design. Uma característica central da proposta é a implementação de **Hard Constraints** na expansão da árvore. Inspirado em **Li et al. (2023)**, o sistema verifica a estabilidade cinemática antes de adicionar nós, com o objetivo de concentrar o esforço computacional em designs viáveis.

### 3.3. O Oráculo Neural: ViT como Função de Valor
Para reduzir o custo computacional do MCTS, substituímos a simulação FEM em cada passo por um *Vision Transformer* treinado para estimar a *compliance* final. Baseado em **Lutheran et al. (2025)**, o ViT atua como uma heurística aprendida para guiar a busca.

## 4. Defesa da Abordagem e Contribuições Esperadas

Esta pesquisa busca contribuir para o campo explorando os seguintes potenciais benefícios:

1.  **Validade Topológica:** A abordagem construtiva tem o potencial de facilitar a geração de grafos conexos e manufaturáveis, mitigando problemas de desconexão observados em métodos generativos baseados em pixel (**BEHZADI, 2023**).
2.  **Eficiência de Amostra e Generalização:** Investiga-se se o planejamento via MCTS e a representação em grafo permitem melhor generalização para domínios maiores do que os vistos no treinamento.
3.  **Física Global via Atenção:** Avalia-se se a incorporação do ViT permite que a heurística de busca considere o fluxo de carga global de forma mais eficaz que CNNs tradicionais (**WANG et al., 2025**).

Este trabalho visa, portanto, investigar a viabilidade dessa abordagem híbrida e propor benchmarks rigorosos baseados em física para a validação de métodos de IA na engenharia.

---

## Referências Bibliográficas

BEHZADI, M. Taming Connectedness in Machine-Learning-based Topology Optimization with Connectivity Graphs. **Computer-Aided Design**, v. 160, 2023.

HAYASHI, K.; OHSAKI, M. Reinforcement learning and graph embedding for binary truss topology optimization under stress and displacement constraints. **Frontiers in Built Environment**, v. 6, p. 59, 2021.

LI, Y. et al. Automatic Truss Design with Reinforcement Learning. **Proceedings of the IJCAI**, 2023.

LUTHERAN, A. et al. Transformer-based Topology Optimization. **arXiv preprint arXiv:2509.05800**, 2025.

NAGAYAMA, T.; SASAKI, H. Predicting torque characteristics of synchronous reluctance motors using Swin Transformer. **COMPEL-The international journal for computation and mathematics in electrical and electronic engineering**, 2025.

ORORBIA, A.; WARN, G. Discrete Structural Design Synthesis: A Hierarchical-Inspired Deep Reinforcement Learning Approach. **Journal of Mechanical Design**, v. 146, n. 9, 2023.

PADHY, S. et al. TreeTOp: Topology Optimization using Constructive Solid Geometry Trees. **arXiv preprint arXiv:2409.02300**, 2024.

SIGMUND, O.; MAUTE, K. Topology optimization approaches. **Structural and Multidisciplinary Optimization**, v. 48, p. 1031-1055, 2013.

TABARRAEI, A.; BHUIYAN, M. Graph Neural Network-Based Topology Optimization for Self-Supporting Structures in Additive Manufacturing. **ResearchGate**, 2025.

WANG, Y. et al. Micrometer: Micromechanics Transformer for Predicting Mechanical Responses of Heterogeneous Materials. **arXiv preprint arXiv:2410.05281**, 2024.

WANG, Z. et al. CViT: Continuous Vision Transformer for Operator Learning. **arXiv preprint arXiv:2405.13998**, 2025.
