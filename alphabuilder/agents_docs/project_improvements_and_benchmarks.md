# Melhorias do Projeto e Novos Benchmarks (Baseado em Deep Research 2021-2025)

Este documento sintetiza as descobertas da pesquisa profunda e propõe um roteiro de melhorias técnicas e protocolos de validação rigorosos para o projeto AlphaBuilder.

## 1. Síntese das Descobertas Críticas

### 1.1. O "Cisma" Arquitetural
A literatura (2021-2025) confirma que **Vision Transformers (ViTs)** e **Graph Neural Networks (GNNs)** estão substituindo CNNs (U-Nets) em tarefas de mecânica.
*   **Por que?** CNNs sofrem de "viés local". Elas lutam para entender que uma carga na ponta de uma viga afeta a tensão no suporte oposto (dependência de longo alcance). ViTs, com atenção global, capturam isso nativamente.
*   **Implicação para AlphaBuilder:** Nossa aposta em ViT para o "Oráculo" (Função de Valor) está correta e alinhada com o estado da arte (ex: *Micrometer*, *CViT*).

### 1.2. A Falácia dos Benchmarks Visuais
A maioria dos papers falha por usar métricas de visão computacional (IoU, Acurácia de Pixel) para problemas de engenharia.
*   **O Problema:** Uma estrutura pode ter 99% de IoU com o ground truth e ainda assim colapsar (infinita compliance) se uma conexão crítica estiver faltando.
*   **Implicação:** Devemos adotar métricas baseadas em física ($\Delta C$, Taxa de Viabilidade) e evitar o "Inverse Crime" (validar com o mesmo solver simplificado que gerou os dados).

### 1.3. A Superioridade da Abordagem Construtiva
Métodos baseados em densidade (SIMP + CNN) geram imagens que precisam de pós-processamento e frequentemente resultam em ilhas desconectadas.
*   **A Vantagem do AlphaBuilder:** Ao "construir" a estrutura sequencialmente (adicionando nós/barras ou voxels conectados), garantimos **conectividade por design**. Isso resolve o maior gargalo identificado na literatura de Generative Design.

---

## 2. Melhorias Propostas para o AlphaBuilder

### 2.1. Refinamento da Arquitetura Neural (O "Oráculo")
*   **Adotar Swin Transformer ou TransUNet:** Em vez de um ViT puro (que pode ser pesado), usar uma arquitetura híbrida ou hierárquica (Swin) para equilibrar a captura de detalhes locais (concentração de tensão) com a estrutura global.
*   **Input Multimodal:** Não alimentar apenas a "imagem" da estrutura. Incorporar as Condições de Contorno (Forças, Suportes) como *tokens* especiais ou canais de atenção dedicados, como sugerido em *Lutheran et al. (2025)*.

### 2.2. Estratégia de Busca Híbrida (MCTS + Hard Constraints)
*   **Poda de Árvore (Pruning):** No MCTS, implementar verificações rígidas de validade. Se um ramo leva a uma estrutura instável (matriz singular), podar imediatamente. Isso evita desperdiçar computação em designs inviáveis (problema da recompensa esparsa).
*   **Graph Embeddings:** Para o agente de RL, considerar migrar da representação visual (imagem da viga) para uma representação de grafo (nós e barras). Isso permitiria invariância de escala e rotação.

### 2.3. Prevenção do "Inverse Crime"
*   **Validação Cruzada:** O ambiente de treino pode usar nosso solver FEM rápido (Python/JAX). Mas a validação final (Benchmark) deve ser feita exportando a malha e rodando em um solver externo robusto (ex: CalculiX ou uma implementação de alta ordem em FEniCSx) para garantir que a IA aprendeu física real, não os bugs do nosso solver.

---

## 3. Novos Benchmarks de Indústria (Protocolo de Validação)

Para publicar ou defender o AlphaBuilder como uma ferramenta séria, devemos seguir este protocolo de 3 estágios:

### Estágio 1: Métricas de Engenharia (Hard Metrics)
Em vez de IoU, reportaremos:
1.  **Erro Relativo de Conformidade ($\Delta C$):** $|C_{pred} - C_{FEM}| / C_{FEM}$. Meta: $< 5\%$.
2.  **Taxa de Viabilidade (Feasibility Rate):** % de designs gerados que formam um caminho de carga válido sem pós-processamento. Meta: $100\%$ (devido à abordagem construtiva).
3.  **Volume de Suporte (Manufatura):** Volume de material de suporte necessário para impressão 3D (estimado pelo ângulo de overhang).

### Estágio 2: Generalização (OOD Testing)
Não testar apenas na viga em balanço padrão.
1.  **Teste de Resolução Cruzada:** Treinar em malha 32x16, testar em 64x32. O modelo deve gerar estruturas coerentes, não pixeladas.
2.  **Geometrias Complexas:** Testar em domínios não-convexos (ex: L-shape, domínios com furos internos).

### Estágio 3: Comparação com Baselines
Comparar o AlphaBuilder contra:
1.  **SIMP Clássico:** (Já implementado). Comparar Compliance final e Tempo de Execução.
2.  **U-Net Baseline:** Treinar uma U-Net simples no mesmo dataset para mostrar que ela falha em conectividade onde o AlphaBuilder (RL) tem sucesso.

## 4. Próximos Passos Imediatos
1.  Implementar o cálculo de **$\Delta C$** no pipeline de validação.
2.  Criar um dataset de teste "OOD" (Out-of-Distribution) com geometrias L-shape.
3.  Ajustar o MCTS para usar "Hard Constraints" na expansão.
