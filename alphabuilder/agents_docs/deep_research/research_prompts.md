# Prompts de Pesquisa Avançada para Otimização Topológica (AlphaBuilder)

Este documento contém prompts detalhados projetados para serem submetidos a assistentes de pesquisa de IA com acesso à web (como GPT-4 com Browsing, Perplexity Pro, ou Gemini Ultra) para garantir o levantamento de referências reais e verificáveis.

---

## Prompt 1: Vision Transformers e Deep Learning "Iteration-Free" em Mecânica
**Objetivo:** Mapear o estado da arte no uso de Transformers para prever campos físicos ou topologias finais sem iterações de solver.

**Prompt:**
> "Atue como um Especialista em Mecânica Computacional e Deep Learning. Realize uma busca profunda na literatura científica (Google Scholar, ArXiv, ScienceDirect, IEEE Xplore) focada no período **2021-2025**.
>
> **Tópico Central:** Aplicação de **Vision Transformers (ViT)** ou arquiteturas de **Self-Attention** para Otimização Topológica (Topology Optimization) ou previsão de campos de tensão (Stress Fields) em problemas de elasticidade linear.
>
> **Requisitos Estritos:**
> 1.  **Zero Alucinação:** Liste apenas papers que você pode verificar a existência (forneça URL do PDF ou DOI).
> 2.  **Foco em "Iteration-Free":** Busque trabalhos que tentam substituir o solver de Elementos Finitos (FEM) tradicional por uma inferência direta de rede neural.
> 3.  **Comparação:** Para cada paper, identifique se eles usam CNNs (U-Net) ou Transformers e qual a vantagem alegada dos Transformers (ex: dependências globais de longo alcance).
>
> **Saída Esperada:** Uma tabela comparativa com: Título, Autores, Ano, Arquitetura (ViT/CNN), Link Verificável e um breve resumo da contribuição."

---

## Prompt 2: Aprendizado por Reforço (RL) para Design Estrutural Construtivo
**Objetivo:** Encontrar trabalhos que usam RL não para "remover" material (como BESO), mas para "construir" ou navegar no espaço de design, garantindo conectividade.

**Prompt:**
> "Atue como um Pesquisador Sênior em IA Aplicada à Engenharia. Investigue a literatura recente (**2020-2025**) sobre **Reinforcement Learning (RL) para Otimização Topológica**.
>
> **Filtro de Pesquisa:**
> *   Estou procurando especificamente por abordagens **Construtivas** (onde o agente adiciona material) ou abordagens que lidam explicitamente com **Restrições de Conectividade** (Connectivity Constraints).
> *   Ignore trabalhos genéricos que apenas usam uma CNN para 'refinar' uma imagem SIMP. Busque trabalhos onde o RL é o 'motor de busca' (Search Strategy).
> *   Palavras-chave: 'Constructive Topology Optimization', 'RL for Truss Optimization', 'Sequential Decision Process in Structural Design', 'Graph Neural Networks for Topology'.
>
> **Saída Esperada:**
> 1.  Lista de 3-5 papers mais relevantes.
> 2.  Para cada um, explique como eles lidam com a recompensa esparsa (o fato de que a estrutura só é útil quando conecta a carga ao suporte).
> 3.  Links diretos para os papers."

---

## Prompt 3: Benchmarking Rigoroso: AI vs. Métodos Clássicos (SIMP/BESO)
**Objetivo:** Estabelecer como a comunidade científica está validando métodos de IA. Não basta ser rápido, tem que ser preciso.

**Prompt:**
> "Realize uma análise crítica sobre **Metodologias de Benchmarking** em Otimização Topológica baseada em Neural Networks (**2022-2025**).
>
> **Perguntas a Responder:**
> 1.  Quais são os datasets padrão usados? (Ex: MNIST of Structures, TopOpt10k?).
> 2.  Como os papers recentes comparam a solução da IA com o 'Ground Truth' do SIMP? Eles usam métricas como 'Compliance Error', 'Volume Fraction Violation' ou 'Hausdorff Distance'?
> 3.  Encontre papers de revisão (Review Papers) publicados em 2023 ou 2024 que façam uma análise quantitativa dessas comparações.
>
> **Entrega:** Um resumo das melhores práticas para validar um novo modelo de IA (AlphaBuilder) contra o método SIMP clássico, com citações reais que fundamentem essas práticas."
