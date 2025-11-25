# **Paradigmas Construtivos e Processos de Decisão Sequencial em Otimização Topológica via Aprendizado por Reforço: Uma Análise Crítica (2020-2025)**

## **1\. Introdução: A Reestruturação do Espaço de Design**

A disciplina de Otimização Topológica (TO) encontra-se em um momento de inflexão metodológica sem precedentes. Historicamente, a busca pela distribuição ótima de material dentro de um domínio de design foi dominada por abordagens baseadas em gradiente, notadamente o método *Solid Isotropic Material with Penalization* (SIMP), e heurísticas evolucionárias como o *Bi-directional Evolutionary Structural Optimization* (BESO). Estes métodos, embora fundamentais para o desenvolvimento da engenharia computacional moderna, operam sob um paradigma que pode ser classificado como "escultural" ou subtrativo dentro de um *continuum* discretizado. Neles, o algoritmo manipula densidades de elementos finitos em uma grade fixa (mesh), onde a conectividade estrutural e a manufaturabilidade são frequentemente tratadas como restrições *a posteriori* ou penalidades suaves, resultando não raro em artefatos numéricos como o *checkerboarding* ou ilhas de material desconectadas que exigem pós-processamento intensivo.  
No entanto, a literatura científica produzida entre 2020 e 2025 evidencia a emergência de uma nova fronteira: a reformulação da otimização topológica como um **Processo de Decisão Sequencial (Sequential Decision Process \- SDP)**. Impulsionada pelos avanços no Aprendizado por Reforço Profundo (Deep Reinforcement Learning \- DRL), esta nova classe de algoritmos abandona a visão do design como uma "imagem" de densidades a ser refinada. Em vez disso, adota a perspectiva de um agente autônomo — um "arquiteto digital" — que navega por um espaço de design combinatorial, construindo a estrutura passo a passo através de ações discretas.  
Esta transição de paradigmas — da otimização de parâmetros contínuos para a tomada de decisão sequencial discreta — é motivada pela necessidade de resolver problemas intrinsecamente discretos que desafiam a diferenciabilidade dos métodos tradicionais. A otimização de treliças (*truss layouts*), o design de estruturas modulares para manufatura aditiva e a configuração de redes de infraestrutura exigem garantias estritas de conectividade que são difíceis de impor em campos de densidade contínua. Neste contexto, o RL não atua meramente como um acelerador de convergência ou um refinador de imagens geradas por CNNs, mas assume o papel de motor de busca primário, explorando a topologia do grafo estrutural.  
A presente análise investiga exaustivamente este corpo literário, com foco estrito em abordagens construtivas e na gestão de restrições de conectividade. Exploraremos como a integração de *Graph Neural Networks* (GNNs) permitiu que agentes de IA "percebessem" a topologia de forma invariante à permutação , como arquiteturas hierárquicas e *Monte Carlo Tree Search* (MCTS) mitigaram o problema da recompensa esparsa , e como a Geometria Sólida Construtiva (CSG) foi revitalizada como um espaço de representação para RL.

## **2\. Fundamentos Teóricos: O Design Estrutural como Processo de Decisão de Markov**

Para compreender a profundidade da inovação proposta pelos métodos construtivos baseados em RL, é imperativo dissecar a tradução dos princípios da mecânica estrutural para a linguagem formal dos Processos de Decisão de Markov (MDPs). Esta reformulação não é meramente sintática; ela altera fundamentalmente a topologia do espaço de busca e a natureza das soluções acessíveis.

### **2.1. A Formalização do MDP em Engenharia Estrutural**

A literatura convergente modela a síntese de estruturas discretas através da tupla clássica (S, A, P, R, \\gamma), mas com especificidades de domínio que merecem detalhamento rigoroso.

#### **2.1.1. O Espaço de Estados (S): Do Pixel ao Grafo**

Em abordagens tradicionais de *Deep Learning* para TO (como as baseadas em U-Nets), o estado s é frequentemente uma representação tensorial euclidiana (uma imagem 2D ou voxel 3D) do domínio de design. Esta representação, no entanto, carrega um viés indutivo inadequado para estruturas esparsas como treliças ou *frames*.  
Nas abordagens construtivas recentes, o estado s\_t é representado predominantemente como um **Grafo Atribuído** G\_t \= (V\_t, E\_t).

* **Nós (V\_t):** Representam as juntas da estrutura. O vetor de características de cada nó x\_v codifica informações críticas como coordenadas espaciais (x, y, z), condições de contorno (indicadores de suporte fixo ou móvel) e vetores de carga externa aplicada.  
* **Arestas (E\_t):** Representam os elementos estruturais (barras, vigas). Seus atributos x\_e incluem propriedades materiais (módulo de Young), área da seção transversal e, em alguns casos avançados, o estado de tensão atual ou a energia de deformação derivada de uma análise de elementos finitos (FEA) intermediária.

A vantagem crítica desta representação baseada em grafos é a **invariância de permutação**. Uma estrutura física mantém suas propriedades mecânicas independentemente da ordem em que seus nós são indexados na matriz de rigidez global. Redes Neurais Convolucionais (CNNs) são sensíveis a translações e rotações no grid de pixels e não capturam naturalmente a topologia não-euclidiana de uma treliça irregular. Em contraste, as GNNs operam através de troca de mensagens (*message passing*) entre vizinhos topológicos, alinhando-se perfeitamente à física da transmissão de forças em sistemas discretos.

#### **2.1.2. O Espaço de Ações (A): A Gramática da Construção**

O núcleo da abordagem construtiva reside na definição do espaço de ações. Diferente do método SIMP, onde a "ação" é a atualização contínua de valores de densidade, o RL construtivo opera através de modificações discretas e topológicas. A literatura identifica classes principais de ações:

1. **Ações Topológicas Aditivas:** Envolvem a criação de novos elementos. Exemplos incluem conectar dois nós existentes com uma barra (u, v \\rightarrow E \\cup \\{(u,v)\\}) ou introduzir um novo nó no domínio e conectá-lo à estrutura existente. Esta abordagem é análoga ao crescimento biológico ou à construção física.  
2. **Ações Topológicas Subtrativas (Pruning):** Partindo de uma *Ground Structure* (uma malha densamente conectada), o agente decide sequencialmente quais elementos remover. A decisão é binária e irreversível no contexto de um passo, exigindo que o agente avalie a criticidade de cada membro para a integridade global.  
3. **Ações Paramétricas:** Envolvem a modificação de atributos de elementos existentes, como aumentar o diâmetro de uma barra ou alterar a posição de um nó (*shape optimization*).  
4. **Operações Booleanas (CSG):** Em contextos volumétricos, a ação pode ser a seleção de uma primitiva geométrica e uma operação lógica (união, interseção, subtração) para compor a forma final, como explorado no framework TreeTOp.

A Tabela 1 resume as diferenças fundamentais entre a formulação de ações em métodos baseados em densidade versus métodos construtivos.

| Característica | Métodos de Densidade (SIMP/CNN) | Métodos Construtivos (RL/Grafo) |
| :---- | :---- | :---- |
| **Natureza da Variável** | Contínua (0.0 a 1.0) | Discreta (Adicionar/Remover/Conectar) |
| **Granularidade** | Elemento Finito / Voxel | Componente Estrutural / Primitiva |
| **Controle de Topologia** | Implícito (emergente da densidade) | Explícito (manipulação direta do grafo) |
| **Espaço de Busca** | Dimensionalidade fixa (Grade N \\times M) | Dimensionalidade variável (Grafo dinâmico) |
| **Garantia de Conectividade** | Baixa (requer filtros/pós-processamento) | Alta (intrínseca às regras de ação) |

*Tabela 1: Comparativo entre formulações de ação em Otimização Topológica.*

#### **2.1.3. A Função de Transição (P) e o Ambiente Determinístico**

No contexto da engenharia estrutural clássica, a função de transição P(s\_{t+1} | s\_t, a\_t) é predominantemente determinística. Se um agente adiciona uma viga de aço entre dois pontos, as propriedades físicas do novo estado são governadas pelas leis da mecânica e são totalmente previsíveis (excluindo-se incertezas estocásticas de material ou carga, que são tratadas em subcampos específicos de TO robusta). No entanto, a complexidade reside no fato de que s\_{t+1} pode não ser um estado válido. A adição de um nó sem suporte ou a remoção de um elemento crítico pode tornar a matriz de rigidez singular, impedindo a avaliação da recompensa. Este aspecto binário da validade do estado é uma das principais fontes de dificuldade no treinamento de agentes de RL neste domínio.

### **2.2. A Engenharia da Recompensa (R) e o Dilema da Esparsidade**

A função de recompensa R(s, a) é o mecanismo de sinalização que orienta o aprendizado da política \\pi. Em TO, o objetivo é geralmente minimizar uma função composta, tipicamente envolvendo *compliance* (o inverso da rigidez), volume (peso) e restrições de tensão ou deslocamento. Uma formulação comum encontrada na literatura recente é:  
Onde C é a *compliance*, V é o volume, e P\_{restrição} é uma penalidade severa aplicada se o estado s\_{t+1} violar limites de tensão ou, crucialmente, se a estrutura se tornar cinematicamente instável.

#### **O Fenômeno da Recompensa Esparsa (Sparse Reward)**

Um dos temas mais recorrentes e desafiadores identificados na pesquisa entre 2020 e 2025 é a **esparsidade da recompensa** em abordagens construtivas. Diferente de jogos de videogame onde pontos são acumulados continuamente, na construção de uma treliça a partir do zero, o agente pode precisar executar dezenas de ações (colocar nós, conectar barras) antes de formar uma estrutura que seja estaticamente determinada e capaz de suportar carga.

* **O "Vale da Morte":** Durante a fase inicial de construção, qualquer tentativa de análise via Elementos Finitos (FEA) falha porque a matriz de rigidez global é singular. Sem FEA, não há cálculo de *compliance* ou tensão. Consequentemente, a recompensa é zero ou um valor de falha padrão.  
* **Implicação:** O agente navega "às cegas" por longos horizontes de tempo. A probabilidade de um agente com política aleatória (fase de exploração inicial) conectar fortuitamente uma sequência de barras que resulte em uma estrutura estável é infinitesimalmente pequena à medida que o número de nós aumenta. Isso torna algoritmos de RL padrão (como PPO ou DQN puros) ineficazes sem mecanismos auxiliares de guia, como será discutido nas seções de estudos de caso.

## **3\. Arquiteturas de Grafos e Representação de Estado**

A primazia das Redes Neurais em Grafos (GNNs) sobre as arquiteturas convolucionais convencionais é um dos achados mais robustos da revisão bibliográfica. A capacidade das GNNs de processar dados não-euclidianos as torna a escolha natural para a representação de estados em TO construtiva.

### **3.1. O Mecanismo de Message Passing em Estruturas**

A operação fundamental em GNNs, o *Message Passing*, possui uma analogia física direta com o comportamento estrutural, o que explica seu sucesso em prever campos de mecânica. Em uma GNN, a atualização do estado latente h\_v de um nó v na camada k é dada por:  
$$ h\_v^{(k)} \= \\sigma \\left( W^{(k)} \\cdot \\text{AGG} \\left( { h\_u^{(k-1)} : u \\in \\mathcal{N}(v) }, h\_v^{(k-1)} \\right) \\right) $$  
Onde \\mathcal{N}(v) são os vizinhos de v.

* **Interpretação Física:** Esta propagação de informações através das arestas da rede neural mimetiza a propagação de forças e deslocamentos através dos membros da treliça física. Em , demonstra-se que uma GNN pode aproximar a solução das equações de equilíbrio elástico com alta precisão, permitindo que o agente de RL tenha uma estimativa rápida da performance estrutural sem rodar uma simulação FEA completa a cada passo, acelerando drasticamente o treinamento.

### **3.2. Graph Embeddings para Generalização**

Trabalhos como o de Hayashi & Ohsaki utilizam *Graph Embeddings* para resolver o problema da dimensionalidade variável. Em uma abordagem baseada em imagem, treinar um agente para um domínio 32 \\times 32 não o capacita para um domínio 64 \\times 64\. Com embeddings de grafo, o agente aprende representações locais. Ele aprende, por exemplo, que "um nó com alta carga e apenas uma conexão é uma situação crítica". Este conhecimento é transferível. Um agente treinado em pequenas treliças pode ser aplicado a estruturas maiores porque opera sobre as características locais da topologia, e não sobre coordenadas globais absolutas. Esta capacidade de generalização é fundamental para a escalabilidade industrial dos métodos de RL construtivo.

## **4\. Estudo de Caso I: Síntese Estrutural Discreta e RL Hierárquico**

O trabalho desenvolvido por Ororbia e Warn entre 2022 e 2024 representa um marco na formulação rigorosa da síntese estrutural como um problema de aprendizado sequencial. Este conjunto de pesquisas ataca diretamente a complexidade do espaço de ações misto (topológico e paramétrico).

### **4.1. Hierarchical Deep Reinforcement Learning (HDRL)**

Em problemas de design realista, o engenheiro não decide apenas "onde colocar uma barra", mas também "qual o tamanho dessa barra". Isso cria um espaço de ação híbrido e combinatorialmente explosivo. Um agente DQN plano ("flat") luta para aprender correlações eficazes quando o espaço de ação tem milhares de dimensões discretas.  
A solução proposta em é uma arquitetura **Hierárquica Inspirada (HDRL)**.

* **O Controlador (High-Level Policy):** Uma rede neural superior que observa o estado global e toma uma decisão abstrata sobre o *tipo* de ação a ser realizada. Por exemplo: "Devemos focar em alterar a topologia agora ou refinar os diâmetros das barras existentes?".  
* **Os Sub-Agentes (Low-Level Policies):** Redes especializadas que executam a diretiva do controlador.  
  * *Agente Topológico:* Decide onde adicionar ou remover elementos.  
  * *Agente Paramétrico:* Decide qual perfil metálico (seção transversal) atribuir a um elemento específico.

Esta decomposição reduz a complexidade da tarefa de aprendizado. O controlador aprende a estratégia de design (o "fluxo" do projeto), enquanto os sub-agentes aprendem a tática (a execução técnica).

### **4.2. Gramáticas de Design como Restrições de Conectividade**

Uma inovação crucial deste trabalho é o uso de **Gramáticas de Design** para definir as ações permitidas. Em vez de permitir que o agente coloque barras arbitrariamente (o que levaria à esparsidade de recompensa discutida anteriormente), as ações são definidas como regras de produção válidas.

* Exemplo de Regra: "Adicionar um triângulo conectando dois nós existentes e um novo nó". Ao restringir o agente a usar apenas regras que tendem a preservar a estabilidade triangular (estruturas isostáticas), a garantia de conectividade torna-se implícita. O agente navega apenas pelo subespaço de designs "construíveis", evitando o desperdício computacional de explorar estruturas instáveis.

## **5\. Estudo de Caso II: A Abordagem Híbrida AutoTruss (MCTS \+ RL)**

Enquanto Ororbia & Warn focam na hierarquia de ações, o framework **AutoTruss**, proposto por Li et al. , propõe uma hibridização de algoritmos para resolver o problema da "busca de agulha no palheiro" em espaços de design válidos.

### **5.1. A Limitação do RL "End-to-End"**

Os autores demonstram empiricamente que a aplicação direta de algoritmos de DRL (como PPO) falha em convergir para designs de treliças complexas 3D. A razão diagnosticada é que a região de designs válidos é uma variedade de baixa dimensão imersa em um vasto espaço de configurações inválidas. O gradiente de política desaparece porque o agente raramente recebe um sinal positivo de "sucesso".

### **5.2. O Framework de Dois Estágios**

A inovação do AutoTruss reside na divisão do trabalho cognitivo entre busca e aprendizado:

1. **Estágio 1: Busca Construtiva via Monte Carlo Tree Search (MCTS)**  
   * O MCTS é utilizado para gerar o "esqueleto" topológico. Diferente do RL, que aprende uma política reativa, o MCTS realiza simulações de *lookahead*. Ele "imagina" múltiplos passos à frente.  
   * Crucialmente, o processo de expansão da árvore no MCTS incorpora verificações de validade rígidas (*Hard Constraints*). Se um ramo da árvore leva a uma estrutura desconexa ou instável, ele é podado imediatamente. Isso garante que a busca se concentre exclusivamente em topologias viáveis.  
   * O resultado deste estágio não é um design final, mas uma população diversificada de topologias válidas candidatas.  
2. **Estágio 2: Refinamento via Reinforcement Learning**  
   * Uma vez estabelecida a topologia, o problema se torna mais tratável e contínuo. Um agente de RL assume o controle para realizar a otimização de dimensionamento (*sizing optimization*) e ajustes finos na posição dos nós.  
   * Nesta fase, a recompensa é densa (pois a estrutura já é válida, então *compliance* e volume podem ser calculados a cada passo), permitindo que o RL brilhe na sua capacidade de otimização local.

### **5.3. Paralelos com Sistemas Multi-Agente**

A literatura recente também aponta para a generalização desta abordagem em sistemas multi-agente. Em e , é descrito o framework **DATTE**, que utiliza dois agentes competitivos/cooperativos: um agente de "criação" e um agente de "deleção".

* Esta dinâmica permite um equilíbrio mais fino na exploração. Enquanto um agente tenta adicionar redundância para robustez, o outro tenta agressivamente remover ineficiências. Este tipo de arquitetura adversarial/cooperativa é particularmente eficaz para manter a conectividade, pois o agente de deleção só é recompensado se conseguir remover um link *sem* quebrar a conectividade garantida pelo agente de criação.

## **6\. Estudo de Caso III: Otimização Topológica via Geometria Sólida Construtiva (TreeTOp)**

Expandindo o escopo das treliças para estruturas contínuas 3D, o trabalho **TreeTOp** (Padhy et al., 2024-2025) introduz uma abordagem construtiva baseada em primitivas volumétricas.

### **6.1. A Árvore CSG como Genótipo**

Em vez de representar o objeto como uma nuvem de voxels (o que consome muita memória e gera superfícies rugosas), o TreeTOp representa o objeto como uma **Árvore de Geometria Sólida Construtiva (CSG Tree)**.

* **Folhas da Árvore:** Primitivas geométricas parametrizáveis (esferas, cubos, cilindros).  
* **Nós Internos:** Operações Booleanas (União \\cup, Interseção \\cap, Diferença \-).

### **6.2. RL na Otimização da Árvore**

O problema de otimização torna-se encontrar a estrutura da árvore (quais operações aplicar e em que ordem) e os parâmetros das folhas (raio da esfera, posição do cubo).

* **Conectividade Intrínseca:** A grande vantagem desta abordagem é a garantia de fechamento e conectividade. A união de dois volumes sólidos sobrepostos resulta, por definição matemática, em um volume sólido único e fechado. Não existem "pixels flutuantes" ou densidades cinzas intermediárias sem significado físico.  
* **Interpretabilidade e Manufatura:** O resultado é, em essência, um arquivo CAD nativo. Diferente de uma malha STL gerada por SIMP que requer suavização e reparo, a árvore CSG é diretamente editável por engenheiros humanos e diretamente convertível em caminhos de ferramenta para manufatura (G-code).

O uso de RL neste contexto é focado na manipulação discreta da estrutura da árvore (adicionar um novo ramo, mudar uma operação de união para subtração), navegando no espaço de complexidade da forma.

## **7\. Desafios de Manufatura Aditiva e a Abordagem GNN**

A Otimização Topológica e a Manufatura Aditiva (AM \- Impressão 3D) são parceiras naturais, mas a AM impõe restrições geométricas severas, como a necessidade de estruturas auto-suportadas (*self-supporting*) para evitar colapso durante a impressão de *overhangs*.

### **7.1. GNNs para Predição de Suportabilidade**

O trabalho de Tabarraei e Bhuiyan (2025) ilustra o uso avançado de GNNs para integrar estas restrições diretamente no processo de otimização.

* Eles propõem uma GNN que atua como um "campo neural" sobre a malha de elementos finitos.  
* A rede aprende a prever não apenas a resposta mecânica, mas também a "impressibilidade" de cada região. A conectividade aqui é analisada camada por camada (layer-wise).  
* Diferente de métodos que geram suportes como pós-processamento, esta abordagem construtiva penaliza a criação de características que exigiriam suportes excessivos, "ensinando" o agente a construir estruturas que se auto-suportam (ex: arcos góticos em vez de vigas planas horizontais).

### **7.2. Suavização SDF e Dual Connectivity Graphs**

Para garantir que as estruturas geradas por métodos de aprendizado (incluindo GANs e RL) sejam livres de desconexões, Behzadi (2023) propõe o uso de **Grafos de Conectividade Dual** e a decomposição em **Maximal Disjoint Balls (MDBD)**.

* Esta técnica geométrica funciona como um filtro de validação. O grafo de tangência das esferas captura a topologia exata. Se o grafo se desconecta, o método identifica imediatamente.  
* Integrado ao loop de RL, isso fornece um sinal de erro preciso sobre *onde* a conectividade falhou, permitindo correções locais mais rápidas do que uma simples penalidade escalar global.

## **8\. Síntese e Perspectivas Futuras**

A análise da literatura de 2020 a 2025 revela que a aplicação de RL em Otimização Topológica ultrapassou a fase de prova de conceito acadêmica e está desenvolvendo metodologias robustas para lidar com a complexidade do mundo real.

### **8.1. Insights de Segunda e Terceira Ordem**

1. **A "Cognição" do Design Híbrido:** A tendência mais forte não é o RL puro, mas o RL híbrido. A combinação de "intuição" (aprendida por redes neurais profundas a partir de dados passados) e "raciocínio" (busca em árvore MCTS ou gramáticas lógicas) parece ser o caminho definitivo para superar a barreira da recompensa esparsa. Isso espelha a evolução da IA em jogos (AlphaGo), sugerindo que o design de engenharia é, computacionalmente, um jogo de estratégia com física embutida.  
2. **O Fim da Dependência do Grid:** A ascensão das GNNs e representações CSG sinaliza o declínio das abordagens baseadas em voxels/pixels para estruturas esparsas. O futuro da TO é *meshless* ou *graph-based*, permitindo uma flexibilidade de resolução infinita e invariância topológica.  
3. **Transferência de Conhecimento:** O uso de *embeddings* permite, pela primeira vez, a transferência de aprendizado real em engenharia estrutural. Um agente que aprende a otimizar pontes pode transferir conceitos de "triangulação" para otimizar fuselagens de aviões, pois os princípios fundamentais de fluxo de carga são capturados nos pesos da GNN, abstraídos da geometria específica.

### **8.2. Direções Futuras**

A pesquisa aponta para a integração de **Modelos de Difusão em Grafos** (Graph Diffusion Models) como uma alternativa gerativa aos MDPs passo-a-passo, potencialmente oferecendo a velocidade da geração *one-shot* com a validade topológica dos métodos construtivos. Além disso, a aplicação de **RL Multi-Agente** para a otimização cooperativa de grandes sistemas (ex: um agente otimiza a estrutura global, outros otimizam as conexões locais) promete resolver o problema de escala que ainda limita a TO a componentes individuais.  
Em conclusão, o paradigma construtivo em RL representa uma evolução necessária para alinhar a otimização topológica com as restrições discretas e conectivas da manufatura e montagem modernas, transformando a otimização de um processo de "escultura cega" para um processo de "construção inteligente".

### **Tabela 2: Resumo Comparativo das Principais Abordagens Investigadas**

| Trabalho / Autores | Paradigma Principal | Representação de Estado | Espaço de Ação | Gestão de Conectividade |
| :---- | :---- | :---- | :---- | :---- |
| **Ororbia & Warn** | **Hierarchical RL (HDRL)** | Grafo Atribuído | Híbrido: Topológico (Regras de Gramática) \+ Paramétrico | Implícita via Gramáticas de Design (regras de produção válidas). |
| **AutoTruss (Li et al.)** | **Hybrid Search (MCTS \+ RL)** | Árvore de Decisão / Grafo | Estágio 1: Construção Topológica. Estágio 2: Sizing. | Garantida pelo *lookahead* do MCTS e poda de ramos inválidos. |
| **Hayashi & Ohsaki** | **Graph Embedding \+ DQN** | Graph Embeddings (GNN) | Sequencial Subtrativa (Remoção de arestas). | Monitoramento passo-a-passo; ações que causam instabilidade são penalizadas. |
| **TreeTOp (Padhy et al.)** | **CSG Tree Optimization** | Árvore CSG | Operações Booleanas e seleção de primitivas. | Intrínseca à geometria sólida (união de volumes gera corpos conectados). |
| **DATTE (Multi-Agent)** | **Multi-Agent RL (MARL)** | Topologia de Rede | Agentes Cooperativos/Competitivos (Criação vs. Deleção). | Agente de deleção restrito pela conectividade mantida pelo agente de criação. |

#### **Referências citadas**

1\. Automatic Truss Design with Reinforcement Learning \- IJCAI, https://www.ijcai.org/proceedings/2023/0407.pdf 2\. Discrete Structural Design Synthesis: A Hierarchical-Inspired Deep Reinforcement Learning Approach Considering Topological and Parametric Actions \- ASME Digital Collection, https://asmedigitalcollection.asme.org/mechanicaldesign/article/146/9/091707/1199815/Discrete-Structural-Design-Synthesis-A 3\. Discrete Structural Design Synthesis: A Hierarchical-Inspired Deep Reinforcement Learning Approach Considering Topological and Parametric Actions | Request PDF \- ResearchGate, https://www.researchgate.net/publication/380449678\_Discrete\_Structural\_Design\_Synthesis\_A\_Hierarchical-Inspired\_Deep\_Reinforcement\_Learning\_Approach\_Considering\_Topological\_and\_Parametric\_Actions 4\. New Approaches for Network Topology Optimization Using Deep Reinforcement Learning and Graph Neural Network \- IEEE Xplore, https://ieeexplore.ieee.org/iel8/6287639/10820123/11000124.pdf 5\. Graph Neural Network-Based Topology Optimization for Self-Supporting Structures in Additive Manufacturing \- ResearchGate, https://www.researchgate.net/publication/394979185\_Graph\_Neural\_Network-Based\_Topology\_Optimization\_for\_Self-Supporting\_Structures\_in\_Additive\_Manufacturing 6\. TreeTOp: Topology Optimization using Constructive Solid Geometry Trees \- arXiv, https://arxiv.org/html/2409.02300v1 7\. Treetop: topology optimization using constructive solid geometry trees \- ResearchGate, https://www.researchgate.net/publication/389397934\_Treetop\_topology\_optimization\_using\_constructive\_solid\_geometry\_trees 8\. A Reinforcement Learning Method for Layout Design of Planar and Spatial Trusses using Kernel Regression \- MDPI, https://www.mdpi.com/2076-3417/12/16/8227 9\. Discrete structural optimization as a sequential decision process solved using deep reinforcement learning \- PSU-ETD, https://etda.libraries.psu.edu/catalog/23314meo9 10\. Reinforcement Learning and Graph Embedding for Binary Truss Topology Optimization Under Stress and Displacement Constraints \- Kyoto University Research Information Repository, https://repository.kulib.kyoto-u.ac.jp/handle/2433/259776 11\. Reinforcement Learning and Graph Embedding for Binary Truss Topology Optimization Under Stress and Displacement Constraints \- Frontiers, https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2020.00059/full 12\. (PDF) Graph Neural Network-Based Topology Optimization for Efficient Support Structure Design in Additive Manufacturing \- ResearchGate, https://www.researchgate.net/publication/391936044\_Graph\_Neural\_Network-Based\_Topology\_Optimization\_for\_Efficient\_Support\_Structure\_Design\_in\_Additive\_Manufacturing 13\. Application of Reinforcement Learning Methods Combining Graph Neural Networks and Self-Attention Mechanisms in Supply Chain Route Optimization \- MDPI, https://www.mdpi.com/1424-8220/25/3/955 14\. Mastering Truss Structure Optimization With Tree Search | J. Mech ..., https://asmedigitalcollection.asme.org/mechanicaldesign/article/147/10/101702/1214374/Mastering-Truss-Structure-Optimization-With-Tree 15\. Safe Reinforcement Learning for Arm Manipulation with Constrained Markov Decision Process \- MDPI, https://www.mdpi.com/2218-6581/13/4/63 16\. Automatic Truss Design with Reinforcement Learning, https://arxiv.org/html/2306.15182 17\. Learning with sparse reward in a gap junction network inspired by the insect mushroom body | PLOS Computational Biology \- Research journals, https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012086 18\. Taming Connectedness in Machine-Learning-based Topology Optimization with Connectivity Graphs \- CDL Lab @ UCONN \- University of Connecticut, https://cdl.engr.uconn.edu/papers/behzadi2023CAD.pdf