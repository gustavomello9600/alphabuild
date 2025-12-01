> [!WARNING]
> **DEPRECATED**: This document has been superseded by [AlphaBuilder v2.0 Spec](alphabuilder_v2_0_spec.md). Do not use this for new development.

# AlphaBuilder v1.1: Especificação de Sistema e Estratégia de Treinamento

| Metadado | Detalhe |
| :---- | :---- |
| **Projeto** | AlphaBuilder |
| **Tipo** | Deep Reinforcement Learning para Otimização Topológica Generativa |
| **Target Hardware** | NVIDIA A100 (Treino) / T4 (Inferência) |
| **Arquitetura Base** | Swin-UNETR (Physics-Aware) |
| **Versão do Doc** | 1.1 |

## **1\. Introdução e Escopo**

O **AlphaBuilder** é um agente autônomo projetado para resolver o problema da Otimização Topológica 3D tratado como um jogo de construção sequencial. Diferente de abordagens tradicionais que iniciam com um bloco sólido cheio (espaço de projeto), o AlphaBuilder opera em dois regimes dinâmicos distintos (**Construção** e **Refinamento**), imitando o raciocínio de um engenheiro humano: primeiro garantir a função (conectividade), depois otimizar a forma (peso/eficiência).  
O núcleo do sistema é uma rede neural **Swin-UNETR** que atua como um oráculo de intuição física, guiando uma busca em árvore (MCTS) para navegar o espaço de estados complexo de voxels tridimensionais.

## **2\. Requisitos Funcionais do Sistema**

* **RF-01 (Bifasia Operacional):** O sistema deve, obrigatoriamente, operar em dois estágios sequenciais. O Estágio de Refinamento (Fase 2\) só pode ser iniciado se, e somente se, os critérios de sucesso do Estágio de Construção (Fase 1\) forem satisfeitos.  
* **RF-02 (Validade Topológica Contínua):** Toda ação tomada pelo agente (Adição ou Remoção) deve ser validada instantaneamente quanto à conectividade. O sistema deve rejeitar ações que resultem em ilhas de material flutuante ou desconexão dos apoios.  
* **RF-03 (Feedback Físico Instrucional):** Durante o treinamento na Fase 2, o sistema deve invocar o Solver FEM externo após cada ação de mudança topológica efetiva para calcular o novo estado de deformação e fornecer a recompensa exata (Dense Reward).  
* **RF-04 (Generalização de Entrada):** A arquitetura neural deve ser invariante às dimensões absolutas do grid (via janelas deslizantes), aceitando volumes de design arbitrários limitados apenas pela memória GPU.  
* **RF-05 (Inputs Físicos Normalizados):** O sistema deve normalizar e codificar vetorialmente as forças e condições de contorno, garantindo que a rede neural responda à magnitude relativa das cargas e não a valores absolutos.

## **3\. Mecânica do Jogo (Environment Dynamics)**

O ambiente de simulação é um grid 3D discretizado. O "Jogo" é episódico e dividido estritamente em duas fases. O agente deve completar a Fase 1 com sucesso para desbloquear a Fase 2\.

### **3.1. O Tabuleiro (State Space)**

O estado $S\_t$ é representado por um Tensor 5D de dimensões (5, D, H, W):

1. **Canal 0 (**$\\rho$**):** Matriz de densidade binária atual (0 \= Ar, 1 \= Material).  
2. **Canal 1 (Mask):** Condições de Contorno de Suporte (1 \= Fixo).  
3. **Canais 2, 3, 4 (**$F\_x, F\_y, F\_z$**):** Campos de Força Normalizados (esparsos, apenas onde há carga).

### **3.2. Ações Válidas (Action Space)**

O espaço de ação é discreto e local.

* **Ação de Adição (**$a\_{add}$**):** Transformar um voxel $v\_{ij} \= 0 \\to 1$.  
  * *Restrição:* O voxel deve ser vizinho (adjacente) a um voxel existente ou a um ponto de suporte/carga (Seeds).  
* **Ação de Remoção (**$a\_{rem}$**):** Transformar um voxel $v\_{ij} \= 1 \\to 0$.  
  * *Restrição A (Conectividade):* A remoção não pode dividir a peça em duas ilhas desconexas.  
  * *Restrição B (BCs):* A remoção não pode desconectar a peça dos Suportes ou Cargas originais.

### **3.3. Fase 1: O Jogo de Conexão (The "Pathfinder")**

* **Estado Inicial:** Espaço vazio. Apenas os voxels de Suporte e Carga existem como "sementes".  
* **Objetivo:** Criar um caminho contínuo de material sólido que conecte **todos** os pontos de carga a **todos** os pontos de suporte necessários para estabilidade estática.  
* **Ações Permitidas:** Predominantemente Adição ($a\_{add}$). Remoção permitida apenas para correção de erros imediatos.  
* **Terminação:** Assim que a verificação de conectividade detectar que todos os componentes de interesse pertencem ao mesmo conjunto conexo.  
* **Recompensa:** \+1 por conexão bem-sucedida, \-0.01 por passo (para incentivar caminhos curtos).

### **3.4. Fase 2: O Jogo de Refinamento (The "Sculptor")**

* **Estado Inicial:** O resultado final da Fase 1 (estrutura válida, mas volumosa/ineficiente).  
* **Objetivo:** Minimizar o volume $Vol$ sujeito a uma restrição de deformação máxima $U\_{max} \\le U\_{limit}$.  
* **Ações Permitidas:** Flexibilidade total entre Remoção ($a\_{rem}$) para redução de massa e Adição ($a\_{add}$) para redistribuição de tensão e reforço estrutural.  
* **Feedback Físico:** Conforme **RF-03**, o Solver FEM é acionado a cada ação efetiva para atualizar o estado de tensão e deformação.  
* **Terminação:** Estabilidade da topologia (agente cessa alterações significativas) ou violação crítica da restrição de deformação (Game Over).  
* **Recompensa Instantânea:** Função da redução de volume ponderada pela manutenção da *Compliance* dentro dos limites.

## **4\. Arquitetura Neural: "Physics-Aware Swin-UNETR"**

Esta arquitetura foi selecionada para atender a dois requisitos conflitantes:

1. **Visão Global (Swin):** Entender que remover um voxel na base afeta a ponta (Conexão e Carga).  
2. **Precisão Local (UNETR):** Decidir exatamente qual pixel da borda remover para alisar a estrutura.

### **4.1. Pipeline de Dados na Rede**

1. **Entrada:** Tensor (Batch, 5, D, H, W).  
   * *Pré-requisito:* Os vetores de força devem estar normalizados em relação à malha.  
2. **Encoder (Swin Transformer 3D):**  
   * Utiliza *Shifted Windows* para capturar dependências físicas de longo alcance com complexidade $O(N)$.  
   * Gera representações hierárquicas em 4 escalas (do detalhe geométrico à física global).  
3. **Decoder (U-Net style):**  
   * Reconstrói a resolução espacial usando Deconvoluções.  
   * Usa **Skip Connections** para injetar a geometria original nos mapas de decisão profunda, garantindo que o agente saiba exatamente onde estão as fronteiras dos voxels.

### **4.2. Saídas (Heads)**

O modelo possui uma estrutura "Two-Headed" compartilhada:

* **Política (Policy Head \-** $\\pi$**):**  
  * **Saída:** Tensor (Batch, 2, D, H, W).  
  * **Canal 0:** Score de probabilidade para Ação $Add$ em cada voxel.  
  * **Canal 1:** Score de probabilidade para Ação $Remove$ em cada voxel.  
  * **Máscara de Ação:** No pós-processamento, zeramos as probabilidades de voxels inválidos (ex: remover onde já é ar, ou adicionar longe da fronteira).  
* **Valor (Value Head \-** $v$**):**  
  * **Saída Fase 1:** Probabilidade de sucesso na conexão (Logit de Classificação).  
  * **Saída Fase 2:** Estimativa da recompensa final negativa (proxy para $U\_{max}$ ou *Compliance*).  
  * *Nota:* O cabeçalho de Valor deve ser treinado para entender contextualmente em qual fase está baseada na densidade do input.

## **5\. Pipeline de Geração de Dados e Cenários Realistas**

Para evitar que o AlphaBuilder aprenda apenas o "Vazio", precisamos de um treinamento curricular.

### **5.1. Engenharia de Cenários Realistas**

O gerador de dados deve criar condições de contorno (BCs) baseadas em bibliotecas de engenharia estrutural:

1. **Cantilever (Viga em Balanço):** Face esquerda fixa, carga na ponta direita.  
2. **Ponte/Viga Bi-apoiada:** Apoios nos cantos inferiores, carga distribuída ou pontual no topo/centro.  
3. **MBB Beam:** Meia viga apoiada (problema clássico de simetria).  
4. **Cargas de Cisalhamento/Torção:** Cargas não alinhadas com eixos principais e cargas binárias opostas para induzir torção.  
5. **Multi-Load:** Múltiplas cargas com direções conflitantes (ex: Vento \+ Gravidade).

### **5.2. Estratégia de Geração (Data Factory)**

* **Dataset Fase 1 (Pathfinding):**  
  * Utilizar algoritmos de **Pathfinding 3D (A\* ou RRT)**. Gerar caminhos aleatórios e volumosos (árvores) que conectam os suportes às cargas. Isso ensina a rede "o que é estar conectado".  
* **Dataset Fase 2 (Otimização):**  
  * Utilizar solver SIMP (Método clássico). Pegar o volume cheio e rodar a otimização.  
  * **Data Augmentation Reverso:** Salvar os estados intermediários do SIMP "de trás para frente". Um estado otimizado é o *alvo*. Um estado anterior (mais gordo) é o *input*. A ação correta é a diferença entre eles (os voxels removidos pelo SIMP).

## **6\. Estratégia de Treinamento**

### **Etapa 1: Supervised Pre-training (Clonagem de Comportamento)**

Antes de jogar, o AlphaBuilder lê os "manuais" (Datasets gerados acima).

* **Objetivo:** Ensinar regras de sintaxe (como conectar) e regras semânticas (onde estão as tensões altas).  
* **Loss Function:**  
  * Entropia Cruzada Ponderada para a Política (Classificação voxel a voxel).  
  * MSE para o Valor (Regressão da deformação/sucesso).

### **Etapa 2: Self-Play com MCTS (Reinforcement Learning)**

Aqui a rede joga contra si mesma para descobrir estratégias que o SIMP ou o A\* não encontraram.

#### **A. A Árvore de Busca (MCTS) Modificada**

* Como o espaço de ações $64^3$ é gigantesco, o MCTS usa a **Política da Rede para podar a árvore**.  
* Apenas os Top-K voxels mais prováveis sugeridos pela rede (Probability Mask) são expandidos como nós filhos.

#### **B. Execução da Rodada (Self-Play)**

1. **Início:** Ambiente gera um cenário (ex: Ponte).  
2. **Fase 1 Loop:** Agente escolhe onde colocar voxels. Verifica *Conectividade*. Se conectado $\\to$ Start Fase 2\.  
3. **Fase 2 Loop:** Agente escolhe a ação ótima (Adicionar ou Remover) para refinar a estrutura. A cada modificação topológica, o ambiente chama o **Solver FEM**, que devolve a deformação real da peça naquele instante. Esse feedback preciso ajusta a estimativa da rede sobre "qual voxel é estruturalmente vital".  
4. **Game Over:** Volume mínimo atingido, estrutura colapsou ou limite de passos excedido.  
5. **Retropropagação:** A recompensa final propaga por todo o caminho, informando quais decisões levaram a uma estrutura ótima e quais causaram falha precoce.

## **7\. Pipeline de Implantação e Validação**

### **Módulo de Validação Topológica**

Para garantir a Fase 1 e a validade das ações na Fase 2 sem custo proibitivo, o sistema utiliza algoritmos de conectividade de grafos (como Union-Find).

* A verificação de conectividade é lógica e não depende do solver físico.  
* Garante em tempo quase constante ($O(1)$) que nenhuma ação viole os Requisitos Funcionais de topologia contínua.

### **Integração do Solver Físico (Oráculo de Valor)**

O Solver FEM (Black Box) é utilizado estritamente como mecanismo de **Ground Truth** durante o treinamento e na Fase 2\. Ele não faz parte da inferência final do modelo em produção (pois o modelo terá internalizado a física), mas é vital no loop de treinamento para fornecer o sinal de erro preciso ($Loss$) entre a previsão da rede neural e a realidade física.

## **8\. Sumário da Justificativa Técnica**

O uso da arquitetura **Swin-UNETR** combinada com as regras de **Jogo Bifásico** soluciona as maiores falhas de TO baseada em IA:

1. **Generalização:** A rede não decora formas, ela aprende a reagir ao campo de forças (Swin Transformer) para construir caminhos de carga (Fase 1).  
2. **Refinamento:** O Decoder UNETR com Skip Connections oferece a resolução espacial necessária para refinar as fronteiras e redistribuir voxels de forma cirúrgica na Fase 2\.  
3. **Consistência Física:** Ao forçar ações locais de vizinhança e validar conectividade logicamente, o sistema garante construtibilidade e impede a rede de criar ilhas de material.  
4. **Aprendizado Supervisionado por FEM:** Ao integrar o solver físico passo a passo no treinamento, transformamos a rede neural em um aproximador universal extremamente preciso das equações de elasticidade, permitindo inferência futura sem o custo computacional do solver numérico.
