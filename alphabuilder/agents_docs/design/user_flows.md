# 🌊 AlphaBuilder: Fluxos de Usuário (User Flows)

**Versão:** 1.1
**Status:** Revisado
**Referência:** `web_interface.md`, `blueprint.md`

Este documento detalha as jornadas de usuário (User Journeys) para o AlphaBuilder. O foco é prover uma experiência de UX moderna, fluida e "profissional", inspirada em ferramentas como Figma, Blender e engines de xadrez (Chess.com).

---

## 1. Fluxo Principal: Otimização Generativa ("The Creator Flow")

Este é o fluxo "Happy Path" onde um engenheiro cria um novo projeto, define o problema físico e obtém uma solução otimizada pela IA.

### 1.1. Criação do Espaço de Projeto
*   **Ação do Usuário:** Na Dashboard, clica em `[+ Novo Projeto]`.
*   **Interface (Modal/Overlay):**
    *   Input de Nome do Projeto.
    *   Seleção de Resolução do Grid (ex: `Low (32x16)`, `Standard (64x32)`, `High (128x64)`).
    *   Definição de Dimensões Físicas (Largura [m], Altura [m], Espessura [m]).
    *   Seleção de Material Base (Dropdown: Aço, Alumínio, Titânio - define Módulo de Young $E$ e Poisson $\nu$).
*   **Feedback do Sistema:** Cria o registro no banco e redireciona para o **Editor**.

### 1.2. Definição de Condições de Contorno (BCs)
*   **Estado Inicial:** Canvas 2D vazio (grid pixelado). Toolbar lateral ativa.
*   **Ferramenta "Suporte" ($\Gamma_u$):**
    *   Usuário seleciona ferramenta `[Fixar / Anchor]` (Ícone de Cadeado ou Triângulo).
    *   *Interação:* Clica ou arrasta (paint) sobre células do grid.
    *   *Visual:* Células ficam **Ciano Neon** (`#00f0ff`). Ícones pequenos de "cadeado" aparecem sobre elas.
*   **Ferramenta "Carga" ($\Gamma_t$):**
    *   Usuário seleciona ferramenta `[Força / Load]` (Ícone de Seta).
    *   *Interação:* Clica em uma célula ou região.
    *   *Pop-up Contextual:* Ao soltar o clique, um mini-modal pede a magnitude e direção do vetor força $(F_x, F_y, F_z)$.
    *   *Visual:* Células ficam **Magenta Neon** (`#ff0055`). Uma seta 3D é renderizada saindo do ponto de aplicação.
*   **Ferramenta "Região Proibida" (Opcional):**
    *   Usuário pinta áreas onde **não** pode haver material (obstáculos).
    *   *Visual:* Hachura vermelha semitransparente.

### 1.3. O Processo de Otimização com "Neural HUD"
*   **Dor do Usuário (Pain Point):** "Black Box Anxiety". O usuário não sabe se a IA travou, se está "pensando", ou se a direção tomada é promissora.
*   **Solução UX:** Visualização em Tempo Real do Raciocínio (Neural HUD).
*   **Layout:** Ao clicar em `[▶ OTIMIZAR]`, a tela se divide ou um painel lateral ("Neural Sidecar") se expande.

#### Componentes do Neural HUD:
1.  **Confidence Graph (Value Head Monitor):**
    *   *O que é:* Um gráfico de linha rolando em tempo real (estilo monitor cardíaco/EKG).
    *   *Dado:* A saída da **Value Head** da rede ($V(s)$), representando a probabilidade estimada de sucesso/viabilidade.
    *   *Feedback:* Se a linha sobe, a IA está confiante. Se cai drasticamente, o usuário vê a IA "percebendo o erro" e tentando corrigir (backtracking).
2.  **MCTS Ghosting (A "Imaginação" da IA):**
    *   *O que é:* Visualização dos caminhos alternativos considerados.
    *   *Visual:* Enquanto a estrutura real (Sólida) cresce, "blocos fantasmas" (amarelo translúcido) piscam brevemente ao redor da fronteira de crescimento.
    *   *Significado:* Representam as simulações do MCTS que foram exploradas mas descartadas. Isso mostra que a IA está ativamente buscando opções, não apenas seguindo um script.
3.  **Policy Heatmap (Intenção vs Ação):**
    *   *O que é:* Um mini-mapa no canto do HUD.
    *   *Visual:* Mostra a distribuição de probabilidade crua da **Policy Head** ($\pi(s)$). Áreas vermelhas são onde a rede *quer* colocar material.
    *   *Utilidade:* Permite ver se a rede está "focada" (um ponto vermelho forte) ou "confusa" (manchas difusas por todo o grid).

### 1.4. Resultado e Inspeção
*   **Conclusão:** Otimização para. Confetes discretos ou brilho dourado na estrutura.
*   **Ação:** Usuário alterna para **Modo 3D** (Toggle Switch).
*   **Visual 3D:** A peça é extrudada. Renderização com sombreamento, oclusão de ambiente (SSAO) e material metálico.
*   **Interação:** Orbit, Pan, Zoom.

---

## 2. Fluxo de Estudo Estrutural ("The Analyst Flow")

Inspirado em engines de xadrez, este fluxo permite entender *por que* a IA tomou certas decisões e onde estão os riscos.

### 2.1. Carregamento e Histórico
*   **Contexto:** Usuário está visualizando uma estrutura pronta.
*   **Timeline:** Na parte inferior, uma linha do tempo (slider) permite "voltar no tempo" para qualquer passo da geração ($t=0$ a $t=Final$).
*   **Ação:** Usuário arrasta o slider para o meio do processo.
*   **Visual:** A estrutura reverte para o estado daquele momento.

### 2.2. Visualização de "Pensamento" (MCTS/Policy)
*   **Toggle:** Ativar `[Show AI Intent]`.
*   **Visual:**
    *   Sobrepõe um **Heatmap** (mapa de calor) sobre o grid.
    *   *Cores Quentes (Vermelho/Laranja):* Regiões onde a Rede Neural (Policy Head) tinha alta certeza de que deveria haver material.
    *   *Cores Frias (Azul/Transparente):* Regiões que a rede queria remover.
    *   *Ghosting:* Mostra "fantasmas" de opções que o MCTS considerou mas descartou (caminhos alternativos semi-transparentes).

### 2.3. Análise de Atenção (Explainability)
*   **Ferramenta:** `[Foco de Atenção]`.
*   **Ação:** Usuário clica em um pixel específico da estrutura (ex: um ponto de conexão crítica).
*   **Resposta do Sistema:**
    *   O sistema consulta os *Attention Weights* do Vision Transformer.
    *   *Visual:* Ilumina outros pixels do grid que a rede "olhou" para decidir sobre o pixel clicado.
    *   *Insight:* "Para decidir manter este nó, a IA focou fortemente no Suporte A e na Carga B". Isso ajuda a entender dependências globais.

### 2.4. Validação Física (FEM Overlay)
*   **Toggle:** Ativar `[Stress Map / Von Mises]`.
*   **Visual:** Colore a estrutura com o gradiente de tensão de Von Mises (Azul = Baixa Tensão, Vermelho = Tensão Crítica).
*   **Interação:** Hover sobre um pixel mostra o valor numérico exato (ex: "250 MPa").
*   **Alerta:** Se alguma região excede o limite de escoamento do material, um ícone de alerta ⚠️ pulsa sobre a região.

---

## 3. Futuro / Não Implementar Agora

*Os fluxos abaixo foram considerados mas estão fora do escopo da implementação atual.*

### 3.1. Fluxo Comparativo ("A/B Testing")
*   Branching de projetos.
*   Comparação lado-a-lado (Split Screen) de duas versões.
*   Diff visual de topologias.

### 3.2. Fluxo de Colaboração e Revisão
*   Anotações em 3D (Pins com comentários).
*   Compartilhamento de Snapshots via link público (WebAssembly viewer).

### 3.3. Fluxo de Exportação e Manufatura
*   Pós-processamento de malha (Marching Cubes/Dual Contouring para suavização).
*   Exportação para STL/STEP.
*   Geração automática de relatório PDF.
