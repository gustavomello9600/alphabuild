### 📂 MISSÃO: AGENTE 04 (WEB_INTERFACE & UX)

**Função:** Especialista em UX/UI, Frontend Moderno e Engenharia de Frontend.
**Paradigma:**
*   **Backend:** Funcional/Declarativo (FastAPI + Pydantic).
*   **Frontend:** Reativo/Funcional (React + Hooks + Canvas API).
**Stack:** Python 3.10+ (FastAPI), TypeScript (React), Three.js (via React-Three-Fiber), HTML5 Canvas, TailwindCSS.

---

## 1. CONTEXTO E FILOSOFIA
Você é responsável pela "Sala de Controle" do **AlphaBuilder**. Sua interface deve ser uma **Single Page Application (SPA)** fluida, sem recarregamentos, comunicando-se assincronamente com o backend.

**Dualidade de Visualização:**
Embora o backend processe volumes, a experiência humana para problemas de placas é inerentemente 2D. Sua interface deve ser **Híbrida**:
1.  **Modo Engenharia (2D):** Renderização leve, ortogonal e precisa usando **HTML5 Canvas**. Ideal para desenhar condições de contorno e visualizar topologia limpa sem ruído visual.
2.  **Modo Volumétrico (3D):** Renderização rica usando **Three.js (Voxels)**. Ideal para visualizar a espessura física, rotação e estética do produto final.

**Sua Primeira Prioridade:**
Antes de codificar, mapeie e valide os fluxos no arquivo `UX_FLOWS.md`.

---

## 2. TAREFA A: DOCUMENTAÇÃO DE FLUXOS (UX JOURNEYS)

Documente os seguintes fluxos mandatórios:

### 2.1. O Fluxo do Arquiteto (Input)
*   **Definição de Canvas:** O usuário define $H \times W$ e a Espessura (parâmetro numérico).
*   **Interação 2D:** A definição de Cargas e Suportes deve ocorrer em um canvas 2D plano. É muito mais preciso clicar em um grid pixelado 2D do que tentar acertar um voxel em um ambiente 3D rotacionável.
*   **Setup:** Configuração de orçamento de passos e restrições.

### 2.2. O Fluxo do Espectador (Monitoramento)
*   **Visualização Padrão:** O sistema inicia mostrando o progresso no **Modo 2D** (Heatmap de Alta Performance). Isso permite ver claramente a conectividade.
*   **Toggle 3D:** O usuário possui um botão "Ver Volume" que alterna instantaneamente para a cena Three.js, mostrando a peça extrudada.
*   **Feedback Real-time:** O grid atualiza via polling sem piscar a tela inteira.

### 2.3. O Fluxo do Analista (Pós-Processamento)
*   **Time Travel:** Slider para navegar pelo histórico do episódio.
*   **Layers:** Checkboxes para ligar/desligar visualização de Cargas, Suportes e Mapa de Tensão.

---

## 3. ESTRUTURA TÉCNICA (ARQUITETURA)

### 3.1. O Servidor de Dados (API Gateway)
Crie uma API RESTful leve (`api/main.py`) usando **FastAPI**.
*   **Função:** Servir dados do SQLite para o Frontend.
*   **Serialização Otimizada:** Para grids grandes, envie a matriz de topologia como *Binary Buffer* ou *Base64* compactado, não como uma lista JSON gigante de `0`s e `1`s.
*   **Endpoints:**
    *   `GET /episodes/{id}/latest`: Retorna o estado atual.
    *   `POST /simulation/start`: Inicia o worker do Agente 02.

### 3.2. O Cliente Visual (React)
Aplicação React moderna gerenciada por **Vite**.

*   **Gerenciamento de Estado:** Use **React Query** para polling e cache.
*   **Roteamento Visual (Smart Component):**
    Crie um componente `<StructureViewer mode="2D|3D" data={grid} />` que condicionalmente renderiza:
    
    *   **Opção A (2D - Pixel Renderer):** Manipulação direta de `<canvas>` via `useRef`. Escreva os dados da matriz diretamente no `Uint8ClampedArray` do Contexto 2D. Isso renderiza milhões de pixels a 60fps com custo zero de GPU. Estilo: "Blueprint Técnico".
    *   **Opção B (3D - Voxel Renderer):** Cena `React-Three-Fiber`. Utilize `InstancedMesh` para desenhar os cubos. Estilo: "Peça Física".

---

## 4. TAREFAS DE IMPLEMENTAÇÃO (CÓDIGO)

### Tarefa B: API Backend
*   Implemente os modelos Pydantic (`ConfigSchema`, `StepResponse`).
*   Garanta CORS habilitado para desenvolvimento local (React porta 5173, API porta 8000).

### Tarefa C: Frontend "AlphaView"
Diretório: `web/`
*   **Componentes Principais:**
    *   `GridInput.tsx`: Canvas interativo. Detecta cliques, converte coordenada de mouse para índice da matriz `(row, col)` e atualiza o estado local de BCs.
    *   `LiveMonitor.tsx`: Container que faz o polling da API.
    *   `PixelCanvas.tsx`: O visualizador 2D de alta performance. Deve usar CSS `image-rendering: pixelated` para garantir que os pixels sejam quadrados nítidos, não borrados.
    *   `VoxelScene.tsx`: O visualizador 3D.
    *   `MetricsChart.tsx`: Gráfico de linha (Recharts/Visx) para a Fitness.

---

## 5. REQUISITOS DE DESIGN SYSTEM
*   **Tema:** Dark Mode Obrigatório (Engenharia Profissional).
*   **Paleta de Cores Funcional:**
    *   Vazio: `#1e1e1e` (Fundo quase preto).
    *   Material: `#e0e0e0` (Branco gelo).
    *   Suporte: `#00f0ff` (Ciano Neon).
    *   Carga: `#ff0055` (Magenta Neon).
    *   Destaque de Refinamento: Piscar em Amarelo quando um bloco é alterado.

---

## 6. VALIDAÇÃO

Inclua os seguintes critérios de aceite no seu report:
1.  **Teste de Nitidez 2D:** O visualizador 2D deve mostrar pixels perfeitamente quadrados e nítidos (crisp edges), sem interpolação linear (blur), mesmo ao dar zoom.
2.  **Teste de Performance de Render:** O visualizador 3D deve aguentar um grid $64 \times 32 \times 16$ sem cair abaixo de 30 FPS.
3.  **Teste de Latência de UX:** O tempo entre clicar em "Iniciar" e ver o primeiro pixel aparecer na tela deve ser imediato (feedback visual de "Aguardando Worker").