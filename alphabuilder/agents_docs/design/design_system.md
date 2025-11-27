# 🎨 AlphaBuilder: Design System & Brand Identity

**Versão:** 1.0
**Status:** Concept
**Artefato:** `agents_docs/design_system.md`

Este documento define a identidade visual do AlphaBuilder e o plano para materializá-la em um **Design System Interativo (Live Styleguide)**.

---

## 1. Manifesto da Marca: "Computational Ethereal"

O AlphaBuilder não é apenas uma ferramenta de engenharia; é uma inteligência artificial que esculpe a matéria. A identidade deve capturar essa tensão entre o **Físico (Pesado, Estático)** e o **Digital (Leve, Dinâmico)**.

*   **Personalidade:** O Oráculo Técnico. Preciso, mas quase mágico.
*   **Conceito Visual (Logo):** "Truss Force". Um 'Alpha' estrutural, formado por treliças metálicas escuras, envolto por linhas de tensão etéreas (campos de força).
*   **Regra 60/30/10:**
    *   **60% (Void):** Preto Profundo (Backgrounds).
    *   **30% (Structure):** Cinza Metálico / Grafite (Elementos UI, Containers).
    *   **10% (Energy):** Ciano/Magenta Neon (Acentos, Dados, Interações).

---

## 2. Átomos Visuais

### 2.1. Paleta de Cores ("The Void & The Energy")

O tema é estritamente **Dark Mode**. A engenharia séria acontece no escuro para focar no dado.

*   **Backgrounds (O Vazio - 60%):**
    *   `Void Black`: `#050505` (Fundo infinito).
    *   `Deep Space`: `#0A0A0A` (Fundo secundário).

*   **Structure (A Matéria - 30%):**
    *   `Matter Grey`: `#121212` (Painéis, Cards).
    *   `Steel Frame`: `#2A2A2A` (Bordas, Divisores).
    *   `Text Primary`: `#E0E0E0` (Branco Gelo - Leitura).

*   **Energy (A Luz - 10%):**
    *   `Support Cyan`: `#00F0FF` (Segurança, Fixo, Frio).
    *   `Load Magenta`: `#FF0055` (Perigo, Força, Quente).
    *   `Neural Purple`: `#7000FF` (IA, Raciocínio).
    *   `Success Green`: `#00FF9D` (Otimizado).

### 2.2. Tipografia ("Data & Display")

*   **Display (Títulos / Impacto):**
    *   *Fonte:* **Space Grotesk** ou **Syne**.
    *   *Características:* Geométrica, com curvas idiossincráticas que lembram tubos ou nós.
*   **Interface (UI / Leitura):**
    *   *Fonte:* **Inter** ou **JetBrains Mono** (para dados numéricos).
    *   *Características:* Legibilidade máxima, tabular nums para tabelas de engenharia.

### 2.3. Iconografia & Formas
*   **Estilo:** "Wireframe". Ícones de linha fina (1.5px), cantos levemente arredondados, mas com terminais retos.
*   **Grid:** Tudo alinhado a um grid de 4px/8px.
*   **Bordas:** Sutis, `1px` com baixa opacidade (`rgba(255,255,255,0.1)`).

---

## 3. Plano de Construção: O Site "AlphaDesign"

O Design System não será um PDF estático. Será um site vivo (`/design`) dentro da própria aplicação, servindo como documentação e teste de componentes.

### 3.1. Stack Tecnológica
*   **Framework:** React + Vite (Mesma da aplicação principal).
*   **Estilização:** TailwindCSS (Utility-first para velocidade).
*   **Animação:** Framer Motion (para interações fluidas).
*   **3D:** React-Three-Fiber (para exibir o logo e elementos 3D interativos no hero).

### 3.2. Estrutura do Site de Identidade

#### **A. Hero Section: "The Living Logo"**
*   **Visual:** O logo do AlphaBuilder (o "A" estrutural) renderizado em 3D no centro.
*   **Interação:** O mouse afeta a iluminação. Ao clicar, o logo se "desmonta" e "remonta" (efeito de otimização topológica).
*   **Texto:** "Sculpting Matter with Intelligence."

#### **B. Seção "Atoms" (Interativa)**
*   **Cores:** Clique na cor para copiar o HEX. As cores pulsam.
*   **Tipografia:** Um editor de texto live para testar as fontes Space Grotesk e Inter.

#### **C. Seção "Components" (Playground)**
*   **Botões:** Botões "Neon" com hover states que emitem brilho (box-shadow).
*   **Inputs:** Campos de entrada que parecem terminais de comando.
*   **Cards:** Cards de vidro (Glassmorphism sutil) sobre fundo escuro.

#### **D. Seção "The Grid" (Demo do Core)**
*   Uma mini-demonstração do canvas 2D. O usuário pode passar o mouse e ver o efeito de "highlight" nos pixels, demonstrando a precisão da UI.

---

## 4. Próximos Passos (Action Plan)

1.  **Setup:** Inicializar o projeto React com Vite e Tailwind.
2.  **Config:** Definir o `tailwind.config.js` com as cores e fontes da marca.
3.  **Assets:** Vetorizar o logo gerado (SVG) e preparar versões para Favicon e Header.
4.  **Dev:** Construir a página `DesignSystem.tsx` implementando as seções acima.
