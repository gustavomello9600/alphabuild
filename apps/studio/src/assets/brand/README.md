# AlphaBuilder Studio — brand assets

Logos programáticos fornecidos pelo projeto e organizados por contexto de uso.
Os SVGs são a fonte de verdade; os ícones nativos do Tauri são gerados a partir do
`p2_prism_brand_mark.svg` renderizado em 1024×1024.

| Variante | Arquivo | Onde é usada |
| :--- | :--- | :--- |
| `p1_studio_neon_squircle` | SVG | Marca principal do shell escuro, empty state/welcome e top bar no tema dark. |
| `p2_prism_brand_mark` | SVG | Ícone do app (`icons/`), favicon, Compute Popover e contextos compactos de alto contraste. |
| `p3_structural_cad` | SVG | Slots vazios do Comparador e superfícies técnicas/CAD. |
| `p4a_minimal_dark` | SVG | Marca minimalista em chrome escuro, biblioteca e ícones inline pequenos. |
| `p4b_minimal_light` | SVG | Variante para `tcc-light` e superfícies claras. |
| `p5_layered_depth` | SVG | Cabeçalho do Estúdio generativo / co-design. |
| `p6_wordmark_dark` | SVG + PNG | Lockup completo (marca + “AlphaBuilder” + STUDIO + tagline) para README, release e materiais de apresentação. |

## Regras de uso

- Não recolora, não estica e não adiciona sombra às variantes; o SVG já contém os gradientes da marca.
- Use `BrandMark` (`src/components/BrandMark.tsx`) para toda marca dentro do app — ele alterna automaticamente entre as variantes dark/light.
- Tamanhos mínimos: 18 px para marcas squircle, 24 px para `p2`, 40 px para `p3`/`p5`. Para tamanhos menores que 18 px, use `p2`.
- O wordmark `p6` é para títulos/README/release; não é usado em botões ou chrome compacto.
- O PNG `p6_wordmark_dark.png` foi renderizado com as fontes locais do projeto para não depender de fontes do sistema.
