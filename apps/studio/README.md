# AlphaBuilder Studio v4.0

Desktop cross-platform para inspecionar e editar estruturas AlphaBuilder v3.3.

## Requisitos

- Node.js 20+ e npm
- Rust stable + Cargo
- Linux: `libwebkit2gtk-4.1-dev`, `libappindicator3-dev`, `librsvg2-dev`, `patchelf`, `libsqlite3-dev`
- Opcional: Python 3.10+ com o Compute Engine (`alphabuilder/server`)

## Desenvolvimento

```bash
npm install
npm run tauri dev
```

O Vite sobe em `http://localhost:1420`; o binário Tauri abre a janela nativa.

## Build

```bash
npm run build          # typecheck + Vite
npm run tauri build    # instaladores nativos
```

## Compute Engine

Local (sidecar gerenciado pelo app ou manual):

```bash
python -m alphabuilder.server --host 127.0.0.1 --port 8765 \
  --checkpoints-dir checkpoints --data-dir data
```

Remoto (Tailscale/LAN/HTTPS):

```bash
python -m alphabuilder.server --host 0.0.0.0 --port 8765 \
  --api-token "<TOKEN>" --checkpoints-dir checkpoints --data-dir data
```

No Studio, abra o pill **Compute**, edite/selecione o perfil e conecte. O token é persistido em `profiles.json` com permissões restritas do usuário.

## Testes

```bash
npm run typecheck
npm run test
cargo test -p alphabuilder-studio
cargo clippy -p alphabuilder-studio --all-targets -- -D warnings
```

A fixture `src/lib/__fixtures__/abmesh_fixture.bin` é gerada por `cargo run -p ab-core --example abmesh_fixture -- <path>`.

## Estrutura

- `src/engine3d/` — ViewportEngine, VoxelMeshLayer, BlueprintLayer, OverlayLayer, BoundaryLayer, VoxelPicker, CameraRig.
- `src/components/` — TopBar, LibraryDrawer, LensToolbar, TimelineDock, Inspector/Studio/Compare e modais.
- `src/store/` — Zustand stores (app, compute, inspector, studio, compare).
- `src/lib/` — IPC tipado, parser zero-copy ABMESH01, Turbo, formatters.
- `src-tauri/` — comandos IPC, cache LRU, sidecar, ws_bridge e contratos snake_case.
