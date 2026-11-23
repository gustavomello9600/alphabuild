# GEMINI.md

## Project Overview

AlphaBuilder combines gradient-free structural topology optimization (SIMP/FEniTop + FEniCSx), a v3.3 PyTorch/MONAI neural architecture, distributed data harvest, and a new cross-platform desktop studio.

The repository is composed of:
- `alphabuilder/src/` — scientific core v3.3 (FEM, MCTS, reward, neural network, storage). Read-only for the Studio.
- `crates/ab-core/` — pure Rust core: v3.3 reward math, `.npy`/`COO1` codecs, ABMESH01/ABWIRE01 protocols, read-only SQLite, surface-culled mesher + AO, BFS topology, strain, exporters.
- `apps/studio/` — AlphaBuilder Studio desktop app (Tauri v2 + React 19 + Zustand + Three.js vanilla).
- `alphabuilder/server/` — `alphabuilder-engine` FastAPI/WebSocket service, thin and versioned over `alphabuilder.src.*`.
- `tests/server/` — API v1, auth, ABWIRE01 and full session-WebSocket tests.

## Building and Running

### Scientific core / data generation
```bash
mamba env create -f environment.yml
mamba activate alphabuilder
mpirun -np 4 python run_data_harvest.py --strategy BEZIER --episodes 1
mpirun -np 4 python run_data_harvest.py --strategy FULL_DOMAIN --episodes 1
```

### Compute Engine
```bash
python -m alphabuilder.server --host 127.0.0.1 --port 8765 \
  --checkpoints-dir checkpoints --data-dir data
```
Remote access requires `--api-token` (or explicit `--allow-insecure-remote`).

### Desktop Studio
```bash
cd apps/studio
npm install
npm run tauri dev
```
Release: `npm run build && npm run tauri build`.

## Verification
```bash
cargo test -p ab-core
cargo clippy -p ab-core --all-targets -- -D warnings
cargo check --manifest-path apps/studio/src-tauri/Cargo.toml
pytest tests/unit/ -v
pytest tests/server/ -v
cd apps/studio && npm run typecheck && npm run test && npm run build
```

## Development Conventions

- Python formatting/linting with `ruff`; tests with `pytest`.
- Rust: `cargo fmt`, `cargo clippy -- -D warnings`; `ab-core` is `#![forbid(unsafe_code)]`.
- All JSON across Rust, Python and TypeScript is **100% snake_case**.
- Tensors cross Tauri IPC as `ABMESH01` and HTTP/WS as `ABWIRE01`; never dense JSON arrays.
- The desktop UI uses Three.js vanilla only (no `@react-three/fiber`, no `drei`, no Framer Motion).
- Local SQLite is always opened read-only (`SQLITE_OPEN_READ_ONLY` + `PRAGMA table_info`).
- FastAPI never performs inference/FEM/MCTS inline; all work runs on the single `ComputeWorker` thread.

## Brand assets

- Fonte: `apps/studio/src/assets/brand/` (SVG) + `p6_wordmark_dark.png`.
- Use `apps/studio/src/components/BrandMark.tsx` para alternar dark/light automaticamente.
- Ícones nativos/favicon: `p2_prism_brand_mark.svg` → `src-tauri/icons/` e `public/favicon.svg`.
- Guia de contexto e tamanhos mínimos: `apps/studio/src/assets/brand/README.md`.

## Key Files

- `specs/app_v4_spec.md`: normative Studio v4.0 specification.
- `crates/ab-core/src/reward.rs`: exact v3.3 reward constants and formulas.
- `crates/ab-core/src/codecs/`: `.npy`, `COO1`, `ABWIRE01`, `ABMESH01`.
- `apps/studio/src/engine3d/ViewportEngine.ts`: imperative Three.js controller.
- `apps/studio/src/types/contracts.ts`: canonical snake_case TS contracts.
- `alphabuilder/server/app.py`: FastAPI factory, auth, error envelopes.
- `alphabuilder/server/worker.py`: single-owner compute worker + sessions.
- `apps/studio/README.md`: local/remote execution details.
