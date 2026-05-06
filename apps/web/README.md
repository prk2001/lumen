# Web app (browser, WASM core)

The browser-targeted build of Lumen. Compiles a subset of the engine to
WebAssembly and renders the same React UI used by `apps/desktop` (via
`ui/`).

**Status:** planned for Phase 6. Browser perf, model size, and codec
licensing make a "full Lumen in the browser" non-trivial; we will start
with a viewer + share-link target before pursuing full editing.

## Why a separate app

Tauri apps run a WebView locally. The web target runs in the user's
browser without Tauri. Same UI, different shell, different capabilities
(no FFmpeg in WASM yet, smaller AI models, etc.).

## When the time comes

```bash
cd ~/Lumen/apps
pnpm create vite@latest web -- --template react-ts
# add wasm-pack-built `lumen-core-wasm` crate and wire it into Vite
```
