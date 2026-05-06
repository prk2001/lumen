# Architecture

This document captures the design decisions behind Lumen's structure.
For the *what to build when*, see [`PLAN.md`](PLAN.md).

## Goals

1. **Single core, many surfaces.** Desktop, CLI, web, and cloud all run the
   same pipeline graph compiled from one Rust core.
2. **Plugin-first from day one.** Built-in effects (`lumen-fx-*`) are
   indistinguishable from third-party plugins — they implement the same
   `Effect` trait and ship through the same registry.
3. **Non-destructive by default.** Every operation is a node in a DAG.
   Source pixels are never overwritten; render is always derived.
4. **GPU when it matters, CPU when it doesn't.** All hot paths can run on
   `wgpu` (Vulkan/Metal/DX12/WebGPU). Falls back to CPU automatically.
5. **Reproducible.** A `.lumenproj` plus the source media re-renders
   bit-identical output across machines, given the same model versions.

## Layered model

```text
┌────────────────────────────────────────────────────────────────────┐
│  Surfaces:  apps/desktop (Tauri)   apps/web (WASM)   crates/cli    │
│             apps/cloud (Axum)      plugins/ (Lua/Py/JS via api)    │
├────────────────────────────────────────────────────────────────────┤
│  Workflow:  workflow · collab · report · export · qa · api         │
├────────────────────────────────────────────────────────────────────┤
│  Effects:   fx-exposure · fx-color · fx-sharpen · fx-denoise · …   │
│             (16 fx-* crates, one per spec category 4–19)           │
├────────────────────────────────────────────────────────────────────┤
│  Domain:    measure · audio · auth · platform · perf               │
├────────────────────────────────────────────────────────────────────┤
│  Engines:   io · color · gpu · ai · playback                       │
├────────────────────────────────────────────────────────────────────┤
│  Core:      lumen-core (types, DAG, project, errors)               │
└────────────────────────────────────────────────────────────────────┘
```

A crate **only depends on layers strictly below it.** This keeps the
dependency graph acyclic and lets us release any layer independently.

## The pipeline DAG

The unit of work is a **`Node`** — an instance of an `Effect` with bound
input and output sockets.

```text
   ┌─────────┐     ┌──────────┐     ┌──────────┐     ┌─────────┐
   │ Source  │────▶│ Denoise  │────▶│ Upscale  │────▶│ Export  │
   │ (io)    │     │ (fx-*)   │     │ (fx-*)   │     │         │
   └─────────┘     └──────────┘     └──────────┘     └─────────┘
        │                                                 ▲
        │           ┌──────────┐     ┌──────────┐         │
        └──────────▶│ Stabilize│────▶│ Color    │─────────┘
                    │ (fx-*)   │     │ (fx-*)   │
                    └──────────┘     └──────────┘
```

Key types live in `lumen-core`:

| Type        | Purpose                                          |
| ----------- | ------------------------------------------------ |
| `Frame`     | A 2D buffer + color space + timestamp            |
| `Clip`      | An ordered sequence of `Frame`s                  |
| `Asset`     | A resolved input source (file, stream, sequence) |
| `Project`   | A serializable graph + asset list + settings     |
| `Effect`    | Trait every node implements                      |
| `Node`      | An `Effect` instance with bound parameters       |
| `Graph`     | A DAG of `Node`s                                 |
| `Scheduler` | Topological executor; GPU+CPU aware              |
| `Cache`     | Frame-level cache, content-addressed             |

## The `Effect` trait (sketch)

```rust
pub trait Effect: Send + Sync {
    fn metadata(&self) -> EffectMetadata;
    fn parameters(&self) -> &[ParamSpec];
    fn capabilities(&self) -> Capabilities;       // GPU? streaming? in-place?
    fn process(
        &self,
        ctx: &mut Context,
        inputs: &[FrameRef],
        params: &ParamValues,
    ) -> Result<Frame, Error>;
}
```

Built-in effects (`lumen-fx-*`) and third-party plugins implement the same
trait — there is no inner/outer API.

## Color management

`lumen-color` owns the **single source of truth for color space**.
Internal compute happens in **scene-linear float32** unless an effect
opts in to a different working space. OpenColorIO drives input
transforms (IDT), view transforms (RRT/ODT), and exports.

| Stage        | Default                                    |
| ------------ | ------------------------------------------ |
| Decode       | Linearize per metadata (sRGB / Rec.709 / …)|
| Working space| ACEScg float32                             |
| Display      | Per-monitor ICC, view-transform applied    |
| Export       | User-selected ODT or LUT                   |

## GPU strategy

`lumen-gpu` exposes a single `GpuContext` backed by `wgpu`. Compute
shaders are written in **WGSL** and compiled at runtime. Each `fx-*`
crate ships one or more WGSL shaders for its hot paths and a
CPU fallback for environments without a GPU (CI, headless servers,
older hardware).

**Why wgpu**: one shader source compiles to Vulkan, Metal, DX12, and
WebGPU. We target all four from day one.

Heavier AI work goes through `lumen-ai` (ONNX Runtime), which routes to
**CUDA**, **CoreML**, **DirectML**, **WebGPU**, or **CPU** based on
hardware probe and model needs.

## Project file format (`.lumenproj`)

Human-readable JSON document, schema-versioned:

```jsonc
{
  "schema": "lumenproj/v1",
  "id": "01JCZ8K2P5R7M3N1Q8…",
  "created": "2026-05-06T18:04:00Z",
  "assets": [ { "id": "…", "uri": "file:///…", "hash": "blake3:…" } ],
  "graph": {
    "nodes": [
      { "id": "n0", "kind": "io.source",     "params": { "asset": "…" } },
      { "id": "n1", "kind": "fx-denoise.dn", "params": { "strength": 0.4 } },
      { "id": "n2", "kind": "io.export",     "params": { "codec": "h265" } }
    ],
    "edges": [ ["n0", "n1"], ["n1", "n2"] ]
  },
  "history": [ /* immutable, append-only */ ],
  "presets":  [ /* user-saved param bundles */ ],
  "models":   { "denoise.dn-v3": "blake3:…" }
}
```

The `models` map pins exact ONNX hashes so a project re-renders
identically forever.

## Concurrency model

- **Per-node parallelism** — independent branches of the DAG run on
  different threads / GPUs.
- **Tile-level parallelism** — large frames split into tiles processed
  by `rayon` (CPU) or shader workgroups (GPU).
- **Frame-level prefetch** — playback engine prefetches N frames ahead.
- **No global GIL** — Rust-only, plus `tokio` for I/O-bound work.

## Logging & telemetry

Everything goes through `tracing`. Three sinks:

1. **stdout** — pretty in dev, JSON in prod.
2. **`.lumen-cache/sessions/<id>.log`** — per-session.
3. **Optional cloud sink** — opt-in remote diagnostics.

No PII or media content is ever logged. See `docs/PRIVACY.md`
(planned).

## Security & integrity

`lumen-auth` (Cat 22) provides:

- BLAKE3 content hashing of all inputs and outputs.
- C2PA-style provenance attached to exports.
- Signed plugin manifests (Ed25519).
- Optional chain-of-custody log for forensic workflows.

## What we deliberately don't do

- We don't reimplement FFmpeg/OpenCV/OCIO — we link them.
- We don't ship a custom UI toolkit — Tauri + React + a small shared
  component library.
- We don't write our own GPU driver layer — `wgpu` does that.
- We don't train models in-app — Python-side training, ONNX export,
  Rust-side inference only.

## Out of scope (for now)

- Native iOS / Android apps (post-Phase 6).
- Real-time multi-user editing (post-Phase 5).
- Custom hardware (capture cards, scopes).

## ADRs

Architecture Decision Records live in `docs/architecture/adr-NNNN-*.md`.
Open questions to capture as we land them:

- `adr-0001` — Rust + Tauri vs. Electron vs. Qt
- `adr-0002` — wgpu vs. CUDA-direct
- `adr-0003` — ONNX Runtime vs. native PyTorch (`tch`)
- `adr-0004` — `.lumenproj` JSON vs. binary format
- `adr-0005` — Plugin scripting language(s)
