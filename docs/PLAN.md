# Build Plan

This is the phased roadmap for taking Lumen from scaffolding to a
production-ready enhancement suite. It is **opinionated** — solo
development demands ruthless prioritization.

> **Reality check.** Topaz Video AI ships ~10 effects with a team of
> dozens of engineers. DaVinci Resolve has hundreds of contributors.
> One person cannot match those products in any timeframe; we *can*
> ship something credible in narrow lanes (forensic clarification,
> CCTV restoration, batch enhancement) within 12–18 months and
> grow outward.

## Status as of this commit

| Phase | Status |
|---|---|
| 0 — Scaffold | ✅ Done |
| 1 — Ingest + preview + export | ✅ Done (still images; FFmpeg in progress) |
| 2 — AI restoration | 🟡 Infrastructure stashed (`lumen-ai-experimental/`) — `ort` 2.0.0-rc has a Rust 1.95 build issue, revisit once upstream releases compatible RC |
| 3 — Color + masking | 🟡 Partial (LUTs deferred; rect mask shipped) |
| 4 — Motion + temporal | 🟡 Partial (translate + motion blur shipped; full stabilize/RIFE deferred) |
| 5–6 — Forensic / cloud / SDK | ⏳ Future |

Effect coverage: **18 effects across 13 of 30 spec categories**.

## Phase 0 — Scaffold *(this session)*

**Goal:** Workspace compiles. Architecture committed. Roadmap clear.

- [x] Rust workspace with 35 crates mapped to spec categories
- [x] Workspace `Cargo.toml`, shared deps, profiles
- [x] `README`, `ARCHITECTURE`, `PLAN`, `FEATURES`, `ROADMAP`
- [x] `.gitignore`, `LICENSE` (Apache-2.0)
- [ ] `cargo check --workspace` passes
- [ ] Initial git commit
- [ ] Tauri desktop shell scaffolded (renders hello-world)
- [ ] CI: GitHub Actions running `cargo check` + `cargo clippy`

**Exit criteria:** `cargo check --workspace` is green; `cli` and `server`
binaries print version strings; first commit pushed.

## Phase 1 — Ingest + preview + export *(months 1–3)*

**Goal:** End-to-end with one happy path. Open a file, see frames,
apply one trivial effect, export.

### Milestone 1.1 — `lumen-io` decoders (real)

- FFmpeg decode for H.264/H.265/AV1/ProRes/VP9/MOV/MP4/MKV
- Image decode for JPEG/PNG/TIFF/WebP via `image` crate
- RAW decode (CR2/NEF/ARW/DNG) via `rawloader`
- HEIC via `libheif-rs`; AVIF via `libavif-rs`; JXL via `jpegxl-rs`
- Probe API: returns codec, dims, fps, duration, bit depth, color space
- Frame iterator that yields `Frame` (RGBA float16) on demand

**Acceptance:** Read 10 sample files of each format; assert metadata
matches FFprobe; pixel-checksum matches `ffmpeg -f rawvideo` reference.

### Milestone 1.2 — `lumen-core` pipeline

- `Effect` trait, `Node`, `Graph`, `Scheduler`
- Topological executor, dirty-flag tracking
- `Project` serialization (`.lumenproj` JSON)
- Frame cache (LRU, content-addressed via BLAKE3)

**Acceptance:** Construct a 3-node graph in code, run it on a 10s clip,
output written to disk. Re-running with no param changes returns from
cache instantly.

### Milestone 1.3 — `lumen-export` writer

- FFmpeg-backed encode for H.264/H.265/ProRes/PNG-sequence/MP4/MOV
- Two-pass + CRF + quality-target presets
- Per-codec parameter validation (`lumen-qa`)

**Acceptance:** Round-trip a 1080p H.264 clip; PSNR ≥ 35 dB at default
H.264 CRF 23.

### Milestone 1.4 — Trivial effects

- `fx-exposure::brightness_contrast` (CPU + GPU shader)
- `fx-color::saturation_hue` (CPU + GPU shader)
- `fx-sharpen::unsharp_mask` (CPU + GPU shader)

Each has CPU and `wgpu` paths, with test parity.

### Milestone 1.5 — Desktop shell

- Tauri 2 + React app
- Open file → preview viewer → effect param panel → export button
- Single-track timeline (no multi-track yet)

**Acceptance:** End-to-end demo: open MP4, drag exposure slider, see
preview update in <100 ms, click Export, file written.

## Phase 2 — AI restoration *(months 4–6)*

**Goal:** Three real AI effects ship with bundled ONNX models.

### Milestone 2.1 — `lumen-ai` inference engine

- ONNX Runtime via `ort` crate with EP routing
- Model registry (`models/registry/*.json` manifests)
- Hash-pinned downloads, BLAKE3 verified
- Tile-based inference for large frames
- Per-model VRAM budget; auto-fallback to CPU EP

### Milestone 2.2 — AI denoise

- Wrap a public restoration model (DnCNN / Restormer / NAFNet)
- Strength slider, tile size auto-tuned
- Goes in `fx-denoise`

### Milestone 2.3 — AI upscale

- 2× / 4× super-resolution using a public ESRGAN-class model
- Goes in `fx-upscale`

### Milestone 2.4 — Face restoration

- Detection (RetinaFace ONNX) → restoration (GFPGAN / CodeFormer-class)
- ROI-only path: face boxes inferred, masked into source
- Goes in `fx-face`

**Acceptance:** Three demo videos showing visible improvement. Each
effect has cached output and resumable runs.

## Phase 3 — Color grading + masking *(months 7–9)*

**Goal:** Actually-usable color and selective application.

- Primary wheels (lift / gamma / gain) — `fx-color`
- LUT loading (.cube, .3dl) — `fx-color`
- HSV / HSL secondaries — `fx-color`
- Curves (per-channel + luma) — `fx-color`
- ROI masks: rect / polygon / freehand / AI segmentation — `fx-mask`
- Mask track keyframing — `lumen-workflow`
- OCIO config support end-to-end — `lumen-color`

**Acceptance:** Replicate a basic Resolve-style grade on a test clip,
selective skin-tone protection via mask.

## Phase 4 — Motion + temporal *(months 10–12)*

- Stabilization (3-DoF + rolling-shutter) — `fx-stabilize`
- Frame interpolation (RIFE-class ONNX) — `fx-temporal`
- Deflicker (CCTV-typical) — `fx-temporal`
- Optical flow API in `lumen-gpu` — used by stabilize/interpolate/denoise

**Acceptance:** Stabilize a handheld 4K clip, slow 60→240 fps without
ghosting on a benchmark scene.

## Phase 5 — Forensic, audio, collaboration *(year 2)*

- **Forensic / Cat 16 / Cat 18:** plate clarification, dehaze, derain,
  CCTV-specific preset chains.
- **Audio (Cat 21):** wrap an audio model (e.g. Demucs/voicefixer
  derivatives) for noise removal & dialog isolation.
- **Cloud (Cat 24/27):** server-side rendering, share-by-link,
  comment threads on the project graph.
- **Authentication (Cat 22):** C2PA on export, chain-of-custody log,
  signed plugins.

## Phase 6 — Plugin SDK + distribution *(year 2+)*

- Stable plugin ABI (C-ABI surface around `Effect`)
- Lua + Python + JS scripting in `lumen-api`
- Marketplace + signing infrastructure
- Auto-update via `tauri-plugin-updater`
- Marketing site, docs site, sample projects

## Cross-cutting tracks (always on)

| Track          | Owner crate(s)            | Tasks                                  |
| -------------- | ------------------------- | -------------------------------------- |
| Performance    | `lumen-perf`, `lumen-gpu` | Profiling, benchmarks, regression gates|
| QA             | `lumen-qa`                | Golden-frame tests, fuzzing, metrics   |
| Docs           | `docs/`                   | Per-phase user + API docs              |
| CI             | `.github/workflows/`      | Build matrix, model-cache caching      |
| Telemetry      | `lumen-perf`              | Opt-in error reporting + perf metrics  |

## Risk register

| Risk                                              | Mitigation                                        |
| ------------------------------------------------- | ------------------------------------------------- |
| ONNX EP coverage is uneven across platforms       | CPU EP everywhere as guaranteed fallback          |
| FFmpeg licensing varies (LGPL vs GPL builds)      | Default LGPL build; offer GPL via separate target |
| Tauri 2 is new — APIs may shift                   | Pin minor; isolate Tauri-specific code in `apps/` |
| Solo bandwidth — 30 categories is a lot           | Phased exits; each phase is independently useful  |
| RAW codec support is a long tail                  | Lean on `rawloader`/`libraw` + community samples  |

## Definition of "credible v1"

Lumen is "credibly v1" when, on a representative consumer machine:

1. Open any common video format (mp4/mov/mkv with H.264/265/AV1).
2. Apply 5 baseline effects (denoise, upscale, sharpen, color, stabilize)
   with both CPU and GPU paths.
3. Export to H.264/H.265/ProRes/PNG-sequence at full quality.
4. Project file round-trips: same `.lumenproj` + media → identical output.
5. Open a CCTV clip and run a "forensic clarification" preset chain.
6. Programmable via CLI and Python plugin.

That is the Phase-4 exit and the first version we cut a real release on.
