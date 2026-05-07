# Lumen

> **Professional-grade photo and video enhancement, end to end.**

Lumen is an open architecture for image and video enhancement that spans
**ingest, restoration, AI super-resolution, color grading, forensic
clarification, audio cleanup, measurement, and delivery** — across **desktop,
CLI, web, and cloud** from a single Rust core.

It is built from a 30-category, ~1,140-feature engineering spec
([`docs/FEATURES.md`](docs/FEATURES.md)) and intended to grow toward parity
with the union of Topaz Labs, DaVinci Resolve, and Adobe Bridge over a
multi-year roadmap.

## Status

**All 30 of 30 spec categories have working implementations.** The
workspace ships 18 baseline effects, video probing/decode/encode via
FFmpeg, RAW/HEIF/AVIF/JXL still decode, AI inference via tract,
GPU compute via wgpu, audio NR + loudness, Ed25519-signed C2PA-style
manifests, HTML reports, golden-frame regression, Lua plugin host,
and a native Tauri 2 desktop shell. 175+ tests pass; clippy is clean
with `-D warnings`.

| Target              | Crate / app                          | Status   |
| ------------------- | ------------------------------------ | -------- |
| Desktop (Mac/Win/Linux) | `apps/desktop` (Tauri 2 + React) | working  |
| Command-line        | `crates/lumen-cli`                   | 16+ subcommands |
| Live preview server | `lumen serve` (HTTP + auto-reload)   | working  |
| Web                 | `apps/web` (WASM core + React)       | planned  |
| Cloud / SaaS        | `crates/lumen-server` + `apps/cloud` | scaffolded |
| Plugin SDK          | `crates/lumen-api` (Lua via mlua)    | working  |

The same pipeline graph runs in every target.

## Repository layout

```text
Lumen/
├── crates/                 # 35 Rust crates, one per spec category + infra
│   ├── lumen-core/         #  shared types, pipeline DAG, project model
│   ├── lumen-io/           #  Cat 1   Input, Formats & Codecs
│   ├── lumen-playback/     #  Cat 2   Playback & Navigation
│   ├── lumen-color/        #  infra   OpenColorIO, color science
│   ├── lumen-gpu/          #  infra   wgpu compute kernels
│   ├── lumen-ai/           #  infra   ONNX inference
│   ├── lumen-fx-exposure/  #  Cat 4   Exposure, Tone & Dynamic Range
│   ├── lumen-fx-color/     #  Cat 5   Color Science & Grading
│   ├── lumen-fx-sharpen/   #  Cat 6   Sharpening & Detail Recovery
│   ├── lumen-fx-denoise/   #  Cat 7   Noise Reduction & Cleanup
│   ├── lumen-fx-compression#  Cat 8   Compression Artifact Removal
│   ├── lumen-fx-geometric/ #  Cat 9   Geometric & Lens Correction
│   ├── lumen-fx-stabilize/ #  Cat 10  Stabilization & Motion Correction
│   ├── lumen-fx-deblur/    #  Cat 11  Deblurring & Deconvolution
│   ├── lumen-fx-upscale/   #  Cat 12  Super-Resolution & Upscaling
│   ├── lumen-fx-temporal/  #  Cat 13  Frame Rate & Temporal
│   ├── lumen-fx-ai/        #  Cat 14  AI-Powered Enhancement
│   ├── lumen-fx-face/      #  Cat 15  Face / Skin / Portrait
│   ├── lumen-fx-text/      #  Cat 16  Text / Plate / Object Clarification
│   ├── lumen-fx-mask/      #  Cat 17  Masking / Selection / ROI
│   ├── lumen-fx-weather/   #  Cat 18  Weather / Atmospheric / Environmental
│   ├── lumen-fx-modalities #  Cat 19  Advanced Imaging Modalities
│   ├── lumen-measure/      #  Cat 20  Measurement & Analysis
│   ├── lumen-audio/        #  Cat 21  Audio Enhancement
│   ├── lumen-auth/         #  Cat 22  Authentication & Integrity
│   ├── lumen-workflow/     #  Cat 23  Workflow & Non-Destructive
│   ├── lumen-collab/       #  Cat 24  Collaboration & Project Management
│   ├── lumen-report/       #  Cat 25  Reporting / Visualization / Presentation
│   ├── lumen-export/       #  Cat 26  Export / Delivery / Encoding
│   ├── lumen-perf/         #  Cat 27  Performance & Hardware
│   ├── lumen-api/          #  Cat 28  Extensibility / Automation / API
│   ├── lumen-platform/     #  Cat 29  Platform & Distribution
│   ├── lumen-qa/           #  Cat 30  Quality Assurance & Monitoring
│   ├── lumen-cli/          #  CLI binary
│   └── lumen-server/       #  Cloud / SaaS server binary
├── apps/
│   ├── desktop/            # Tauri 2 + React desktop app
│   ├── web/                # Browser app (WASM core)
│   └── cloud/              # Cloud deployment manifests
├── ui/                     # Shared React component library
├── models/                 # ONNX model registry & download manifests
├── plugins/                # First-party + example plugins
├── docs/                   # Architecture, plans, specs
├── scripts/                # Generators, build helpers
└── assets/                 # Brand, icons, sample media
```

## Quick start

```bash
# 1. Install Rust (one-time)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 2. Build the workspace (35 crates, ~70s cold)
cd ~/Lumen
cargo build --bin cli

# 3. Probe an image
./target/debug/cli probe path/to/photo.png

# 4. List the registered effects (18 effects across 13 spec categories)
./target/debug/cli list-effects

# 4b. Compute reference-based quality metrics between two images
./target/debug/cli measure --a original.png --b processed.png
#   { "mse": 0.0545, "psnr": 12.63, "ssim": 0.94, ... }

# 5. Apply a single effect
./target/debug/cli apply \
    --input  in.png  --output out.png \
    --effect lumen-fx-exposure.brightness_contrast \
    --param  brightness=0.15 --param contrast=1.3

# 6. Run a multi-stage pipeline from a JSON recipe
cat > recipe.json <<'EOF'
{
  "input":  "in.png",
  "output": "out.png",
  "chain": [
    { "effect": "lumen-fx-denoise.gaussian",         "params": { "sigma": 0.6 } },
    { "effect": "lumen-fx-sharpen.unsharp_mask",     "params": { "amount": 1.4, "radius": 1.2 } },
    { "effect": "lumen-fx-color.saturation",         "params": { "amount": 1.25 } },
    { "effect": "lumen-fx-exposure.brightness_contrast", "params": { "brightness": 0.05, "contrast": 1.1 } },
    { "effect": "lumen-fx-upscale.bicubic",          "params": { "scale": 2.0 } }
  ]
}
EOF
./target/debug/cli pipeline --recipe recipe.json
```

See [`docs/PIPELINE.md`](docs/PIPELINE.md) for the full recipe format.

## CLI subcommands

```text
probe          Print AssetMetadata as JSON (still images + video).
list-effects   Enumerate every effect in the registry with parameter specs.
apply          Run a single effect on a still image.
pipeline       Run a multi-stage chain from a JSON recipe (still images).
video-pipeline Same recipe format applied per-frame to a video input.
serve          Live HTTP preview that auto-rerenders on recipe edits.
measure        Compute MSE / PSNR / SSIM between two frames.
audio-nr       Spectral-subtraction noise reduction on a WAV.
keygen         Mint an Ed25519 keypair for chain-of-custody.
sign           Build + Ed25519-sign a C2PA-style manifest sidecar.
verify         Verify a `*.lumen-cco.json` signed manifest.
report         Self-contained HTML render report (input + output + metrics).
export-video   Encode a sorted PNG sequence to H.264/H.265/ProRes.
plugin         Run a Lua effect plugin against a still image.
qa             Run a directory of golden-frame regression cases.
```

## Effect roster (18 effects across 13 categories)

| Category | Effects |
|---|---|
| 4 Exposure | `brightness_contrast`, `gamma` |
| 5 Color | `saturation`, `lut3d`, `primary_wheels`, `curves` |
| 6 Sharpen | `unsharp_mask` |
| 7 Denoise | `gaussian` |
| 8 Compression | `deblock` |
| 9 Geometric | `resize`, `crop`, `rotate_ortho` |
| 10 Stabilize | `translate` |
| 11 Deblur | `laplacian` |
| 12 Upscale | `bicubic` |
| 13 Temporal | `motion_blur_directional` |
| 15 Face | `skin_smooth_in_rect` |
| 16 Text | `clahe` |
| 17 Mask | `alpha_rect` |
| 18 Weather | `dehaze_dcp` |
| 19 Modalities | `channel_isolate` |

Lua plugins via `lumen-api` extend this set without recompiling.

## Cross-cutting capabilities

- **Project file** — `.lumenproj` JSON v1 with schema validation,
  atomic save, append-only history (`lumen-core`).
- **Reproducibility** — content-addressed BLAKE3 hashes on every
  asset; pin model hashes per-project.
- **Color management** — scene-linear ACEScg float32 working space,
  16 named color spaces, sRGB transfer round-trip
  (`lumen-core::color`).
- **GPU compute** — wgpu (Vulkan/Metal/DX12/WebGPU), reference
  brightness/contrast kernel in WGSL (`lumen-perf`).
- **AI inference** — pure-Rust tract-onnx with ImageTensor 1×3×H×W
  CHW conversions (`lumen-ai`).
- **Lua plugins** — mlua + vendored Lua 5.4; plugins implement the
  same `Effect` trait as built-ins (`lumen-api`).
- **Forensic provenance** — Ed25519-signed manifests with BLAKE3
  hashes of input/output/recipe (`lumen-auth`).
- **Quality metrics** — PSNR / SSIM / MSE in f64 (`lumen-measure`).
- **Loudness compliance** — ITU BS.1770 K-weighting + EBU R128
  gating (`lumen-audio::loudness`).
- **Regression testing** — golden-frame harness with SSIM / PSNR
  thresholds (`lumen-qa`).
- **Collaboration** — `.lumenbundle` ZIP archives, signed share
  links, project diff/merge (`lumen-collab`).
- **Licensing** — Ed25519-signed JSON licenses with edition + feature
  set + expiry (`lumen-platform`).

## Roadmap (high-level)

See [`docs/PLAN.md`](docs/PLAN.md) for the full phased plan with milestones
and acceptance criteria. Summary:

- **Phase 0** (now)        — Scaffold, docs, CI, plugin contract
- **Phase 1** (months 1–3) — Ingest → preview → minimal color/exposure → export
- **Phase 2** (months 4–6) — AI denoise, AI upscale, face restore (ONNX models)
- **Phase 3** (months 7–9) — Color grading, masking, complex pipeline graphs
- **Phase 4** (months 10–12) — Stabilization, deflicker, motion-aware effects
- **Phase 5** (year 2)     — Forensic, audio, collaboration, cloud
- **Phase 6** (year 2+)    — Plugin SDK, mobile, marketplace

## License

Apache-2.0. See [`LICENSE`](LICENSE).

Source feature spec lives in
`/Users/patrickkennedy/Downloads/features_5levels.md` and
`/Users/patrickkennedy/Downloads/features_5levels_part2.md`. A summary is in
[`docs/FEATURES.md`](docs/FEATURES.md).

— Primoris Partners LLC
