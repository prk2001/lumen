#!/usr/bin/env bash
# One-shot stub generator for the Lumen workspace.
# Creates Cargo.toml + src/lib.rs for every crate.
# Idempotent: safe to re-run; will overwrite stubs but never source code we add later.

set -euo pipefail

LUMEN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LUMEN_ROOT"

# Crate name -> { category number (or "infra"/"bin"), one-line description }
# Format: name|kind|description
crates=(
  "lumen-core|infra|Core types, pipeline DAG, error model, project file format"
  "lumen-io|cat1|Input handling: FFmpeg, RAW, HEIF/HEIC, AVIF, JPEG XL, TIFF, EXR, DPX, ProRes"
  "lumen-color|infra|OpenColorIO, color science, transforms, ICC profiles, gamuts"
  "lumen-gpu|infra|wgpu compute kernels and render pipelines (Vulkan/Metal/DX12/WebGPU)"
  "lumen-ai|infra|ONNX Runtime inference, model registry, hardware EP routing"
  "lumen-playback|cat2|Playback engine: scrubbing, frame cache, A/B compare, timecode"
  "lumen-fx-exposure|cat4|Exposure, tone, dynamic range, HDR mapping"
  "lumen-fx-color|cat5|Color grading: primaries, secondaries, LUTs, OCIO views"
  "lumen-fx-sharpen|cat6|Sharpening and detail recovery"
  "lumen-fx-denoise|cat7|Spatial / temporal / chroma noise reduction"
  "lumen-fx-compression|cat8|Compression artifact removal: blocking, ringing, mosquito"
  "lumen-fx-geometric|cat9|Lens distortion, perspective, chromatic aberration"
  "lumen-fx-stabilize|cat10|Stabilization, rolling-shutter, motion correction"
  "lumen-fx-deblur|cat11|Deblurring, deconvolution, motion-blur removal"
  "lumen-fx-upscale|cat12|Super-resolution, AI upscaling, classical resamplers"
  "lumen-fx-temporal|cat13|Frame interpolation, retiming, deflicker"
  "lumen-fx-ai|cat14|AI-powered enhancement: generative restoration, style"
  "lumen-fx-face|cat15|Face detection, skin retouch, portrait enhancement"
  "lumen-fx-text|cat16|Text/plate/object clarification (forensic & OCR-aware)"
  "lumen-fx-mask|cat17|Masking, ROI selection, segmentation, matting"
  "lumen-fx-weather|cat18|Atmospheric: dehaze, derain, defog, glare, smoke"
  "lumen-fx-modalities|cat19|Multi-spectral, IR, UV, polarization, stereo"
  "lumen-measure|cat20|Measurement & analysis: scopes, metrics (PSNR/SSIM/VMAF/LPIPS)"
  "lumen-audio|cat21|Audio enhancement: NR, EQ, loudness, restoration"
  "lumen-auth|cat22|Authentication & integrity: chain-of-custody, hashing, C2PA"
  "lumen-workflow|cat23|Non-destructive editing: history, branches, presets"
  "lumen-collab|cat24|Collaboration & project management: shares, reviews, locks"
  "lumen-report|cat25|Reporting / visualization / presentation builders"
  "lumen-export|cat26|Export / delivery / encoding pipelines"
  "lumen-perf|cat27|Performance & hardware: GPU/CPU scheduling, memory, telemetry"
  "lumen-api|cat28|Extensibility: plugin host, scripting (Lua/Python/JS), REST/GraphQL"
  "lumen-platform|cat29|Platform & distribution: installers, updates, licensing"
  "lumen-qa|cat30|QA & monitoring: golden frames, fuzz, regression, telemetry"
  "lumen-cli|bin|Command-line interface binary"
  "lumen-server|bin|Cloud / SaaS HTTP server binary"
)

write_cargo_toml() {
  local name="$1" kind="$2" desc="$3" path="crates/$1/Cargo.toml"
  case "$kind" in
    infra)
      if [[ "$name" == "lumen-core" ]]; then
        cat > "$path" <<EOF
[package]
name = "$name"
description = "$desc"
version.workspace = true
edition.workspace = true
license.workspace = true
authors.workspace = true
repository.workspace = true

[dependencies]
thiserror.workspace = true
serde.workspace = true
serde_json.workspace = true
uuid.workspace = true
tracing.workspace = true
EOF
      else
        cat > "$path" <<EOF
[package]
name = "$name"
description = "$desc"
version.workspace = true
edition.workspace = true
license.workspace = true
authors.workspace = true
repository.workspace = true

[dependencies]
lumen-core = { path = "../lumen-core" }
thiserror.workspace = true
tracing.workspace = true
EOF
      fi
      ;;
    cat*|bin)
      cat > "$path" <<EOF
[package]
name = "$name"
description = "$desc"
version.workspace = true
edition.workspace = true
license.workspace = true
authors.workspace = true
repository.workspace = true

[dependencies]
lumen-core = { path = "../lumen-core" }
thiserror.workspace = true
tracing.workspace = true
EOF
      ;;
  esac

  # Add bin section for binaries
  if [[ "$kind" == "bin" ]]; then
    cat >> "$path" <<EOF

[[bin]]
name = "${name#lumen-}"
path = "src/main.rs"
EOF
  fi
}

write_lib_rs() {
  local name="$1" kind="$2" desc="$3" path="crates/$1/src/lib.rs"
  cat > "$path" <<EOF
//! # $name
//!
//! $desc
//!
//! Status: scaffolding stub. See \`docs/PLAN.md\` for the implementation roadmap.

#![forbid(unsafe_op_in_unsafe_fn)]

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
EOF
}

write_main_rs() {
  local name="$1" path="crates/$1/src/main.rs"
  case "$name" in
    lumen-cli)
      cat > "$path" <<'EOF'
//! Lumen CLI entry point.
//!
//! This is a placeholder stub. The real CLI will dispatch to `lumen-core`
//! pipelines and stream results to stdout / files.

fn main() {
    println!(
        "lumen v{} — CLI scaffold. See `docs/PLAN.md` for milestones.",
        env!("CARGO_PKG_VERSION")
    );
}
EOF
      ;;
    lumen-server)
      cat > "$path" <<'EOF'
//! Lumen cloud / SaaS server entry point.
//!
//! Stub — will host the same pipelines as desktop, exposed over HTTP/gRPC.

fn main() {
    println!(
        "lumen-server v{} — server scaffold. See `docs/PLAN.md` for milestones.",
        env!("CARGO_PKG_VERSION")
    );
}
EOF
      ;;
  esac
}

for entry in "${crates[@]}"; do
  IFS='|' read -r name kind desc <<<"$entry"
  write_cargo_toml "$name" "$kind" "$desc"
  write_lib_rs "$name" "$kind" "$desc"
  if [[ "$kind" == "bin" ]]; then
    write_main_rs "$name"
    # binaries also get a (mostly empty) lib.rs so workspace tooling works
  fi
done

echo "Generated stubs for ${#crates[@]} crates."
