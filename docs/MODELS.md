# Models

Lumen ships infrastructure for ONNX inference (via `tract-onnx` in
[`lumen-ai`](../crates/lumen-ai/)) but **does not bundle any pretrained
model weights** — partly to keep the repo small, partly because
real colorization / denoise / super-resolution models have varying
licensing terms.

This document tells you where to drop a model file and how to use it.

## What's implemented today

- **Heuristic colorization** via `lumen colorize --palette night|day|…`.
  No model needed, runs instantly. Five palettes:
  `night`, `day`, `sepia`, `cyan-orange`, `noir`.
  Uses the [`lumen-fx-color.duotone`](../crates/lumen-fx-color/src/duotone.rs)
  effect (Rec.709 luma-driven shadow/highlight color mapping).

- **Identity ONNX smoke test** via `lumen-ai`'s
  [`run_identity_check`](../crates/lumen-ai/src/session.rs). Loads a
  bundled 135-byte 1×3×4×4 Identity model, runs it on a synthetic
  tensor, asserts the output equals the input. Proves the
  inference pipeline works end-to-end without a "real" model.

- **`ImageTensor` plumbing** for converting `Frame` ↔ `1×3×H×W` CHW
  float tensor (alpha preserved, drop-into-model-ready).

## What needs a model file (planned)

Real ML colorization, denoise, and super-resolution are wired into
the architecture but require you to supply the ONNX file.

### Colorization (planned)

Replaces the heuristic `lumen colorize --palette night` with a true
content-aware colorization. Most published colorization models predict
**ab channels in CIE Lab space** from an L (luminance) input. Common
compatible models:

| Model | Size | Notes |
| --- | --- | --- |
| [DDColor](https://github.com/piddnad/DDColor) | ~600 MB | Best quality. Apache-2.0 (research-only weights). |
| [SIGGRAPH 2017 Colorization](https://richzhang.github.io/colorization/) | ~135 MB | Classical baseline. CC-BY-NC. |
| [Deep-Coloring](https://github.com/junyanz/interactive-deep-colorization) | ~50 MB | Smaller; lower quality. |

Drop the converted ONNX file at `models/colorize.onnx` and run:

```bash
# (planned, not yet wired)
lumen colorize --input X.jpg --output Y.png --model models/colorize.onnx
```

The CLI will lift the input to scene-linear, convert to L, run the
model, combine L + ab, convert Lab → RGB, encode.

### AI denoise (planned)

Drop a NAFNet / Restormer / DnCNN ONNX into `models/denoise.onnx`
and call:

```bash
# (planned)
lumen ai-denoise --input X.png --output Y.png --model models/denoise.onnx
```

### AI super-resolution (planned)

ESRGAN / Real-ESRGAN exported to ONNX, dropped in
`models/upscale.onnx`:

```bash
# (planned)
lumen ai-upscale --input X.png --output Y.png --model models/upscale.onnx --scale 4
```

## How to convert a PyTorch / TF model to ONNX

```bash
# PyTorch
import torch
torch.onnx.export(model, dummy_input, "out.onnx",
                  opset_version=13, input_names=["input"], output_names=["output"])

# TensorFlow
python -m tf2onnx.convert --saved-model SAVED_MODEL_DIR --output out.onnx --opset 13
```

Once exported, sanity-check that tract can load it:

```bash
cargo run --bin cli -- ai-check --model your-model.onnx   # planned subcommand
```

## Why ONNX (and tract specifically)

`tract-onnx` is **pure Rust, no system dependencies, no Python at
runtime**. It supports a useful subset of ONNX (most CV models work).
Trade-off vs. ONNX Runtime / PyTorch:

| | tract | ort (ONNX Runtime) | torch (PyTorch) |
| --- | --- | --- | --- |
| Rust integration | ✅ native | ✅ via `ort` crate | ⚠ via `tch`, harder |
| System deps | none | bundled DLLs | system PyTorch |
| GPU | no (yet) | CUDA / CoreML / DX | full GPU |
| Build time | fast | slower (downloads) | slowest |
| WASM support | yes | no | no |

We tried `ort` 2.0.0-rc earlier; the rc had a build break on Rust
1.95 (ureq tls_config API change). `tract` is the right call for now;
when we want GPU we can swap to `ort` for AI-heavy effects without
changing the public API of `lumen-ai`.

## Reproducibility

Project files (`*.lumenproj`) carry a
`models: { id: "blake3:..." }` map. When you ship a project that
depends on a particular model, the BLAKE3 hash is pinned so a
re-render years later produces bit-identical output (given the same
model file).

## License

Lumen is Apache-2.0. Model weights you supply are governed by their
own licenses — check each model's terms before redistributing.
