# Public Roadmap

A trimmed, milestone-only view of [`PLAN.md`](PLAN.md). This is the version
suitable for outside readers, status pages, and pitch decks.

## v0.0 — Scaffold *(now)*

- Workspace compiles
- 35-crate skeleton committed
- Architecture, plan, and feature map documented

## v0.1 — "Open it, see it, save it"

- FFmpeg-backed decode for the common formats
- Single-track timeline preview at 1080p
- 3 trivial effects (brightness/contrast, saturation, unsharp mask)
- Export to H.264 / H.265 / ProRes / PNG-sequence
- Tauri desktop shell on macOS, Windows, Linux

## v0.2 — "AI that helps"

- AI denoise
- AI 2× / 4× upscale
- Face restoration

## v0.3 — "Color you can grade with"

- Primary wheels, curves, secondaries
- LUT support (`.cube`, `.3dl`)
- ROI masks (rect, polygon, AI segmentation)
- OCIO end-to-end

## v0.4 — "Motion handled"

- 3-DoF stabilization + rolling-shutter
- Frame interpolation
- Deflicker

## v0.5 — "Forensic & audio"

- Plate / text clarification, dehaze, derain
- Audio NR + dialog isolation
- C2PA on export, chain-of-custody log

## v0.6 — "Cloud + plugins"

- Cloud render with shareable links
- Stable plugin ABI
- Lua / Python / JS plugin host

## v1.0 — "Credible release"

Definition of done lives in `PLAN.md` § *Definition of "credible v1"*.

## Beyond v1.0

- Mobile clients
- Real-time collaborative editing
- Integrated marketplace
- Hardware partnerships (capture cards, NAS targets)
