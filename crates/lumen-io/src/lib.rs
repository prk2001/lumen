//! # lumen-io
//!
//! Input and output for Lumen.
//!
//! **Always available**:
//! * Still images via the `image` crate — PNG, JPEG, TIFF, WebP, BMP.
//! * Camera RAW via `rawloader` (pure Rust) — CR2/CR3, NEF, ARW, DNG,
//!   RAF, ORF, RW2, PEF, SRW, IIQ, 3FR, X3F, etc. See [`raw`].
//! * FFmpeg-backed video decode/encode for H.264/H.265/AV1/ProRes/MOV/
//!   MP4/MKV/WebM/etc. See [`video`].
//!
//! **Behind Cargo features (off by default)** — these need system
//! libraries to build, so they're opt-in:
//! * [`heif`] — `--features heif` requires `libheif`.
//! * [`jxl`]  — `--features jxl`  requires `libjxl`.
//! * [`avif`] — `--features avif` requires `libdav1d` (it forwards to
//!   `image/avif-native`).
//!
//! When a feature is off, the corresponding `decode_*` / `probe_*`
//! function still exists and returns a clear
//! [`Error::UnsupportedFormat`](lumen_core::Error::UnsupportedFormat).
//! This keeps the fallback chain in [`probe::probe_path`] /
//! [`still::decode_image`] uniform — callers don't have to cfg-gate
//! their code.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod avif;
pub mod hash;
pub mod heif;
pub mod jxl;
pub mod probe;
pub mod raw;
pub mod still;
pub mod video;

pub use avif::{decode_avif, probe_avif};
pub use hash::hash_file;
pub use heif::{decode_heif, probe_heif};
pub use jxl::{decode_jxl, probe_jxl};
pub use probe::{probe, probe_path};
pub use raw::{decode_raw, probe_raw};
pub use still::{decode_image, decode_still, encode_image, ImageEncodeOptions};
pub use video::{decode_video_frame, decode_video_range, probe_video, VideoProbe};

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
