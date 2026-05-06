//! # lumen-io
//!
//! Input and output for Lumen.
//!
//! **Phase 1 scope** (this milestone): still images via the `image` crate
//! — PNG, JPEG, TIFF, WebP, BMP. Probe + decode + encode round-trips.
//!
//! **Coming in Milestone 1.1.b**: FFmpeg-backed video decode/encode for
//! H.264/H.265/AV1/ProRes/MOV/MP4/MKV.
//!
//! **Coming later**: RAW (CR2/NEF/ARW/DNG) via `rawloader`/`libraw`,
//! HEIF/HEIC via `libheif-rs`, AVIF via `libavif-rs`, JXL via `jpegxl-rs`,
//! image sequences (DPX/EXR), CCTV-vendor formats (Cat 1.2 of spec).

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod hash;
pub mod probe;
pub mod still;

pub use hash::hash_file;
pub use probe::{probe, probe_path};
pub use still::{decode_image, encode_image, ImageEncodeOptions};

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
