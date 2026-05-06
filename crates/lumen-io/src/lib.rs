//! # lumen-io
//!
//! Input handling: FFmpeg, RAW, HEIF/HEIC, AVIF, JPEG XL, TIFF, EXR, DPX, ProRes
//!
//! Status: scaffolding stub. See `docs/PLAN.md` for the implementation roadmap.

#![forbid(unsafe_op_in_unsafe_fn)]

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
