//! # lumen-export
//!
//! Export / delivery / encoding pipelines.
//!
//! Phase 1 video encoder. See [`VideoEncoder`] for the public entry point;
//! it wraps an `ffmpeg-next` muxer + encoder pair into a frame-by-frame
//! interface that accepts any [`lumen_core::Frame`] and converts to the
//! encoder's pixel format internally via swscale.
//!
//! ## Build requirements
//!
//! `ffmpeg-next` is a thin wrapper over `ffmpeg-sys-next`, which uses
//! pkg-config to find the system FFmpeg shared libraries. On macOS with
//! Homebrew, ensure `pkg-config` is installed and the FFmpeg `.pc` files
//! are reachable, e.g.:
//!
//! ```sh
//! brew install pkg-config
//! export PKG_CONFIG_PATH=/usr/local/opt/ffmpeg/lib/pkgconfig:$PKG_CONFIG_PATH
//! ```
//!
//! Tested against system FFmpeg 8.1.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod encoder;

pub use encoder::{Codec, VideoEncoder, VideoEncoderOptions};

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
