//! # lumen-ai
//!
//! Pure-Rust ONNX inference foundation for Lumen's Phase-2 AI effects
//! (denoise, upscale, face restore, …).
//!
//! This crate wraps [`tract-onnx`] — a pure-Rust ONNX runtime — plus a
//! handful of conveniences for converting between Lumen's [`Frame`] type
//! and the `1×3×H×W` float tensors most computer-vision models expect.
//!
//! ## Why tract instead of ONNX-Runtime
//!
//! The previous attempt used the `ort` crate (bindings to ONNX-Runtime
//! C). Its build script depended on a `ureq`/`tls_config` API combination
//! that broke under Rust 1.95, with no released fix at the time. tract is
//! pure Rust, has no system dependencies, and supports the ONNX op
//! subset Lumen needs for vision effects. The previous experimental code
//! is preserved at `crates/lumen-ai-experimental/` for reference but is
//! not built.
//!
//! ## Public surface
//!
//! - [`Session`] — a loaded, optimized, runnable ONNX model.
//! - [`load_session`] — load a `.onnx` file from disk.
//! - [`run_identity_check`] — end-to-end smoke test using a tiny
//!   embedded Identity model.
//! - [`ImageTensor`] — `1×C×H×W` (CHW) float tensor.
//! - [`from_frame_chw`] / [`to_frame`] — bridge to and from
//!   [`lumen_core::Frame`].
//! - [`run`] — execute a [`Session`] on an [`ImageTensor`].
//!
//! ## Layout choice
//!
//! CHW (`1×3×H×W`) is the default because it is the PyTorch / ONNX
//! convention and what virtually every vision model we ship will use.
//! Models that want HWC can permute axes themselves.
//!
//! [`Frame`]: lumen_core::frame::Frame

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, rust_2018_idioms)]

mod error;
mod session;
mod tensor;

pub use error::AiError;
pub use session::{load_session, run, run_identity_check, Session};
pub use tensor::{from_frame_chw, to_frame, ImageTensor};

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
