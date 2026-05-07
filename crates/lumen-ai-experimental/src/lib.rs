//! # lumen-ai
//!
//! ONNX Runtime inference foundation for Lumen's Phase-2 AI effects
//! (denoise, upscale, face restore, …).
//!
//! This crate is a thin wrapper around [`ort`] (ONNX Runtime) plus a
//! handful of conveniences for converting between Lumen's [`Frame`] type
//! and the `1×3×H×W` float tensors that most computer-vision models
//! expect.
//!
//! ## What's here
//!
//! - [`Session`]    — wrapper around an [`ort::session::Session`] with
//!                    macOS-aware execution-provider routing (CoreML, CPU
//!                    fallback).
//! - [`load_session`] — load a `.onnx` file from disk into a [`Session`].
//! - [`run_identity_check`] — end-to-end smoke test that proves the
//!                    runtime is wired correctly without needing an
//!                    external model file.
//! - [`ImageTensor`] — `1×3×H×W` float32 tensor with conversions to and
//!                    from [`Frame`].
//! - [`registry`]    — Phase-1 stub for verifying / downloading models.
//!
//! ## Layout choice
//!
//! The tensor helpers ship the **CHW** layout (`1×3×H×W`) because that
//! is what the vast majority of ONNX vision models use (PyTorch
//! convention). Models that want HWC can permute axes themselves.
//!
//! [`Frame`]: lumen_core::Frame

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, rust_2018_idioms)]

pub mod error;
pub mod registry;
pub mod session;
pub mod tensor;

pub use error::AiError;
pub use session::{load_session, run_identity_check, Session};
pub use tensor::ImageTensor;

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
