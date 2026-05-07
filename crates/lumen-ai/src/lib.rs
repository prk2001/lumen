//! # lumen-ai
//!
//! ONNX Runtime inference, model registry, hardware EP routing
//!
//! Status: scaffolding stub. See `docs/PLAN.md` for the implementation roadmap.

#![forbid(unsafe_op_in_unsafe_fn)]

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
// NOTE: lumen-ai experimental ONNX work stashed in lumen-ai-experimental/
// Reason: ort-sys 2.0.0-rc.12 build script fails on Rust 1.95 (ureq tls_config api change)
// Revisit when ort issues a compatible release or with load-dynamic feature.
