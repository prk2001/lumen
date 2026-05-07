//! # lumen-fx-face
//!
//! Face detection, skin retouch, portrait enhancement — Cat 15 of the spec.
//!
//! Phase 1 ships [`SkinSmoothInRect`] — a rect-bounded skin smoother. The
//! rectangle stands in for the bounding box that real face detection will
//! produce in Phase 2 (via `lumen-fx-ai` + an ONNX model). The AI face
//! pipeline will eventually call into the same primitive.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod skin_smooth;

pub use skin_smooth::SkinSmoothInRect;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

/// Register every effect this crate provides.
pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(SkinSmoothInRect))?;
    Ok(())
}

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
