//! # lumen-fx-denoise
//!
//! Noise reduction & cleanup — Cat 7 of the spec.
//!
//! Phase 1 ships [`GaussianDenoise`] — a baseline spatial filter that's
//! intentionally simple. Phase 2 adds bilateral and NL-means; Phase 2
//! also adds AI denoise via `lumen-fx-ai` + ONNX Runtime.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod gaussian;

pub use gaussian::GaussianDenoise;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

/// Register every effect this crate provides.
pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(GaussianDenoise))?;
    Ok(())
}

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
