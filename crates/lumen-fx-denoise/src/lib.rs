//! # lumen-fx-denoise
//!
//! Noise reduction & cleanup — Cat 7 of the spec.
//!
//! Phase 1 ships [`GaussianDenoise`] — a baseline spatial filter that's
//! intentionally simple. Phase 2 adds bilateral and NL-means; Phase 2
//! also adds AI denoise via `lumen-fx-ai` + ONNX Runtime.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod bilateral;
pub mod gaussian;
pub mod median;

pub use bilateral::Bilateral;
pub use gaussian::GaussianDenoise;
pub use median::Median;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

/// Register every effect this crate provides.
pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(GaussianDenoise))?;
    registry.register(Arc::new(Bilateral))?;
    registry.register(Arc::new(Median))?;
    Ok(())
}

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
