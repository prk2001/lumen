//! # lumen-fx-deblur
//!
//! Deblurring, deconvolution, motion-blur removal — Cat 11 of the spec.
//!
//! Phase 1 ships [`LaplacianSharpen`], a Difference-of-Gaussians edge
//! enhancer that compensates mild defocus blur. Full Wiener and
//! Richardson–Lucy deconvolution land in Phase 4.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod laplacian;
pub mod wiener;

pub use laplacian::LaplacianSharpen;
pub use wiener::Wiener;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

/// Register every effect this crate provides.
pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(LaplacianSharpen))?;
    registry.register(Arc::new(Wiener))?;
    Ok(())
}

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
