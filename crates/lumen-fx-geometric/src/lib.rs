//! # lumen-fx-geometric
//!
//! Geometric & lens correction — Cat 9 of the spec.
//!
//! Phase 1 ships [`Resize`], [`Crop`], and [`RotateOrtho`]. Future
//! milestones add: lens distortion correction (k1/k2/k3), perspective
//! warps, chromatic aberration removal, vignetting, and barrel/pincushion.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod crop;
pub mod resize;
pub mod rotate;

pub use crop::Crop;
pub use resize::Resize;
pub use rotate::RotateOrtho;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

/// Register every effect this crate provides.
pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(Resize))?;
    registry.register(Arc::new(Crop))?;
    registry.register(Arc::new(RotateOrtho))?;
    Ok(())
}

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
