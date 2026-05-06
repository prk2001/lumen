//! # lumen-fx-color
//!
//! Color science & grading — Cat 5 of the spec.
//!
//! Phase 1 ships [`Saturation`]. Future milestones add primary wheels
//! (lift/gamma/gain), curves, secondaries (HSL/HSV), LUT loaders
//! (.cube/.3dl), and full OCIO view transforms (in concert with
//! `lumen-color`).

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod saturation;

pub use saturation::Saturation;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

/// Register every effect this crate provides.
pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(Saturation))?;
    Ok(())
}

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
