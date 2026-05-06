//! # lumen-fx-temporal
//!
//! Frame interpolation, retiming, deflicker — Cat 13 of the spec.
//!
//! Phase 1 ships [`MotionBlurDirectional`] — a still-image directional
//! motion blur primitive. Multi-frame interpolation and retiming arrive
//! in later phases.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod motion_blur;

pub use motion_blur::MotionBlurDirectional;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

/// Register every effect this crate provides.
pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(MotionBlurDirectional))?;
    Ok(())
}

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
