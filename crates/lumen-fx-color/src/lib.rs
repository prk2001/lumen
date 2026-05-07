//! # lumen-fx-color
//!
//! Color science & grading — Cat 5 of the spec.
//!
//! Currently ships:
//!
//! - [`Saturation`] — luminance-axis saturation control.
//! - [`Lut3d`] — 3D `.cube` LUT loader with trilinear interpolation.
//! - [`PrimaryWheels`] — per-channel Lift / Gamma / Gain.
//! - [`Curves`] — per-channel piecewise-linear curves with optional
//!   luma-preserving master.
//!
//! Future milestones add secondaries (HSL/HSV) and full OCIO view
//! transforms (in concert with `lumen-color`).

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod curves;
pub mod duotone;
pub mod lut3d;
pub mod primary_wheels;
pub mod saturation;

pub use curves::Curves;
pub use duotone::Duotone;
pub use lut3d::Lut3d;
pub use primary_wheels::PrimaryWheels;
pub use saturation::Saturation;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

/// Register every effect this crate provides.
pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(Saturation))?;
    registry.register(Arc::new(Lut3d))?;
    registry.register(Arc::new(PrimaryWheels))?;
    registry.register(Arc::new(Curves))?;
    registry.register(Arc::new(Duotone))?;
    Ok(())
}

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
