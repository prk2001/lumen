//! # lumen-fx-exposure
//!
//! Exposure, tone, and dynamic range adjustments — Cat 4 of the spec.
//!
//! Phase 1 ships [`BrightnessContrast`]. Future milestones add tone
//! curves, log/linear conversions, HDR-to-SDR tone mapping, gamma,
//! and per-zone exposure (zone system).

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod brightness_contrast;

pub use brightness_contrast::BrightnessContrast;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

/// Register every effect this crate provides into the supplied registry.
pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(BrightnessContrast))?;
    Ok(())
}

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
