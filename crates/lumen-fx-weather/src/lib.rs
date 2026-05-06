//! # lumen-fx-weather
//!
//! Atmospheric: dehaze, derain, defog, glare, smoke.
//!
//! Phase 1 ships [`DehazeDcp`] — a classical, no-AI dehaze based on
//! He et al.'s Dark Channel Prior (CVPR 2009). Future phases add
//! derain / defog / glare / smoke removal.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod dehaze;

pub use dehaze::DehazeDcp;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

/// Register every effect this crate provides.
pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(DehazeDcp))?;
    Ok(())
}

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
