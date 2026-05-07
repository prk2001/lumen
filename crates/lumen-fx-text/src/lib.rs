//! # lumen-fx-text
//!
//! Text/plate/object clarification (forensic & OCR-aware) — Cat 16.
//!
//! Phase 1 ships [`Clahe`] (contrast-limited adaptive histogram
//! equalization), the classical first pass for plate, sign, and text
//! clarification on CCTV-style imagery.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod clahe;

pub use clahe::Clahe;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

/// Register every effect this crate provides.
pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(Clahe))?;
    Ok(())
}

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
