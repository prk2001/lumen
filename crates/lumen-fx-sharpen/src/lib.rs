//! # lumen-fx-sharpen
//!
//! Sharpening & detail recovery — Cat 6 of the spec.
//!
//! Phase 1 ships [`UnsharpMask`]. Future milestones add high-pass, AI
//! detail recovery, frequency-separated sharpening, and per-channel
//! detail layers.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod unsharp_mask;

pub use unsharp_mask::UnsharpMask;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

/// Register every effect this crate provides.
pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(UnsharpMask))?;
    Ok(())
}

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
