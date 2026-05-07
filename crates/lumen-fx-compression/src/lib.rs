//! # lumen-fx-compression
//!
//! Compression artifact removal: blocking, ringing, mosquito.
//!
//! Phase 1 ships [`Deblock`] — a simple post-process that softens
//! visible JPEG-style 8x8 block boundaries with a 1-D Gaussian. Phase 4
//! adds DCT-domain deblocking and AI artifact removal.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod deblock;

pub use deblock::Deblock;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

/// Register every effect this crate provides.
pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(Deblock))?;
    Ok(())
}

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
