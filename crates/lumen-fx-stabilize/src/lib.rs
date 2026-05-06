//! # lumen-fx-stabilize
//!
//! Stabilization, rolling-shutter, motion correction — Cat 10 of the spec.
//!
//! Phase 1 ships [`Translate`] — a single-frame translation primitive
//! with edge-clamp fill. Real (multi-frame) stabilization in Phase 4
//! composes per-frame motion estimates with this warp.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod translate;

pub use translate::Translate;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

/// Register every effect this crate provides.
pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(Translate))?;
    Ok(())
}

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
