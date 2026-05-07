//! # lumen-fx-modalities
//!
//! Multi-spectral, IR, UV, polarization, stereo — Cat 19 of the spec.
//!
//! Phase 1 ships [`ChannelIsolate`], which extracts a single channel
//! (R, G, B, alpha, or Rec.709 luma) and emits it as a grayscale RGB
//! image. True multi-spectral / IR / UV / polarization workflows
//! require source data we don't have in Phase 1; this effect is the
//! debug / inspection / IR-like-from-visible workhorse.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod channel_isolate;

pub use channel_isolate::ChannelIsolate;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

/// Register every effect this crate provides.
pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(ChannelIsolate))?;
    Ok(())
}

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
