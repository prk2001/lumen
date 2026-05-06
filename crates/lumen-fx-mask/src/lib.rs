//! # lumen-fx-mask
//!
//! Masking / Selection / ROI — Cat 17 of the spec.
//!
//! Phase 1 ships [`AlphaRect`] — a rectangular alpha mask. Future
//! milestones add: polygon, freehand, AI semantic segmentation, hair-
//! aware matting, mask track keyframing, and feathered edges.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod alpha_rect;

pub use alpha_rect::AlphaRect;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(AlphaRect))?;
    Ok(())
}

pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
