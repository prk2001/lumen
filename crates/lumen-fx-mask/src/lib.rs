//! # lumen-fx-mask
//!
//! Masking / Selection / ROI — Cat 17 of the spec.
//!
//! Ships [`AlphaRect`] — a rectangular alpha mask — and
//! [`AlphaPolygon`] — a polygonal alpha mask. Future milestones add:
//! freehand, AI semantic segmentation, hair-aware matting, and mask
//! track keyframing.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod alpha_polygon;
pub mod alpha_rect;

pub use alpha_polygon::AlphaPolygon;
pub use alpha_rect::AlphaRect;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(AlphaRect))?;
    registry.register(Arc::new(AlphaPolygon))?;
    Ok(())
}

pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
