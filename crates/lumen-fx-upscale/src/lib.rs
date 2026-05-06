//! # lumen-fx-upscale
//!
//! Super-resolution & upscaling — Cat 12 of the spec.
//!
//! Phase 1 ships [`Bicubic`] — a classical resampler with a Mitchell
//! cubic kernel. Phase 2 adds AI super-resolution (ESRGAN-class) via
//! `lumen-fx-ai` + ONNX Runtime in `lumen-ai`.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod bicubic;

pub use bicubic::Bicubic;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(Bicubic))?;
    Ok(())
}

pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
