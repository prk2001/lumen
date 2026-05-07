//! # lumen-fx-upscale
//!
//! Super-resolution & upscaling — Cat 12 of the spec.
//!
//! Phase 1 ships three classical resamplers: [`Bicubic`] (Mitchell-
//! Netravali, B = C = 1/3), [`CatmullRom`] (sharper bicubic, B = 0,
//! C = 1/2), and [`Lanczos`] (windowed-sinc with selectable lobes).
//! Phase 2 adds AI super-resolution (ESRGAN-class) via `lumen-fx-ai`
//! + ONNX Runtime in `lumen-ai`.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod bicubic;
pub mod catmull_rom;
pub mod lanczos;

pub use bicubic::Bicubic;
pub use catmull_rom::CatmullRom;
pub use lanczos::Lanczos;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(Bicubic))?;
    registry.register(Arc::new(Lanczos))?;
    registry.register(Arc::new(CatmullRom))?;
    Ok(())
}

pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
