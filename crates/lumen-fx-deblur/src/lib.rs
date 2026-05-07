//! # lumen-fx-deblur
//!
//! Deblurring, deconvolution, motion-blur removal — Cat 11 of the spec.
//!
//! Ships three deblur effects:
//!
//! * [`LaplacianSharpen`] — Difference-of-Gaussians edge enhancer for
//!   mild defocus.
//! * [`Wiener`] — single-pass FFT inverse filter against an assumed
//!   Gaussian PSF.
//! * [`RichardsonLucy`] — iterative ML deconvolution against an assumed
//!   Gaussian PSF; strictly more powerful than Wiener for known-PSF
//!   work, used in astronomy / medical imaging / forensics.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod laplacian;
pub mod richardson_lucy;
pub mod wiener;

pub use laplacian::LaplacianSharpen;
pub use richardson_lucy::RichardsonLucy;
pub use wiener::Wiener;

use lumen_core::{EffectRegistry, Result};
use std::sync::Arc;

/// Register every effect this crate provides.
pub fn register_all(registry: &EffectRegistry) -> Result<()> {
    registry.register(Arc::new(LaplacianSharpen))?;
    registry.register(Arc::new(Wiener))?;
    registry.register(Arc::new(RichardsonLucy))?;
    Ok(())
}

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
