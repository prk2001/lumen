//! # lumen-measure
//!
//! Measurement & analysis: scopes, metrics (PSNR/SSIM/VMAF/LPIPS).
//!
//! Round 1 covers the three classic full-reference image-quality metrics:
//!
//! - [`mse`] — mean-squared error in `[0, 1]^2`
//! - [`psnr`] — peak signal-to-noise ratio in dB
//! - [`ssim`] — structural similarity index in `[-1, 1]`
//!
//! All three operate on RGB channels (alpha is ignored) after lifting
//! both inputs to linear-light f32 via [`Frame::into_rgba_f32_linear`],
//! so callers can mix sRGB-encoded and linear inputs freely.
//!
//! See [`Metrics`] / [`all_metrics`] for the bundled output.

#![forbid(unsafe_op_in_unsafe_fn)]

mod psnr;
mod ssim;

pub use psnr::{mse, psnr};
pub use ssim::ssim;

use lumen_core::{Frame, Result};

/// Bundle of reference-based image-quality metrics for a single
/// reference/test pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    /// Mean-squared error in `[0, 1]^2`.
    pub mse: f64,
    /// Peak signal-to-noise ratio in dB. `f64::INFINITY` for identical inputs.
    pub psnr: f64,
    /// Structural similarity index in `[-1, 1]`. `1.0` for identical inputs.
    pub ssim: f64,
}

/// Compute MSE, PSNR, and SSIM in one call.
///
/// Returns `Err(Error::Layout(...))` if the frame dimensions differ.
pub fn all_metrics(a: &Frame, b: &Frame) -> Result<Metrics> {
    let m = mse(a, b)?;
    let p = psnr(a, b)?;
    let s = ssim(a, b)?;
    Ok(Metrics { mse: m, psnr: p, ssim: s })
}

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, Frame, PixelData};

    fn solid_rgba8(w: u32, h: u32, r: u8, g: u8, b: u8) -> Frame {
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            data.extend_from_slice(&[r, g, b, 255]);
        }
        Frame::new(w, h, PixelData::Rgba8(data), ColorSpace::SRgb, None).unwrap()
    }

    #[test]
    fn identical_frames_yield_infinite_psnr_and_unit_ssim() {
        let a = solid_rgba8(16, 16, 128, 64, 200);
        let b = a.clone();

        let m = all_metrics(&a, &b).unwrap();
        assert_eq!(m.mse, 0.0);
        assert!(m.psnr.is_infinite() && m.psnr.is_sign_positive());
        // SSIM of identical inputs collapses to 1.0 within FP rounding.
        assert!((m.ssim - 1.0).abs() < 1e-9, "ssim was {}", m.ssim);
    }

    #[test]
    fn uniform_shift_yields_finite_psnr() {
        // Two solid frames differing by a small constant in linear
        // space — MSE > 0, PSNR finite.
        let a = solid_rgba8(8, 8, 100, 100, 100);
        let b = solid_rgba8(8, 8, 110, 110, 110);

        let m = mse(&a, &b).unwrap();
        let p = psnr(&a, &b).unwrap();
        assert!(m > 0.0, "expected nonzero MSE, got {m}");
        assert!(p.is_finite(), "expected finite PSNR, got {p}");
        // Sanity-check: MSE should be well below 1 for a tiny shift.
        assert!(m < 0.05, "MSE unexpectedly large: {m}");
    }

    #[test]
    fn dimension_mismatch_returns_layout_error() {
        let a = solid_rgba8(8, 8, 0, 0, 0);
        let b = solid_rgba8(4, 8, 0, 0, 0);

        for err in [
            mse(&a, &b).unwrap_err(),
            psnr(&a, &b).unwrap_err(),
            ssim(&a, &b).unwrap_err(),
            all_metrics(&a, &b).unwrap_err(),
        ] {
            assert_eq!(err.code(), "LAYOUT", "got non-layout error: {err}");
        }
    }

    #[test]
    fn tiny_4x4_frame_does_not_panic() {
        // SSIM uses an 11×11 window; a 4×4 frame is smaller than the
        // window. Must not panic; result must be finite and in range.
        let a = solid_rgba8(4, 4, 50, 150, 250);
        let mut b = a.clone();
        // Perturb a single pixel so the frames aren't identical.
        if let PixelData::Rgba8(ref mut v) = b.data {
            v[0] = v[0].saturating_add(20);
        }

        let m = all_metrics(&a, &b).unwrap();
        assert!(m.mse.is_finite() && m.mse > 0.0);
        assert!(m.psnr.is_finite());
        assert!(m.ssim.is_finite() && m.ssim <= 1.0 && m.ssim >= -1.0);
    }
}
