//! Mean-squared error and peak signal-to-noise ratio between two frames.
//!
//! Both metrics operate on RGB channels (alpha skipped) in **linear-light
//! float**. Frames are lifted via [`Frame::into_rgba_f32_linear`] before
//! comparison, so callers don't need to worry about the input encoding.
//!
//! # Math
//!
//! For two N-pixel images A and B with three RGB channels each (3N
//! samples total),
//!
//! ```text
//!   MSE  = (1 / 3N) * Σ (a_i − b_i)^2
//!   PSNR = 10 * log10(L^2 / MSE)              with L = 1.0 (peak signal)
//! ```
//!
//! For identical frames, MSE is 0 and PSNR is `+∞`.

use lumen_core::{Error, Frame, Result};

/// Mean-squared error in `[0, 1]^2`. Computed over RGB channels only.
///
/// Returns `Err(Error::Layout(...))` if frame dimensions disagree.
pub fn mse(a: &Frame, b: &Frame) -> Result<f64> {
    if a.width != b.width || a.height != b.height {
        return Err(Error::Layout(format!(
            "frame dims differ: {}x{} vs {}x{}",
            a.width, a.height, b.width, b.height
        )));
    }
    if a.is_empty() {
        return Ok(0.0);
    }

    let af = a.clone().into_rgba_f32_linear();
    let bf = b.clone().into_rgba_f32_linear();
    let ap = af.as_f32().expect("RgbaF32 after lift");
    let bp = bf.as_f32().expect("RgbaF32 after lift");

    let n = ap.len() / 4;
    debug_assert_eq!(n, bp.len() / 4);

    let mut sse = 0.0f64;
    for i in 0..n {
        let off = i * 4;
        for c in 0..3 {
            let d = ap[off + c] as f64 - bp[off + c] as f64;
            sse += d * d;
        }
    }
    Ok(sse / (3.0 * n as f64))
}

/// Peak signal-to-noise ratio in decibels (RGB only, peak signal `L = 1.0`).
///
/// Returns `f64::INFINITY` when the frames are bit-identical (MSE = 0).
/// Returns `Err(Error::Layout(...))` if dimensions differ.
pub fn psnr(a: &Frame, b: &Frame) -> Result<f64> {
    let m = mse(a, b)?;
    if m == 0.0 {
        return Ok(f64::INFINITY);
    }
    // L = 1.0, so 10 * log10(1 / MSE) = -10 * log10(MSE).
    Ok(-10.0 * m.log10())
}
