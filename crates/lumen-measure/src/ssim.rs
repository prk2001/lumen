//! Structural similarity index (SSIM) — Wang, Bovik, Sheikh, Simoncelli, 2004.
//!
//! Operates on the **luma** of each frame (Rec.709 weights on the linear
//! RGB channels) using **11×11 Gaussian-weighted local windows** with
//! σ ≈ 1.5 px.
//!
//! Constants follow the original paper:
//!
//! - `K1 = 0.01`, `K2 = 0.03`
//! - `L = 1.0` (dynamic range of normalized linear-light float pixels)
//! - `C1 = (K1 · L)^2`, `C2 = (K2 · L)^2`
//!
//! Per-pixel SSIM is averaged over the whole image to produce a scalar
//! in `[-1, 1]`, where `1.0` indicates identical signals. Identical
//! frames return values within float-rounding of `1.0`.
//!
//! # Edge handling
//!
//! The Gaussian smoothing pass is separable. We use **clamp-to-edge**
//! (replicate) sampling at image borders, matching the convention used
//! by [`lumen-fx-denoise`]'s Gaussian filter. SSIM is then evaluated at
//! every pixel (no boundary cropping), which is the most common
//! reference-implementation choice when the window is small relative to
//! the image; for very small images this slightly biases borders but
//! does not produce undefined values.

use lumen_core::{Error, Frame, Result};

const K1: f64 = 0.01;
const K2: f64 = 0.03;
const L: f64 = 1.0;

/// Compute the mean SSIM index over the luma channel.
///
/// Returns a scalar in `[-1, 1]`. Returns `Err(Error::Layout(...))` if
/// frame dimensions disagree.
pub fn ssim(a: &Frame, b: &Frame) -> Result<f64> {
    if a.width != b.width || a.height != b.height {
        return Err(Error::Layout(format!(
            "frame dims differ: {}x{} vs {}x{}",
            a.width, a.height, b.width, b.height
        )));
    }
    if a.is_empty() {
        return Ok(1.0);
    }

    let w = a.width as usize;
    let h = a.height as usize;

    let af = a.clone().into_rgba_f32_linear();
    let bf = b.clone().into_rgba_f32_linear();
    let ap = af.as_f32().expect("RgbaF32 after lift");
    let bp = bf.as_f32().expect("RgbaF32 after lift");

    // Extract luma planes (Rec.709 coefficients) in f64 so that the
    // running products and the Gaussian-weighted sums don't accumulate
    // f32 rounding error — SSIM dynamic range is small near 1.0 and
    // tests rely on identical-input symmetry.
    let la0 = luma_plane(ap, w, h);
    let lb0 = luma_plane(bp, w, h);

    // Pre-compute per-pixel products before any blurring.
    let a_sq0: Vec<f64> = la0.iter().map(|x| x * x).collect();
    let b_sq0: Vec<f64> = lb0.iter().map(|x| x * x).collect();
    let ab0: Vec<f64> = la0.iter().zip(lb0.iter()).map(|(x, y)| x * y).collect();

    // 11×11 Gaussian, σ = 1.5 — the canonical SSIM window from Wang et al.
    let kernel = build_gaussian_kernel(1.5, 5);

    let mu_a = gaussian_blur_plane(&la0, w, h, &kernel);
    let mu_b = gaussian_blur_plane(&lb0, w, h, &kernel);
    let mu_a_sq_blur = gaussian_blur_plane(&a_sq0, w, h, &kernel);
    let mu_b_sq_blur = gaussian_blur_plane(&b_sq0, w, h, &kernel);
    let mu_ab_blur = gaussian_blur_plane(&ab0, w, h, &kernel);

    let c1 = (K1 * L).powi(2);
    let c2 = (K2 * L).powi(2);

    let n = w * h;
    let mut sum = 0.0f64;
    for i in 0..n {
        let ma = mu_a[i];
        let mb = mu_b[i];
        let ma_sq = ma * ma;
        let mb_sq = mb * mb;
        let mab = ma * mb;

        // var/cov estimates; clamp to non-negative to absorb FP noise.
        let var_a = (mu_a_sq_blur[i] - ma_sq).max(0.0);
        let var_b = (mu_b_sq_blur[i] - mb_sq).max(0.0);
        let cov_ab = mu_ab_blur[i] - mab;

        let num = (2.0 * mab + c1) * (2.0 * cov_ab + c2);
        let den = (ma_sq + mb_sq + c1) * (var_a + var_b + c2);
        sum += num / den;
    }
    Ok(sum / n as f64)
}

/// Rec.709 luma from linear RGBA f32 pixels (alpha ignored). Output is
/// f64 to keep the SSIM accumulators numerically tight.
fn luma_plane(rgba: &[f32], w: usize, h: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(w * h);
    for chunk in rgba.chunks_exact(4) {
        let r = chunk[0] as f64;
        let g = chunk[1] as f64;
        let b = chunk[2] as f64;
        out.push(0.2126 * r + 0.7152 * g + 0.0722 * b);
    }
    debug_assert_eq!(out.len(), w * h);
    out
}

/// Build a normalized 1-D Gaussian kernel of length `2*half + 1`.
fn build_gaussian_kernel(sigma: f64, half: usize) -> Vec<f64> {
    let len = 2 * half + 1;
    let mut k = Vec::with_capacity(len);
    let inv2sigma2 = 1.0 / (2.0 * sigma * sigma);
    let mut sum = 0.0f64;
    for i in 0..len {
        let x = i as f64 - half as f64;
        let v = (-x * x * inv2sigma2).exp();
        k.push(v);
        sum += v;
    }
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// Separable Gaussian blur on a single-channel f64 plane with
/// clamp-to-edge, returning a fresh buffer.
///
/// Adapted from `lumen-fx-denoise/src/gaussian.rs::gaussian_blur_rgba`,
/// reduced to one channel and promoted to f64 for SSIM accuracy.
fn gaussian_blur_plane(input: &[f64], w: usize, h: usize, kernel: &[f64]) -> Vec<f64> {
    let half = kernel.len() / 2;

    // Horizontal pass.
    let mut temp = vec![0.0f64; input.len()];
    for y in 0..h {
        let row_in = &input[y * w..(y + 1) * w];
        let row_out = &mut temp[y * w..(y + 1) * w];
        for (x, slot) in row_out.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for (i, &k) in kernel.iter().enumerate() {
                let xi = (x as isize + i as isize - half as isize)
                    .clamp(0, w as isize - 1) as usize;
                acc += row_in[xi] * k;
            }
            *slot = acc;
        }
    }
    // Vertical pass.
    let mut out = vec![0.0f64; input.len()];
    for y in 0..h {
        let row_out = &mut out[y * w..(y + 1) * w];
        for (x, slot) in row_out.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for (i, &k) in kernel.iter().enumerate() {
                let yi = (y as isize + i as isize - half as isize)
                    .clamp(0, h as isize - 1) as usize;
                acc += temp[yi * w + x] * k;
            }
            *slot = acc;
        }
    }
    out
}
