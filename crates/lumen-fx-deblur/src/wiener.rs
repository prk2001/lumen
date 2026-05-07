//! Wiener deconvolution — classical frequency-domain inverse filter.
//!
//! Given a blurred image `g = h * f + n` where `h` is a known PSF and
//! `n` is additive noise with noise-to-signal ratio `K`, the Wiener
//! filter that minimizes mean-squared error is:
//!
//! ```text
//! F_hat = G * conj(H) / (|H|^2 + K)
//! ```
//!
//! Here we assume `h` is a Gaussian PSF with user-supplied `sigma`. The
//! regularization parameter `K` (called `nsr` in the UI) trades ringing
//! against detail recovery — higher `nsr` → more regularization → less
//! ringing but less sharpening.
//!
//! Implementation:
//!
//! 1. Operate on linear-light float RGBA. Alpha is left untouched.
//! 2. Pad each color channel to the next power-of-two in both dims to
//!    avoid wrap-around artifacts (the FFT is implicitly periodic).
//! 3. Run a 2D FFT (row pass + column pass with `rustfft`) per channel.
//! 4. Build the Gaussian PSF kernel of the same padded size, FFT it.
//! 5. Multiply by the Wiener filter `conj(H) / (|H|^2 + K)`.
//! 6. Inverse-FFT back, crop to original dims, clamp to [0, 1].
//!
//! `iterations` is currently fixed to 1; it's reserved for future
//! Richardson–Lucy-style iterative refinement.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;
use tracing::instrument;

#[derive(Debug, Default)]
pub struct Wiener;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-deblur.wiener",
    display_name: "Wiener Deconvolution",
    description: "Frequency-domain inverse filter against an assumed Gaussian PSF.",
    category: Category::Deblur,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "sigma",
        display_name: "Sigma",
        description: "Assumed Gaussian PSF sigma in pixels (the blur we're inverting).",
        kind: ParamKind::Float {
            default: 1.0,
            min: Some(0.3),
            max: Some(8.0),
        },
    },
    ParamSpec {
        id: "nsr",
        display_name: "Noise/Signal",
        description: "Noise-to-signal ratio (regularization). Higher = less ringing, less detail.",
        kind: ParamKind::Float {
            default: 0.01,
            min: Some(0.0001),
            max: Some(0.5),
        },
    },
    ParamSpec {
        id: "iterations",
        display_name: "Iterations",
        description: "Reserved for future iterative refinement; fixed at 1 today.",
        kind: ParamKind::Int {
            default: 1,
            min: Some(1),
            max: Some(1),
        },
    },
];

impl Effect for Wiener {
    fn metadata(&self) -> &EffectMetadata {
        &META
    }
    fn parameters(&self) -> &[ParamSpec] {
        PARAMS
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            deterministic: true,
            gpu: false,
            streamable: false, // needs full frame for FFT
            temporal: false,
        }
    }

    #[instrument(skip_all, fields(effect = META.id))]
    fn apply(&self, _ctx: &mut Context, input: Frame, params: &ParamValues) -> Result<Frame> {
        let sigma = (params.get_float("sigma").unwrap_or(1.0) as f32).max(0.3);
        let nsr = (params.get_float("nsr").unwrap_or(0.01) as f32).max(1e-6);

        let mut frame = input.into_rgba_f32_linear();
        let w = frame.width as usize;
        let h = frame.height as usize;
        if w == 0 || h == 0 {
            return Ok(frame);
        }

        // Pad to next power of two in each dim — `rustfft` handles
        // arbitrary sizes but pow2 is fastest, and zero-padding avoids
        // wrap-around contamination at the borders.
        let pw = next_pow2(w.max(2));
        let ph = next_pow2(h.max(2));

        // Build planner once and reuse for all three channels.
        let mut planner = FftPlanner::<f32>::new();
        let fft_row_fwd = planner.plan_fft_forward(pw);
        let fft_row_inv = planner.plan_fft_inverse(pw);
        let fft_col_fwd = planner.plan_fft_forward(ph);
        let fft_col_inv = planner.plan_fft_inverse(ph);

        // Compute H (FFT of the Gaussian PSF) once — same for all channels.
        let h_fft = build_psf_spectrum(sigma, pw, ph, &fft_row_fwd, &fft_col_fwd);

        // Precompute the Wiener filter coefficient per frequency bin.
        // W = conj(H) / (|H|^2 + K). Multiplying G by W gives the estimate.
        //
        // We force the DC coefficient (bin 0) to exactly 1 so the image's
        // mean brightness is preserved regardless of regularization. The
        // unregularized formula already gives ~1 at DC for a normalized
        // PSF (where H[0]=1); pinning it makes the high-K limit a true
        // pass-through at DC, which is what users expect from a sharpener.
        let mut wiener = vec![Complex32::default(); pw * ph];
        for (i, hi) in h_fft.iter().enumerate() {
            let mag2 = hi.re * hi.re + hi.im * hi.im;
            let denom = mag2 + nsr;
            wiener[i] = hi.conj() / denom;
        }
        wiener[0] = Complex32::new(1.0, 0.0);

        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");
        let mut buf = vec![Complex32::default(); pw * ph];

        for c in 0..3 {
            // Load channel into padded complex buffer (zero-padded edges).
            for v in buf.iter_mut() {
                *v = Complex32::default();
            }
            for y in 0..h {
                for x in 0..w {
                    let v = pixels[(y * w + x) * 4 + c];
                    buf[y * pw + x] = Complex32::new(v, 0.0);
                }
            }

            // Forward 2D FFT: rows then columns.
            fft2d_inplace(&mut buf, pw, ph, &fft_row_fwd, &fft_col_fwd);

            // Apply the Wiener filter.
            for (b, w) in buf.iter_mut().zip(wiener.iter()) {
                *b *= *w;
            }

            // Inverse 2D FFT.
            fft2d_inplace(&mut buf, pw, ph, &fft_row_inv, &fft_col_inv);

            // rustfft's inverse is unnormalized; divide by N = pw*ph.
            let scale = 1.0 / ((pw * ph) as f32);

            // Write cropped, clamped real part back into the channel.
            for y in 0..h {
                for x in 0..w {
                    let v = buf[y * pw + x].re * scale;
                    pixels[(y * w + x) * 4 + c] = v.clamp(0.0, 1.0);
                }
            }
        }

        Ok(frame)
    }
}

/// Compute the 2D FFT of a Gaussian PSF centered at the origin (so
/// that convolution by `h` becomes a no-shift multiplication in the
/// frequency domain).
fn build_psf_spectrum(
    sigma: f32,
    pw: usize,
    ph: usize,
    fft_row_fwd: &Arc<dyn Fft<f32>>,
    fft_col_fwd: &Arc<dyn Fft<f32>>,
) -> Vec<Complex32> {
    let mut k = vec![Complex32::default(); pw * ph];

    // Gaussian sampled with origin at (0, 0), wrapping coordinates so
    // that the kernel straddles the four corners of the buffer. This
    // matches the periodic FFT convention and keeps the filter zero-phase.
    let inv2sigma2 = 1.0 / (2.0 * sigma * sigma);
    let half_w = pw as isize / 2;
    let half_h = ph as isize / 2;
    let mut sum = 0.0f32;
    for y in 0..ph {
        // Map y -> signed offset in [-ph/2, ph/2)
        let dy = if (y as isize) <= half_h {
            y as isize
        } else {
            y as isize - ph as isize
        };
        for x in 0..pw {
            let dx = if (x as isize) <= half_w {
                x as isize
            } else {
                x as isize - pw as isize
            };
            let d2 = (dx * dx + dy * dy) as f32;
            let v = (-d2 * inv2sigma2).exp();
            k[y * pw + x] = Complex32::new(v, 0.0);
            sum += v;
        }
    }
    // Normalize so the PSF sums to 1 (DC gain = 1).
    let inv = 1.0 / sum;
    for v in &mut k {
        v.re *= inv;
    }

    fft2d_inplace(&mut k, pw, ph, fft_row_fwd, fft_col_fwd);
    k
}

/// 2D FFT in-place via row pass then column pass. `buf` is row-major
/// `pw × ph` complex.
fn fft2d_inplace(
    buf: &mut [Complex32],
    pw: usize,
    ph: usize,
    fft_row: &Arc<dyn Fft<f32>>,
    fft_col: &Arc<dyn Fft<f32>>,
) {
    debug_assert_eq!(buf.len(), pw * ph);

    // Row pass: each row is contiguous, so transform in place.
    for row in buf.chunks_exact_mut(pw) {
        fft_row.process(row);
    }

    // Column pass: gather each column into a scratch vec, transform,
    // and scatter back. (rustfft only operates on contiguous slices.)
    let mut col = vec![Complex32::default(); ph];
    for x in 0..pw {
        for y in 0..ph {
            col[y] = buf[y * pw + x];
        }
        fft_col.process(&mut col);
        for y in 0..ph {
            buf[y * pw + x] = col[y];
        }
    }
}

fn next_pow2(n: usize) -> usize {
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue, PixelData};

    /// Build a u8 RGBA frame from a per-pixel grayscale generator.
    fn gray_frame(w: usize, h: usize, mut f: impl FnMut(usize, usize) -> u8) -> Frame {
        let mut data = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let v = f(x, y);
                let off = (y * w + x) * 4;
                data[off] = v;
                data[off + 1] = v;
                data[off + 2] = v;
                data[off + 3] = 255;
            }
        }
        Frame::new(
            w as u32,
            h as u32,
            PixelData::Rgba8(data),
            ColorSpace::SRgb,
            None,
        )
        .unwrap()
    }

    /// Convolve a single grayscale plane (linear-light, [0,1]) with a
    /// separable Gaussian. Used to build a known-blurred ground truth
    /// for the recovery test below.
    fn gaussian_blur_gray(plane: &mut [f32], w: usize, h: usize, sigma: f32) {
        let half = (3.0 * sigma).ceil() as usize;
        let len = 2 * half + 1;
        let inv2s2 = 1.0 / (2.0 * sigma * sigma);
        let mut k = vec![0.0f32; len];
        let mut sum = 0.0f32;
        for (i, ki) in k.iter_mut().enumerate() {
            let x = i as f32 - half as f32;
            let v = (-x * x * inv2s2).exp();
            *ki = v;
            sum += v;
        }
        for v in &mut k {
            *v /= sum;
        }

        // Horizontal.
        let mut tmp = vec![0.0f32; plane.len()];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0f32;
                for (i, &kv) in k.iter().enumerate() {
                    let xi =
                        (x as isize + i as isize - half as isize).clamp(0, w as isize - 1) as usize;
                    acc += plane[y * w + xi] * kv;
                }
                tmp[y * w + x] = acc;
            }
        }
        // Vertical.
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0f32;
                for (i, &kv) in k.iter().enumerate() {
                    let yi =
                        (y as isize + i as isize - half as isize).clamp(0, h as isize - 1) as usize;
                    acc += tmp[yi * w + x] * kv;
                }
                plane[y * w + x] = acc;
            }
        }
    }

    #[test]
    fn heavy_regularization_is_near_passthrough_on_constant() {
        // With sigma small (mild assumed PSF) and nsr large (heavy
        // regularization), the filter is dominated by the |H|^2/(|H|^2+K)
        // factor which collapses toward the identity at DC. A constant
        // image should round-trip with no drift beyond f32 noise.
        let wf = Wiener;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("sigma", ParamValue::Float(0.3));
        p.insert("nsr", ParamValue::Float(0.5));
        p.validate_and_fill(wf.parameters()).unwrap();

        let frame = gray_frame(16, 16, |_, _| 128);
        let out = wf.apply(&mut ctx, frame, &p).unwrap();
        let out8 = out.into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out8.data else {
            panic!()
        };
        // Allow small rounding tolerance (FFT + sRGB round-trip).
        assert!(
            px.chunks_exact(4).all(|p| {
                (p[0] as i32 - 128).abs() <= 2
                    && (p[1] as i32 - 128).abs() <= 2
                    && (p[2] as i32 - 128).abs() <= 2
                    && p[3] == 255
            }),
            "constant image drifted under heavy-reg Wiener: first={:?}",
            &px[..16]
        );
    }

    #[test]
    fn recovers_detail_from_known_blur() {
        // 1) Build a high-contrast 32x32 pattern (vertical bars).
        // 2) Apply a known Gaussian blur with sigma_blur to make a
        //    "blurred" reference.
        // 3) Run Wiener with the matching sigma and small nsr.
        // 4) Compare per-pixel deltas: |output - blurred| should have
        //    non-trivial variance (Wiener actually moved pixels toward
        //    the original sharp pattern instead of returning the blur).
        let w = 32usize;
        let h = 32usize;
        let sigma_blur = 1.5f32;

        // Sharp pattern in linear-light [0,1]: 4-pixel-wide vertical bars.
        let mut sharp = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                sharp[y * w + x] = if (x / 4) % 2 == 0 { 0.0 } else { 1.0 };
            }
        }

        // Blur it.
        let mut blurred = sharp.clone();
        gaussian_blur_gray(&mut blurred, w, h, sigma_blur);

        // Pack the blurred plane into an RgbaF32 linear-light Frame.
        let mut data = vec![0.0f32; w * h * 4];
        for i in 0..w * h {
            data[i * 4] = blurred[i];
            data[i * 4 + 1] = blurred[i];
            data[i * 4 + 2] = blurred[i];
            data[i * 4 + 3] = 1.0;
        }
        let frame = Frame::new(
            w as u32,
            h as u32,
            PixelData::RgbaF32(data),
            ColorSpace::LinearSRgb,
            None,
        )
        .unwrap();

        let wf = Wiener;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("sigma", ParamValue::Float(sigma_blur as f64));
        p.insert("nsr", ParamValue::Float(0.005));
        p.validate_and_fill(wf.parameters()).unwrap();

        let out = wf.apply(&mut ctx, frame, &p).unwrap();
        let out_pixels = out.as_f32().unwrap();

        // Variance of (output - blurred) on the red channel. If Wiener
        // produced anything other than the blurred input, this will be
        // non-trivial. (A pass-through filter would give variance ≈ 0.)
        let mut diffs = Vec::with_capacity(w * h);
        for i in 0..w * h {
            diffs.push(out_pixels[i * 4] - blurred[i]);
        }
        let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
        let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / diffs.len() as f32;
        assert!(
            var > 1e-3,
            "Wiener output too close to blurred input: var={var}"
        );

        // Sanity: output should also move *toward* the sharp ground
        // truth — i.e. error against sharp drops compared to the blur's
        // error against sharp. (Not a strict guarantee for all sigmas /
        // patterns, but holds for this clean case.)
        let blurred_err: f32 = blurred
            .iter()
            .zip(sharp.iter())
            .map(|(b, s)| (b - s).powi(2))
            .sum();
        let out_err: f32 = (0..w * h)
            .map(|i| (out_pixels[i * 4] - sharp[i]).powi(2))
            .sum();
        assert!(
            out_err < blurred_err,
            "Wiener didn't reduce error toward sharp: blur_err={blurred_err} out_err={out_err}"
        );
    }

    #[test]
    fn small_image_does_not_panic() {
        // 4x4 forces padding (next_pow2 -> 4) and exercises the full
        // FFT path on a degenerate-but-valid frame.
        let wf = Wiener;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("sigma", ParamValue::Float(1.2));
        p.insert("nsr", ParamValue::Float(0.02));
        p.validate_and_fill(wf.parameters()).unwrap();

        let frame = gray_frame(4, 4, |x, y| ((x + y) * 32) as u8);
        let out = wf.apply(&mut ctx, frame, &p).unwrap();
        assert_eq!(out.width, 4);
        assert_eq!(out.height, 4);
    }
}
