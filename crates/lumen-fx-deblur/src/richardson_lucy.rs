//! Richardson–Lucy iterative deconvolution.
//!
//! Given an observed image `O` and a known PSF `H`, Richardson–Lucy is an
//! expectation-maximization scheme that iteratively maximizes the
//! Poisson likelihood of the recovered image:
//!
//! ```text
//! estimate_0 = O
//! for k in 0..iterations {
//!     relative   = O / conv(estimate_k, H)        // element-wise divide
//!     correction = conv(relative, flip(H))         // correlation, not convolution
//!     estimate_{k+1} = estimate_k * correction     // element-wise multiply
//! }
//! ```
//!
//! For symmetric PSFs (like an isotropic Gaussian) `flip(H) == H` so the
//! correlation reduces to another convolution by `H`. We exploit that
//! here and just convolve twice per iteration.
//!
//! The algorithm is strictly more powerful than a single-pass Wiener
//! inverse for known-PSF deblurring — it recovers more high-frequency
//! detail at the cost of amplifying noise as iterations grow. 5–10
//! iterations is the typical sweet spot for forensic / astronomy /
//! medical imaging work.
//!
//! Implementation mirrors [`crate::wiener`]:
//!
//! 1. Operate on linear-light Float32 RGBA. Alpha is left untouched.
//! 2. Pad each color channel to the next power-of-two in both dims.
//! 3. Forward 2D FFT per channel via row pass + column pass.
//! 4. Build the Gaussian PSF spectrum once.
//! 5. Iterate the EM update entirely in spectral form for the
//!    convolutions, hopping back to spatial form for the element-wise
//!    multiply / divide steps.
//! 6. Crop and clamp to [0, 1].
//!
//! Numerical guards: the divisor `conv(estimate, H)` is clamped to a
//! small epsilon so a near-zero region cannot blow the ratio up.
//! Optional Biggs–Andrews-style damping is supported via the `damping`
//! parameter — values >0 pull the correction toward 1, trading detail
//! for noise suppression.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;
use tracing::instrument;

#[derive(Debug, Default)]
pub struct RichardsonLucy;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-deblur.richardson_lucy",
    display_name: "Richardson–Lucy Deconvolution",
    description: "Iterative ML deconvolution against an assumed Gaussian PSF.",
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
        id: "iterations",
        display_name: "Iterations",
        description: "Number of EM iterations. More = sharper, but more noise amplification.",
        kind: ParamKind::Int {
            default: 8,
            min: Some(1),
            max: Some(50),
        },
    },
    ParamSpec {
        id: "damping",
        display_name: "Damping",
        description:
            "Biggs–Andrews-style correction damping. 0 = standard RL; higher = less noise, less detail.",
        kind: ParamKind::Float {
            default: 0.0,
            min: Some(0.0),
            max: Some(1.0),
        },
    },
];

/// Floor for the divisor in the EM update, in linear light. Anything
/// below this collapses to `EPS_DIV`, preventing 0/0 blow-ups when the
/// estimate convolves to ~0 in dark regions.
const EPS_DIV: f32 = 1.0e-6;

impl Effect for RichardsonLucy {
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
        let iterations = params.get_int("iterations").unwrap_or(8).clamp(1, 50) as usize;
        let damping = (params.get_float("damping").unwrap_or(0.0) as f32).clamp(0.0, 1.0);

        let mut frame = input.into_rgba_f32_linear();
        let w = frame.width as usize;
        let h = frame.height as usize;
        if w == 0 || h == 0 {
            return Ok(frame);
        }

        // Pad to next power of two in each dim — same strategy as Wiener.
        let pw = next_pow2(w.max(2));
        let ph = next_pow2(h.max(2));
        let n = pw * ph;
        let inv_n = 1.0 / (n as f32);

        // Plan all four FFTs once and reuse across channels + iterations.
        let mut planner = FftPlanner::<f32>::new();
        let fft_row_fwd = planner.plan_fft_forward(pw);
        let fft_row_inv = planner.plan_fft_inverse(pw);
        let fft_col_fwd = planner.plan_fft_forward(ph);
        let fft_col_inv = planner.plan_fft_inverse(ph);

        // H = FFT of the (DC-normalized) Gaussian PSF. The Gaussian is
        // symmetric so flip(H) == H in real space, which means H_fft for
        // the correlation step is identical to H_fft for the convolution
        // step (the conjugate of a real-symmetric kernel's spectrum
        // equals the spectrum itself for symmetric kernels — but to be
        // numerically safe we use conj(H) explicitly for the correlation
        // step). For a true symmetric Gaussian this is a no-op; it costs
        // nothing and stays correct if a future change introduces an
        // asymmetric PSF.
        let h_fft = build_psf_spectrum(sigma, pw, ph, &fft_row_fwd, &fft_col_fwd);

        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");

        // Reusable scratch buffers — one per channel iteration.
        let mut estimate = vec![Complex32::default(); n]; // current f_k
        let mut conv_buf = vec![Complex32::default(); n]; // convolutions go here
        let observed = &mut vec![0.0f32; n][..]; // padded observed channel

        for c in 0..3 {
            // Load channel (zero-padded) into both `observed` (real) and
            // `estimate` (complex). estimate_0 = observed.
            for v in estimate.iter_mut() {
                *v = Complex32::default();
            }
            for v in observed.iter_mut() {
                *v = 0.0;
            }
            for y in 0..h {
                for x in 0..w {
                    let v = pixels[(y * w + x) * 4 + c];
                    let idx = y * pw + x;
                    observed[idx] = v;
                    estimate[idx] = Complex32::new(v, 0.0);
                }
            }

            for _ in 0..iterations {
                // Step 1: conv(estimate, H) — forward FFT, multiply by H, inverse.
                conv_buf.copy_from_slice(&estimate);
                fft2d_inplace(&mut conv_buf, pw, ph, &fft_row_fwd, &fft_col_fwd);
                for (b, hi) in conv_buf.iter_mut().zip(h_fft.iter()) {
                    *b *= *hi;
                }
                fft2d_inplace(&mut conv_buf, pw, ph, &fft_row_inv, &fft_col_inv);

                // Step 2: relative = observed / conv(estimate, H), with
                // a numerical guard on the divisor.
                //
                // We reuse `conv_buf` to hold the relative blur (real
                // part filled, imag zeroed) so we can FFT it next.
                for (i, b) in conv_buf.iter_mut().enumerate() {
                    let denom = (b.re * inv_n).max(EPS_DIV);
                    let r = observed[i] / denom;
                    *b = Complex32::new(r, 0.0);
                }

                // Step 3: correction = conv(relative, flip(H)).
                // For a symmetric PSF, conj(H) in spectral form equals
                // the FFT of flip(H). Since our Gaussian is symmetric,
                // we could just multiply by `h_fft`, but multiplying by
                // `conj(h_fft)` is mathematically the correct general
                // form and costs the same — keep it correct.
                fft2d_inplace(&mut conv_buf, pw, ph, &fft_row_fwd, &fft_col_fwd);
                for (b, hi) in conv_buf.iter_mut().zip(h_fft.iter()) {
                    *b *= hi.conj();
                }
                fft2d_inplace(&mut conv_buf, pw, ph, &fft_row_inv, &fft_col_inv);

                // Step 4: estimate *= correction (with optional Biggs–
                // Andrews-style damping pulling the correction toward 1).
                //
                // damping=0 -> standard RL update.
                // damping=1 -> correction collapses to 1 (no update).
                let one_minus_damp = 1.0 - damping;
                for (e, b) in estimate.iter_mut().zip(conv_buf.iter()) {
                    let mut corr = b.re * inv_n;
                    if damping > 0.0 {
                        corr = 1.0 + (corr - 1.0) * one_minus_damp;
                    }
                    e.re *= corr;
                    // Imag part should remain ~0 from real-symmetric
                    // arithmetic; force it to keep numerical drift out
                    // of subsequent FFT rounds.
                    e.im = 0.0;
                }
            }

            // Write final estimate back, cropped + clamped.
            for y in 0..h {
                for x in 0..w {
                    let v = estimate[y * pw + x].re;
                    pixels[(y * w + x) * 4 + c] = v.clamp(0.0, 1.0);
                }
            }
        }

        Ok(frame)
    }
}

/// Compute the 2D FFT of a Gaussian PSF centered at the origin (so
/// convolution by `h` becomes a no-shift multiplication in spectral
/// form).
///
/// Identical kernel construction to [`crate::wiener::build_psf_spectrum`];
/// kept in this file to avoid cross-module visibility juggling.
fn build_psf_spectrum(
    sigma: f32,
    pw: usize,
    ph: usize,
    fft_row_fwd: &Arc<dyn Fft<f32>>,
    fft_col_fwd: &Arc<dyn Fft<f32>>,
) -> Vec<Complex32> {
    let mut k = vec![Complex32::default(); pw * ph];

    let inv2sigma2 = 1.0 / (2.0 * sigma * sigma);
    let half_w = pw as isize / 2;
    let half_h = ph as isize / 2;
    let mut sum = 0.0f32;
    for y in 0..ph {
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
    let inv = 1.0 / sum;
    for v in &mut k {
        v.re *= inv;
    }

    fft2d_inplace(&mut k, pw, ph, fft_row_fwd, fft_col_fwd);
    k
}

/// 2D FFT in-place via row pass then column pass. Mirrors the helper in
/// [`crate::wiener`].
fn fft2d_inplace(
    buf: &mut [Complex32],
    pw: usize,
    ph: usize,
    fft_row: &Arc<dyn Fft<f32>>,
    fft_col: &Arc<dyn Fft<f32>>,
) {
    debug_assert_eq!(buf.len(), pw * ph);

    for row in buf.chunks_exact_mut(pw) {
        fft_row.process(row);
    }

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
    /// separable Gaussian. Used to synthesize a known-blurred reference.
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
    fn heavy_iterations_on_constant_image_stays_constant() {
        // sigma=1.0, iterations=10 on a flat 16x16 mid-gray image.
        // RL is multiplicative, so a perfectly constant image is a
        // fixed point: relative = O / conv(O, H) = 1, correction = 1,
        // estimate is unchanged. We allow ±2/255 drift for FFT round-off
        // and the sRGB round-trip.
        let rl = RichardsonLucy;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("sigma", ParamValue::Float(1.0));
        p.insert("iterations", ParamValue::Int(10));
        p.insert("damping", ParamValue::Float(0.0));
        p.validate_and_fill(rl.parameters()).unwrap();

        let frame = gray_frame(16, 16, |_, _| 128);
        let out = rl.apply(&mut ctx, frame, &p).unwrap();
        let out8 = out.into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out8.data else {
            panic!()
        };
        assert!(
            px.chunks_exact(4).all(|p| {
                (p[0] as i32 - 128).abs() <= 2
                    && (p[1] as i32 - 128).abs() <= 2
                    && (p[2] as i32 - 128).abs() <= 2
                    && p[3] == 255
            }),
            "constant image drifted under 10-iter RL: first={:?}",
            &px[..16]
        );
    }

    #[test]
    fn recovers_detail_from_known_blur() {
        // 1) Build a sharp 32x32 high-contrast bar pattern.
        // 2) Apply a known Gaussian blur (sigma=1.5).
        // 3) Run RL with the matching sigma, 10 iterations, damping=0.
        // 4) Assert RL output's MSE vs the sharp ground truth is LOWER
        //    than the blurred input's MSE — i.e. RL recovered detail.
        let w = 32usize;
        let h = 32usize;
        let sigma_blur = 1.5f32;

        let mut sharp = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                sharp[y * w + x] = if (x / 4) % 2 == 0 { 0.0 } else { 1.0 };
            }
        }

        let mut blurred = sharp.clone();
        gaussian_blur_gray(&mut blurred, w, h, sigma_blur);

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

        let rl = RichardsonLucy;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("sigma", ParamValue::Float(sigma_blur as f64));
        p.insert("iterations", ParamValue::Int(10));
        p.insert("damping", ParamValue::Float(0.0));
        p.validate_and_fill(rl.parameters()).unwrap();

        let out = rl.apply(&mut ctx, frame, &p).unwrap();
        let out_pixels = out.as_f32().unwrap();

        let blurred_mse: f32 = blurred
            .iter()
            .zip(sharp.iter())
            .map(|(b, s)| (b - s).powi(2))
            .sum::<f32>()
            / (w * h) as f32;
        let out_mse: f32 = (0..w * h)
            .map(|i| (out_pixels[i * 4] - sharp[i]).powi(2))
            .sum::<f32>()
            / (w * h) as f32;

        assert!(
            out_mse < blurred_mse,
            "RL did not reduce MSE toward sharp: blurred_mse={blurred_mse} out_mse={out_mse}"
        );
    }

    #[test]
    fn small_image_does_not_panic() {
        // 4x4 forces minimum FFT padding (next_pow2 -> 4) and exercises
        // every code path on a degenerate-but-valid frame.
        let rl = RichardsonLucy;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("sigma", ParamValue::Float(0.8));
        p.insert("iterations", ParamValue::Int(5));
        p.insert("damping", ParamValue::Float(0.0));
        p.validate_and_fill(rl.parameters()).unwrap();

        let frame = gray_frame(4, 4, |x, y| ((x + y) * 32) as u8);
        let out = rl.apply(&mut ctx, frame, &p).unwrap();
        assert_eq!(out.width, 4);
        assert_eq!(out.height, 4);
    }
}
