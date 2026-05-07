//! Laplacian-of-Gaussian sharpening — a real but simple deblur baseline.
//!
//! Algorithm (Difference-of-Gaussians approximation of the Laplacian):
//!
//! 1. Convert input to linear-light float RGBA.
//! 2. Build `inner = gaussian(input, sigma)`.
//! 3. Build `outer = gaussian(input, sigma * sigma_ratio)`.
//! 4. `lap = inner - outer` is a band-pass / Laplacian-like signal.
//! 5. `out = input + amount * lap`, clamped to [0, 1].
//!
//! This compensates mild defocus blur by injecting back the band of
//! frequencies the (assumed) Gaussian PSF most attenuated. Full Wiener
//! and Richardson–Lucy deconvolution land in Phase 4; this is the CPU
//! baseline that ships in Phase 1.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct LaplacianSharpen;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-deblur.laplacian",
    display_name: "Laplacian Sharpen",
    description: "Difference-of-Gaussians Laplacian edge enhancement to compensate mild blur.",
    category: Category::Deblur,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "amount",
        display_name: "Amount",
        description: "Strength of the Laplacian addition.",
        kind: ParamKind::Float {
            default: 1.0,
            min: Some(0.0),
            max: Some(4.0),
        },
    },
    ParamSpec {
        id: "sigma",
        display_name: "Sigma",
        description: "Inner Gaussian sigma in pixels.",
        kind: ParamKind::Float {
            default: 1.0,
            min: Some(0.1),
            max: Some(10.0),
        },
    },
    ParamSpec {
        id: "sigma_ratio",
        display_name: "Sigma Ratio",
        description: "Outer Gaussian sigma is sigma * sigma_ratio.",
        kind: ParamKind::Float {
            default: 1.6,
            min: Some(1.1),
            max: Some(5.0),
        },
    },
];

impl Effect for LaplacianSharpen {
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
            streamable: false, // needs full frame for blur
            temporal: false,
        }
    }

    #[instrument(skip_all, fields(effect = META.id))]
    fn apply(&self, _ctx: &mut Context, input: Frame, params: &ParamValues) -> Result<Frame> {
        let amount = params.get_float("amount").unwrap_or(1.0) as f32;
        let sigma = params.get_float("sigma").unwrap_or(1.0).max(0.1) as f32;
        let sigma_ratio = params.get_float("sigma_ratio").unwrap_or(1.6).max(1.1) as f32;

        if amount == 0.0 {
            return Ok(input);
        }

        let mut frame = input.into_rgba_f32_linear();
        let w = frame.width as usize;
        let h = frame.height as usize;
        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");

        // Build separable 1D Gaussian kernels at two scales.
        let kernel_inner = build_gaussian_kernel(sigma);
        let kernel_outer = build_gaussian_kernel(sigma * sigma_ratio);

        // Inner blur (smaller sigma).
        let mut inner = pixels.to_vec();
        gaussian_blur_rgba(&mut inner, w, h, &kernel_inner);

        // Outer blur (larger sigma).
        let mut outer = pixels.to_vec();
        gaussian_blur_rgba(&mut outer, w, h, &kernel_outer);

        // out = input + amount * (inner - outer), clamped to [0, 1].
        // Alpha is left untouched.
        for ((px, ic), oc) in pixels
            .chunks_exact_mut(4)
            .zip(inner.chunks_exact(4))
            .zip(outer.chunks_exact(4))
        {
            for c in 0..3 {
                let lap = ic[c] - oc[c];
                let v = px[c] + amount * lap;
                px[c] = v.clamp(0.0, 1.0);
            }
        }
        Ok(frame)
    }
}

/// Half-width of the kernel (samples to each side of center). Capped to
/// keep things sane on huge sigmas.
fn kernel_half_width(sigma: f32) -> usize {
    ((3.0 * sigma).ceil() as usize).clamp(1, 64)
}

fn build_gaussian_kernel(sigma: f32) -> Vec<f32> {
    let half = kernel_half_width(sigma);
    let len = 2 * half + 1;
    let mut k = Vec::with_capacity(len);
    let inv2sigma2 = 1.0 / (2.0 * sigma * sigma);
    let mut sum = 0.0f32;
    for i in 0..len {
        let x = i as f32 - half as f32;
        let v = (-x * x * inv2sigma2).exp();
        k.push(v);
        sum += v;
    }
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// In-place separable Gaussian on a packed RGBA f32 buffer.
fn gaussian_blur_rgba(buf: &mut [f32], w: usize, h: usize, kernel: &[f32]) {
    let half = kernel.len() / 2;
    let stride = w * 4;

    // Horizontal pass into temp.
    let mut temp = vec![0.0f32; buf.len()];
    for y in 0..h {
        let row = &buf[y * stride..(y + 1) * stride];
        let row_out = &mut temp[y * stride..(y + 1) * stride];
        for x in 0..w {
            let mut acc = [0.0f32; 4];
            for (i, &k) in kernel.iter().enumerate() {
                let xi =
                    (x as isize + i as isize - half as isize).clamp(0, w as isize - 1) as usize;
                let p = &row[xi * 4..xi * 4 + 4];
                acc[0] += p[0] * k;
                acc[1] += p[1] * k;
                acc[2] += p[2] * k;
                acc[3] += p[3] * k;
            }
            let off = x * 4;
            row_out[off] = acc[0];
            row_out[off + 1] = acc[1];
            row_out[off + 2] = acc[2];
            row_out[off + 3] = acc[3];
        }
    }

    // Vertical pass back into buf.
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 4];
            for (i, &k) in kernel.iter().enumerate() {
                let yi =
                    (y as isize + i as isize - half as isize).clamp(0, h as isize - 1) as usize;
                let off = yi * stride + x * 4;
                acc[0] += temp[off] * k;
                acc[1] += temp[off + 1] * k;
                acc[2] += temp[off + 2] * k;
                acc[3] += temp[off + 3] * k;
            }
            let off = y * stride + x * 4;
            buf[off] = acc[0];
            buf[off + 1] = acc[1];
            buf[off + 2] = acc[2];
            buf[off + 3] = acc[3];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue, PixelData};

    #[test]
    fn amount_zero_passthrough() {
        // amount=0 short-circuits and returns the input frame unchanged.
        let lap = LaplacianSharpen;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("amount", ParamValue::Float(0.0));
        p.validate_and_fill(lap.parameters()).unwrap();

        let frame = Frame::new(
            8,
            8,
            PixelData::Rgba8(vec![100; 8 * 8 * 4]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();
        let layout_before = frame.layout();
        let out = lap.apply(&mut ctx, frame, &p).unwrap();
        assert_eq!(out.layout(), layout_before);

        // Pixel data should be byte-for-byte unchanged.
        let out8 = out.into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out8.data else {
            panic!()
        };
        assert!(px.iter().all(|&v| v == 100));
    }

    #[test]
    fn solid_image_is_unchanged() {
        // Laplacian of a flat image is zero everywhere, so the output
        // must equal the input within rounding error.
        let lap = LaplacianSharpen;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("amount", ParamValue::Float(2.0));
        p.insert("sigma", ParamValue::Float(1.5));
        p.insert("sigma_ratio", ParamValue::Float(1.6));
        p.validate_and_fill(lap.parameters()).unwrap();

        let solid = Frame::new(
            16,
            16,
            PixelData::Rgba8(vec![80; 16 * 16 * 4]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();
        let out = lap.apply(&mut ctx, solid, &p).unwrap();
        let out8 = out.into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out8.data else {
            panic!()
        };
        // Allow ±1 for f32 round-trip through the sRGB transfer.
        assert!(
            px.iter().all(|&v| (v as i32 - 80).abs() <= 1),
            "constant image drifted: {:?}",
            &px[..16]
        );
    }

    #[test]
    fn edge_detail_is_amplified() {
        // On a clean black/white step, Laplacian sharpening should make
        // the bright side brighter (overshoot) and the dark side darker
        // (undershoot) just inside the edge — relative to the same pixel
        // in the unprocessed input.
        let lap = LaplacianSharpen;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("amount", ParamValue::Float(2.0));
        p.insert("sigma", ParamValue::Float(1.0));
        p.insert("sigma_ratio", ParamValue::Float(1.6));
        p.validate_and_fill(lap.parameters()).unwrap();

        // Build a 32x4 image where the right half is mid-gray and the
        // left half is darker gray. Using non-saturated values so we can
        // observe overshoot/undershoot without clamping hiding it.
        let dark = 80u8;
        let bright = 176u8;
        let w = 32usize;
        let h = 4usize;
        let mut data = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let off = (y * w + x) * 4;
                let v = if x < w / 2 { dark } else { bright };
                data[off] = v;
                data[off + 1] = v;
                data[off + 2] = v;
                data[off + 3] = 255;
            }
        }
        let f = Frame::new(
            w as u32,
            h as u32,
            PixelData::Rgba8(data),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();
        let out = lap.apply(&mut ctx, f, &p).unwrap();
        let out8 = out.into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out8.data else {
            panic!()
        };

        // Sample the bright pixel just to the right of the edge and the
        // dark pixel just to the left of the edge, on the middle row.
        let row = h / 2;
        let bright_idx = (row * w + (w / 2)) * 4; // first bright col
        let dark_idx = (row * w + (w / 2 - 1)) * 4; // last dark col
        let far_bright_idx = (row * w + (w - 1)) * 4; // far end of bright side
        let far_dark_idx = (row * w) * 4; // far end of dark side

        // Edge pixels should overshoot the flat-side baseline.
        assert!(
            px[bright_idx] > px[far_bright_idx],
            "expected bright-side overshoot near edge: edge={} far={}",
            px[bright_idx],
            px[far_bright_idx]
        );
        assert!(
            px[dark_idx] < px[far_dark_idx],
            "expected dark-side undershoot near edge: edge={} far={}",
            px[dark_idx],
            px[far_dark_idx]
        );
    }
}
