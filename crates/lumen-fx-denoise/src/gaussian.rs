//! Gaussian denoise — separable convolution with a Gaussian kernel.
//!
//! This is the simplest possible spatial denoiser. It blurs detail
//! along with noise, so it's only useful as a fallback when better
//! options aren't available. Phase 2 replaces it with a proper
//! bilateral / NL-means / AI path.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct GaussianDenoise;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-denoise.gaussian",
    display_name: "Gaussian Denoise",
    description: "Spatial Gaussian blur. Baseline noise reduction.",
    category: Category::Denoise,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[ParamSpec {
    id: "sigma",
    display_name: "Sigma",
    description: "Gaussian sigma in pixels. Larger = stronger smoothing and detail loss.",
    kind: ParamKind::Float { default: 1.0, min: Some(0.1), max: Some(20.0) },
}];

impl Effect for GaussianDenoise {
    fn metadata(&self) -> &EffectMetadata { &META }
    fn parameters(&self) -> &[ParamSpec] { PARAMS }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            deterministic: true,
            gpu: false,
            streamable: false,
            temporal: false,
        }
    }

    #[instrument(skip_all, fields(effect = META.id))]
    fn apply(&self, _ctx: &mut Context, input: Frame, params: &ParamValues) -> Result<Frame> {
        let sigma = params.get_float("sigma").unwrap_or(1.0).max(0.1) as f32;

        let mut frame = input.into_rgba_f32_linear();
        let w = frame.width as usize;
        let h = frame.height as usize;
        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");

        let kernel = build_gaussian_kernel(sigma);
        gaussian_blur_rgba(pixels, w, h, &kernel);

        Ok(frame)
    }
}

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

fn gaussian_blur_rgba(buf: &mut [f32], w: usize, h: usize, kernel: &[f32]) {
    let half = kernel.len() / 2;
    let stride = w * 4;

    let mut temp = vec![0.0f32; buf.len()];
    for y in 0..h {
        let row_in = &buf[y * stride..(y + 1) * stride];
        let row_out = &mut temp[y * stride..(y + 1) * stride];
        for x in 0..w {
            let mut acc = [0.0f32; 4];
            for (i, &k) in kernel.iter().enumerate() {
                let xi = (x as isize + i as isize - half as isize)
                    .clamp(0, w as isize - 1) as usize;
                let p = &row_in[xi * 4..xi * 4 + 4];
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

    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 4];
            for (i, &k) in kernel.iter().enumerate() {
                let yi = (y as isize + i as isize - half as isize)
                    .clamp(0, h as isize - 1) as usize;
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
    fn solid_image_unchanged() {
        let g = GaussianDenoise;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("sigma", ParamValue::Float(2.0));
        p.validate_and_fill(g.parameters()).unwrap();
        let f =
            Frame::new(16, 16, PixelData::Rgba8(vec![100; 16 * 16 * 4]), ColorSpace::SRgb, None)
                .unwrap();
        let out = g.apply(&mut ctx, f, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!() };
        assert!(px.iter().all(|&v| (v as i32 - 100).abs() <= 1));
    }

    #[test]
    fn salt_and_pepper_attenuated() {
        // Create an 8x8 image with one bright pixel; check it's blurred
        // outward (energy spread reduces peak).
        let g = GaussianDenoise;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("sigma", ParamValue::Float(1.5));
        p.validate_and_fill(g.parameters()).unwrap();

        let mut data = vec![0u8; 8 * 8 * 4];
        for off in (3..data.len()).step_by(4) {
            data[off] = 255; // alpha
        }
        let center = (4 * 8 + 4) * 4;
        data[center] = 255;
        data[center + 1] = 255;
        data[center + 2] = 255;
        let f = Frame::new(8, 8, PixelData::Rgba8(data), ColorSpace::SRgb, None).unwrap();
        let out = g.apply(&mut ctx, f, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!() };
        // Peak should be lower after blur.
        assert!(px[center] < 255, "expected peak attenuation, got {}", px[center]);
    }
}
