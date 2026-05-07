//! Rect-bounded skin smoothing.
//!
//! Applies a separable Gaussian blur across the entire frame, then mixes
//! the blurred result back into the input only inside a user-specified
//! rectangle. An optional `feather` parameter softens the rect edges
//! with a linear ramp so smoothing fades at the boundary.
//!
//! In Phase 1 the rect is supplied directly by the caller. In Phase 2,
//! `lumen-fx-ai` will run a face-detection model and feed the resulting
//! bounding box into this primitive, so the AI face path reuses the
//! same code rather than reimplementing the blur+blend.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct SkinSmoothInRect;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-face.skin_smooth_in_rect",
    display_name: "Skin Smooth (Rect)",
    description: "Soft Gaussian smoothing applied only inside a user rect.",
    category: Category::Face,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "x",
        display_name: "X",
        description: "Rect left in pixels.",
        kind: ParamKind::Int { default: 0, min: Some(0), max: None },
    },
    ParamSpec {
        id: "y",
        display_name: "Y",
        description: "Rect top in pixels.",
        kind: ParamKind::Int { default: 0, min: Some(0), max: None },
    },
    ParamSpec {
        id: "width",
        display_name: "Width",
        description: "Rect width. 0 = whole frame.",
        kind: ParamKind::Int { default: 0, min: Some(0), max: None },
    },
    ParamSpec {
        id: "height",
        display_name: "Height",
        description: "Rect height. 0 = whole frame.",
        kind: ParamKind::Int { default: 0, min: Some(0), max: None },
    },
    ParamSpec {
        id: "radius",
        display_name: "Radius",
        description: "Gaussian sigma in pixels.",
        kind: ParamKind::Float { default: 2.0, min: Some(0.1), max: Some(20.0) },
    },
    ParamSpec {
        id: "amount",
        display_name: "Amount",
        description: "How much smoothing to mix in (0=none, 1=full blur).",
        kind: ParamKind::Float { default: 0.5, min: Some(0.0), max: Some(1.0) },
    },
    ParamSpec {
        id: "feather",
        display_name: "Feather",
        description: "Linear-ramp width at rect edges.",
        kind: ParamKind::Float { default: 4.0, min: Some(0.0), max: Some(256.0) },
    },
];

impl Effect for SkinSmoothInRect {
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
        let img_w = input.width as f32;
        let img_h = input.height as f32;
        let rx = params.get_int("x").unwrap_or(0).max(0) as f32;
        let ry = params.get_int("y").unwrap_or(0).max(0) as f32;
        let rw_raw = params.get_int("width").unwrap_or(0);
        let rh_raw = params.get_int("height").unwrap_or(0);
        let rw = if rw_raw <= 0 { img_w } else { rw_raw as f32 };
        let rh = if rh_raw <= 0 { img_h } else { rh_raw as f32 };
        let sigma = params.get_float("radius").unwrap_or(2.0).max(0.1) as f32;
        let amount = params
            .get_float("amount")
            .unwrap_or(0.5)
            .clamp(0.0, 1.0) as f32;
        let feather = params.get_float("feather").unwrap_or(4.0).max(0.0) as f32;

        let mut frame = input.into_rgba_f32_linear();
        let w = frame.width as usize;
        let h = frame.height as usize;

        // Fast path: no smoothing requested or zero-area image.
        if amount <= 0.0 || w == 0 || h == 0 {
            return Ok(frame);
        }

        // Build the blurred copy from a snapshot of the input pixels.
        let blurred = {
            let pixels = frame.as_f32().expect("RgbaF32 after lift");
            let mut buf = pixels.to_vec();
            let kernel = build_gaussian_kernel(sigma);
            gaussian_blur_rgba(&mut buf, w, h, &kernel);
            buf
        };

        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");
        for py in 0..h {
            for px in 0..w {
                let cx = px as f32 + 0.5;
                let cy = py as f32 + 0.5;
                let m = mask_value(cx, cy, rx, ry, rw, rh, feather);
                if m <= 0.0 {
                    continue;
                }
                let t = m * amount;
                let off = (py * w + px) * 4;
                for c in 0..4 {
                    let a = pixels[off + c];
                    let b = blurred[off + c];
                    pixels[off + c] = a + (b - a) * t;
                }
            }
        }
        Ok(frame)
    }
}

/// 1.0 fully inside, 0.0 fully outside, linear ramp in feather band.
///
/// Adapted from `lumen-fx-mask::alpha_rect::mask_value` so the blend
/// weight inside the rect ramps smoothly to 0 at the boundary.
fn mask_value(cx: f32, cy: f32, rx: f32, ry: f32, rw: f32, rh: f32, feather: f32) -> f32 {
    if feather <= 0.0 {
        let inside = cx >= rx && cx < rx + rw && cy >= ry && cy < ry + rh;
        return if inside { 1.0 } else { 0.0 };
    }
    let dx_left = cx - rx;
    let dx_right = (rx + rw) - cx;
    let dy_top = cy - ry;
    let dy_bot = (ry + rh) - cy;
    let dist = dx_left.min(dx_right).min(dy_top).min(dy_bot);
    if dist >= feather {
        1.0
    } else if dist <= -feather {
        0.0
    } else {
        ((dist + feather) / (2.0 * feather)).clamp(0.0, 1.0)
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

    fn checker_frame(w: u32, h: u32) -> Frame {
        let mut data = vec![0u8; (w as usize) * (h as usize) * 4];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let on = (x + y) % 2 == 0;
                let v = if on { 240 } else { 16 };
                let off = (y * w as usize + x) * 4;
                data[off] = v;
                data[off + 1] = v;
                data[off + 2] = v;
                data[off + 3] = 255;
            }
        }
        Frame::new(w, h, PixelData::Rgba8(data), ColorSpace::SRgb, None).unwrap()
    }

    fn variance(pixels: &[f32]) -> f32 {
        if pixels.is_empty() {
            return 0.0;
        }
        let n = pixels.len() as f32;
        let mean: f32 = pixels.iter().sum::<f32>() / n;
        pixels.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n
    }

    #[test]
    fn amount_zero_is_passthrough() {
        let fx = SkinSmoothInRect;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("amount", ParamValue::Float(0.0));
        p.insert("radius", ParamValue::Float(3.0));
        p.insert("feather", ParamValue::Float(0.0));
        p.validate_and_fill(fx.parameters()).unwrap();

        let original = checker_frame(8, 8);
        let expected = original.clone();
        let out = fx.apply(&mut ctx, original, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(out_px) = out.data else { panic!() };
        let PixelData::Rgba8(in_px) = expected.data else { panic!() };
        // amount=0 should leave the image bit-exact (subject to round-trip
        // sRGB precision, so allow a 1-LSB tolerance).
        for (a, b) in out_px.iter().zip(in_px.iter()) {
            assert!(
                (*a as i32 - *b as i32).abs() <= 1,
                "amount=0 should be passthrough, got {a} vs {b}"
            );
        }
    }

    #[test]
    fn solid_image_unchanged() {
        let fx = SkinSmoothInRect;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("x", ParamValue::Int(2));
        p.insert("y", ParamValue::Int(2));
        p.insert("width", ParamValue::Int(8));
        p.insert("height", ParamValue::Int(8));
        p.insert("radius", ParamValue::Float(4.0));
        p.insert("amount", ParamValue::Float(1.0));
        p.insert("feather", ParamValue::Float(2.0));
        p.validate_and_fill(fx.parameters()).unwrap();

        let f = Frame::new(
            16,
            16,
            PixelData::Rgba8(vec![100; 16 * 16 * 4]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();
        let out = fx.apply(&mut ctx, f, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!() };
        assert!(
            px.iter().all(|&v| (v as i32 - 100).abs() <= 1),
            "smoothing a solid image must not change it"
        );
    }

    #[test]
    fn smoothing_reduces_variance_only_inside_rect() {
        // 16x16 high-frequency checker. Smooth a centered 8x8 rect.
        // Variance of luma should drop sharply inside the rect and stay
        // close to original outside.
        let fx = SkinSmoothInRect;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("x", ParamValue::Int(4));
        p.insert("y", ParamValue::Int(4));
        p.insert("width", ParamValue::Int(8));
        p.insert("height", ParamValue::Int(8));
        p.insert("radius", ParamValue::Float(2.0));
        p.insert("amount", ParamValue::Float(1.0));
        p.insert("feather", ParamValue::Float(0.0));
        p.validate_and_fill(fx.parameters()).unwrap();

        let original = checker_frame(16, 16);
        // Compute reference variance on the lifted (linear-float) buffer
        // so the comparison is apples-to-apples with the output.
        let lifted = original.clone().into_rgba_f32_linear();
        let in_pixels = lifted.as_f32().unwrap();

        let collect_red = |pixels: &[f32], inside_rect: bool| -> Vec<f32> {
            let mut out = Vec::new();
            for y in 0..16 {
                for x in 0..16 {
                    let in_rect = (4..12).contains(&x) && (4..12).contains(&y);
                    if in_rect == inside_rect {
                        out.push(pixels[(y * 16 + x) * 4]);
                    }
                }
            }
            out
        };

        let in_var_inside = variance(&collect_red(in_pixels, true));
        let in_var_outside = variance(&collect_red(in_pixels, false));

        let out_frame = fx.apply(&mut ctx, original, &p).unwrap();
        let out_pixels = out_frame.as_f32().unwrap();
        let out_var_inside = variance(&collect_red(out_pixels, true));
        let out_var_outside = variance(&collect_red(out_pixels, false));

        // Inside the rect: variance must drop significantly (smoothing
        // collapses the checker toward the local mean).
        assert!(
            out_var_inside < in_var_inside * 0.5,
            "expected variance inside rect to drop by >=50%, got {in_var_inside} -> {out_var_inside}"
        );
        // Outside the rect: variance must stay close to the original.
        // Allow a small tolerance because edges of the rect interact
        // with neighbors during the global blur (but pixels outside
        // the rect are themselves untouched).
        assert!(
            (out_var_outside - in_var_outside).abs() < in_var_outside * 0.05,
            "expected variance outside rect to be unchanged, got {in_var_outside} -> {out_var_outside}"
        );
    }
}
