//! Bilateral filter — edge-preserving spatial smoother.
//!
//! Tomasi & Manduchi 1998. For each output pixel we average a
//! square neighborhood, weighting each contributor by *both*:
//!
//!   * a spatial Gaussian on pixel distance, and
//!   * a range Gaussian on luma difference (Rec.709 luma) to the
//!     center.
//!
//! Pixels on the far side of an edge have very different luma and
//! get a vanishingly small range weight, so edges survive while
//! flat regions get smoothed. This is the workhorse classical
//! denoiser when you don't want a neural net.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct Bilateral;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-denoise.bilateral",
    display_name: "Bilateral Denoise",
    description: "Edge-preserving smoother — Gaussian on distance + luma.",
    category: Category::Denoise,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "sigma_spatial",
        display_name: "Spatial sigma",
        description: "Gaussian sigma in pixels for the spatial weight.",
        kind: ParamKind::Float { default: 2.0, min: Some(0.5), max: Some(16.0) },
    },
    ParamSpec {
        id: "sigma_range",
        display_name: "Range sigma",
        description: "Gaussian sigma in linear-luma units for the range weight.",
        kind: ParamKind::Float { default: 0.10, min: Some(0.01), max: Some(1.0) },
    },
    ParamSpec {
        id: "radius",
        display_name: "Radius",
        description: "Patch half-width in pixels (clamped to ~3 * sigma_spatial).",
        kind: ParamKind::Int { default: 3, min: Some(1), max: Some(32) },
    },
];

/// Rec.709 luma — matches what the human eye weights.
#[inline]
fn luma709(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

impl Effect for Bilateral {
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
        let sigma_spatial =
            (params.get_float("sigma_spatial").unwrap_or(2.0) as f32).max(0.5);
        let sigma_range = (params.get_float("sigma_range").unwrap_or(0.10) as f32).max(0.001);
        let req_radius = params.get_int("radius").unwrap_or(3).max(1) as usize;
        // Clamp radius to ~3*sigma_spatial — beyond that, weights are
        // negligible and we waste work.
        let cap = ((3.0 * sigma_spatial).ceil() as usize).max(1);
        let radius = req_radius.min(cap).min(32);

        let mut frame = input.into_rgba_f32_linear();
        let w = frame.width as usize;
        let h = frame.height as usize;
        if w == 0 || h == 0 {
            return Ok(frame);
        }
        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");

        let kernel = build_spatial_kernel(sigma_spatial, radius);
        let inv2sr2 = 1.0 / (2.0 * sigma_range * sigma_range);

        let stride = w * 4;
        let src = pixels.to_vec();
        bilateral_filter_rgba(&src, pixels, w, h, stride, radius, &kernel, inv2sr2);

        Ok(frame)
    }
}

/// 2D spatial weight table indexed as `[(dy + radius) * (2r+1) + (dx + radius)]`.
fn build_spatial_kernel(sigma: f32, radius: usize) -> Vec<f32> {
    let len = 2 * radius + 1;
    let mut k = Vec::with_capacity(len * len);
    let inv2s2 = 1.0 / (2.0 * sigma * sigma);
    for dy in 0..len {
        for dx in 0..len {
            let x = dx as f32 - radius as f32;
            let y = dy as f32 - radius as f32;
            k.push((-(x * x + y * y) * inv2s2).exp());
        }
    }
    k
}

#[allow(clippy::too_many_arguments)]
fn bilateral_filter_rgba(
    src: &[f32],
    dst: &mut [f32],
    w: usize,
    h: usize,
    stride: usize,
    radius: usize,
    spatial: &[f32],
    inv2sr2: f32,
) {
    let kw = 2 * radius + 1;
    for y in 0..h {
        for x in 0..w {
            let center_off = y * stride + x * 4;
            let cr = src[center_off];
            let cg = src[center_off + 1];
            let cb = src[center_off + 2];
            let cl = luma709(cr, cg, cb);

            let mut sum_r = 0.0f32;
            let mut sum_g = 0.0f32;
            let mut sum_b = 0.0f32;
            let mut sum_w = 0.0f32;

            for dy in 0..kw {
                let yi = (y as isize + dy as isize - radius as isize)
                    .clamp(0, h as isize - 1) as usize;
                for dx in 0..kw {
                    let xi = (x as isize + dx as isize - radius as isize)
                        .clamp(0, w as isize - 1) as usize;
                    let off = yi * stride + xi * 4;
                    let r = src[off];
                    let g = src[off + 1];
                    let b = src[off + 2];
                    let l = luma709(r, g, b);
                    let dl = l - cl;
                    let ws = spatial[dy * kw + dx];
                    let wr = (-dl * dl * inv2sr2).exp();
                    let wgt = ws * wr;
                    sum_r += r * wgt;
                    sum_g += g * wgt;
                    sum_b += b * wgt;
                    sum_w += wgt;
                }
            }

            let inv = if sum_w > 0.0 { 1.0 / sum_w } else { 0.0 };
            dst[center_off] = sum_r * inv;
            dst[center_off + 1] = sum_g * inv;
            dst[center_off + 2] = sum_b * inv;
            // Preserve alpha.
            dst[center_off + 3] = src[center_off + 3];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue, PixelData};

    fn run(b: &Bilateral, frame: Frame, sigma_spatial: f64, sigma_range: f64, radius: i64)
        -> Frame
    {
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("sigma_spatial", ParamValue::Float(sigma_spatial));
        p.insert("sigma_range", ParamValue::Float(sigma_range));
        p.insert("radius", ParamValue::Int(radius));
        p.validate_and_fill(b.parameters()).unwrap();
        b.apply(&mut ctx, frame, &p).unwrap()
    }

    #[test]
    fn solid_image_unchanged() {
        let b = Bilateral;
        let f = Frame::new(
            16,
            16,
            PixelData::Rgba8(vec![80; 16 * 16 * 4]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();
        let out = run(&b, f, 3.0, 0.1, 4).into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!() };
        assert!(
            px.iter().all(|&v| (v as i32 - 80).abs() <= 1),
            "solid image must round-trip; got out-of-range value"
        );
    }

    #[test]
    fn pass_through_when_sigma_range_huge() {
        // With sigma_range much larger than any luma difference in the
        // image, the range weight is ~1 everywhere and the bilateral
        // degenerates to a Gaussian-weighted box average. Use a
        // low-contrast image so this regime is reachable within the
        // sigma_range upper bound (1.0). We compare to an independent
        // spatial-only reference computed in linear space.
        let b = Bilateral;
        let w: usize = 12;
        let h: usize = 12;
        let stride = w * 4;
        // Low-contrast pattern: gray ± a tiny ripple. Max luma diff in
        // linear space is well under 0.05, so sigma_range = 1.0 makes
        // every range weight > 0.998.
        let mut data = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let off = (y * w + x) * 4;
                let v: u8 = 128_u8.saturating_add(((x + y) % 5) as u8);
                data[off] = v;
                data[off + 1] = v;
                data[off + 2] = v;
                data[off + 3] = 255;
            }
        }
        let radius_i: i64 = 4;
        let radius = radius_i as usize;
        let sigma_spatial = 2.0f64;
        let f = Frame::new(w as u32, h as u32, PixelData::Rgba8(data.clone()), ColorSpace::SRgb,
            None).unwrap();
        let out_f32 = run(&b, f, sigma_spatial, 1.0, radius_i).as_f32().unwrap().to_vec();

        // Independent reference: spatial-only weighted average in linear
        // space, on the SAME linearized input.
        let lin: Vec<f32> = data.iter().enumerate().map(|(i, &b)| {
            if i % 4 == 3 {
                b as f32 / 255.0
            } else {
                let f = b as f32 / 255.0;
                if f <= 0.04045 { f / 12.92 } else { ((f + 0.055) / 1.055).powf(2.4) }
            }
        }).collect();
        let kernel = build_spatial_kernel(sigma_spatial as f32, radius);
        let kw = 2 * radius + 1;
        let mut ref_lin = lin.clone();
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0.0f32; 3];
                let mut s = 0.0f32;
                for dy in 0..kw {
                    let yi = (y as isize + dy as isize - radius as isize)
                        .clamp(0, h as isize - 1) as usize;
                    for dx in 0..kw {
                        let xi = (x as isize + dx as isize - radius as isize)
                            .clamp(0, w as isize - 1) as usize;
                        let off = yi * stride + xi * 4;
                        let wgt = kernel[dy * kw + dx];
                        acc[0] += lin[off] * wgt;
                        acc[1] += lin[off + 1] * wgt;
                        acc[2] += lin[off + 2] * wgt;
                        s += wgt;
                    }
                }
                let off = y * stride + x * 4;
                ref_lin[off] = acc[0] / s;
                ref_lin[off + 1] = acc[1] / s;
                ref_lin[off + 2] = acc[2] / s;
            }
        }

        // Compare in linear-float space — much tighter than going
        // through u8 sRGB.
        let mut max_diff = 0.0f32;
        for y in 0..h {
            for x in 0..w {
                let off = y * stride + x * 4;
                for c in 0..3 {
                    let d = (out_f32[off + c] - ref_lin[off + c]).abs();
                    if d > max_diff { max_diff = d; }
                }
            }
        }
        assert!(
            max_diff < 1e-3,
            "expected near-pass-through to spatial blur with huge sigma_range; max_diff = {max_diff}"
        );
    }

    #[test]
    fn high_contrast_edge_preserved() {
        // 16x16: left half dark, right half bright. After a small-sigma_range
        // bilateral, the average of each half should be ~unchanged because
        // pixels on the other side of the step get tiny range weights.
        let b = Bilateral;
        let w: usize = 16;
        let h: usize = 16;
        let mut data = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let off = (y * w + x) * 4;
                let v: u8 = if x < w / 2 { 30 } else { 220 };
                data[off] = v;
                data[off + 1] = v;
                data[off + 2] = v;
                data[off + 3] = 255;
            }
        }
        let f =
            Frame::new(w as u32, h as u32, PixelData::Rgba8(data), ColorSpace::SRgb, None)
                .unwrap();
        // sigma_range chosen smaller than the gap between the two halves
        // in linear units. (linear(220/255) >> linear(30/255).)
        let out = run(&b, f, 3.0, 0.05, 5).into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!() };

        let mut sum_left = 0u64;
        let mut sum_right = 0u64;
        let mut count_left = 0u64;
        let mut count_right = 0u64;
        // Sample columns away from the edge to avoid the transition zone.
        for y in 0..h {
            for x in 0..3 {
                let off = (y * w + x) * 4;
                sum_left += px[off] as u64;
                count_left += 1;
            }
            for x in (w - 3)..w {
                let off = (y * w + x) * 4;
                sum_right += px[off] as u64;
                count_right += 1;
            }
        }
        let avg_left = sum_left as f32 / count_left as f32;
        let avg_right = sum_right as f32 / count_right as f32;
        // Originals were 30 and 220. After edge-preserving filter, both
        // halves should still be very close to their originals.
        assert!(
            (avg_left - 30.0).abs() < 5.0,
            "left half should remain dark; got {avg_left}"
        );
        assert!(
            (avg_right - 220.0).abs() < 5.0,
            "right half should remain bright; got {avg_right}"
        );
        // And the contrast must still be huge.
        assert!(
            avg_right - avg_left > 150.0,
            "edge contrast collapsed: left={avg_left}, right={avg_right}"
        );
    }
}
