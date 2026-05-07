//! Deblock — soft post-process that smooths visible JPEG-style 8x8 block
//! boundaries.
//!
//! Algorithm:
//!
//! 1. Lift to linear RGBA float.
//! 2. For each horizontal block boundary (rows `block_size`, `2*block_size`,
//!    …), apply a 1-D vertical Gaussian along the column direction to the
//!    rows within `radius` of that boundary. Same for vertical block
//!    boundaries (columns).
//! 3. Other pixels are left untouched.
//!
//! This is a deliberately simple deblocking baseline. DCT-domain
//! deblocking and AI artifact removal land in Phase 4.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct Deblock;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-compression.deblock",
    display_name: "Deblock",
    description: "Soft 1-D Gaussian smoothing across JPEG-style block boundaries.",
    category: Category::Compression,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "block_size",
        display_name: "Block Size",
        description: "Block size to assume (8 for JPEG).",
        kind: ParamKind::Int { default: 8, min: Some(4), max: Some(32) },
    },
    ParamSpec {
        id: "strength",
        display_name: "Strength",
        description: "Gaussian sigma along boundary rows/columns.",
        kind: ParamKind::Float { default: 0.6, min: Some(0.0), max: Some(4.0) },
    },
];

impl Effect for Deblock {
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
        let block_size = params.get_int("block_size").unwrap_or(8).max(1) as usize;
        let strength = params.get_float("strength").unwrap_or(0.6) as f32;

        if strength <= 0.0 {
            return Ok(input);
        }

        let mut frame = input.into_rgba_f32_linear();
        let w = frame.width as usize;
        let h = frame.height as usize;
        if w == 0 || h == 0 {
            return Ok(frame);
        }
        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");

        let kernel = build_gaussian_kernel(strength);
        let radius = kernel.len() / 2;

        // Snapshot of source pixels so all reads come from the original
        // image rather than a partially-written buffer.
        let src = pixels.to_vec();

        // Smooth pixels near horizontal block boundaries (rows that are
        // multiples of block_size, excluding row 0). For each affected
        // row, blend along the vertical (Y) direction.
        smooth_horizontal_boundaries(pixels, &src, w, h, block_size, &kernel, radius);

        // Snapshot again so the vertical-boundary pass also reads from
        // the post-horizontal-pass image (the two passes compose).
        let src2 = pixels.to_vec();

        // Smooth pixels near vertical block boundaries (columns that are
        // multiples of block_size, excluding column 0). For each affected
        // column, blend along the horizontal (X) direction.
        smooth_vertical_boundaries(pixels, &src2, w, h, block_size, &kernel, radius);

        Ok(frame)
    }
}

fn kernel_half_width(sigma: f32) -> usize {
    ((2.0 * sigma).ceil() as usize).clamp(1, 32)
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

/// For each horizontal block boundary at row `b * block_size` (b >= 1),
/// recompute the pixels in rows `[boundary - radius, boundary + radius - 1]`
/// (clamped to image extent) by blending vertically with the 1-D Gaussian.
fn smooth_horizontal_boundaries(
    dst: &mut [f32],
    src: &[f32],
    w: usize,
    h: usize,
    block_size: usize,
    kernel: &[f32],
    radius: usize,
) {
    let stride = w * 4;
    let half = kernel.len() / 2;

    let mut boundary = block_size;
    while boundary < h {
        // Affected rows straddle the boundary line. Touch `radius` rows
        // above (the last rows of the block above) and `radius` rows
        // below (the first rows of the block below).
        let y_lo = boundary.saturating_sub(radius);
        let y_hi = (boundary + radius).min(h); // exclusive
        for y in y_lo..y_hi {
            for x in 0..w {
                let mut acc = [0.0f32; 4];
                for (i, &k) in kernel.iter().enumerate() {
                    let yi = (y as isize + i as isize - half as isize)
                        .clamp(0, h as isize - 1) as usize;
                    let off = yi * stride + x * 4;
                    acc[0] += src[off] * k;
                    acc[1] += src[off + 1] * k;
                    acc[2] += src[off + 2] * k;
                    acc[3] += src[off + 3] * k;
                }
                let off = y * stride + x * 4;
                dst[off] = acc[0];
                dst[off + 1] = acc[1];
                dst[off + 2] = acc[2];
                dst[off + 3] = acc[3];
            }
        }
        boundary += block_size;
    }
}

/// Mirror of [`smooth_horizontal_boundaries`] for vertical block edges.
fn smooth_vertical_boundaries(
    dst: &mut [f32],
    src: &[f32],
    w: usize,
    h: usize,
    block_size: usize,
    kernel: &[f32],
    radius: usize,
) {
    let stride = w * 4;
    let half = kernel.len() / 2;

    let mut boundary = block_size;
    while boundary < w {
        let x_lo = boundary.saturating_sub(radius);
        let x_hi = (boundary + radius).min(w); // exclusive
        for y in 0..h {
            for x in x_lo..x_hi {
                let mut acc = [0.0f32; 4];
                for (i, &k) in kernel.iter().enumerate() {
                    let xi = (x as isize + i as isize - half as isize)
                        .clamp(0, w as isize - 1) as usize;
                    let off = y * stride + xi * 4;
                    acc[0] += src[off] * k;
                    acc[1] += src[off + 1] * k;
                    acc[2] += src[off + 2] * k;
                    acc[3] += src[off + 3] * k;
                }
                let off = y * stride + x * 4;
                dst[off] = acc[0];
                dst[off + 1] = acc[1];
                dst[off + 2] = acc[2];
                dst[off + 3] = acc[3];
            }
        }
        boundary += block_size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue, PixelData};

    fn make_solid(w: u32, h: u32, value: u8) -> Frame {
        let n = (w as usize) * (h as usize) * 4;
        let mut data = vec![value; n];
        // Force alpha to 255 so sRGB round-trip is well-defined.
        for i in (3..n).step_by(4) {
            data[i] = 255;
        }
        Frame::new(w, h, PixelData::Rgba8(data), ColorSpace::SRgb, None).unwrap()
    }

    #[test]
    fn strength_zero_passthrough() {
        // strength = 0 short-circuits — frame is returned unchanged.
        let d = Deblock;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("strength", ParamValue::Float(0.0));
        p.validate_and_fill(d.parameters()).unwrap();

        let frame = make_solid(16, 16, 100);
        let original = frame.clone();
        let out = d.apply(&mut ctx, frame, &p).unwrap();
        assert_eq!(out.layout(), original.layout());
        assert_eq!(out.data, original.data);
    }

    #[test]
    fn solid_image_is_unchanged() {
        // Deblocking a constant-value image must not change it.
        let d = Deblock;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("strength", ParamValue::Float(1.5));
        p.insert("block_size", ParamValue::Int(8));
        p.validate_and_fill(d.parameters()).unwrap();

        let frame = make_solid(16, 16, 80);
        let out = d.apply(&mut ctx, frame, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!() };
        assert!(
            px.chunks_exact(4).all(|p| {
                (p[0] as i32 - 80).abs() <= 1
                    && (p[1] as i32 - 80).abs() <= 1
                    && (p[2] as i32 - 80).abs() <= 1
                    && p[3] == 255
            }),
            "deblock altered a solid image",
        );
    }

    #[test]
    fn synthetic_blocking_edge_is_reduced() {
        // Build a 16x16 RGBA image with a sharp horizontal step at the
        // 8-row block boundary: rows 0..8 are dark gray (40), rows 8..16
        // are light gray (200). This is the kind of artifact deblock
        // should soften.
        let w = 16usize;
        let h = 16usize;
        let mut data = vec![0u8; w * h * 4];
        for y in 0..h {
            let v = if y < 8 { 40u8 } else { 200u8 };
            for x in 0..w {
                let off = (y * w + x) * 4;
                data[off] = v;
                data[off + 1] = v;
                data[off + 2] = v;
                data[off + 3] = 255;
            }
        }
        let frame = Frame::new(
            w as u32,
            h as u32,
            PixelData::Rgba8(data.clone()),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();

        let d = Deblock;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("strength", ParamValue::Float(1.5));
        p.insert("block_size", ParamValue::Int(8));
        p.validate_and_fill(d.parameters()).unwrap();

        let out = d.apply(&mut ctx, frame, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!() };

        // Edge magnitude across the 8-row boundary, averaged over
        // columns. Compare row 7 (just above) and row 8 (just below).
        let edge_before: i32 = (0..w)
            .map(|x| {
                let a = data[(7 * w + x) * 4] as i32;
                let b = data[(8 * w + x) * 4] as i32;
                (b - a).abs()
            })
            .sum();
        let edge_after: i32 = (0..w)
            .map(|x| {
                let a = px[(7 * w + x) * 4] as i32;
                let b = px[(8 * w + x) * 4] as i32;
                (b - a).abs()
            })
            .sum();

        assert!(
            edge_after < edge_before,
            "expected boundary edge to soften: before={edge_before} after={edge_after}",
        );

        // Pixels far from any boundary (row 3, col 3 — interior of top
        // block, away from any row/col multiple of 8 ± radius) should
        // remain at their original value.
        let untouched_off = (3 * w + 3) * 4;
        assert!(
            (px[untouched_off] as i32 - 40).abs() <= 1,
            "interior pixel unexpectedly altered: {} (was 40)",
            px[untouched_off],
        );
    }
}
