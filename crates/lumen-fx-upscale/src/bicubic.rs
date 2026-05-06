//! Bicubic upscale with the Mitchell-Netravali kernel.
//!
//! Default `B=1/3, C=1/3` — Mitchell's recommended defaults. Sharper
//! profiles (Catmull-Rom B=0,C=0.5) and softer (B=1,C=0) are
//! parameterizable but kept minimal for Phase 1; AI upscalers in Phase
//! 2 supersede this for high-quality work.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, PixelData, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct Bicubic;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-upscale.bicubic",
    display_name: "Bicubic Upscale",
    description: "Mitchell-Netravali bicubic resample to a scale factor.",
    category: Category::Upscale,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[ParamSpec {
    id: "scale",
    display_name: "Scale",
    description: "Multiplier applied to both dimensions. 2.0 doubles size.",
    kind: ParamKind::Float { default: 2.0, min: Some(0.25), max: Some(8.0) },
}];

impl Effect for Bicubic {
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
        let scale = params.get_float("scale").unwrap_or(2.0).max(0.25) as f32;

        let frame = input.into_rgba_f32_linear();
        let src_w = frame.width as usize;
        let src_h = frame.height as usize;
        let src = frame.as_f32().expect("RgbaF32 after lift");

        let dst_w = ((src_w as f32 * scale).round() as usize).max(1);
        let dst_h = ((src_h as f32 * scale).round() as usize).max(1);

        let mut out = vec![0.0f32; dst_w * dst_h * 4];
        let stride_src = src_w * 4;

        let x_ratio = src_w as f32 / dst_w as f32;
        let y_ratio = src_h as f32 / dst_h as f32;

        // Precompute kernel weights per output column for the horizontal
        // pass (separable bicubic: sample 4 rows, then weight vertically).
        let mut row_buf = vec![0.0f32; dst_w * 4];
        let mut sampled_rows = vec![0.0f32; 4 * dst_w * 4]; // 4 rows x dst_w pixels x 4 channels

        for ty in 0..dst_h {
            let sy = (ty as f32 + 0.5) * y_ratio - 0.5;
            let sy_floor = sy.floor() as isize;
            let fy = sy - sy_floor as f32;

            // Sample 4 source rows, doing horizontal interpolation into
            // sampled_rows (sy_floor-1, sy_floor, sy_floor+1, sy_floor+2).
            for k in 0..4 {
                let row_idx = (sy_floor + k - 1).clamp(0, src_h as isize - 1) as usize;
                let src_row = &src[row_idx * stride_src..(row_idx + 1) * stride_src];
                let dst_row = &mut sampled_rows[k as usize * dst_w * 4..(k as usize + 1) * dst_w * 4];

                for tx in 0..dst_w {
                    let sx = (tx as f32 + 0.5) * x_ratio - 0.5;
                    let sx_floor = sx.floor() as isize;
                    let fx = sx - sx_floor as f32;

                    let mut acc = [0.0f32; 4];
                    for j in 0..4 {
                        let xi = (sx_floor + j - 1).clamp(0, src_w as isize - 1) as usize;
                        let w = mitchell((j as f32 - 1.0) - fx);
                        let p = &src_row[xi * 4..xi * 4 + 4];
                        acc[0] += p[0] * w;
                        acc[1] += p[1] * w;
                        acc[2] += p[2] * w;
                        acc[3] += p[3] * w;
                    }
                    let off = tx * 4;
                    dst_row[off] = acc[0];
                    dst_row[off + 1] = acc[1];
                    dst_row[off + 2] = acc[2];
                    dst_row[off + 3] = acc[3];
                }
            }

            // Vertical pass through the 4 rows.
            for tx in 0..dst_w {
                let mut acc = [0.0f32; 4];
                for k in 0..4 {
                    let w = mitchell((k as f32 - 1.0) - fy);
                    let off = k * dst_w * 4 + tx * 4;
                    acc[0] += sampled_rows[off] * w;
                    acc[1] += sampled_rows[off + 1] * w;
                    acc[2] += sampled_rows[off + 2] * w;
                    acc[3] += sampled_rows[off + 3] * w;
                }
                let off = tx * 4;
                row_buf[off] = acc[0].clamp(0.0, 1.0);
                row_buf[off + 1] = acc[1].clamp(0.0, 1.0);
                row_buf[off + 2] = acc[2].clamp(0.0, 1.0);
                row_buf[off + 3] = acc[3].clamp(0.0, 1.0);
            }

            let row_dst = &mut out[ty * dst_w * 4..(ty + 1) * dst_w * 4];
            row_dst.copy_from_slice(&row_buf);
        }

        Frame::new(
            dst_w as u32,
            dst_h as u32,
            PixelData::RgbaF32(out),
            frame.color_space.clone(),
            frame.pts,
        )
    }
}

/// Mitchell-Netravali kernel with B = C = 1/3.
fn mitchell(x: f32) -> f32 {
    const B: f32 = 1.0 / 3.0;
    const C: f32 = 1.0 / 3.0;
    let x = x.abs();
    if x < 1.0 {
        ((12.0 - 9.0 * B - 6.0 * C) * x.powi(3)
            + (-18.0 + 12.0 * B + 6.0 * C) * x.powi(2)
            + (6.0 - 2.0 * B))
            / 6.0
    } else if x < 2.0 {
        ((-B - 6.0 * C) * x.powi(3)
            + (6.0 * B + 30.0 * C) * x.powi(2)
            + (-12.0 * B - 48.0 * C) * x
            + (8.0 * B + 24.0 * C))
            / 6.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue};

    #[test]
    fn doubles_dimensions() {
        let b = Bicubic;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("scale", ParamValue::Float(2.0));
        p.validate_and_fill(b.parameters()).unwrap();

        let f =
            Frame::new(8, 6, PixelData::Rgba8(vec![100; 8 * 6 * 4]), ColorSpace::SRgb, None)
                .unwrap();
        let out = b.apply(&mut ctx, f, &p).unwrap();
        assert_eq!(out.width, 16);
        assert_eq!(out.height, 12);
    }

    #[test]
    fn solid_image_unchanged_under_upscale() {
        let b = Bicubic;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("scale", ParamValue::Float(2.0));
        p.validate_and_fill(b.parameters()).unwrap();

        let f =
            Frame::new(8, 8, PixelData::Rgba8(vec![80; 8 * 8 * 4]), ColorSpace::SRgb, None)
                .unwrap();
        let out = b.apply(&mut ctx, f, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!() };
        // Bicubic on a constant image should produce the same constant.
        assert!(px.iter().all(|&v| (v as i32 - 80).abs() <= 1));
    }
}
