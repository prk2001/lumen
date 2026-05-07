//! Catmull-Rom upscale.
//!
//! Catmull-Rom is the standard sharper bicubic — `B = 0, C = 0.5` in the
//! Mitchell-Netravali parameterization. It gives crisper edges than the
//! Mitchell defaults at the cost of mild ringing/overshoot, which is
//! often desirable for photographic upscales.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, PixelData, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct CatmullRom;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-upscale.catmull_rom",
    display_name: "Catmull-Rom Upscale",
    description: "Catmull-Rom bicubic resample (B=0, C=1/2) — sharper than Mitchell.",
    category: Category::Upscale,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[ParamSpec {
    id: "scale",
    display_name: "Scale",
    description: "Multiplier applied to both dimensions. 2.0 doubles size.",
    kind: ParamKind::Float {
        default: 2.0,
        min: Some(0.25),
        max: Some(8.0),
    },
}];

impl Effect for CatmullRom {
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

        let mut row_buf = vec![0.0f32; dst_w * 4];
        let mut sampled_rows = vec![0.0f32; 4 * dst_w * 4];

        for ty in 0..dst_h {
            let sy = (ty as f32 + 0.5) * y_ratio - 0.5;
            let sy_floor = sy.floor() as isize;
            let fy = sy - sy_floor as f32;

            for k in 0..4 {
                let row_idx = (sy_floor + k - 1).clamp(0, src_h as isize - 1) as usize;
                let src_row = &src[row_idx * stride_src..(row_idx + 1) * stride_src];
                let dst_row =
                    &mut sampled_rows[k as usize * dst_w * 4..(k as usize + 1) * dst_w * 4];

                for tx in 0..dst_w {
                    let sx = (tx as f32 + 0.5) * x_ratio - 0.5;
                    let sx_floor = sx.floor() as isize;
                    let fx = sx - sx_floor as f32;

                    let mut acc = [0.0f32; 4];
                    for j in 0..4 {
                        let xi = (sx_floor + j - 1).clamp(0, src_w as isize - 1) as usize;
                        let w = catmull_rom((j as f32 - 1.0) - fx);
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

            for tx in 0..dst_w {
                let mut acc = [0.0f32; 4];
                for k in 0..4 {
                    let w = catmull_rom((k as f32 - 1.0) - fy);
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

/// Catmull-Rom kernel — Mitchell-Netravali with `B = 0, C = 1/2`.
///
/// Inlined rather than calling a generic Mitchell helper so the constants
/// fold at compile time.
fn catmull_rom(x: f32) -> f32 {
    const B: f32 = 0.0;
    const C: f32 = 0.5;
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
    use crate::Bicubic;
    use lumen_core::{ColorSpace, ParamValue};

    #[test]
    fn doubles_dimensions() {
        let cr = CatmullRom;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("scale", ParamValue::Float(2.0));
        p.validate_and_fill(cr.parameters()).unwrap();

        let f = Frame::new(
            8,
            6,
            PixelData::Rgba8(vec![100; 8 * 6 * 4]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();
        let out = cr.apply(&mut ctx, f, &p).unwrap();
        assert_eq!(out.width, 16);
        assert_eq!(out.height, 12);
    }

    #[test]
    fn sharper_than_mitchell_on_step_edge() {
        // A 1D step edge replicated vertically. Use mid-tones (0.4 -> 0.6
        // in linear) so that the negative-lobe overshoot/undershoot
        // stays inside the [0, 1] clamp range and is preserved in the
        // output. After 2x upscale, Catmull-Rom (C=0.5) should produce
        // a larger output range than Mitchell (C=1/3) because of its
        // deeper negative lobes.
        let w = 16;
        let h = 4;
        // 0.4 and 0.6 in linear, encoded to sRGB-u8.
        // (Frame stores u8 sRGB; `into_rgba_f32_linear` decodes back.)
        let lo_lin = 0.4f32;
        let hi_lin = 0.6f32;
        let lo_u8 = (linear_to_srgb_u8(lo_lin)) as u8;
        let hi_u8 = (linear_to_srgb_u8(hi_lin)) as u8;

        let mut buf = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let v = if x >= w / 2 { hi_u8 } else { lo_u8 };
                let i = (y * w + x) * 4;
                buf[i] = v;
                buf[i + 1] = v;
                buf[i + 2] = v;
                buf[i + 3] = 255;
            }
        }

        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("scale", ParamValue::Float(2.0));
        p.validate_and_fill(CatmullRom.parameters()).unwrap();

        let f_cr = Frame::new(
            w as u32,
            h as u32,
            PixelData::Rgba8(buf.clone()),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();
        let f_mn = Frame::new(
            w as u32,
            h as u32,
            PixelData::Rgba8(buf.clone()),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();

        let cr_out = CatmullRom.apply(&mut ctx, f_cr, &p).unwrap();
        let mn_out = Bicubic.apply(&mut ctx, f_mn, &p).unwrap();

        assert_eq!(cr_out.width, mn_out.width);
        assert_eq!(cr_out.height, mn_out.height);

        // Compare in linear-f32 space (what the kernels operate in).
        let cr_lin = cr_out.into_rgba_f32_linear();
        let mn_lin = mn_out.into_rgba_f32_linear();
        let cr_px = cr_lin.as_f32().unwrap();
        let mn_px = mn_lin.as_f32().unwrap();

        let stride = cr_lin.width as usize * 4;
        // Pick an interior row to avoid vertical edge clamp at top/bottom.
        let row = 2 * stride;
        let cr_row = &cr_px[row..row + stride];
        let mn_row = &mn_px[row..row + stride];

        let cr_min = cr_row
            .iter()
            .step_by(4)
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let cr_max = cr_row
            .iter()
            .step_by(4)
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mn_min = mn_row
            .iter()
            .step_by(4)
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let mn_max = mn_row
            .iter()
            .step_by(4)
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let cr_range = cr_max - cr_min;
        let mn_range = mn_max - mn_min;

        // Catmull-Rom's deeper negative lobes produce both a larger
        // overshoot (above hi) and a deeper undershoot (below lo) than
        // Mitchell — strictly larger output range.
        assert!(
            cr_range > mn_range + 1e-4,
            "expected Catmull-Rom range {cr_range} > Mitchell range {mn_range}",
        );
    }

    /// Linear -> sRGB-u8 round, reusing the standard piecewise formula.
    fn linear_to_srgb_u8(c: f32) -> u32 {
        let s = if c <= 0.003_130_8 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (s.clamp(0.0, 1.0) * 255.0).round() as u32
    }
}
