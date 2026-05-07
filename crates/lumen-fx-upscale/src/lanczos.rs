//! Lanczos (windowed-sinc) upscale.
//!
//! Sinc-based resampler with a Lanczos window of `a` lobes (`a = 2, 3, 4`
//! commonly). General consensus best-quality classical resampler for
//! photographic content; mild ringing on hard edges.
//!
//! Kernel:
//!   L(x) = sinc(x) * sinc(x / a)   for |x| < a
//!   L(x) = 0                        otherwise
//!   L(0) = 1
//!
//! Implementation is the same separable-2D structure as the bicubic
//! resampler, just with a wider 2a-tap support window and a normalization
//! step (the Lanczos kernel does not exactly sum to 1 at arbitrary
//! sub-pixel offsets, so we renormalize per-tap-set to avoid brightness
//! shifts).

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, PixelData, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct Lanczos;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-upscale.lanczos",
    display_name: "Lanczos Upscale",
    description: "Windowed-sinc resample with selectable lobe count (Lanczos2/3/4).",
    category: Category::Upscale,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "scale",
        display_name: "Scale",
        description: "Multiplier applied to both dimensions. 2.0 doubles size.",
        kind: ParamKind::Float {
            default: 2.0,
            min: Some(0.25),
            max: Some(8.0),
        },
    },
    ParamSpec {
        id: "lobes",
        display_name: "Lobes",
        description: "Lanczos parameter `a`: kernel support is +/- a source pixels.",
        kind: ParamKind::Int {
            default: 3,
            min: Some(2),
            max: Some(6),
        },
    },
];

impl Effect for Lanczos {
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
        let lobes = params.get_int("lobes").unwrap_or(3).clamp(2, 6) as isize;
        let taps = 2 * lobes as usize; // total taps per axis: 2a
        let a_f = lobes as f32;

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

        // Precompute horizontal weights and source indices per output column.
        // weights_x: dst_w * taps; xidx: dst_w * taps.
        let mut weights_x = vec![0.0f32; dst_w * taps];
        let mut xidx = vec![0usize; dst_w * taps];
        for tx in 0..dst_w {
            let sx = (tx as f32 + 0.5) * x_ratio - 0.5;
            let sx_floor = sx.floor() as isize;
            let fx = sx - sx_floor as f32;

            let off = tx * taps;
            let mut sum = 0.0f32;
            for j in 0..taps {
                // Tap j corresponds to source x = sx_floor + (j - lobes + 1).
                let dx = (j as isize - (lobes - 1)) as f32 - fx;
                let w = lanczos(dx, a_f);
                weights_x[off + j] = w;
                sum += w;

                let xi =
                    (sx_floor + j as isize - (lobes - 1)).clamp(0, src_w as isize - 1) as usize;
                xidx[off + j] = xi;
            }
            // Normalize so the row weights sum to 1 (avoids DC drift).
            if sum.abs() > 1e-8 {
                let inv = 1.0 / sum;
                for j in 0..taps {
                    weights_x[off + j] *= inv;
                }
            }
        }

        // Buffer holding `taps` horizontally-resampled rows.
        let mut sampled_rows = vec![0.0f32; taps * dst_w * 4];
        let mut row_buf = vec![0.0f32; dst_w * 4];

        for ty in 0..dst_h {
            let sy = (ty as f32 + 0.5) * y_ratio - 0.5;
            let sy_floor = sy.floor() as isize;
            let fy = sy - sy_floor as f32;

            // Compute and normalize vertical weights for this output row.
            let mut wy = [0.0f32; 12]; // upper bound: 2*6 = 12
            let mut wy_sum = 0.0f32;
            for (k, slot) in wy.iter_mut().enumerate().take(taps) {
                let dy = (k as isize - (lobes - 1)) as f32 - fy;
                let w = lanczos(dy, a_f);
                *slot = w;
                wy_sum += w;
            }
            if wy_sum.abs() > 1e-8 {
                let inv = 1.0 / wy_sum;
                for slot in wy.iter_mut().take(taps) {
                    *slot *= inv;
                }
            }

            // Sample `taps` source rows, doing horizontal interpolation
            // into sampled_rows.
            for k in 0..taps {
                let row_idx =
                    (sy_floor + k as isize - (lobes - 1)).clamp(0, src_h as isize - 1) as usize;
                let src_row = &src[row_idx * stride_src..(row_idx + 1) * stride_src];
                let dst_row = &mut sampled_rows[k * dst_w * 4..(k + 1) * dst_w * 4];

                for tx in 0..dst_w {
                    let woff = tx * taps;
                    let mut acc = [0.0f32; 4];
                    for j in 0..taps {
                        let xi = xidx[woff + j];
                        let w = weights_x[woff + j];
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

            // Vertical pass through the `taps` rows.
            for tx in 0..dst_w {
                let mut acc = [0.0f32; 4];
                for (k, &w) in wy.iter().take(taps).enumerate() {
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

/// Normalized sinc: `sin(pi*x) / (pi*x)`, with `sinc(0) = 1`.
#[inline]
fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        1.0
    } else {
        let pix = std::f32::consts::PI * x;
        pix.sin() / pix
    }
}

/// Lanczos kernel with `a` lobes.
#[inline]
fn lanczos(x: f32, a: f32) -> f32 {
    let ax = x.abs();
    if ax < 1e-8 {
        1.0
    } else if ax < a {
        sinc(x) * sinc(x / a)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue};

    #[test]
    fn doubles_dimensions_8x6() {
        let l = Lanczos;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("scale", ParamValue::Float(2.0));
        p.insert("lobes", ParamValue::Int(3));
        p.validate_and_fill(l.parameters()).unwrap();

        let f = Frame::new(
            8,
            6,
            PixelData::Rgba8(vec![100; 8 * 6 * 4]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();
        let out = l.apply(&mut ctx, f, &p).unwrap();
        assert_eq!(out.width, 16);
        assert_eq!(out.height, 12);
    }

    #[test]
    fn solid_image_unchanged_under_upscale() {
        let l = Lanczos;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("scale", ParamValue::Float(2.0));
        p.insert("lobes", ParamValue::Int(3));
        p.validate_and_fill(l.parameters()).unwrap();

        let f = Frame::new(
            8,
            8,
            PixelData::Rgba8(vec![80; 8 * 8 * 4]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();
        let out = l.apply(&mut ctx, f, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else {
            panic!()
        };
        // A constant source under any sane resampler with normalized
        // weights must stay constant within a 1-LSB float-rounding margin.
        assert!(
            px.iter().all(|&v| (v as i32 - 80).abs() <= 1),
            "expected ~constant 80, saw range {}..{}",
            px.iter().min().unwrap(),
            px.iter().max().unwrap(),
        );
    }
}
