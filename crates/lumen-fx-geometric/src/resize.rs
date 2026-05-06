//! Bilinear resize.
//!
//! Resamples the input to an explicit target width/height. Phase 1
//! ships bilinear only; Lanczos/Mitchell/AI upscalers come from
//! `lumen-fx-upscale` (Cat 12) in Phase 2.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, PixelData, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct Resize;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-geometric.resize",
    display_name: "Resize",
    description: "Bilinear resample to an explicit pixel size.",
    category: Category::Geometric,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "width",
        display_name: "Width",
        description: "Target width in pixels.",
        kind: ParamKind::Int { default: 1024, min: Some(1), max: Some(65535) },
    },
    ParamSpec {
        id: "height",
        display_name: "Height",
        description: "Target height in pixels.",
        kind: ParamKind::Int { default: 1024, min: Some(1), max: Some(65535) },
    },
];

impl Effect for Resize {
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
        let target_w = params.get_int("width").unwrap_or(1024).max(1) as u32;
        let target_h = params.get_int("height").unwrap_or(1024).max(1) as u32;

        let frame = input.into_rgba_f32_linear();
        let src_w = frame.width as usize;
        let src_h = frame.height as usize;
        let src = frame.as_f32().expect("RgbaF32 after lift");

        let mut out = vec![0.0f32; (target_w as usize) * (target_h as usize) * 4];
        let stride_src = src_w * 4;

        // Bilinear resampling. Map output pixel center to source space.
        let x_ratio = src_w as f32 / target_w as f32;
        let y_ratio = src_h as f32 / target_h as f32;

        for ty in 0..target_h as usize {
            let sy = (ty as f32 + 0.5) * y_ratio - 0.5;
            let sy0 = sy.floor().clamp(0.0, (src_h - 1) as f32) as usize;
            let sy1 = (sy0 + 1).min(src_h - 1);
            let fy = (sy - sy0 as f32).clamp(0.0, 1.0);

            for tx in 0..target_w as usize {
                let sx = (tx as f32 + 0.5) * x_ratio - 0.5;
                let sx0 = sx.floor().clamp(0.0, (src_w - 1) as f32) as usize;
                let sx1 = (sx0 + 1).min(src_w - 1);
                let fx = (sx - sx0 as f32).clamp(0.0, 1.0);

                let p00 = &src[sy0 * stride_src + sx0 * 4..sy0 * stride_src + sx0 * 4 + 4];
                let p10 = &src[sy0 * stride_src + sx1 * 4..sy0 * stride_src + sx1 * 4 + 4];
                let p01 = &src[sy1 * stride_src + sx0 * 4..sy1 * stride_src + sx0 * 4 + 4];
                let p11 = &src[sy1 * stride_src + sx1 * 4..sy1 * stride_src + sx1 * 4 + 4];

                let off = (ty * target_w as usize + tx) * 4;
                for c in 0..4 {
                    let top = p00[c] * (1.0 - fx) + p10[c] * fx;
                    let bot = p01[c] * (1.0 - fx) + p11[c] * fx;
                    out[off + c] = top * (1.0 - fy) + bot * fy;
                }
            }
        }

        Frame::new(
            target_w,
            target_h,
            PixelData::RgbaF32(out),
            frame.color_space.clone(),
            frame.pts,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue};

    #[test]
    fn halve_size() {
        let r = Resize;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("width", ParamValue::Int(2));
        p.insert("height", ParamValue::Int(2));
        p.validate_and_fill(r.parameters()).unwrap();

        let f =
            Frame::new(4, 4, PixelData::Rgba8(vec![100; 4 * 4 * 4]), ColorSpace::SRgb, None)
                .unwrap();
        let out = r.apply(&mut ctx, f, &p).unwrap();
        assert_eq!(out.width, 2);
        assert_eq!(out.height, 2);
    }
}
