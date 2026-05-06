//! Gamma adjustment.
//!
//! `out = in ^ (1 / gamma)`, applied per RGB channel in linear-light
//! space (alpha untouched). `gamma = 1.0` is identity, `gamma > 1.0`
//! brightens midtones, `gamma < 1.0` darkens them.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct Gamma;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-exposure.gamma",
    display_name: "Gamma",
    description: "Power-law brightness curve. Operates in linear light.",
    category: Category::Exposure,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[ParamSpec {
    id: "gamma",
    display_name: "Gamma",
    description: ">1 brightens midtones; <1 darkens. 1.0 is pass-through.",
    kind: ParamKind::Float { default: 1.0, min: Some(0.05), max: Some(10.0) },
}];

impl Effect for Gamma {
    fn metadata(&self) -> &EffectMetadata { &META }
    fn parameters(&self) -> &[ParamSpec] { PARAMS }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            deterministic: true,
            gpu: false,
            streamable: true,
            temporal: false,
        }
    }

    #[instrument(skip_all, fields(effect = META.id))]
    fn apply(&self, _ctx: &mut Context, input: Frame, params: &ParamValues) -> Result<Frame> {
        let gamma = params.get_float("gamma").unwrap_or(1.0).max(0.05) as f32;
        let inv = 1.0 / gamma;

        let mut frame = input.into_rgba_f32_linear();
        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");
        for px in pixels.chunks_exact_mut(4) {
            for c in &mut px[..3] {
                *c = c.max(0.0).powf(inv).clamp(0.0, 1.0);
            }
        }
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue, PixelData};

    #[test]
    fn gamma_one_passthrough() {
        let g = Gamma;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.validate_and_fill(g.parameters()).unwrap();
        let f = Frame::new(
            1,
            1,
            PixelData::Rgba8(vec![64, 128, 192, 255]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();
        let out = g.apply(&mut ctx, f, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!() };
        assert!((px[0] as i32 - 64).abs() <= 1);
        assert!((px[1] as i32 - 128).abs() <= 1);
        assert!((px[2] as i32 - 192).abs() <= 1);
    }

    #[test]
    fn gamma_two_brightens() {
        // gamma=2.0 => exponent 0.5 => sqrt of linear values, brighter.
        let g = Gamma;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("gamma", ParamValue::Float(2.0));
        p.validate_and_fill(g.parameters()).unwrap();
        let f = Frame::new(
            1,
            1,
            PixelData::Rgba8(vec![64, 64, 64, 255]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();
        let out = g.apply(&mut ctx, f, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!() };
        // Output should be brighter than input.
        assert!(px[0] > 64, "expected brightening, got {}", px[0]);
    }
}
