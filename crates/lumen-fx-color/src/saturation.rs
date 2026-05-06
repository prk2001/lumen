//! Saturation adjustment around the Rec.709 luminance axis.
//!
//! Math:
//!
//! ```text
//! Y   = 0.2126*R + 0.7152*G + 0.0722*B
//! out = mix(Y, in, amount)
//! ```
//!
//! `amount = 1.0` is pass-through, `0.0` is fully desaturated, `>1.0`
//! oversaturates.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct Saturation;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-color.saturation",
    display_name: "Saturation",
    description: "Mix between luminance-only (0.0) and original (1.0).",
    category: Category::Color,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[ParamSpec {
    id: "amount",
    display_name: "Amount",
    description: "0.0 = grayscale, 1.0 = identity, >1.0 boosts.",
    kind: ParamKind::Float { default: 1.0, min: Some(0.0), max: Some(2.0) },
}];

impl Effect for Saturation {
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
        let amount = params.get_float("amount").unwrap_or(1.0) as f32;

        let mut frame = input.into_rgba_f32_linear();
        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");
        for px in pixels.chunks_exact_mut(4) {
            let y = 0.212_6 * px[0] + 0.715_2 * px[1] + 0.072_2 * px[2];
            for c in &mut px[..3] {
                let v = y + (*c - y) * amount;
                *c = v.clamp(0.0, 1.0);
            }
        }
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue, PixelData};

    fn solid(r: u8, g: u8, b: u8) -> Frame {
        Frame::new(
            1,
            1,
            PixelData::Rgba8(vec![r, g, b, 255]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap()
    }

    #[test]
    fn amount_one_passthrough() {
        let sat = Saturation;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.validate_and_fill(sat.parameters()).unwrap();

        let red = solid(200, 30, 30);
        let out = sat.apply(&mut ctx, red, &p).unwrap();
        let out8 = out.into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out8.data else { panic!() };
        assert!((px[0] as i32 - 200).abs() <= 1);
        assert!((px[1] as i32 - 30).abs() <= 2);
        assert!((px[2] as i32 - 30).abs() <= 2);
    }

    #[test]
    fn amount_zero_grays_out() {
        let sat = Saturation;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("amount", ParamValue::Float(0.0));
        p.validate_and_fill(sat.parameters()).unwrap();

        let red = solid(255, 0, 0);
        let out = sat.apply(&mut ctx, red, &p).unwrap();
        let out8 = out.into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out8.data else { panic!() };
        // Equal R/G/B since collapsed to luminance.
        assert_eq!(px[0], px[1]);
        assert_eq!(px[1], px[2]);
    }
}
