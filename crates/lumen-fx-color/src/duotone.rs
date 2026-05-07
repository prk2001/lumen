//! Duotone — luma-driven color map.
//!
//! For each pixel, compute Rec.709 luma, then linearly interpolate
//! between a `shadow_color` (at Y=0) and a `highlight_color` (at Y=1):
//!
//! ```text
//! Y     = 0.2126*R + 0.7152*G + 0.0722*B
//! mapped = shadow + (highlight - shadow) * Y
//! out    = mix(in, mapped, amount)
//! ```
//!
//! This is the classical "colorize by luminance" technique used to
//! turn a grayscale or low-color image into a duotone — pair with
//! `channel_isolate(luma)` upstream for a clean monochrome input.
//! Works as the heuristic backbone of `lumen colorize`'s
//! `--mode heuristic` path.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct Duotone;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-color.duotone",
    display_name: "Duotone",
    description: "Luma-driven shadow/highlight color map. Heuristic colorize.",
    category: Category::Color,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "shadow_r",
        display_name: "Shadow R",
        description: "Red channel of the shadow color.",
        kind: ParamKind::Float { default: 0.04, min: Some(0.0), max: Some(1.0) },
    },
    ParamSpec {
        id: "shadow_g",
        display_name: "Shadow G",
        description: "Green channel of the shadow color.",
        kind: ParamKind::Float { default: 0.07, min: Some(0.0), max: Some(1.0) },
    },
    ParamSpec {
        id: "shadow_b",
        display_name: "Shadow B",
        description: "Blue channel of the shadow color.",
        kind: ParamKind::Float { default: 0.18, min: Some(0.0), max: Some(1.0) },
    },
    ParamSpec {
        id: "highlight_r",
        display_name: "Highlight R",
        description: "Red channel of the highlight color.",
        kind: ParamKind::Float { default: 0.95, min: Some(0.0), max: Some(1.0) },
    },
    ParamSpec {
        id: "highlight_g",
        display_name: "Highlight G",
        description: "Green channel of the highlight color.",
        kind: ParamKind::Float { default: 0.78, min: Some(0.0), max: Some(1.0) },
    },
    ParamSpec {
        id: "highlight_b",
        display_name: "Highlight B",
        description: "Blue channel of the highlight color.",
        kind: ParamKind::Float { default: 0.42, min: Some(0.0), max: Some(1.0) },
    },
    ParamSpec {
        id: "amount",
        display_name: "Amount",
        description: "0 = pass-through, 1 = fully duotoned.",
        kind: ParamKind::Float { default: 0.85, min: Some(0.0), max: Some(1.0) },
    },
];

impl Effect for Duotone {
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
        let sr = params.get_float("shadow_r").unwrap_or(0.04) as f32;
        let sg = params.get_float("shadow_g").unwrap_or(0.07) as f32;
        let sb = params.get_float("shadow_b").unwrap_or(0.18) as f32;
        let hr = params.get_float("highlight_r").unwrap_or(0.95) as f32;
        let hg = params.get_float("highlight_g").unwrap_or(0.78) as f32;
        let hb = params.get_float("highlight_b").unwrap_or(0.42) as f32;
        let amount = params.get_float("amount").unwrap_or(0.85).clamp(0.0, 1.0) as f32;

        if amount == 0.0 {
            return Ok(input);
        }

        let mut frame = input.into_rgba_f32_linear();
        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");
        for px in pixels.chunks_exact_mut(4) {
            let y = 0.212_6 * px[0] + 0.715_2 * px[1] + 0.072_2 * px[2];
            let dr = sr + (hr - sr) * y;
            let dg = sg + (hg - sg) * y;
            let db = sb + (hb - sb) * y;
            px[0] = (px[0] * (1.0 - amount) + dr * amount).clamp(0.0, 1.0);
            px[1] = (px[1] * (1.0 - amount) + dg * amount).clamp(0.0, 1.0);
            px[2] = (px[2] * (1.0 - amount) + db * amount).clamp(0.0, 1.0);
        }
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue, PixelData};

    fn solid(rgb: [u8; 3]) -> Frame {
        Frame::new(
            1,
            1,
            PixelData::Rgba8(vec![rgb[0], rgb[1], rgb[2], 255]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap()
    }

    #[test]
    fn amount_zero_is_passthrough() {
        let d = Duotone;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("amount", ParamValue::Float(0.0));
        p.validate_and_fill(d.parameters()).unwrap();
        let f = solid([100, 150, 200]);
        let out = d.apply(&mut ctx, f, &p).unwrap();
        assert_eq!(out.layout(), lumen_core::PixelLayout::Rgba8);
    }

    #[test]
    fn black_input_maps_toward_shadow_color() {
        let d = Duotone;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("amount", ParamValue::Float(1.0));
        // navy shadow
        p.insert("shadow_r", ParamValue::Float(0.10));
        p.insert("shadow_g", ParamValue::Float(0.10));
        p.insert("shadow_b", ParamValue::Float(0.40));
        p.validate_and_fill(d.parameters()).unwrap();
        let black = solid([0, 0, 0]);
        let out = d.apply(&mut ctx, black, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!() };
        // Black → Y=0 → mapped = shadow color exactly. Then re-encoded via sRGB.
        // The blue channel must be the largest in the output.
        assert!(px[2] > px[0], "expected shadow tint to dominate B over R, got {:?}", &px[..3]);
        assert!(px[2] > px[1], "expected shadow tint to dominate B over G, got {:?}", &px[..3]);
    }

    #[test]
    fn white_input_maps_toward_highlight_color() {
        let d = Duotone;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("amount", ParamValue::Float(1.0));
        // amber highlight
        p.insert("highlight_r", ParamValue::Float(1.00));
        p.insert("highlight_g", ParamValue::Float(0.70));
        p.insert("highlight_b", ParamValue::Float(0.20));
        p.validate_and_fill(d.parameters()).unwrap();
        let white = solid([255, 255, 255]);
        let out = d.apply(&mut ctx, white, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!() };
        // White → Y=1 → mapped = highlight color exactly.
        assert!(px[0] > px[2], "expected highlight tint to dominate R over B, got {:?}", &px[..3]);
        assert!(px[1] > px[2], "expected highlight tint to dominate G over B, got {:?}", &px[..3]);
    }
}
