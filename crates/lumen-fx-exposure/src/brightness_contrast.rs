//! Brightness + contrast adjustment.
//!
//! This is the simplest possible exposure effect — a linear remap of
//! pixel values. Despite its simplicity it's the canonical demo for the
//! [`Effect`] trait, used throughout the test suite.
//!
//! Math (per RGB channel, alpha untouched):
//!
//! ```text
//! out = (in - 0.5) * contrast + 0.5 + brightness
//! ```
//!
//! - `brightness ∈ [-1.0, 1.0]` shifts the entire range up/down.
//! - `contrast   ∈ [ 0.0, 4.0]` scales around 0.5; `1.0` is a no-op.
//!
//! Operates in linear-light when fed a linear frame (recommended), and
//! in sRGB-encoded space if not — both are valid; just specify your
//! input.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use tracing::instrument;

/// Brightness/contrast effect — linear remap with no spatial component.
#[derive(Debug, Default)]
pub struct BrightnessContrast;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-exposure.brightness_contrast",
    display_name: "Brightness / Contrast",
    description: "Linearly remap each RGB channel; alpha is preserved.",
    category: Category::Exposure,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "brightness",
        display_name: "Brightness",
        description: "Additive offset applied after contrast. -1 = black, +1 = white.",
        kind: ParamKind::Float { default: 0.0, min: Some(-1.0), max: Some(1.0) },
    },
    ParamSpec {
        id: "contrast",
        display_name: "Contrast",
        description: "Multiplicative scale around 0.5. 1.0 = pass-through.",
        kind: ParamKind::Float { default: 1.0, min: Some(0.0), max: Some(4.0) },
    },
];

impl Effect for BrightnessContrast {
    fn metadata(&self) -> &EffectMetadata { &META }
    fn parameters(&self) -> &[ParamSpec] { PARAMS }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            deterministic: true,
            gpu: false, // GPU shader added later; CPU path is correct now
            streamable: true,
            temporal: false,
        }
    }

    #[instrument(skip_all, fields(effect = META.id))]
    fn apply(
        &self,
        _ctx: &mut Context,
        input: Frame,
        params: &ParamValues,
    ) -> Result<Frame> {
        let brightness = params.get_float("brightness").unwrap_or(0.0) as f32;
        let contrast = params.get_float("contrast").unwrap_or(1.0) as f32;

        // Lift to f32 RGBA for arithmetic.
        let mut frame = input.into_rgba_f32_linear();
        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");
        for px in pixels.chunks_exact_mut(4) {
            for c in &mut px[..3] {
                let v = (*c - 0.5) * contrast + 0.5 + brightness;
                *c = v.clamp(0.0, 1.0);
            }
            // Alpha (px[3]) untouched.
        }
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue, PixelData};

    fn unit_frame(rgb: [u8; 3]) -> Frame {
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
    fn defaults_are_passthrough() {
        let bc = BrightnessContrast;
        let mut ctx = Context::for_still_srgb();
        let mut params = ParamValues::new();
        params.validate_and_fill(bc.parameters()).unwrap();

        let mid = unit_frame([128, 128, 128]);
        let out = bc.apply(&mut ctx, mid, &params).unwrap();
        // After conversion to linear and back to sRGB the round-trip
        // should be lossless within ~1 ulp.
        let out8 = out.into_rgba_u8_srgb();
        let PixelData::Rgba8(p) = out8.data else { panic!() };
        assert!((p[0] as i32 - 128).abs() <= 1);
        assert!((p[1] as i32 - 128).abs() <= 1);
        assert!((p[2] as i32 - 128).abs() <= 1);
        assert_eq!(p[3], 255);
    }

    #[test]
    fn brightness_plus_one_clamps_white() {
        let bc = BrightnessContrast;
        let mut ctx = Context::for_still_srgb();
        let mut params = ParamValues::new();
        params.insert("brightness", ParamValue::Float(1.0));
        params.validate_and_fill(bc.parameters()).unwrap();

        let dark = unit_frame([10, 20, 30]);
        let out = bc.apply(&mut ctx, dark, &params).unwrap();
        let out8 = out.into_rgba_u8_srgb();
        let PixelData::Rgba8(p) = out8.data else { panic!() };
        assert_eq!(p[0], 255);
        assert_eq!(p[1], 255);
        assert_eq!(p[2], 255);
    }

    #[test]
    fn contrast_zero_collapses_to_mid() {
        let bc = BrightnessContrast;
        let mut ctx = Context::for_still_srgb();
        let mut params = ParamValues::new();
        params.insert("contrast", ParamValue::Float(0.0));
        params.validate_and_fill(bc.parameters()).unwrap();

        let any = unit_frame([10, 200, 100]);
        let out = bc.apply(&mut ctx, any, &params).unwrap();
        // Linear mid-gray (0.5) → sRGB 188 (linear_to_srgb(0.5) ≈ 0.7354 → 188).
        let out8 = out.into_rgba_u8_srgb();
        let PixelData::Rgba8(p) = out8.data else { panic!() };
        assert_eq!(p[0], p[1]);
        assert_eq!(p[1], p[2]);
        assert!((p[0] as i32 - 188).abs() <= 1);
    }
}
