//! Single-channel isolation as grayscale RGB.
//!
//! Pulls one channel out of an RGBA frame — R, G, B, alpha, or
//! Rec.709 luma — and emits it as a grayscale image (R = G = B = value).
//! Useful for IR-like analysis on visible-light imagery, channel
//! inspection, and debug visualization. True multi-spectral / IR / UV
//! / polarization support needs source data we don't have in Phase 1.
//!
//! Math:
//!
//! ```text
//! v   = pick(channel, R, G, B, A)
//!     = 0.2126*R + 0.7152*G + 0.0722*B   (when channel == "luma")
//! out = (v, v, v, A)                     (or 1 - v if invert)
//! ```

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct ChannelIsolate;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-modalities.channel_isolate",
    display_name: "Channel Isolate",
    description: "Extract a single channel (R/G/B/A/luma) as grayscale RGB.",
    category: Category::Modalities,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "channel",
        display_name: "Channel",
        description: "Which channel to isolate.",
        kind: ParamKind::Choice {
            default: "luma",
            options: &["r", "g", "b", "a", "luma"],
        },
    },
    ParamSpec {
        id: "invert",
        display_name: "Invert",
        description: "If true, output 1 - value (negative).",
        kind: ParamKind::Bool { default: false },
    },
];

impl Effect for ChannelIsolate {
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
        let channel = params.get_string("channel").unwrap_or("luma");
        let invert = params.get_bool("invert").unwrap_or(false);

        let mut frame = input.into_rgba_f32_linear();
        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");
        for px in pixels.chunks_exact_mut(4) {
            let v = match channel {
                "r" => px[0],
                "g" => px[1],
                "b" => px[2],
                "a" => px[3],
                // "luma" and any unknown value fall back to Rec.709 luma.
                _ => 0.212_6 * px[0] + 0.715_2 * px[1] + 0.072_2 * px[2],
            };
            let v = if invert { 1.0 - v } else { v };
            let v = v.clamp(0.0, 1.0);
            px[0] = v;
            px[1] = v;
            px[2] = v;
            // px[3] (alpha) preserved.
        }
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue, PixelData};

    fn solid_f32(r: f32, g: f32, b: f32, a: f32) -> Frame {
        Frame::new(
            1,
            1,
            PixelData::RgbaF32(vec![r, g, b, a]),
            ColorSpace::LinearSRgb,
            None,
        )
        .unwrap()
    }

    #[test]
    fn red_channel_isolation() {
        let fx = ChannelIsolate;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("channel", ParamValue::String("r".into()));
        p.validate_and_fill(fx.parameters()).unwrap();

        let input = solid_f32(1.0, 0.0, 0.0, 1.0);
        let out = fx.apply(&mut ctx, input, &p).unwrap();
        let px = out.as_f32().unwrap();
        assert!((px[0] - 1.0).abs() < 1e-6);
        assert!((px[1] - 1.0).abs() < 1e-6);
        assert!((px[2] - 1.0).abs() < 1e-6);
        assert!((px[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn luma_default_for_pure_green() {
        let fx = ChannelIsolate;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        // No "channel" insert — should fall through to default "luma".
        p.validate_and_fill(fx.parameters()).unwrap();
        assert_eq!(p.get_string("channel"), Some("luma"));

        let input = solid_f32(0.0, 1.0, 0.0, 1.0);
        let out = fx.apply(&mut ctx, input, &p).unwrap();
        let px = out.as_f32().unwrap();
        // Rec.709 luma of (0, 1, 0) is 0.7152.
        let expected = 0.715_2_f32;
        assert!((px[0] - expected).abs() < 1e-5);
        assert!((px[1] - expected).abs() < 1e-5);
        assert!((px[2] - expected).abs() < 1e-5);
        // Alpha preserved.
        assert!((px[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn invert_flips_values() {
        let fx = ChannelIsolate;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("channel", ParamValue::String("b".into()));
        p.insert("invert", ParamValue::Bool(true));
        p.validate_and_fill(fx.parameters()).unwrap();

        let input = solid_f32(0.0, 0.0, 0.25, 1.0);
        let out = fx.apply(&mut ctx, input, &p).unwrap();
        let px = out.as_f32().unwrap();
        // Picked B = 0.25, inverted to 0.75.
        assert!((px[0] - 0.75).abs() < 1e-6);
        assert!((px[1] - 0.75).abs() < 1e-6);
        assert!((px[2] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn alpha_channel_isolation() {
        let fx = ChannelIsolate;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("channel", ParamValue::String("a".into()));
        p.validate_and_fill(fx.parameters()).unwrap();

        let input = solid_f32(0.9, 0.9, 0.9, 0.4);
        let out = fx.apply(&mut ctx, input, &p).unwrap();
        let px = out.as_f32().unwrap();
        assert!((px[0] - 0.4).abs() < 1e-6);
        assert!((px[1] - 0.4).abs() < 1e-6);
        assert!((px[2] - 0.4).abs() < 1e-6);
        // Alpha untouched.
        assert!((px[3] - 0.4).abs() < 1e-6);
    }
}
