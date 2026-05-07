//! Primary color wheels — independent Lift / Gamma / Gain controls per
//! RGB channel.
//!
//! Math (per channel, in scene-linear light, clamped to [0, 1]):
//!
//! ```text
//! out = pow(in * gain + lift, 1 / gamma)
//! ```
//!
//! - `lift` shifts shadows (added before the curve).
//! - `gain` scales highlights (multiplied before lift).
//! - `gamma` reshapes midtones via a power curve.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct PrimaryWheels;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-color.primary_wheels",
    display_name: "Primary Wheels",
    description: "Per-channel Lift / Gamma / Gain (DaVinci-style log/offset wheels).",
    category: Category::Color,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "lift_r",
        display_name: "Lift R",
        description: "Shadow offset for the red channel.",
        kind: ParamKind::Float { default: 0.0, min: Some(-1.0), max: Some(1.0) },
    },
    ParamSpec {
        id: "lift_g",
        display_name: "Lift G",
        description: "Shadow offset for the green channel.",
        kind: ParamKind::Float { default: 0.0, min: Some(-1.0), max: Some(1.0) },
    },
    ParamSpec {
        id: "lift_b",
        display_name: "Lift B",
        description: "Shadow offset for the blue channel.",
        kind: ParamKind::Float { default: 0.0, min: Some(-1.0), max: Some(1.0) },
    },
    ParamSpec {
        id: "gamma_r",
        display_name: "Gamma R",
        description: "Midtone power for the red channel.",
        kind: ParamKind::Float { default: 1.0, min: Some(0.05), max: Some(10.0) },
    },
    ParamSpec {
        id: "gamma_g",
        display_name: "Gamma G",
        description: "Midtone power for the green channel.",
        kind: ParamKind::Float { default: 1.0, min: Some(0.05), max: Some(10.0) },
    },
    ParamSpec {
        id: "gamma_b",
        display_name: "Gamma B",
        description: "Midtone power for the blue channel.",
        kind: ParamKind::Float { default: 1.0, min: Some(0.05), max: Some(10.0) },
    },
    ParamSpec {
        id: "gain_r",
        display_name: "Gain R",
        description: "Highlight multiplier for the red channel.",
        kind: ParamKind::Float { default: 1.0, min: Some(0.0), max: Some(4.0) },
    },
    ParamSpec {
        id: "gain_g",
        display_name: "Gain G",
        description: "Highlight multiplier for the green channel.",
        kind: ParamKind::Float { default: 1.0, min: Some(0.0), max: Some(4.0) },
    },
    ParamSpec {
        id: "gain_b",
        display_name: "Gain B",
        description: "Highlight multiplier for the blue channel.",
        kind: ParamKind::Float { default: 1.0, min: Some(0.0), max: Some(4.0) },
    },
];

#[inline]
fn apply_lgg(c: f32, lift: f32, gamma: f32, gain: f32) -> f32 {
    let pre = (c * gain + lift).max(0.0);
    let inv = 1.0 / gamma;
    pre.powf(inv).clamp(0.0, 1.0)
}

impl Effect for PrimaryWheels {
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
        let lift = [
            params.get_float("lift_r").unwrap_or(0.0) as f32,
            params.get_float("lift_g").unwrap_or(0.0) as f32,
            params.get_float("lift_b").unwrap_or(0.0) as f32,
        ];
        let gamma = [
            params.get_float("gamma_r").unwrap_or(1.0) as f32,
            params.get_float("gamma_g").unwrap_or(1.0) as f32,
            params.get_float("gamma_b").unwrap_or(1.0) as f32,
        ];
        let gain = [
            params.get_float("gain_r").unwrap_or(1.0) as f32,
            params.get_float("gain_g").unwrap_or(1.0) as f32,
            params.get_float("gain_b").unwrap_or(1.0) as f32,
        ];

        let mut frame = input.into_rgba_f32_linear();
        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");
        for px in pixels.chunks_exact_mut(4) {
            for ch in 0..3 {
                px[ch] = apply_lgg(px[ch], lift[ch], gamma[ch], gain[ch]);
            }
        }
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue, PixelData};

    fn frame_f32(pixels: Vec<f32>) -> Frame {
        let n = pixels.len() / 4;
        let w = n as u32;
        Frame::new(w, 1, PixelData::RgbaF32(pixels), ColorSpace::LinearSRgb, None).unwrap()
    }

    #[test]
    fn defaults_are_passthrough() {
        let fx = PrimaryWheels;
        let mut p = ParamValues::new();
        p.validate_and_fill(fx.parameters()).unwrap();

        let pixels = vec![0.10, 0.50, 0.90, 1.0, 0.25, 0.75, 0.30, 1.0];
        let frame = frame_f32(pixels.clone());
        let mut ctx = Context::for_still_srgb();
        let out = fx.apply(&mut ctx, frame, &p).unwrap();
        let out_px = out.as_f32().unwrap();
        for (a, b) in pixels.iter().zip(out_px.iter()) {
            assert!((a - b).abs() < 1e-6, "drift: {a} vs {b}");
        }
    }

    #[test]
    fn lift_brightens_shadows() {
        let fx = PrimaryWheels;
        let mut p = ParamValues::new();
        p.insert("lift_r", ParamValue::Float(0.1));
        p.insert("lift_g", ParamValue::Float(0.1));
        p.insert("lift_b", ParamValue::Float(0.1));
        p.validate_and_fill(fx.parameters()).unwrap();

        // Black pixel — only lift can move it.
        let frame = frame_f32(vec![0.0, 0.0, 0.0, 1.0]);
        let mut ctx = Context::for_still_srgb();
        let out = fx.apply(&mut ctx, frame, &p).unwrap();
        let out_px = out.as_f32().unwrap();
        assert!(out_px[0] > 0.05, "R should be lifted, got {}", out_px[0]);
        assert!(out_px[1] > 0.05, "G should be lifted, got {}", out_px[1]);
        assert!(out_px[2] > 0.05, "B should be lifted, got {}", out_px[2]);
        // Alpha untouched.
        assert!((out_px[3] - 1.0).abs() < 1e-6);
    }
}
