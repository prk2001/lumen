//! Rectangular alpha mask with optional feathering.
//!
//! Inside the rect, alpha is multiplied by `inside`. Outside, by
//! `outside`. A `feather` parameter expressed in pixels softens the
//! transition with a linear ramp. Useful as a building block for
//! vignettes, ROI passes, and per-region effect masking.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct AlphaRect;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-mask.alpha_rect",
    display_name: "Alpha Rect Mask",
    description: "Rectangular alpha mask with feathered edges.",
    category: Category::Mask,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "x",
        display_name: "X",
        description: "Mask rect left edge in pixels.",
        kind: ParamKind::Int { default: 0, min: Some(0), max: None },
    },
    ParamSpec {
        id: "y",
        display_name: "Y",
        description: "Mask rect top edge in pixels.",
        kind: ParamKind::Int { default: 0, min: Some(0), max: None },
    },
    ParamSpec {
        id: "width",
        display_name: "Width",
        description: "Mask rect width in pixels.",
        kind: ParamKind::Int { default: 0, min: Some(1), max: None },
    },
    ParamSpec {
        id: "height",
        display_name: "Height",
        description: "Mask rect height in pixels.",
        kind: ParamKind::Int { default: 0, min: Some(1), max: None },
    },
    ParamSpec {
        id: "inside",
        display_name: "Inside α",
        description: "Alpha multiplier inside the rect (0..1).",
        kind: ParamKind::Float { default: 1.0, min: Some(0.0), max: Some(1.0) },
    },
    ParamSpec {
        id: "outside",
        display_name: "Outside α",
        description: "Alpha multiplier outside the rect (0..1).",
        kind: ParamKind::Float { default: 0.0, min: Some(0.0), max: Some(1.0) },
    },
    ParamSpec {
        id: "feather",
        display_name: "Feather",
        description: "Linear-ramp width in pixels at each edge.",
        kind: ParamKind::Float { default: 0.0, min: Some(0.0), max: Some(512.0) },
    },
];

impl Effect for AlphaRect {
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
        let img_w = input.width as f32;
        let img_h = input.height as f32;
        let rx = params.get_int("x").unwrap_or(0).max(0) as f32;
        let ry = params.get_int("y").unwrap_or(0).max(0) as f32;
        let rw_raw = params.get_int("width").unwrap_or(0);
        let rh_raw = params.get_int("height").unwrap_or(0);
        // If width/height are zero or unset, default to whole image.
        let rw = if rw_raw <= 0 { img_w } else { rw_raw as f32 };
        let rh = if rh_raw <= 0 { img_h } else { rh_raw as f32 };
        let inside = params.get_float("inside").unwrap_or(1.0) as f32;
        let outside = params.get_float("outside").unwrap_or(0.0) as f32;
        let feather = params.get_float("feather").unwrap_or(0.0).max(0.0) as f32;

        let mut frame = input.into_rgba_f32_linear();
        let w = frame.width as usize;
        let h = frame.height as usize;
        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");

        for py in 0..h {
            for px in 0..w {
                let cx = px as f32 + 0.5;
                let cy = py as f32 + 0.5;
                let m = mask_value(cx, cy, rx, ry, rw, rh, feather);
                let alpha_mul = outside + (inside - outside) * m;
                let off = (py * w + px) * 4;
                pixels[off + 3] = (pixels[off + 3] * alpha_mul).clamp(0.0, 1.0);
            }
        }
        Ok(frame)
    }
}

/// 1.0 fully inside, 0.0 fully outside, linear ramp in feather band.
fn mask_value(cx: f32, cy: f32, rx: f32, ry: f32, rw: f32, rh: f32, feather: f32) -> f32 {
    if feather <= 0.0 {
        let inside = cx >= rx && cx < rx + rw && cy >= ry && cy < ry + rh;
        return if inside { 1.0 } else { 0.0 };
    }
    let dx_left = cx - rx;
    let dx_right = (rx + rw) - cx;
    let dy_top = cy - ry;
    let dy_bot = (ry + rh) - cy;
    let dist = dx_left.min(dx_right).min(dy_top).min(dy_bot);
    if dist >= feather {
        1.0
    } else if dist <= -feather {
        0.0
    } else {
        ((dist + feather) / (2.0 * feather)).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue, PixelData};

    #[test]
    fn outside_zero_kills_alpha_outside() {
        let m = AlphaRect;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("x", ParamValue::Int(2));
        p.insert("y", ParamValue::Int(2));
        p.insert("width", ParamValue::Int(2));
        p.insert("height", ParamValue::Int(2));
        p.insert("inside", ParamValue::Float(1.0));
        p.insert("outside", ParamValue::Float(0.0));
        p.insert("feather", ParamValue::Float(0.0));
        p.validate_and_fill(m.parameters()).unwrap();

        let f =
            Frame::new(6, 6, PixelData::Rgba8(vec![255; 6 * 6 * 4]), ColorSpace::SRgb, None)
                .unwrap();
        let out = m.apply(&mut ctx, f, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!() };
        // Center 2x2 alpha = 255; everywhere else alpha = 0.
        for y in 0..6 {
            for x in 0..6 {
                let a = px[(y * 6 + x) * 4 + 3];
                let in_rect = (2..4).contains(&x) && (2..4).contains(&y);
                if in_rect {
                    assert_eq!(a, 255, "expected 255 at ({x},{y}), got {a}");
                } else {
                    assert_eq!(a, 0, "expected 0 at ({x},{y}), got {a}");
                }
            }
        }
    }
}
