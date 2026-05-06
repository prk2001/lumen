//! Directional motion blur — single-frame primitive.
//!
//! Simulates motion blur on a still frame by averaging N samples taken
//! along a line at a user-specified angle. This is the still-image
//! primitive that temporal motion blur reduces to per-frame; multi-frame
//! interpolation belongs to a later phase.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, PixelData, Result,
};
use tracing::instrument;

/// Directional motion blur effect.
#[derive(Debug, Default)]
pub struct MotionBlurDirectional;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-temporal.motion_blur_directional",
    display_name: "Directional Motion Blur",
    description: "Averages bilinear samples along a line to simulate motion blur on a still frame.",
    category: Category::Temporal,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "length",
        display_name: "Length",
        description: "Blur length in pixels.",
        kind: ParamKind::Float { default: 8.0, min: Some(0.0), max: Some(256.0) },
    },
    ParamSpec {
        id: "angle_degrees",
        display_name: "Angle",
        description: "Angle in degrees, 0 = horizontal right.",
        kind: ParamKind::Float { default: 0.0, min: Some(-360.0), max: Some(360.0) },
    },
    ParamSpec {
        id: "samples",
        display_name: "Samples",
        description: "Number of samples averaged along the line.",
        kind: ParamKind::Int { default: 16, min: Some(2), max: Some(256) },
    },
];

impl Effect for MotionBlurDirectional {
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
        let length = params.get_float("length").unwrap_or(8.0) as f32;
        let angle_degrees = params.get_float("angle_degrees").unwrap_or(0.0) as f32;
        let samples = params.get_int("samples").unwrap_or(16);

        // Passthrough on degenerate inputs.
        if length < 0.5 || samples < 2 {
            return Ok(input);
        }

        let frame = input.into_rgba_f32_linear();
        let w = frame.width as usize;
        let h = frame.height as usize;
        let src = frame.as_f32().expect("RgbaF32 after lift");

        if w == 0 || h == 0 {
            return Frame::new(
                frame.width,
                frame.height,
                frame.data.clone(),
                frame.color_space.clone(),
                frame.pts,
            );
        }

        let samples = samples as usize;
        let angle_rad = angle_degrees.to_radians();
        let dx = angle_rad.cos();
        let dy = angle_rad.sin();
        let half_len = length * 0.5;

        let mut out = vec![0.0f32; w * h * 4];

        // Pre-compute parametric offsets along [-half_len, +half_len].
        // Step from t=0 (start) to t=samples-1 (end), inclusive of both endpoints.
        let denom = (samples - 1) as f32;
        let inv_samples = 1.0f32 / samples as f32;

        for y in 0..h {
            for x in 0..w {
                let mut acc = [0.0f32; 4];
                for s in 0..samples {
                    let t = s as f32 / denom; // 0..=1
                    let offset = -half_len + t * length;
                    let sx = x as f32 + offset * dx;
                    let sy = y as f32 + offset * dy;
                    let p = sample_bilinear_clamp(src, w, h, sx, sy);
                    acc[0] += p[0];
                    acc[1] += p[1];
                    acc[2] += p[2];
                    acc[3] += p[3];
                }
                let off = (y * w + x) * 4;
                out[off] = acc[0] * inv_samples;
                out[off + 1] = acc[1] * inv_samples;
                out[off + 2] = acc[2] * inv_samples;
                out[off + 3] = acc[3] * inv_samples;
            }
        }

        Frame::new(
            frame.width,
            frame.height,
            PixelData::RgbaF32(out),
            frame.color_space.clone(),
            frame.pts,
        )
    }
}

/// Bilinear sample with edge-clamp out-of-bounds handling.
fn sample_bilinear_clamp(src: &[f32], w: usize, h: usize, sx: f32, sy: f32) -> [f32; 4] {
    let max_x = (w - 1) as f32;
    let max_y = (h - 1) as f32;
    let cx = sx.clamp(0.0, max_x);
    let cy = sy.clamp(0.0, max_y);

    let x0 = cx.floor() as usize;
    let y0 = cy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = cx - x0 as f32;
    let fy = cy - y0 as f32;

    let stride = w * 4;
    let p00 = &src[y0 * stride + x0 * 4..y0 * stride + x0 * 4 + 4];
    let p10 = &src[y0 * stride + x1 * 4..y0 * stride + x1 * 4 + 4];
    let p01 = &src[y1 * stride + x0 * 4..y1 * stride + x0 * 4 + 4];
    let p11 = &src[y1 * stride + x1 * 4..y1 * stride + x1 * 4 + 4];

    let mut out = [0.0f32; 4];
    for c in 0..4 {
        let top = p00[c] * (1.0 - fx) + p10[c] * fx;
        let bot = p01[c] * (1.0 - fx) + p11[c] * fx;
        out[c] = top * (1.0 - fy) + bot * fy;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue, PixelData};

    fn make_frame_rgba8(w: u32, h: u32, data: Vec<u8>) -> Frame {
        Frame::new(w, h, PixelData::Rgba8(data), ColorSpace::SRgb, None).unwrap()
    }

    #[test]
    fn zero_length_passthrough() {
        // length < 0.5 should return input unchanged (ignoring color-space lift).
        let mb = MotionBlurDirectional;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("length", ParamValue::Float(0.0));
        p.insert("angle_degrees", ParamValue::Float(45.0));
        p.insert("samples", ParamValue::Int(16));
        p.validate_and_fill(mb.parameters()).unwrap();

        let original: Vec<u8> = (0..(8 * 8 * 4)).map(|i| (i % 256) as u8).collect();
        let f = make_frame_rgba8(8, 8, original.clone());
        let out = mb.apply(&mut ctx, f, &p).unwrap();
        // The frame should still be Rgba8 (no float lift) and equal to the original.
        let PixelData::Rgba8(px) = out.data else {
            panic!("expected Rgba8 passthrough — got {:?}", out.layout());
        };
        assert_eq!(px, original);
    }

    #[test]
    fn samples_below_min_passthrough() {
        // samples < 2 also bypasses processing.
        let mb = MotionBlurDirectional;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        // Bypass validation (which would reject samples < 2) by inserting
        // values directly without filling defaults, then filling only the
        // missing entries — here we want to test the runtime guard.
        p.insert("length", ParamValue::Float(8.0));
        p.insert("angle_degrees", ParamValue::Float(0.0));
        p.insert("samples", ParamValue::Int(1));
        // Skip validate_and_fill so the out-of-range int isn't rejected.
        let original = vec![200u8; 4 * 4 * 4];
        let f = make_frame_rgba8(4, 4, original.clone());
        let out = mb.apply(&mut ctx, f, &p).unwrap();
        let PixelData::Rgba8(px) = out.data else { panic!("expected Rgba8 passthrough") };
        assert_eq!(px, original);
    }

    #[test]
    fn horizontal_blur_widens_vertical_edge() {
        // Build a 16x16 image with a sharp vertical edge at x=8: black on
        // the left, white on the right (alpha = 255 everywhere). After a
        // strong horizontal blur, columns near the edge should become
        // mid-gray instead of pure black/white.
        let w = 16u32;
        let h = 16u32;
        let mut data = vec![0u8; (w as usize) * (h as usize) * 4];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let off = (y * w as usize + x) * 4;
                let v = if x >= 8 { 255 } else { 0 };
                data[off] = v;
                data[off + 1] = v;
                data[off + 2] = v;
                data[off + 3] = 255;
            }
        }
        let f = make_frame_rgba8(w, h, data);

        let mb = MotionBlurDirectional;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("length", ParamValue::Float(8.0));
        p.insert("angle_degrees", ParamValue::Float(0.0));
        p.insert("samples", ParamValue::Int(16));
        p.validate_and_fill(mb.parameters()).unwrap();

        let out = mb.apply(&mut ctx, f, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!("expected Rgba8") };

        // Sample pixels at the edge boundary — they should now be
        // intermediate values, not pure 0 or 255.
        let mid_y = (h / 2) as usize;
        let off_left = (mid_y * w as usize + 7) * 4; // just left of edge
        let off_right = (mid_y * w as usize + 8) * 4; // just right of edge

        let l = px[off_left] as i32;
        let r = px[off_right] as i32;
        assert!(
            l > 10 && l < 245,
            "expected blurred mid-tone left of edge, got {l}"
        );
        assert!(
            r > 10 && r < 245,
            "expected blurred mid-tone right of edge, got {r}"
        );

        // Far from the edge, values should remain near their original.
        let off_far_left = (mid_y * w as usize) * 4;
        let off_far_right = (mid_y * w as usize + (w as usize - 1)) * 4;
        assert!(
            (px[off_far_left] as i32) < 20,
            "far-left should stay near black, got {}",
            px[off_far_left]
        );
        assert!(
            (px[off_far_right] as i32) > 235,
            "far-right should stay near white, got {}",
            px[off_far_right]
        );
    }

    #[test]
    fn solid_image_unchanged() {
        // Isotropic / sanity check: a constant image must remain constant
        // after blur, regardless of length and angle. (Analogue of the
        // gaussian "sigma=0 sanity check".)
        let mb = MotionBlurDirectional;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("length", ParamValue::Float(16.0));
        p.insert("angle_degrees", ParamValue::Float(37.0));
        p.insert("samples", ParamValue::Int(32));
        p.validate_and_fill(mb.parameters()).unwrap();

        let f = make_frame_rgba8(12, 12, vec![123u8; 12 * 12 * 4]);
        let out = mb.apply(&mut ctx, f, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!("expected Rgba8") };
        // Allow a 1-LSB tolerance for sRGB <-> linear round-trip noise.
        assert!(
            px.iter().all(|&v| (v as i32 - 123).abs() <= 1),
            "expected solid image to remain ~constant; min={} max={}",
            px.iter().min().unwrap(),
            px.iter().max().unwrap()
        );
    }
}
