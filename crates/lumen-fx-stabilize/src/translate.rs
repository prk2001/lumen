//! Translate — shift a frame by `(dx, dy)` pixels with edge-clamp fill.
//!
//! This is the single-frame primitive that real stabilization composes:
//! given a per-frame motion estimate, you push the inverse translation
//! through this effect to register the frame against a reference. Phase 4
//! adds the multi-frame motion estimation; for now, this effect just
//! exposes the warp as a user-facing knob.
//!
//! # Algorithm
//!
//! For an output pixel `(x, y)` we look up the source pixel at
//! `(x - dx, y - dy)`. Out-of-bounds source coordinates are clamped to the
//! nearest edge pixel ("clamp-to-edge" sampling), which avoids introducing
//! a black/transparent border when shifting.
//!
//! When `subpixel = true` we bilinearly interpolate the four source pixels
//! straddling the fractional source coordinate. When `subpixel = false`
//! the source coordinate is rounded to the nearest integer and a single
//! pixel is copied — faster, but visibly steppy for non-integer shifts.
//!
//! Computation runs on linearized RGBA f32 pixels (via
//! [`Frame::into_rgba_f32_linear`]) so blending is gamma-correct.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, PixelData, Result,
};
use tracing::instrument;

/// Maximum absolute shift in pixels — kept generous so an 8K frame can be
/// translated half its width if the user really wants to.
const MAX_SHIFT: f64 = 4096.0;

#[derive(Debug, Default)]
pub struct Translate;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-stabilize.translate",
    display_name: "Translate",
    description: "Shift the frame by (dx, dy) pixels with edge-clamp fill.",
    category: Category::Stabilize,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "dx",
        display_name: "DX",
        description: "Horizontal shift in pixels (positive = right).",
        kind: ParamKind::Float { default: 0.0, min: Some(-MAX_SHIFT), max: Some(MAX_SHIFT) },
    },
    ParamSpec {
        id: "dy",
        display_name: "DY",
        description: "Vertical shift in pixels (positive = down).",
        kind: ParamKind::Float { default: 0.0, min: Some(-MAX_SHIFT), max: Some(MAX_SHIFT) },
    },
    ParamSpec {
        id: "subpixel",
        display_name: "Subpixel",
        description: "If true, bilinear interpolation between source pixels.",
        kind: ParamKind::Bool { default: true },
    },
];

impl Effect for Translate {
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
        let dx = params.get_float("dx").unwrap_or(0.0) as f32;
        let dy = params.get_float("dy").unwrap_or(0.0) as f32;
        let subpixel = params.get_bool("subpixel").unwrap_or(true);

        let frame = input.into_rgba_f32_linear();
        let w = frame.width as usize;
        let h = frame.height as usize;

        // Empty frame: nothing to do.
        if w == 0 || h == 0 {
            return Ok(frame);
        }

        let src = frame.as_f32().expect("RgbaF32 after lift");
        let mut dst = vec![0.0f32; src.len()];

        if subpixel {
            translate_bilinear(src, &mut dst, w, h, dx, dy);
        } else {
            translate_nearest(src, &mut dst, w, h, dx, dy);
        }

        Ok(Frame {
            width: frame.width,
            height: frame.height,
            data: PixelData::RgbaF32(dst),
            color_space: frame.color_space,
            pts: frame.pts,
        })
    }
}

/// Nearest-neighbor translate: each output pixel copies a single source
/// pixel located at the rounded `(x - dx, y - dy)` with edge clamping.
fn translate_nearest(src: &[f32], dst: &mut [f32], w: usize, h: usize, dx: f32, dy: f32) {
    let stride = w * 4;
    let dx_round = dx.round() as isize;
    let dy_round = dy.round() as isize;
    let max_x = w as isize - 1;
    let max_y = h as isize - 1;

    for y in 0..h {
        let sy = (y as isize - dy_round).clamp(0, max_y) as usize;
        let row_in = &src[sy * stride..sy * stride + stride];
        let row_out = &mut dst[y * stride..y * stride + stride];
        for x in 0..w {
            let sx = (x as isize - dx_round).clamp(0, max_x) as usize;
            let src_off = sx * 4;
            let dst_off = x * 4;
            row_out[dst_off] = row_in[src_off];
            row_out[dst_off + 1] = row_in[src_off + 1];
            row_out[dst_off + 2] = row_in[src_off + 2];
            row_out[dst_off + 3] = row_in[src_off + 3];
        }
    }
}

/// Bilinear translate: sample the source at fractional `(x - dx, y - dy)`
/// using the four neighboring pixels. Coordinates outside the frame are
/// clamped to the nearest edge pixel.
fn translate_bilinear(src: &[f32], dst: &mut [f32], w: usize, h: usize, dx: f32, dy: f32) {
    let stride = w * 4;
    let max_x = w as isize - 1;
    let max_y = h as isize - 1;

    for y in 0..h {
        let sy = y as f32 - dy;
        let y0 = sy.floor();
        let fy = sy - y0;
        let y0i = (y0 as isize).clamp(0, max_y) as usize;
        let y1i = ((y0 as isize) + 1).clamp(0, max_y) as usize;

        for x in 0..w {
            let sx = x as f32 - dx;
            let x0 = sx.floor();
            let fx = sx - x0;
            let x0i = (x0 as isize).clamp(0, max_x) as usize;
            let x1i = ((x0 as isize) + 1).clamp(0, max_x) as usize;

            let p00 = &src[y0i * stride + x0i * 4..y0i * stride + x0i * 4 + 4];
            let p01 = &src[y0i * stride + x1i * 4..y0i * stride + x1i * 4 + 4];
            let p10 = &src[y1i * stride + x0i * 4..y1i * stride + x0i * 4 + 4];
            let p11 = &src[y1i * stride + x1i * 4..y1i * stride + x1i * 4 + 4];

            let w00 = (1.0 - fx) * (1.0 - fy);
            let w01 = fx * (1.0 - fy);
            let w10 = (1.0 - fx) * fy;
            let w11 = fx * fy;

            let dst_off = y * stride + x * 4;
            for c in 0..4 {
                dst[dst_off + c] =
                    p00[c] * w00 + p01[c] * w01 + p10[c] * w10 + p11[c] * w11;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue, PixelData};

    /// Build a 4x4 frame whose red channel encodes the (x, y) position so
    /// shifts are easy to verify. R = x*16 + y, G = 0, B = 0, A = 255.
    fn position_frame() -> Frame {
        let w = 4u32;
        let h = 4u32;
        let mut data = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let off = ((y * w + x) * 4) as usize;
                data[off] = (x as u8) * 16 + y as u8;
                data[off + 3] = 255;
            }
        }
        Frame::new(w, h, PixelData::Rgba8(data), ColorSpace::SRgb, None).unwrap()
    }

    fn run(t: &Translate, f: Frame, dx: f64, dy: f64, subpixel: bool) -> Frame {
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("dx", ParamValue::Float(dx));
        p.insert("dy", ParamValue::Float(dy));
        p.insert("subpixel", ParamValue::Bool(subpixel));
        p.validate_and_fill(t.parameters()).unwrap();
        t.apply(&mut ctx, f, &p).unwrap()
    }

    #[test]
    fn identity_zero_shift_preserves_pixels() {
        let t = Translate;
        let original = position_frame();
        let expected = original.clone().into_rgba_u8_srgb();
        let out = run(&t, original, 0.0, 0.0, true).into_rgba_u8_srgb();

        let PixelData::Rgba8(got) = out.data else { panic!("expected Rgba8") };
        let PixelData::Rgba8(want) = expected.data else { panic!("expected Rgba8") };
        // Allow ±1 LSB drift from sRGB round-trip through linear f32.
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((*g as i32 - *w as i32).abs() <= 1, "got {g} want {w}");
        }
    }

    #[test]
    fn pure_horizontal_shift_moves_pixels_right() {
        // Shift right by 1 pixel, nearest-neighbor: column x=1 in the
        // output should match column x=0 in the input.
        let t = Translate;
        let out = run(&t, position_frame(), 1.0, 0.0, false).into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!("expected Rgba8") };

        let w = 4usize;
        // For each row y, output(1, y).R should equal input(0, y).R = 0*16 + y = y.
        for y in 0..4usize {
            let off = (y * w + 1) * 4;
            assert_eq!(px[off], y as u8, "row {y}: expected {} got {}", y, px[off]);
        }
        // And output(2, y).R should equal input(1, y).R = 16 + y.
        for y in 0..4usize {
            let off = (y * w + 2) * 4;
            assert_eq!(px[off], 16 + y as u8);
        }
    }

    #[test]
    fn out_of_bounds_clamps_to_edge() {
        // Shift right by 10 pixels in a 4-wide frame: the entire output
        // should be filled with the leftmost source column repeated, since
        // every source x coordinate becomes negative and clamps to x=0.
        let t = Translate;
        let out = run(&t, position_frame(), 10.0, 0.0, false).into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!("expected Rgba8") };

        let w = 4usize;
        for y in 0..4usize {
            // Source column 0 at row y has R = y.
            let expected = y as u8;
            for x in 0..4usize {
                let off = (y * w + x) * 4;
                assert_eq!(
                    px[off], expected,
                    "row {y} col {x}: expected {expected} got {}",
                    px[off]
                );
            }
        }
    }

    #[test]
    fn subpixel_half_shift_blends_neighbors() {
        // With dx=0.5 subpixel on, output column x should be the average
        // of source columns x-1 and x (or clamped equivalents). For
        // interior columns this is a clean midpoint blend.
        let t = Translate;
        let out = run(&t, position_frame(), 0.5, 0.0, true).into_rgba_f32_linear();
        let pixels = out.as_f32().unwrap();
        let w = 4usize;

        // Output(2, 0).R should be ~halfway between input(1, 0).R=16/255
        // and input(2, 0).R=32/255 in linear space. Just check that it
        // sits strictly between the two source values. Row 0 starts at
        // offset 0, so `(y*w + x) * 4` for y=0 simplifies to `x * 4`.
        let _ = w;
        let mid = pixels[8]; // x=2, y=0 -> 2*4
        // Grab the f32-linear input by re-linearizing.
        let in_linear = position_frame().into_rgba_f32_linear();
        let lin = in_linear.as_f32().unwrap();
        let left = lin[4];  // x=1, y=0
        let right = lin[8]; // x=2, y=0
        let lo = left.min(right);
        let hi = left.max(right);
        assert!(mid > lo && mid < hi, "expected {lo} < {mid} < {hi}");
    }
}
