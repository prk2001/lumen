//! Median filter — pure (2r+1) × (2r+1) per-channel median.
//!
//! Median filtering preserves edges (a step edge is fixed under
//! median) and crushes salt-and-pepper / impulse noise. Different
//! tradeoffs from bilateral: no parameters to tune beyond radius,
//! and it can wipe out fine speckles a Gaussian or bilateral
//! cannot. Bad at Gaussian noise; great at outliers.
//!
//! Implementation is the textbook O(N · k²) per pixel, per channel.
//! That's fine for the small radii we expose (radius ≤ 8).

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct Median;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-denoise.median",
    display_name: "Median Denoise",
    description: "Per-channel median filter. Edge-preserving; great on impulse noise.",
    category: Category::Denoise,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[ParamSpec {
    id: "radius",
    display_name: "Radius",
    description: "Half-width of the window. radius=1 is 3x3, radius=2 is 5x5, etc.",
    kind: ParamKind::Int { default: 1, min: Some(1), max: Some(8) },
}];

impl Effect for Median {
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
        let radius = params.get_int("radius").unwrap_or(1).clamp(1, 8) as usize;

        let mut frame = input.into_rgba_f32_linear();
        let w = frame.width as usize;
        let h = frame.height as usize;
        if w == 0 || h == 0 {
            return Ok(frame);
        }
        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");
        let stride = w * 4;
        let src = pixels.to_vec();

        median_filter_rgba(&src, pixels, w, h, stride, radius);

        Ok(frame)
    }
}

fn median_filter_rgba(
    src: &[f32],
    dst: &mut [f32],
    w: usize,
    h: usize,
    stride: usize,
    radius: usize,
) {
    let kw = 2 * radius + 1;
    let n = kw * kw;
    let mut buf_r: Vec<f32> = Vec::with_capacity(n);
    let mut buf_g: Vec<f32> = Vec::with_capacity(n);
    let mut buf_b: Vec<f32> = Vec::with_capacity(n);

    for y in 0..h {
        for x in 0..w {
            buf_r.clear();
            buf_g.clear();
            buf_b.clear();
            for dy in 0..kw {
                let yi = (y as isize + dy as isize - radius as isize)
                    .clamp(0, h as isize - 1) as usize;
                for dx in 0..kw {
                    let xi = (x as isize + dx as isize - radius as isize)
                        .clamp(0, w as isize - 1) as usize;
                    let off = yi * stride + xi * 4;
                    buf_r.push(src[off]);
                    buf_g.push(src[off + 1]);
                    buf_b.push(src[off + 2]);
                }
            }
            let off = y * stride + x * 4;
            dst[off] = median_of(&mut buf_r);
            dst[off + 1] = median_of(&mut buf_g);
            dst[off + 2] = median_of(&mut buf_b);
            // Preserve alpha.
            dst[off + 3] = src[off + 3];
        }
    }
}

/// Median by partial sort. `buf.len()` is always odd here (window is
/// (2r+1)²), but if it ever isn't we just take the lower middle —
/// no panic, no NaN gymnastics.
fn median_of(buf: &mut [f32]) -> f32 {
    let n = buf.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    buf[mid]
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue, PixelData};

    fn run(m: &Median, frame: Frame, radius: i64) -> Frame {
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("radius", ParamValue::Int(radius));
        p.validate_and_fill(m.parameters()).unwrap();
        m.apply(&mut ctx, frame, &p).unwrap()
    }

    #[test]
    fn solid_image_unchanged() {
        let m = Median;
        let f = Frame::new(
            12,
            12,
            PixelData::Rgba8(vec![160; 12 * 12 * 4]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();
        let out = run(&m, f, 2).into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!() };
        assert!(
            px.iter().all(|&v| (v as i32 - 160).abs() <= 1),
            "median of constant must be the constant"
        );
    }

    #[test]
    fn salt_and_pepper_attenuated() {
        // Flat gray with a sprinkle of black-and-white impulses.
        // After a 3x3 median, the impulses must be removed because
        // each impulse is a minority in its 9-pixel window.
        let m = Median;
        let w: usize = 16;
        let h: usize = 16;
        let mut data = vec![128u8; w * h * 4];
        for off in (3..data.len()).step_by(4) {
            data[off] = 255; // alpha
        }
        // Sparse impulses: pixels (3,3) -> 0, (8,5) -> 255, (12,11) -> 0.
        let impulses: &[((usize, usize), u8)] = &[
            ((3, 3), 0),
            ((8, 5), 255),
            ((12, 11), 0),
        ];
        for &((x, y), v) in impulses {
            let off = (y * w + x) * 4;
            data[off] = v;
            data[off + 1] = v;
            data[off + 2] = v;
        }
        let f =
            Frame::new(w as u32, h as u32, PixelData::Rgba8(data), ColorSpace::SRgb, None)
                .unwrap();
        let out = run(&m, f, 1).into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!() };
        for &((x, y), _v) in impulses {
            let off = (y * w + x) * 4;
            // After 3x3 median, impulse pixel becomes the majority value (~128).
            let r = px[off] as i32;
            assert!(
                (r - 128).abs() <= 2,
                "impulse at ({x},{y}) should be replaced by surrounding 128, got {r}"
            );
        }
    }

    #[test]
    fn larger_radius_does_not_panic() {
        // radius=2 -> 5x5 window. Includes the "even radius" question:
        // here radius is the half-width, always producing an odd
        // window (2r+1). But also exercise the largest allowed
        // value to be sure the buffer math is fine.
        let m = Median;
        let w: usize = 8;
        let h: usize = 8;
        let mut data = vec![0u8; w * h * 4];
        for i in 0..(w * h) {
            data[i * 4] = (i * 7 % 256) as u8;
            data[i * 4 + 1] = (i * 13 % 256) as u8;
            data[i * 4 + 2] = (i * 17 % 256) as u8;
            data[i * 4 + 3] = 255;
        }
        for r in [2i64, 4, 8] {
            let f = Frame::new(
                w as u32,
                h as u32,
                PixelData::Rgba8(data.clone()),
                ColorSpace::SRgb,
                None,
            )
            .unwrap();
            let _ = run(&m, f, r);
        }
    }

    #[test]
    fn median_of_handles_even_length() {
        // Direct sanity check on the helper — no panic on an even-length
        // input (defensive: window is always odd, but the helper must
        // not assume so).
        let mut v = vec![1.0f32, 5.0, 2.0, 4.0];
        let m = median_of(&mut v);
        // Sorted: [1,2,4,5]; lower-mid index 2 -> 4.0.
        assert_eq!(m, 4.0);
    }
}
