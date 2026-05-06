//! Dehaze via Dark Channel Prior (He, Sun, Tang — CVPR 2009).
//!
//! Classical, no-AI dehaze. Operates on linear-light RGBA float pixels.
//!
//! Algorithm:
//!
//! 1. **Dark channel.** For each pixel take `min(R, G, B)`, then a
//!    min-pool over a `(2*patch_radius + 1)` square window. In a
//!    haze-free outdoor patch at least one channel is usually very
//!    dark; a high dark-channel value therefore signals haze.
//!
//! 2. **Atmospheric light A.** Take the top 0.1% brightest pixels in
//!    the dark channel; among those, pick the one whose RGB intensity
//!    (sum of channels) is highest in the *original* image and use its
//!    RGB as `A`.
//!
//! 3. **Transmission t(x).**
//!    `t(x) = 1 - omega * dark_channel(I / A)`. `omega ≈ 0.95` keeps a
//!    small amount of haze for naturalness.
//!
//! 4. **Scene radiance.**
//!    `J(x) = (I(x) - A) / max(t(x), t0) + A`, clamped to `[0, 1]`.
//!    The floor `t0` prevents division-by-zero blow-up in dark pixels.
//!
//! Edge handling is clamp-to-edge for the dark-channel min-pool. The
//! min-pool is naive O(W * H * patch²); a soft-matting / guided-filter
//! refinement and an O(W * H) min-pool come in Phase 4.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct DehazeDcp;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-weather.dehaze_dcp",
    display_name: "Dehaze (Dark Channel Prior)",
    description: "Classical haze removal using He et al.'s 2009 Dark Channel Prior.",
    category: Category::Weather,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "omega",
        display_name: "Omega",
        description: "Strength factor — 1.0 removes more haze, may oversaturate.",
        kind: ParamKind::Float { default: 0.95, min: Some(0.5), max: Some(1.0) },
    },
    ParamSpec {
        id: "t0",
        display_name: "Transmission floor",
        description: "Transmission floor to prevent dark-pixel blow-up.",
        kind: ParamKind::Float { default: 0.1, min: Some(0.01), max: Some(0.5) },
    },
    ParamSpec {
        id: "patch_radius",
        display_name: "Patch radius",
        description: "Half-width of the dark-channel min-pool window.",
        kind: ParamKind::Int { default: 7, min: Some(1), max: Some(31) },
    },
];

impl Effect for DehazeDcp {
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
        let omega = params.get_float("omega").unwrap_or(0.95).clamp(0.5, 1.0) as f32;
        let t0 = params.get_float("t0").unwrap_or(0.1).clamp(0.01, 0.5) as f32;
        let patch_radius = params
            .get_int("patch_radius")
            .unwrap_or(7)
            .clamp(1, 31) as usize;

        let mut frame = input.into_rgba_f32_linear();
        let w = frame.width as usize;
        let h = frame.height as usize;

        if w == 0 || h == 0 {
            return Ok(frame);
        }

        // Stage 1: per-pixel min over RGB.
        let pixel_min = pixel_min_rgb(frame.as_f32().expect("RgbaF32 after lift"), w, h);

        // Stage 2: min-pool to get dark channel.
        let dark = min_pool(&pixel_min, w, h, patch_radius);

        // Stage 3: atmospheric light A.
        let a = atmospheric_light(
            frame.as_f32().expect("RgbaF32 after lift"),
            &dark,
            w,
            h,
        );

        // Stage 4 + 5: transmission t(x) and scene recovery J(x).
        // We need a per-pixel transmission, which is computed from the
        // dark channel of `I / A`. Reuse the same min-pool over the
        // normalized-image min.
        let inv_a = [
            if a[0] > 1e-6 { 1.0 / a[0] } else { 1.0 },
            if a[1] > 1e-6 { 1.0 / a[1] } else { 1.0 },
            if a[2] > 1e-6 { 1.0 / a[2] } else { 1.0 },
        ];
        let pixel_min_norm = pixel_min_rgb_normalized(
            frame.as_f32().expect("RgbaF32 after lift"),
            w,
            h,
            inv_a,
        );
        let dark_norm = min_pool(&pixel_min_norm, w, h, patch_radius);

        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");
        for (i, px) in pixels.chunks_exact_mut(4).enumerate() {
            let t = (1.0 - omega * dark_norm[i]).max(t0);
            for c in 0..3 {
                let recovered = (px[c] - a[c]) / t + a[c];
                px[c] = recovered.clamp(0.0, 1.0);
            }
            // alpha untouched
        }

        Ok(frame)
    }
}

/// Per-pixel `min(R, G, B)` over an RGBA float buffer.
fn pixel_min_rgb(buf: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(w * h);
    for px in buf.chunks_exact(4) {
        out.push(px[0].min(px[1]).min(px[2]));
    }
    out
}

/// Per-pixel `min(R/Ar, G/Ag, B/Ab)` over an RGBA float buffer.
fn pixel_min_rgb_normalized(
    buf: &[f32],
    w: usize,
    h: usize,
    inv_a: [f32; 3],
) -> Vec<f32> {
    let mut out = Vec::with_capacity(w * h);
    for px in buf.chunks_exact(4) {
        let r = px[0] * inv_a[0];
        let g = px[1] * inv_a[1];
        let b = px[2] * inv_a[2];
        out.push(r.min(g).min(b));
    }
    out
}

/// Naive square-window min-pool with clamp-to-edge.
fn min_pool(src: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let r = radius as isize;
    let wi = w as isize;
    let hi = h as isize;
    for y in 0..hi {
        let y0 = (y - r).max(0) as usize;
        let y1 = (y + r).min(hi - 1) as usize;
        for x in 0..wi {
            let x0 = (x - r).max(0) as usize;
            let x1 = (x + r).min(wi - 1) as usize;
            let mut m = f32::INFINITY;
            for yy in y0..=y1 {
                let row = &src[yy * w..yy * w + w];
                let slice = &row[x0..=x1];
                for &v in slice {
                    if v < m {
                        m = v;
                    }
                }
            }
            out[(y as usize) * w + (x as usize)] = m;
        }
    }
    out
}

/// Pick atmospheric light `A` per He et al.: take the top 0.1% of
/// dark-channel pixels, then among those pick the one with highest RGB
/// intensity in the source.
fn atmospheric_light(buf: &[f32], dark: &[f32], w: usize, h: usize) -> [f32; 3] {
    let total = w * h;
    if total == 0 {
        return [1.0, 1.0, 1.0];
    }
    // At least 1 pixel even on tiny images.
    let count = ((total as f32 * 0.001).ceil() as usize).max(1);

    // Indices sorted by dark channel descending. For typical images
    // this is fine; a partial sort would be faster but we keep it
    // simple for the Phase 1 implementation.
    let mut idx: Vec<usize> = (0..total).collect();
    idx.sort_unstable_by(|&i, &j| {
        dark[j]
            .partial_cmp(&dark[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut best = [0.0f32; 3];
    let mut best_intensity = -1.0f32;
    for &i in idx.iter().take(count) {
        let off = i * 4;
        let r = buf[off];
        let g = buf[off + 1];
        let b = buf[off + 2];
        let intensity = r + g + b;
        if intensity > best_intensity {
            best_intensity = intensity;
            best = [r, g, b];
        }
    }
    // Avoid pathological zero atmospheric light.
    for c in &mut best {
        if !c.is_finite() {
            *c = 1.0;
        }
        // Keep within sensible range.
        *c = c.clamp(1e-3, 1.0);
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, PixelData};

    fn defaults_for(eff: &DehazeDcp) -> ParamValues {
        let mut p = ParamValues::new();
        p.validate_and_fill(eff.parameters()).unwrap();
        p
    }

    #[test]
    fn solid_color_near_identity() {
        // A solid, fully-saturated color has a per-pixel dark channel
        // of 0 (one channel is exactly zero), so the transmission is 1
        // everywhere and the recovery is the identity.
        let eff = DehazeDcp;
        let mut ctx = Context::for_still_srgb();
        let p = defaults_for(&eff);

        let mut data = vec![0u8; 16 * 16 * 4];
        for px in data.chunks_exact_mut(4) {
            px[0] = 220;
            px[1] = 40;
            px[2] = 40;
            px[3] = 255;
        }
        let f = Frame::new(16, 16, PixelData::Rgba8(data), ColorSpace::SRgb, None).unwrap();
        let out = eff.apply(&mut ctx, f, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!() };

        for chunk in px.chunks_exact(4) {
            assert!((chunk[0] as i32 - 220).abs() <= 2, "R drifted: {}", chunk[0]);
            assert!((chunk[1] as i32 - 40).abs() <= 2, "G drifted: {}", chunk[1]);
            assert!((chunk[2] as i32 - 40).abs() <= 2, "B drifted: {}", chunk[2]);
            assert_eq!(chunk[3], 255);
        }
    }

    #[test]
    fn hazy_gradient_increases_contrast() {
        // Synthesize a hazy scene: a pure-dark gradient from left to
        // right, then composited with a uniform white "haze" via the
        // physical model `I = J * t + A * (1 - t)` with A = 1.0 and a
        // moderate transmission t = 0.4. Dehaze should pull the dark
        // end darker (recover J), increasing dynamic range.
        let eff = DehazeDcp;
        let mut ctx = Context::for_still_srgb();
        let p = defaults_for(&eff);

        let w = 64usize;
        let h = 16usize;
        let t = 0.4f32;
        let a = 1.0f32;
        let mut data = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let j = (x as f32) / (w as f32 - 1.0); // 0..1
                let i_val = j * t + a * (1.0 - t);
                let off = (y * w + x) * 4;
                data[off] = i_val;
                data[off + 1] = i_val;
                data[off + 2] = i_val;
                data[off + 3] = 1.0;
            }
        }
        let frame = Frame::new(
            w as u32,
            h as u32,
            PixelData::RgbaF32(data.clone()),
            ColorSpace::LinearSRgb,
            None,
        )
        .unwrap();

        let in_min = data
            .chunks_exact(4)
            .map(|p| p[0])
            .fold(f32::INFINITY, f32::min);
        let in_max = data
            .chunks_exact(4)
            .map(|p| p[0])
            .fold(f32::NEG_INFINITY, f32::max);
        let in_range = in_max - in_min;

        let out = eff.apply(&mut ctx, frame, &p).unwrap();
        let pixels = out.as_f32().expect("RgbaF32 out");

        let out_min = pixels
            .chunks_exact(4)
            .map(|p| p[0])
            .fold(f32::INFINITY, f32::min);
        let out_max = pixels
            .chunks_exact(4)
            .map(|p| p[0])
            .fold(f32::NEG_INFINITY, f32::max);
        let out_range = out_max - out_min;

        assert!(
            out_range > in_range + 0.05,
            "expected dehaze to expand dynamic range: in={in_range}, out={out_range}"
        );
        // The dark side should get darker.
        assert!(out_min < in_min - 0.05, "dark side did not deepen: {out_min} vs {in_min}");
    }

    #[test]
    fn tiny_image_does_not_panic() {
        // 4x4 with default patch radius (7) — patch extends well past
        // the image bounds. The clamp-to-edge min-pool must handle this.
        let eff = DehazeDcp;
        let mut ctx = Context::for_still_srgb();
        let p = defaults_for(&eff);

        let mut data = Vec::with_capacity(4 * 4 * 4);
        for i in 0..16 {
            let v = (i * 16) as u8;
            data.extend_from_slice(&[v, v, v, 255]);
        }
        let f = Frame::new(4, 4, PixelData::Rgba8(data), ColorSpace::SRgb, None).unwrap();
        let out = eff.apply(&mut ctx, f, &p).unwrap();
        // Should still be a valid 4x4 frame.
        assert_eq!(out.width, 4);
        assert_eq!(out.height, 4);
        assert_eq!(out.pixel_count(), 16);
    }

    #[test]
    fn one_by_one_image_no_panic() {
        // Pathological: 1x1 image — sort-by-dark-channel still has to
        // produce a finite atmospheric light.
        let eff = DehazeDcp;
        let mut ctx = Context::for_still_srgb();
        let p = defaults_for(&eff);

        let f = Frame::new(
            1,
            1,
            PixelData::Rgba8(vec![200, 180, 150, 255]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();
        let out = eff.apply(&mut ctx, f, &p).unwrap();
        assert_eq!(out.pixel_count(), 1);
    }
}
