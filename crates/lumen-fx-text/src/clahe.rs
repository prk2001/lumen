//! Contrast-Limited Adaptive Histogram Equalization (CLAHE).
//!
//! Boosts local contrast in dark / low-contrast tiles without blowing out
//! already-bright regions — the classical first pass for plate, sign, and
//! text clarification on CCTV-style imagery.
//!
//! Algorithm (per frame, on the Rec.709 luma channel):
//!
//! 1. Lift to linear RGBA float.
//! 2. Compute `Y = 0.2126*R + 0.7152*G + 0.0722*B` per pixel.
//! 3. Tile the luma plane into `tiles_x * tiles_y` non-overlapping regions
//!    and build a 256-bin histogram per tile.
//! 4. Clip each bin to `clip_limit * (tile_pixels / 256)` and redistribute
//!    the clipped excess uniformly across all bins.
//! 5. Build a normalized CDF (range 0..1) per tile.
//! 6. For each output pixel bilinearly interpolate the four nearest tile
//!    CDFs to produce the equalized luma `Y_new`.
//! 7. Re-color the pixel by `R,G,B *= Y_new / max(Y_old, eps)` so chroma
//!    is preserved. Final RGB is clamped to `[0, 1]`.
//!
//! `clip_limit < 0.01` switches to global histogram equalization (no
//! clipping). `tiles_x = tiles_y = 1` is global histogram equalization
//! over the whole frame.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use tracing::instrument;

/// CLAHE — contrast-limited adaptive histogram equalization on luma.
#[derive(Debug, Default)]
pub struct Clahe;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-text.clahe",
    display_name: "CLAHE",
    description: "Contrast-limited adaptive histogram equalization for plate/text clarity.",
    category: Category::Text,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "tiles_x",
        display_name: "Tiles X",
        description: "Horizontal tile count.",
        kind: ParamKind::Int { default: 8, min: Some(1), max: Some(64) },
    },
    ParamSpec {
        id: "tiles_y",
        display_name: "Tiles Y",
        description: "Vertical tile count.",
        kind: ParamKind::Int { default: 8, min: Some(1), max: Some(64) },
    },
    ParamSpec {
        id: "clip_limit",
        display_name: "Clip Limit",
        description: "Histogram clip multiplier (1.0 = no clipping).",
        kind: ParamKind::Float { default: 2.5, min: Some(0.0), max: Some(16.0) },
    },
];

const BINS: usize = 256;
const BINS_F: f32 = 255.0;
const EPS: f32 = 1.0e-6;

impl Effect for Clahe {
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
        let tiles_x = params.get_int("tiles_x").unwrap_or(8).max(1) as usize;
        let tiles_y = params.get_int("tiles_y").unwrap_or(8).max(1) as usize;
        let clip_limit = params.get_float("clip_limit").unwrap_or(2.5).max(0.0) as f32;

        let mut frame = input.into_rgba_f32_linear();
        let width = frame.width as usize;
        let height = frame.height as usize;

        if width == 0 || height == 0 {
            return Ok(frame);
        }

        // Cap tiles so each tile holds at least 1 pixel.
        let tx = tiles_x.min(width).max(1);
        let ty = tiles_y.min(height).max(1);

        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");

        // 1. Compute luma plane.
        let mut luma = vec![0.0f32; width * height];
        for (i, px) in pixels.chunks_exact(4).enumerate() {
            luma[i] = 0.212_6 * px[0] + 0.715_2 * px[1] + 0.072_2 * px[2];
        }

        // 2. Build per-tile clipped CDFs (each CDF is BINS entries in 0..=1).
        let cdfs = build_tile_cdfs(&luma, width, height, tx, ty, clip_limit);

        // 3. Apply per-pixel bilinear interpolation across neighboring tiles.
        for y in 0..height {
            // Map pixel y to tile-center coordinates; tiles span tile_h rows.
            let (ty0, ty1, fy) = tile_neighbors(y, height, ty);
            for x in 0..width {
                let (tx0, tx1, fx) = tile_neighbors(x, width, tx);
                let i = y * width + x;
                let y_old = luma[i];
                let bin = (y_old.clamp(0.0, 1.0) * BINS_F).round() as usize;
                let bin = bin.min(BINS - 1);

                // Sample the 4 surrounding tile CDFs.
                let c00 = cdfs[ty0 * tx + tx0][bin];
                let c10 = cdfs[ty0 * tx + tx1][bin];
                let c01 = cdfs[ty1 * tx + tx0][bin];
                let c11 = cdfs[ty1 * tx + tx1][bin];

                let top = c00 * (1.0 - fx) + c10 * fx;
                let bot = c01 * (1.0 - fx) + c11 * fx;
                let y_new = top * (1.0 - fy) + bot * fy;

                let scale = y_new / y_old.max(EPS);
                let pi = i * 4;
                pixels[pi] = (pixels[pi] * scale).clamp(0.0, 1.0);
                pixels[pi + 1] = (pixels[pi + 1] * scale).clamp(0.0, 1.0);
                pixels[pi + 2] = (pixels[pi + 2] * scale).clamp(0.0, 1.0);
                // alpha untouched
            }
        }

        Ok(frame)
    }
}

/// Bilinear-interp coordinates: returns `(tile_a, tile_b, frac)` along one
/// axis. The pixel at coordinate `p` is treated as sitting between tiles
/// whose centers are at `(tile_idx + 0.5) * tile_size`. Pixels in the outer
/// half-tile collapse `tile_a == tile_b` so they sample a single CDF.
fn tile_neighbors(p: usize, total: usize, tiles: usize) -> (usize, usize, f32) {
    let tile_size = total as f32 / tiles as f32;
    // Position in "tile center" coordinates.
    let t = (p as f32 + 0.5) / tile_size - 0.5;
    if t <= 0.0 {
        return (0, 0, 0.0);
    }
    let max_t = (tiles - 1) as f32;
    if t >= max_t {
        let last = tiles - 1;
        return (last, last, 0.0);
    }
    let a = t.floor();
    let frac = t - a;
    let ia = a as usize;
    let ib = (ia + 1).min(tiles - 1);
    (ia, ib, frac)
}

/// Build one clipped-redistributed CDF per tile. Returns `tiles_y * tiles_x`
/// CDFs in row-major order; each CDF has [`BINS`] entries normalized to
/// `[0, 1]`.
fn build_tile_cdfs(
    luma: &[f32],
    width: usize,
    height: usize,
    tx: usize,
    ty: usize,
    clip_limit: f32,
) -> Vec<[f32; BINS]> {
    let mut cdfs = vec![[0.0f32; BINS]; tx * ty];

    for tj in 0..ty {
        let y0 = tj * height / ty;
        let y1 = (tj + 1) * height / ty;
        for ti in 0..tx {
            let x0 = ti * width / tx;
            let x1 = (ti + 1) * width / tx;

            let tile_w = x1 - x0;
            let tile_h = y1 - y0;
            let tile_pixels = (tile_w * tile_h).max(1);

            // Histogram.
            let mut hist = [0u32; BINS];
            for yy in y0..y1 {
                let row = &luma[yy * width + x0..yy * width + x1];
                for &v in row {
                    let bin = (v.clamp(0.0, 1.0) * BINS_F).round() as usize;
                    let bin = bin.min(BINS - 1);
                    hist[bin] += 1;
                }
            }

            // Clip + redistribute. clip_limit < 0.01 -> plain hist eq.
            if clip_limit >= 0.01 {
                let limit = ((clip_limit * tile_pixels as f32) / BINS as f32).max(1.0) as u32;
                let mut excess: u64 = 0;
                for h in &mut hist {
                    if *h > limit {
                        excess += (*h - limit) as u64;
                        *h = limit;
                    }
                }
                if excess > 0 {
                    let bonus = (excess / BINS as u64) as u32;
                    let mut leftover = (excess % BINS as u64) as u32;
                    for h in &mut hist {
                        *h += bonus;
                    }
                    // Spread the remainder evenly across the histogram by
                    // stepping with a stride — avoids biasing the low end
                    // when the leftover is small relative to BINS.
                    if leftover > 0 {
                        let stride = (BINS as u32)
                            .checked_div(leftover)
                            .unwrap_or(1)
                            .max(1) as usize;
                        let mut i = 0usize;
                        while leftover > 0 && i < BINS {
                            hist[i] += 1;
                            leftover -= 1;
                            i += stride;
                        }
                    }
                }
            }

            // Build the per-tile look-up table. We use the standard
            // shifted-CDF form
            //
            //     lut[b] = (cdf[b] - cdf_min) / (1 - cdf_min)
            //
            // where `cdf_min` is the smallest non-zero CDF value (i.e. the
            // CDF at the first occupied bin). This stretches the output
            // across the full [0, 1] range — proper histogram equalization
            // — while still mapping a single-valued tile to itself when
            // the histogram is degenerate.
            let bin_min = hist.iter().position(|&c| c > 0).unwrap_or(0);
            let bin_max = hist.iter().rposition(|&c| c > 0).unwrap_or(BINS - 1);

            let cdf = &mut cdfs[tj * tx + ti];
            if bin_min == bin_max {
                // Single-bin tile: identity map at that bin.
                for (b, slot) in cdf.iter_mut().enumerate() {
                    *slot = (b as f32 / BINS_F).clamp(0.0, 1.0);
                }
            } else {
                let total = tile_pixels as f32;
                let cdf_min = hist[bin_min] as f32 / total;
                let denom = (1.0 - cdf_min).max(EPS);
                let mut acc: u64 = 0;
                for b in 0..BINS {
                    acc += hist[b] as u64;
                    let raw = acc as f32 / total;
                    cdf[b] = ((raw - cdf_min) / denom).clamp(0.0, 1.0);
                }
            }
        }
    }

    cdfs
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue, PixelData};

    fn frame_from_gray(width: u32, height: u32, levels: &[u8]) -> Frame {
        assert_eq!(levels.len(), (width * height) as usize);
        let mut buf = Vec::with_capacity(levels.len() * 4);
        for &v in levels {
            buf.extend_from_slice(&[v, v, v, 255]);
        }
        Frame::new(width, height, PixelData::Rgba8(buf), ColorSpace::SRgb, None).unwrap()
    }

    fn extract_gray(frame: Frame) -> Vec<u8> {
        let out = frame.into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else { panic!("expected u8") };
        px.chunks_exact(4).map(|p| p[0]).collect()
    }

    #[test]
    fn solid_image_unchanged() {
        // Constant-luma input has a degenerate histogram (all mass in one
        // bin). With clipping disabled the midpoint-CDF maps every pixel
        // back to its original value.
        let clahe = Clahe;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        // Disable clipping so the histogram is the raw count — the
        // midpoint CDF is then guaranteed identity on a single-bin
        // distribution.
        p.insert("clip_limit", ParamValue::Float(0.0));
        p.validate_and_fill(clahe.parameters()).unwrap();

        let input = frame_from_gray(16, 16, &[128u8; 256]);
        let out = clahe.apply(&mut ctx, input, &p).unwrap();
        let pixels = extract_gray(out);
        for v in pixels {
            assert!(
                (v as i32 - 128).abs() <= 2,
                "solid image drifted: got {v}, expected 128"
            );
        }
    }

    #[test]
    fn low_contrast_gradient_gains_contrast() {
        // 32x32 horizontal ramp packed into the [100..156] band — a
        // narrow-dynamic-range gradient. After CLAHE the spread should
        // widen because hist eq stretches mid-range.
        let w = 32u32;
        let h = 32u32;
        let mut levels = Vec::with_capacity((w * h) as usize);
        for _y in 0..h {
            for x in 0..w {
                let v = 100 + (x * 56 / (w - 1)) as u8;
                levels.push(v);
            }
        }

        let in_min = *levels.iter().min().unwrap();
        let in_max = *levels.iter().max().unwrap();
        let in_range = (in_max - in_min) as i32;

        let clahe = Clahe;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        // Use 1x1 tiling so this becomes global hist eq — easy to reason about.
        p.insert("tiles_x", ParamValue::Int(1));
        p.insert("tiles_y", ParamValue::Int(1));
        p.insert("clip_limit", ParamValue::Float(0.0));
        p.validate_and_fill(clahe.parameters()).unwrap();

        let input = frame_from_gray(w, h, &levels);
        let out = clahe.apply(&mut ctx, input, &p).unwrap();
        let pixels = extract_gray(out);

        let out_min = *pixels.iter().min().unwrap();
        let out_max = *pixels.iter().max().unwrap();
        let out_range = (out_max - out_min) as i32;

        assert!(
            out_range > in_range,
            "expected CLAHE to widen dynamic range: in={in_range} out={out_range}"
        );
    }

    #[test]
    fn tiny_image_does_not_panic() {
        // 4x4 with default 8x8 tiling: more tiles than pixels — should
        // collapse gracefully without panicking.
        let clahe = Clahe;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.validate_and_fill(clahe.parameters()).unwrap();

        let levels: Vec<u8> = (0..16).map(|i| (i * 16) as u8).collect();
        let input = frame_from_gray(4, 4, &levels);
        let out = clahe.apply(&mut ctx, input, &p).unwrap();
        let pixels = extract_gray(out);
        assert_eq!(pixels.len(), 16);
    }

    #[test]
    fn zero_alpha_is_handled() {
        // A fully-black, alpha-0 pixel hits the eps-guard divide path.
        let clahe = Clahe;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.validate_and_fill(clahe.parameters()).unwrap();

        let buf = vec![0u8; 4 * 4 * 4]; // 4x4, all zero RGBA
        let input =
            Frame::new(4, 4, PixelData::Rgba8(buf), ColorSpace::SRgb, None).unwrap();
        let out = clahe.apply(&mut ctx, input, &p).unwrap();
        let _ = out.into_rgba_u8_srgb();
    }
}
