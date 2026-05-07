//! Auto-enhance — analyze an input frame, pick optimal effect parameters,
//! produce a chain that mirrors the in-browser demo at `apps/web/demo.js`.
//!
//! The algorithm is intentionally simple and explainable:
//!
//! 1. **Brightness/contrast** stretches the input so the 0.5 / 99.5
//!    percentiles of luma map to 0.05 / 0.95.
//! 2. **Gamma** pulls the post-BC median toward 0.5.
//! 3. **Saturation** boosts when chroma is low, tames when very high.
//! 4. **Unsharp mask** is sized by an edge-density proxy.
//!
//! Each stage is suppressed if it would be a near-no-op. The chain is
//! capped at four steps; effects beyond what the spec calls "tone +
//! color + sharpen" stay in the user's hands.

use anyhow::Result;
use lumen_core::Frame;
use serde_json::json;

use crate::{Recipe, RecipeStep};

#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub p01: f64,
    pub p50: f64,
    pub p99: f64,
    pub chroma_mean: f64,
    pub edge_mean: f64,
    pub luminance_mean: f64,
}

/// Single-pass analysis of a frame in linear-light space.
pub fn analyze_frame(frame: &Frame) -> Stats {
    let lifted = frame.clone().into_rgba_f32_linear();
    let pixels = lifted.as_f32().expect("RgbaF32 after lift");
    let w = lifted.width as usize;
    let h = lifted.height as usize;
    let n = w * h;

    if n == 0 {
        return Stats {
            p01: 0.0, p50: 0.0, p99: 0.0,
            chroma_mean: 0.0, edge_mean: 0.0, luminance_mean: 0.0,
        };
    }

    let mut y_arr = Vec::with_capacity(n);
    let mut r_sum = 0.0f64;
    let mut g_sum = 0.0f64;
    let mut b_sum = 0.0f64;
    let mut chroma_sum = 0.0f64;

    for px in pixels.chunks_exact(4) {
        let (r, g, b) = (px[0], px[1], px[2]);
        r_sum += r as f64;
        g_sum += g as f64;
        b_sum += b as f64;
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        chroma_sum += (mx - mn) as f64;
        y_arr.push(0.212_6_f32 * r + 0.715_2_f32 * g + 0.072_2_f32 * b);
    }

    // Percentiles via sort. n ~ 200K-2M; partial_sort would be cheaper
    // but full sort is fine for an analyzer that runs once.
    let mut sorted = y_arr.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pct = |q: f64| -> f64 {
        let idx = ((q * (sorted.len() - 1) as f64) as usize).min(sorted.len() - 1);
        sorted[idx] as f64
    };

    let mut edge_sum = 0.0f64;
    let mut edge_count = 0u64;
    for y in 0..h {
        let row_off = y * w;
        for x in 1..w {
            edge_sum += (y_arr[row_off + x] - y_arr[row_off + x - 1]).abs() as f64;
            edge_count += 1;
        }
    }

    Stats {
        p01: pct(0.005),
        p50: pct(0.500),
        p99: pct(0.995),
        chroma_mean: chroma_sum / n as f64,
        edge_mean: if edge_count > 0 {
            edge_sum / edge_count as f64
        } else {
            0.0
        },
        luminance_mean: (r_sum + g_sum + b_sum) / (3.0 * n as f64),
    }
}

/// Build a recipe chain from analyzer output. Mirrors `buildAutoChain`
/// in `apps/web/demo.js`.
pub fn build_auto_chain(stats: &Stats) -> Vec<RecipeStep> {
    let mut chain = Vec::new();
    let round2 = |x: f64| (x * 100.0).round() / 100.0;
    let round3 = |x: f64| (x * 1000.0).round() / 1000.0;
    let clamp = |v: f64, lo: f64, hi: f64| v.max(lo).min(hi);

    // 1. Brightness/contrast: stretch p01 -> 0.05 and p99 -> 0.95.
    let range = (stats.p99 - stats.p01).max(0.02);
    let c = clamp(0.90 / range, 0.7, 2.5);
    let b = clamp(0.05 - (stats.p01 - 0.5) * c - 0.5, -0.4, 0.4);
    if (c - 1.0).abs() > 0.04 || b.abs() > 0.02 {
        chain.push(RecipeStep {
            effect: "lumen-fx-exposure.brightness_contrast".to_string(),
            label: Some("auto bc".to_string()),
            params: json!({ "brightness": round3(b), "contrast": round2(c) }),
        });
    }

    // 2. Gamma: pull post-BC p50 toward 0.5.
    let new_p50 = clamp((stats.p50 - 0.5) * c + 0.5 + b, 0.02, 0.98);
    if (new_p50 - 0.5).abs() > 0.04 {
        let g = clamp(new_p50.ln() / 0.5_f64.ln(), 0.5, 2.0);
        if (g - 1.0).abs() > 0.03 {
            chain.push(RecipeStep {
                effect: "lumen-fx-exposure.gamma".to_string(),
                label: Some("auto gamma".to_string()),
                params: json!({ "gamma": round2(g) }),
            });
        }
    }

    // 3. Saturation by chroma magnitude.
    let amount: f64 = if stats.chroma_mean < 0.05 {
        1.15
    } else if stats.chroma_mean < 0.10 {
        1.30
    } else if stats.chroma_mean < 0.20 {
        1.20
    } else if stats.chroma_mean > 0.40 {
        0.92
    } else {
        1.0
    };
    if (amount - 1.0).abs() > 0.03 {
        chain.push(RecipeStep {
            effect: "lumen-fx-color.saturation".to_string(),
            label: Some("auto sat".to_string()),
            params: json!({ "amount": round2(amount) }),
        });
    }

    // 4. Unsharp mask sized by edge density.
    let (s_amount, s_radius) = if stats.edge_mean < 0.04 {
        (0.9, 1.2)
    } else if stats.edge_mean < 0.08 {
        (0.6, 1.1)
    } else if stats.edge_mean > 0.18 {
        (0.25, 0.9)
    } else {
        (0.5, 1.0)
    };
    chain.push(RecipeStep {
        effect: "lumen-fx-sharpen.unsharp_mask".to_string(),
        label: Some("auto sharpen".to_string()),
        params: json!({
            "amount": round2(s_amount),
            "radius": round2(s_radius),
            "threshold": 0.0,
        }),
    });

    chain
}

/// Build the analyzed recipe for a given input path. Output path is
/// included so a returned `Recipe` round-trips through `cmd_pipeline`
/// without further fixup.
pub fn analyze_path(input: &std::path::Path, output: &std::path::Path) -> Result<(Stats, Recipe)> {
    let frame = lumen_io::decode_image(input).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    let stats = analyze_frame(&frame);
    let recipe = Recipe {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        chain: build_auto_chain(&stats),
    };
    Ok((stats, recipe))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, PixelData};

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> Frame {
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            data.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
        }
        Frame::new(w, h, PixelData::Rgba8(data), ColorSpace::SRgb, None).unwrap()
    }

    #[test]
    fn analyze_empty_does_not_panic() {
        let f = Frame::new(0, 0, PixelData::Rgba8(vec![]), ColorSpace::SRgb, None).unwrap();
        let _ = analyze_frame(&f);
    }

    #[test]
    fn analyze_solid_gray_has_zero_chroma_and_edges() {
        let f = solid(32, 32, [128, 128, 128]);
        let s = analyze_frame(&f);
        assert!(s.chroma_mean < 1e-6);
        assert!(s.edge_mean < 1e-6);
        // p01 ≈ p50 ≈ p99 because every pixel is identical.
        assert!((s.p01 - s.p50).abs() < 1e-5);
        assert!((s.p99 - s.p50).abs() < 1e-5);
    }

    #[test]
    fn auto_chain_for_low_contrast_image_includes_bc() {
        // A 4x4 image where every pixel is mid-gray — p99 == p01, so
        // the BC step IS suppressed (no useful stretch). Use a small
        // gradient instead so p99 - p01 > 0.
        let mut data = Vec::with_capacity(16 * 4);
        for i in 0..16u8 {
            // Pack values around mid-gray with a tiny range.
            let v = 100 + (i % 4) * 8; // 100..124, narrow band
            data.extend_from_slice(&[v, v, v, 255]);
        }
        let f = Frame::new(4, 4, PixelData::Rgba8(data), ColorSpace::SRgb, None).unwrap();
        let stats = analyze_frame(&f);
        let chain = build_auto_chain(&stats);
        // Narrow-band gray should always at least produce an unsharp mask
        // step (we always include sharpen) plus a stretch.
        assert!(!chain.is_empty(), "chain is empty for {:?}", stats);
        assert!(
            chain
                .iter()
                .any(|s| s.effect == "lumen-fx-exposure.brightness_contrast"),
            "expected BC step for narrow luma range, got chain {:?}",
            chain.iter().map(|s| &s.effect).collect::<Vec<_>>()
        );
    }
}
