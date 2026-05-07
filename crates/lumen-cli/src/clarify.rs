//! Surveillance / forensic clarification preset.
//!
//! Targeted at low-quality stills — CCTV, dashcam, cell-phone screen grabs,
//! compressed video frames. The preset chains the existing forensic-lane
//! effects in an order that's been useful in practice:
//!
//! 1. **Gaussian denoise** — kill sensor + ISO grain that would amplify
//!    in later stages.
//! 2. **Deblock** — soften visible 8x8 JPEG/H.264 block boundaries.
//! 3. **Dehaze (Dark Channel Prior)** — remove glare / atmospheric scatter
//!    / IR sensor wash.
//! 4. **CLAHE** — localized contrast amplification on the luma channel,
//!    the classical plate / text clarification go-to.
//! 5. **Laplacian deblur** — DoG-based edge enhancement to recover detail
//!    that the previous stages flatten.
//! 6. **Unsharp mask** — final detail kicker.
//! 7. **Brightness/contrast stretch** — clamp the histogram into the
//!    visible range.
//! 8. **(Optional) bicubic 2x upscale** — for "magnify and clarify"
//!    workflows.
//!
//! Three strength tiers (light/standard/aggressive) just nudge the
//! parameters; the chain shape is identical.

use anyhow::Result;
use serde_json::json;

use crate::{Recipe, RecipeStep};

#[derive(Debug, Clone, Copy)]
struct Params {
    nr_sigma: f64,
    deblock_strength: f64,
    dehaze_omega: f64,
    clahe_clip: f64,
    clahe_tiles: i64,
    laplacian_amount: f64,   // 0.0 disables the laplacian step
    unsharp_amount: f64,
    bc_contrast: f64,
    /// Second-pass Gaussian denoise sigma applied AFTER dehaze + CLAHE
    /// to kill amplified noise. The single biggest fidelity safeguard.
    cleanup_sigma: f64,
}

/// Tuned for fidelity: dehaze strengths capped, with a second denoise
/// pass AFTER dehaze + CLAHE specifically to kill amplified noise.
/// The previous version over-sharpened and hallucinated structure on
/// low-information inputs (e.g. dehaze omega=0.95 + CLAHE clip=4.0
/// would expose JPEG noise). These parameters favor "don't make stuff
/// up" over "reveal everything possible".
fn params_for(strength: &str) -> Params {
    match strength {
        "light" => Params {
            nr_sigma: 0.5,
            deblock_strength: 0.25,
            dehaze_omega: 0.55,
            clahe_clip: 1.2,
            clahe_tiles: 8,
            laplacian_amount: 0.0,   // off — too noise-amplifying for low-info inputs
            unsharp_amount: 0.35,
            bc_contrast: 1.04,
            cleanup_sigma: 0.35,
        },
        "aggressive" => Params {
            nr_sigma: 0.9,
            deblock_strength: 0.6,
            dehaze_omega: 0.80,      // was 0.95 — over-amplified noise
            clahe_clip: 2.5,         // was 4.0
            clahe_tiles: 10,
            laplacian_amount: 0.4,   // was 1.4 — keep mild
            unsharp_amount: 0.7,     // was 1.3
            bc_contrast: 1.10,       // was 1.20
            cleanup_sigma: 0.55,
        },
        _ => Params {
            // "standard"
            nr_sigma: 0.7,
            deblock_strength: 0.45,
            dehaze_omega: 0.65,
            clahe_clip: 1.8,
            clahe_tiles: 8,
            laplacian_amount: 0.0,
            unsharp_amount: 0.5,
            bc_contrast: 1.06,
            cleanup_sigma: 0.45,
        },
    }
}

pub fn build_clarify_recipe(
    input: &std::path::Path,
    output: &std::path::Path,
    strength: &str,
    upscale: bool,
) -> Result<Recipe> {
    let p = params_for(strength);
    let mut chain: Vec<RecipeStep> = vec![
        // 1. Pre-clean: kill sensor noise that would amplify in dehaze.
        RecipeStep {
            effect: "lumen-fx-denoise.gaussian".into(),
            label: Some("clarify denoise (pre)".into()),
            params: json!({ "sigma": p.nr_sigma }),
        },
        // 2. Soften JPEG/H.264 block edges.
        RecipeStep {
            effect: "lumen-fx-compression.deblock".into(),
            label: Some("clarify deblock".into()),
            params: json!({ "block_size": 8, "strength": p.deblock_strength }),
        },
        // 3. Dehaze with capped omega — Phase 1 used 0.95 which exposed
        //    every bit of noise; 0.55-0.80 keeps it honest.
        RecipeStep {
            effect: "lumen-fx-weather.dehaze_dcp".into(),
            label: Some("clarify dehaze".into()),
            params: json!({
                "omega": p.dehaze_omega,
                "t0": 0.15,
                "patch_radius": 5,
            }),
        },
        // 4. CLAHE for local contrast — clip-limit reduced from 4.0 → 2.5
        //    so quiet regions don't get histogram-blown into noise fields.
        RecipeStep {
            effect: "lumen-fx-text.clahe".into(),
            label: Some("clarify CLAHE".into()),
            params: json!({
                "tiles_x": p.clahe_tiles,
                "tiles_y": p.clahe_tiles,
                "clip_limit": p.clahe_clip,
            }),
        },
        // 5. Cleanup denoise — the single biggest fidelity safeguard.
        //    Dehaze + CLAHE inevitably amplify noise; this pass mops it up.
        RecipeStep {
            effect: "lumen-fx-denoise.gaussian".into(),
            label: Some("clarify denoise (post)".into()),
            params: json!({ "sigma": p.cleanup_sigma }),
        },
    ];

    // 6. Optional Laplacian deblur — off by default in fidelity mode.
    //    Stacking laplacian + unsharp on a low-info input hallucinates structure.
    if p.laplacian_amount > 0.0 {
        chain.push(RecipeStep {
            effect: "lumen-fx-deblur.laplacian".into(),
            label: Some("clarify deblur".into()),
            params: json!({
                "amount": p.laplacian_amount,
                "sigma": 0.9,
                "sigma_ratio": 1.6,
            }),
        });
    }

    // 7. Final unsharp — restore edge crispness. Threshold > 0 protects
    //    flat regions from getting their texture amplified.
    chain.push(RecipeStep {
        effect: "lumen-fx-sharpen.unsharp_mask".into(),
        label: Some("clarify sharpen".into()),
        params: json!({
            "amount": p.unsharp_amount,
            "radius": 1.0,
            "threshold": 0.015,
        }),
    });

    // 8. Mild tone stretch.
    chain.push(RecipeStep {
        effect: "lumen-fx-exposure.brightness_contrast".into(),
        label: Some("clarify tone".into()),
        params: json!({ "brightness": 0.0, "contrast": p.bc_contrast }),
    });

    if upscale {
        chain.push(RecipeStep {
            effect: "lumen-fx-upscale.bicubic".into(),
            label: Some("clarify upscale".into()),
            params: json!({ "scale": 2.0 }),
        });
    }
    Ok(Recipe {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        chain,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_chain_includes_post_dehaze_denoise_for_fidelity() {
        let r = build_clarify_recipe(
            std::path::Path::new("/tmp/in.png"),
            std::path::Path::new("/tmp/out.png"),
            "standard",
            false,
        )
        .unwrap();
        // Two denoise passes — pre-dehaze and post-dehaze cleanup.
        let denoise_count = r
            .chain
            .iter()
            .filter(|s| s.effect == "lumen-fx-denoise.gaussian")
            .count();
        assert_eq!(
            denoise_count, 2,
            "fidelity mode requires two denoise passes; got {denoise_count}"
        );
    }

    #[test]
    fn aggressive_no_upscale_chain_size_is_bounded() {
        // After fidelity retuning, the aggressive chain (with laplacian
        // re-enabled) is at most: denoise + deblock + dehaze + CLAHE +
        // denoise + laplacian + unsharp + tone = 8 steps.
        let r = build_clarify_recipe(
            std::path::Path::new("/tmp/in.png"),
            std::path::Path::new("/tmp/out.png"),
            "aggressive",
            false,
        )
        .unwrap();
        assert!(r.chain.len() <= 8);
    }

    #[test]
    fn aggressive_strength_pushes_dehaze_higher_than_light() {
        let light = build_clarify_recipe(
            std::path::Path::new("/tmp/in.png"),
            std::path::Path::new("/tmp/out.png"),
            "light",
            false,
        )
        .unwrap();
        let aggro = build_clarify_recipe(
            std::path::Path::new("/tmp/in.png"),
            std::path::Path::new("/tmp/out.png"),
            "aggressive",
            false,
        )
        .unwrap();
        let omega_of = |r: &Recipe| -> f64 {
            r.chain
                .iter()
                .find(|s| s.effect == "lumen-fx-weather.dehaze_dcp")
                .and_then(|s| s.params.get("omega"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
        };
        assert!(omega_of(&aggro) > omega_of(&light));
        // Cap: even aggressive must not exceed 0.85 — anything higher
        // hallucinates structure on low-information inputs.
        assert!(omega_of(&aggro) <= 0.85, "aggressive omega should be capped");
    }
}
