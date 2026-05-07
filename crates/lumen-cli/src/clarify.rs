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
    laplacian_amount: f64,
    unsharp_amount: f64,
    bc_contrast: f64,
}

fn params_for(strength: &str) -> Params {
    match strength {
        "light" => Params {
            nr_sigma: 0.6,
            deblock_strength: 0.3,
            dehaze_omega: 0.6,
            clahe_clip: 1.5,
            clahe_tiles: 8,
            laplacian_amount: 0.5,
            unsharp_amount: 0.5,
            bc_contrast: 1.05,
        },
        "aggressive" => Params {
            nr_sigma: 1.4,
            deblock_strength: 0.9,
            dehaze_omega: 0.95,
            clahe_clip: 4.0,
            clahe_tiles: 12,
            laplacian_amount: 1.4,
            unsharp_amount: 1.3,
            bc_contrast: 1.20,
        },
        _ => Params {
            // "standard"
            nr_sigma: 0.9,
            deblock_strength: 0.6,
            dehaze_omega: 0.8,
            clahe_clip: 2.5,
            clahe_tiles: 8,
            laplacian_amount: 0.9,
            unsharp_amount: 0.8,
            bc_contrast: 1.10,
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
        RecipeStep {
            effect: "lumen-fx-denoise.gaussian".into(),
            label: Some("clarify denoise".into()),
            params: json!({ "sigma": p.nr_sigma }),
        },
        RecipeStep {
            effect: "lumen-fx-compression.deblock".into(),
            label: Some("clarify deblock".into()),
            params: json!({ "block_size": 8, "strength": p.deblock_strength }),
        },
        RecipeStep {
            effect: "lumen-fx-weather.dehaze_dcp".into(),
            label: Some("clarify dehaze".into()),
            params: json!({
                "omega": p.dehaze_omega,
                "t0": 0.1,
                "patch_radius": 5,
            }),
        },
        RecipeStep {
            effect: "lumen-fx-text.clahe".into(),
            label: Some("clarify CLAHE".into()),
            params: json!({
                "tiles_x": p.clahe_tiles,
                "tiles_y": p.clahe_tiles,
                "clip_limit": p.clahe_clip,
            }),
        },
        RecipeStep {
            effect: "lumen-fx-deblur.laplacian".into(),
            label: Some("clarify deblur".into()),
            params: json!({
                "amount": p.laplacian_amount,
                "sigma": 0.8,
                "sigma_ratio": 1.6,
            }),
        },
        RecipeStep {
            effect: "lumen-fx-sharpen.unsharp_mask".into(),
            label: Some("clarify sharpen".into()),
            params: json!({
                "amount": p.unsharp_amount,
                "radius": 1.0,
                "threshold": 0.0,
            }),
        },
        RecipeStep {
            effect: "lumen-fx-exposure.brightness_contrast".into(),
            label: Some("clarify tone".into()),
            params: json!({ "brightness": 0.0, "contrast": p.bc_contrast }),
        },
    ];
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
    fn standard_chain_has_seven_or_eight_steps() {
        let r = build_clarify_recipe(
            std::path::Path::new("/tmp/in.png"),
            std::path::Path::new("/tmp/out.png"),
            "standard",
            true,
        )
        .unwrap();
        assert_eq!(r.chain.len(), 8);
        assert_eq!(r.chain.last().unwrap().effect, "lumen-fx-upscale.bicubic");
    }

    #[test]
    fn no_upscale_is_one_shorter() {
        let r = build_clarify_recipe(
            std::path::Path::new("/tmp/in.png"),
            std::path::Path::new("/tmp/out.png"),
            "standard",
            false,
        )
        .unwrap();
        assert_eq!(r.chain.len(), 7);
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
    }
}
