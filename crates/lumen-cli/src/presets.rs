//! Stylistic presets — fixed chains for one-click looks.
//!
//! Mirrors `buildStyleChain` in `apps/web/demo.js` so the demo's
//! "same engine, same recipes" claim is byte-for-byte true.
//!
//! Available preset names:
//!
//! - **pop** — saturation boost + crisp sharpen + slight contrast
//! - **bw** — luma-only grayscale conversion with mild contrast
//! - **vintage** — desaturated + warm gamma lift + soft sharpen
//! - **sharpen** — laplacian deblur + unsharp mask
//! - **restore** — denoise + CLAHE + saturation + sharpen
//!
//! Use `lumen style --name <preset> --input X --output Y` to apply.

use anyhow::{anyhow, Result};
use serde_json::json;

use crate::{Recipe, RecipeStep};

pub fn build_style_recipe(
    input: &std::path::Path,
    output: &std::path::Path,
    name: &str,
) -> Result<Recipe> {
    let chain = match name {
        "pop" => vec![
            RecipeStep {
                effect: "lumen-fx-color.saturation".into(),
                label: Some("pop sat".into()),
                params: json!({ "amount": 1.35 }),
            },
            RecipeStep {
                effect: "lumen-fx-sharpen.unsharp_mask".into(),
                label: Some("pop sharpen".into()),
                params: json!({ "amount": 1.0, "radius": 1.0, "threshold": 0.0 }),
            },
            RecipeStep {
                effect: "lumen-fx-exposure.brightness_contrast".into(),
                label: Some("pop bc".into()),
                params: json!({ "brightness": 0.02, "contrast": 1.1 }),
            },
        ],
        "bw" => vec![
            RecipeStep {
                effect: "lumen-fx-modalities.channel_isolate".into(),
                label: Some("bw luma".into()),
                params: json!({ "channel": "luma", "invert": false }),
            },
            RecipeStep {
                effect: "lumen-fx-exposure.brightness_contrast".into(),
                label: Some("bw bc".into()),
                params: json!({ "brightness": 0.0, "contrast": 1.08 }),
            },
            RecipeStep {
                effect: "lumen-fx-sharpen.unsharp_mask".into(),
                label: Some("bw sharpen".into()),
                params: json!({ "amount": 0.5, "radius": 1.0, "threshold": 0.0 }),
            },
        ],
        "vintage" => vec![
            RecipeStep {
                effect: "lumen-fx-color.saturation".into(),
                label: Some("vintage desat".into()),
                params: json!({ "amount": 0.78 }),
            },
            RecipeStep {
                effect: "lumen-fx-exposure.gamma".into(),
                label: Some("vintage gamma".into()),
                params: json!({ "gamma": 1.08 }),
            },
            RecipeStep {
                effect: "lumen-fx-exposure.brightness_contrast".into(),
                label: Some("vintage bc".into()),
                params: json!({ "brightness": 0.02, "contrast": 0.92 }),
            },
            RecipeStep {
                effect: "lumen-fx-sharpen.unsharp_mask".into(),
                label: Some("vintage sharpen".into()),
                params: json!({ "amount": 0.3, "radius": 1.4, "threshold": 0.02 }),
            },
        ],
        "sharpen" => vec![
            RecipeStep {
                effect: "lumen-fx-deblur.laplacian".into(),
                label: Some("laplacian".into()),
                params: json!({ "amount": 0.8, "sigma": 0.8, "sigma_ratio": 1.6 }),
            },
            RecipeStep {
                effect: "lumen-fx-sharpen.unsharp_mask".into(),
                label: Some("unsharp".into()),
                params: json!({ "amount": 1.2, "radius": 0.9, "threshold": 0.0 }),
            },
        ],
        "restore" => vec![
            RecipeStep {
                effect: "lumen-fx-denoise.gaussian".into(),
                label: Some("restore denoise".into()),
                params: json!({ "sigma": 0.6 }),
            },
            RecipeStep {
                effect: "lumen-fx-text.clahe".into(),
                label: Some("restore CLAHE".into()),
                params: json!({ "tiles_x": 8, "tiles_y": 8, "clip_limit": 1.5 }),
            },
            RecipeStep {
                effect: "lumen-fx-color.saturation".into(),
                label: Some("restore sat".into()),
                params: json!({ "amount": 1.10 }),
            },
            RecipeStep {
                effect: "lumen-fx-sharpen.unsharp_mask".into(),
                label: Some("restore sharpen".into()),
                params: json!({ "amount": 0.5, "radius": 1.0, "threshold": 0.0 }),
            },
        ],
        other => {
            return Err(anyhow!(
                "unknown preset '{other}' — try one of: pop, bw, vintage, sharpen, restore"
            ));
        }
    };
    Ok(Recipe {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        chain,
    })
}

/// Names of every built-in preset, in display order.
#[allow(dead_code)] // re-exported indirectly via test; useful for future `lumen style --list`
pub const KNOWN_PRESETS: &[&str] = &["pop", "bw", "vintage", "sharpen", "restore"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_preset_builds_a_nonempty_recipe() {
        for name in KNOWN_PRESETS {
            let r = build_style_recipe(
                std::path::Path::new("/tmp/in.png"),
                std::path::Path::new("/tmp/out.png"),
                name,
            )
            .unwrap();
            assert!(
                !r.chain.is_empty(),
                "preset '{name}' produced an empty chain"
            );
        }
    }

    #[test]
    fn unknown_preset_errors() {
        let r = build_style_recipe(
            std::path::Path::new("/tmp/in.png"),
            std::path::Path::new("/tmp/out.png"),
            "nope",
        );
        assert!(r.is_err());
    }
}
