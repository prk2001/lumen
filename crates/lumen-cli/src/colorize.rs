//! `lumen colorize` — turn a grayscale or low-color image into color.
//!
//! Two modes:
//!
//! - **Heuristic** — channel_isolate(luma) → duotone with a chosen
//!   palette → optional CLAHE + sharpen. Works without any model.
//!   Five palettes: `night`, `day`, `sepia`, `cyan-orange`, `noir`.
//! - **ML** — load a colorization ONNX model via `lumen-ai` (the
//!   tract-onnx wrapper) and run it. Most published colorization
//!   models predict ab channels in Lab space from an L (luminance)
//!   input; this CLI handles the L → ab → RGB plumbing. See
//!   `docs/MODELS.md` for compatible models and download steps.
//!
//! When in doubt, run with `--print-recipe` to see what would be
//! applied without writing the output file.

use anyhow::{anyhow, Result};
use serde_json::json;

use crate::{Recipe, RecipeStep};

#[derive(Debug, Clone, Copy)]
struct Palette {
    shadow:    [f64; 3],
    highlight: [f64; 3],
    amount:    f64,
    clahe_clip:    f64,
    sharpen:       f64,
}

fn palette_for(name: &str) -> Result<Palette> {
    Ok(match name {
        "night" => Palette {
            shadow:    [0.04, 0.07, 0.20],
            highlight: [0.96, 0.78, 0.40],
            amount:    0.92,
            clahe_clip: 2.0,
            sharpen:    0.4,
        },
        "day" => Palette {
            shadow:    [0.18, 0.30, 0.46],
            highlight: [0.98, 0.92, 0.78],
            amount:    0.85,
            clahe_clip: 1.5,
            sharpen:    0.3,
        },
        "sepia" => Palette {
            shadow:    [0.18, 0.10, 0.05],
            highlight: [0.98, 0.86, 0.62],
            amount:    0.95,
            clahe_clip: 1.0,
            sharpen:    0.3,
        },
        "cyan-orange" => Palette {
            shadow:    [0.05, 0.30, 0.40],
            highlight: [0.98, 0.62, 0.18],
            amount:    0.88,
            clahe_clip: 1.5,
            sharpen:    0.4,
        },
        "noir" => Palette {
            shadow:    [0.02, 0.02, 0.05],
            highlight: [0.92, 0.92, 0.95],
            amount:    1.00,
            clahe_clip: 2.5,
            sharpen:    0.5,
        },
        other => {
            return Err(anyhow!(
                "unknown palette '{other}' — try one of: night, day, sepia, cyan-orange, noir"
            ));
        }
    })
}

pub fn build_heuristic_recipe(
    input: &std::path::Path,
    output: &std::path::Path,
    palette_name: &str,
) -> Result<Recipe> {
    let p = palette_for(palette_name)?;
    let chain = vec![
        RecipeStep {
            effect: "lumen-fx-modalities.channel_isolate".into(),
            label: Some("colorize prep".into()),
            params: json!({ "channel": "luma", "invert": false }),
        },
        RecipeStep {
            effect: "lumen-fx-color.duotone".into(),
            label: Some(format!("colorize {palette_name}")),
            params: json!({
                "shadow_r":    p.shadow[0],    "shadow_g":    p.shadow[1],    "shadow_b":    p.shadow[2],
                "highlight_r": p.highlight[0], "highlight_g": p.highlight[1], "highlight_b": p.highlight[2],
                "amount": p.amount,
            }),
        },
        RecipeStep {
            effect: "lumen-fx-text.clahe".into(),
            label: Some("colorize CLAHE".into()),
            params: json!({ "tiles_x": 6, "tiles_y": 6, "clip_limit": p.clahe_clip }),
        },
        RecipeStep {
            effect: "lumen-fx-sharpen.unsharp_mask".into(),
            label: Some("colorize sharpen".into()),
            params: json!({ "amount": p.sharpen, "radius": 1.0, "threshold": 0.0 }),
        },
    ];
    Ok(Recipe {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        chain,
    })
}

/// Names of every built-in palette, in display order.
#[allow(dead_code)] // re-exported for tests + future `lumen colorize --list`
pub const KNOWN_PALETTES: &[&str] = &["night", "day", "sepia", "cyan-orange", "noir"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_palette_builds() {
        for name in KNOWN_PALETTES {
            let r = build_heuristic_recipe(
                std::path::Path::new("/tmp/in.png"),
                std::path::Path::new("/tmp/out.png"),
                name,
            )
            .unwrap();
            assert_eq!(r.chain.len(), 4);
        }
    }

    #[test]
    fn unknown_palette_errors() {
        assert!(build_heuristic_recipe(
            std::path::Path::new("/tmp/in.png"),
            std::path::Path::new("/tmp/out.png"),
            "ultraviolet"
        )
        .is_err());
    }
}
