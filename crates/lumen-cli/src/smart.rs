//! Smart Auto — one-shot intelligent enhancement.
//!
//! Analyzes the input frame, scores how "degraded" it looks, and picks
//! the right preset:
//!
//! - **Heavily degraded** (low edges + low contrast + low chroma) →
//!   `clarify --strength aggressive`.
//! - **Moderately degraded** → `clarify --strength standard`.
//! - **Otherwise** → `auto-enhance`.
//!
//! Mirrors `pickSmartStrategy` in `apps/web/demo.js`.

use anyhow::Result;

use crate::{auto, clarify, Recipe};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    AutoEnhance,
    ClarifyStandard,
    ClarifyAggressive,
}

impl Strategy {
    pub fn label(self) -> &'static str {
        match self {
            Strategy::AutoEnhance       => "auto-enhance",
            Strategy::ClarifyStandard   => "clarify (standard)",
            Strategy::ClarifyAggressive => "clarify (aggressive)",
        }
    }
}

pub fn pick_strategy(stats: &auto::Stats) -> Strategy {
    let mut score = 0;
    if stats.edge_mean   < 0.06 { score += 1; }
    if stats.edge_mean   < 0.03 { score += 1; }
    if (stats.p99 - stats.p01) < 0.50 { score += 1; }
    if (stats.p99 - stats.p01) < 0.30 { score += 1; }
    if stats.chroma_mean < 0.10 { score += 1; }
    if stats.chroma_mean < 0.05 { score += 1; }
    if score >= 4 { Strategy::ClarifyAggressive }
    else if score >= 2 { Strategy::ClarifyStandard }
    else { Strategy::AutoEnhance }
}

pub fn build_smart_recipe(
    input: &std::path::Path,
    output: &std::path::Path,
    upscale: bool,
) -> Result<(auto::Stats, Strategy, Recipe)> {
    let frame = lumen_io::decode_image(input)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    let stats = auto::analyze_frame(&frame);
    let strategy = pick_strategy(&stats);
    let recipe = match strategy {
        Strategy::AutoEnhance => Recipe {
            input: input.to_path_buf(),
            output: output.to_path_buf(),
            chain: auto::build_auto_chain(&stats),
        },
        Strategy::ClarifyStandard => {
            clarify::build_clarify_recipe(input, output, "standard", upscale)?
        }
        Strategy::ClarifyAggressive => {
            clarify::build_clarify_recipe(input, output, "aggressive", upscale)?
        }
    };
    Ok((stats, strategy, recipe))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats_with(edge: f64, chroma: f64, range: f64) -> auto::Stats {
        auto::Stats {
            p01: 0.5 - range / 2.0,
            p50: 0.5,
            p99: 0.5 + range / 2.0,
            chroma_mean: chroma,
            edge_mean: edge,
            luminance_mean: 0.5,
        }
    }

    #[test]
    fn clean_photo_picks_auto_enhance() {
        let s = stats_with(0.15, 0.25, 0.85);
        assert_eq!(pick_strategy(&s), Strategy::AutoEnhance);
    }

    #[test]
    fn moderately_degraded_picks_standard_clarify() {
        // edges < 0.06 (1) + chroma < 0.10 (1) = 2 -> ClarifyStandard
        let s = stats_with(0.05, 0.08, 0.65);
        assert_eq!(pick_strategy(&s), Strategy::ClarifyStandard);
    }

    #[test]
    fn heavily_degraded_picks_aggressive_clarify() {
        // all six markers fire -> 6 -> ClarifyAggressive
        let s = stats_with(0.02, 0.04, 0.20);
        assert_eq!(pick_strategy(&s), Strategy::ClarifyAggressive);
    }
}
