//! Curves effect — per-channel piecewise-linear remap, optionally
//! followed by a luma-preserving master curve.
//!
//! Each curve is given as a comma-separated list of `x:y` control points
//! (sorted ascending by x). Empty string means "identity" for that
//! channel.
//!
//! When `points_luma` is non-empty, after the per-channel pass we
//! recompute the Rec.709 luma `Y`, look up `Y'` on the luma curve, and
//! scale all three RGB channels by `Y' / max(Y, eps)` so hue is
//! preserved.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Error, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct Curves;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-color.curves",
    display_name: "Curves",
    description: "Per-channel piecewise-linear curves with luma-preserving master.",
    category: Category::Color,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "points_r",
        display_name: "Red Points",
        description: "Comma-separated x:y pairs for the red channel; empty = identity.",
        kind: ParamKind::String { default: "" },
    },
    ParamSpec {
        id: "points_g",
        display_name: "Green Points",
        description: "Comma-separated x:y pairs for the green channel; empty = identity.",
        kind: ParamKind::String { default: "" },
    },
    ParamSpec {
        id: "points_b",
        display_name: "Blue Points",
        description: "Comma-separated x:y pairs for the blue channel; empty = identity.",
        kind: ParamKind::String { default: "" },
    },
    ParamSpec {
        id: "points_luma",
        display_name: "Luma Points",
        description: "Comma-separated x:y pairs applied to Rec.709 luma; empty = no master curve.",
        kind: ParamKind::String { default: "" },
    },
];

const LUMA_EPS: f32 = 1e-6;

/// A monotonic-x piecewise-linear curve. Empty list = identity.
#[derive(Debug, Clone, Default)]
pub struct PiecewiseCurve {
    pub points: Vec<(f32, f32)>,
}

impl PiecewiseCurve {
    pub fn parse(text: &str) -> Result<Self> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(Self::default());
        }
        let mut points: Vec<(f32, f32)> = Vec::new();
        for raw in trimmed.split(',') {
            let pair = raw.trim();
            if pair.is_empty() {
                continue;
            }
            let mut split = pair.split(':');
            let x_str = split.next().ok_or_else(|| Error::InvalidParameter {
                name: "curve".into(),
                reason: format!("missing x in {pair:?}"),
            })?;
            let y_str = split.next().ok_or_else(|| Error::InvalidParameter {
                name: "curve".into(),
                reason: format!("missing y in {pair:?}"),
            })?;
            if split.next().is_some() {
                return Err(Error::InvalidParameter {
                    name: "curve".into(),
                    reason: format!("expected x:y, got {pair:?}"),
                });
            }
            let x: f32 = x_str.trim().parse().map_err(|_| Error::InvalidParameter {
                name: "curve".into(),
                reason: format!("bad x in {pair:?}"),
            })?;
            let y: f32 = y_str.trim().parse().map_err(|_| Error::InvalidParameter {
                name: "curve".into(),
                reason: format!("bad y in {pair:?}"),
            })?;
            points.push((x, y));
        }
        for w in points.windows(2) {
            if w[1].0 < w[0].0 {
                return Err(Error::InvalidParameter {
                    name: "curve".into(),
                    reason: "control points must be sorted ascending by x".into(),
                });
            }
        }
        Ok(Self { points })
    }

    pub fn is_identity(&self) -> bool { self.points.is_empty() }

    /// Sample the curve at `x`. Identity if no points; clamps to the
    /// endpoints outside the declared x-range.
    pub fn sample(&self, x: f32) -> f32 {
        if self.points.is_empty() {
            return x;
        }
        if x <= self.points[0].0 {
            return self.points[0].1;
        }
        if x >= self.points[self.points.len() - 1].0 {
            return self.points[self.points.len() - 1].1;
        }
        // Binary search for the upper bracket.
        let pts = &self.points;
        let mut lo = 0usize;
        let mut hi = pts.len() - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if pts[mid].0 <= x { lo = mid } else { hi = mid }
        }
        let (x0, y0) = pts[lo];
        let (x1, y1) = pts[hi];
        let dx = x1 - x0;
        if dx.abs() < f32::EPSILON {
            return y0;
        }
        let t = (x - x0) / dx;
        y0 + t * (y1 - y0)
    }
}

#[inline]
fn rec709_luma(r: f32, g: f32, b: f32) -> f32 {
    0.212_6 * r + 0.715_2 * g + 0.072_2 * b
}

impl Effect for Curves {
    fn metadata(&self) -> &EffectMetadata { &META }
    fn parameters(&self) -> &[ParamSpec] { PARAMS }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            deterministic: true,
            gpu: false,
            streamable: true,
            temporal: false,
        }
    }

    #[instrument(skip_all, fields(effect = META.id))]
    fn apply(&self, _ctx: &mut Context, input: Frame, params: &ParamValues) -> Result<Frame> {
        let cr = PiecewiseCurve::parse(params.get_string("points_r").unwrap_or(""))?;
        let cg = PiecewiseCurve::parse(params.get_string("points_g").unwrap_or(""))?;
        let cb = PiecewiseCurve::parse(params.get_string("points_b").unwrap_or(""))?;
        let cl = PiecewiseCurve::parse(params.get_string("points_luma").unwrap_or(""))?;

        let any_per_channel = !(cr.is_identity() && cg.is_identity() && cb.is_identity());
        if !any_per_channel && cl.is_identity() {
            return Ok(input);
        }

        let mut frame = input.into_rgba_f32_linear();
        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");

        for px in pixels.chunks_exact_mut(4) {
            if any_per_channel {
                px[0] = cr.sample(px[0]).clamp(0.0, 1.0);
                px[1] = cg.sample(px[1]).clamp(0.0, 1.0);
                px[2] = cb.sample(px[2]).clamp(0.0, 1.0);
            }
            if !cl.is_identity() {
                let y = rec709_luma(px[0], px[1], px[2]);
                let y_new = cl.sample(y);
                let scale = y_new / y.max(LUMA_EPS);
                px[0] = (px[0] * scale).clamp(0.0, 1.0);
                px[1] = (px[1] * scale).clamp(0.0, 1.0);
                px[2] = (px[2] * scale).clamp(0.0, 1.0);
            }
        }
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue, PixelData};

    fn frame_f32(pixels: Vec<f32>) -> Frame {
        let n = pixels.len() / 4;
        Frame::new(n as u32, 1, PixelData::RgbaF32(pixels), ColorSpace::LinearSRgb, None)
            .unwrap()
    }

    #[test]
    fn empty_points_passthrough() {
        let fx = Curves;
        let mut p = ParamValues::new();
        p.validate_and_fill(fx.parameters()).unwrap();

        let pixels = vec![0.10, 0.40, 0.80, 1.0];
        let frame = frame_f32(pixels.clone());
        let mut ctx = Context::for_still_srgb();
        let out = fx.apply(&mut ctx, frame, &p).unwrap();
        // No curves -> input is returned untouched (Rgba8 input would
        // also pass through; here we used f32 to keep the comparison
        // exact).
        let out_px = out.as_f32().unwrap();
        for (a, b) in pixels.iter().zip(out_px.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn mid_gray_pull_darkens_midtones() {
        let fx = Curves;
        let mut p = ParamValues::new();
        // Identity at endpoints, but pull 0.5 -> 0.3 on red channel.
        p.insert(
            "points_r",
            ParamValue::String("0:0, 0.5:0.3, 1:1".to_string()),
        );
        p.validate_and_fill(fx.parameters()).unwrap();

        let frame = frame_f32(vec![0.5, 0.5, 0.5, 1.0]);
        let mut ctx = Context::for_still_srgb();
        let out = fx.apply(&mut ctx, frame, &p).unwrap();
        let out_px = out.as_f32().unwrap();
        assert!(
            (out_px[0] - 0.3).abs() < 1e-5,
            "expected ~0.3 from curve, got {}",
            out_px[0]
        );
        // Other channels unaffected.
        assert!((out_px[1] - 0.5).abs() < 1e-6);
        assert!((out_px[2] - 0.5).abs() < 1e-6);
    }
}
