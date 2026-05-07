//! Polygon alpha mask with optional feathering.
//!
//! Inside the polygon, alpha is multiplied by `inside`. Outside, by
//! `outside`. A `feather` parameter expressed in pixels softens the
//! transition with a linear ramp around the polygon boundary.
//!
//! The polygon is described by a `points` string of the form
//! `"x1,y1; x2,y2; x3,y3; ..."` in pixel coordinates. The polygon is
//! implicitly closed (last vertex connects to first). Inside-ness is
//! determined by the standard even-odd / ray-casting test.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct AlphaPolygon;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-mask.alpha_polygon",
    display_name: "Alpha Polygon Mask",
    description: "Polygonal alpha mask with feathered edges.",
    category: Category::Mask,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "points",
        display_name: "Points",
        description: "Polygon vertices as 'x1,y1; x2,y2; ...' in pixels.",
        kind: ParamKind::String { default: "" },
    },
    ParamSpec {
        id: "inside",
        display_name: "Inside α",
        description: "Alpha multiplier inside the polygon (0..1).",
        kind: ParamKind::Float {
            default: 1.0,
            min: Some(0.0),
            max: Some(1.0),
        },
    },
    ParamSpec {
        id: "outside",
        display_name: "Outside α",
        description: "Alpha multiplier outside the polygon (0..1).",
        kind: ParamKind::Float {
            default: 0.0,
            min: Some(0.0),
            max: Some(1.0),
        },
    },
    ParamSpec {
        id: "feather",
        display_name: "Feather",
        description: "Linear-ramp width in pixels around the polygon edge.",
        kind: ParamKind::Float {
            default: 0.0,
            min: Some(0.0),
            max: Some(256.0),
        },
    },
];

impl Effect for AlphaPolygon {
    fn metadata(&self) -> &EffectMetadata {
        &META
    }
    fn parameters(&self) -> &[ParamSpec] {
        PARAMS
    }
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
        let points_str = params.get_string("points").unwrap_or("");
        let polygon = parse_points(points_str);
        let inside = params.get_float("inside").unwrap_or(1.0) as f32;
        let outside = params.get_float("outside").unwrap_or(0.0) as f32;
        let feather = params.get_float("feather").unwrap_or(0.0).max(0.0) as f32;

        // Empty polygon → treat the whole frame as inside (no-op when
        // inside == 1.0 / outside == 0.0 are the defaults).
        let empty = polygon.len() < 3;

        let mut frame = input.into_rgba_f32_linear();
        let w = frame.width as usize;
        let h = frame.height as usize;
        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");

        for py in 0..h {
            for px in 0..w {
                let cx = px as f32 + 0.5;
                let cy = py as f32 + 0.5;
                let m = if empty {
                    1.0
                } else {
                    mask_value(cx, cy, &polygon, feather)
                };
                let alpha_mul = outside + (inside - outside) * m;
                let off = (py * w + px) * 4;
                pixels[off + 3] = (pixels[off + 3] * alpha_mul).clamp(0.0, 1.0);
            }
        }
        Ok(frame)
    }
}

/// Parse `"x1,y1; x2,y2; ..."` → list of (x, y). Robust to whitespace.
/// Bad / empty input returns an empty list.
fn parse_points(s: &str) -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    if s.trim().is_empty() {
        return out;
    }
    for tok in s.split(';') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        let mut it = tok.split(',');
        let x = it.next().map(str::trim).and_then(|t| t.parse::<f32>().ok());
        let y = it.next().map(str::trim).and_then(|t| t.parse::<f32>().ok());
        // Reject if a third comma-separated component exists.
        if it.next().is_some() {
            return Vec::new();
        }
        match (x, y) {
            (Some(x), Some(y)) => out.push((x, y)),
            _ => return Vec::new(),
        }
    }
    out
}

/// 1.0 fully inside the polygon, 0.0 fully outside, linear ramp in
/// the feather band.
fn mask_value(cx: f32, cy: f32, poly: &[(f32, f32)], feather: f32) -> f32 {
    let inside = point_in_polygon(cx, cy, poly);
    if feather <= 0.0 {
        return if inside { 1.0 } else { 0.0 };
    }
    // Signed distance: positive inside, negative outside.
    let dist = if inside {
        distance_to_boundary(cx, cy, poly)
    } else {
        -distance_to_boundary(cx, cy, poly)
    };
    if dist >= feather {
        1.0
    } else if dist <= -feather {
        0.0
    } else {
        ((dist + feather) / (2.0 * feather)).clamp(0.0, 1.0)
    }
}

/// Even-odd ray-casting point-in-polygon test. Casts a horizontal ray
/// from `(cx, cy)` to +∞ and counts edge crossings.
fn point_in_polygon(cx: f32, cy: f32, poly: &[(f32, f32)]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        // Does the edge straddle the horizontal ray at y = cy?
        let straddles = (yi > cy) != (yj > cy);
        if straddles {
            // x-coordinate of edge intersection with y = cy.
            let x_isect = xi + (cy - yi) * (xj - xi) / (yj - yi);
            if cx < x_isect {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Unsigned distance from point to the closed polygon boundary
/// (minimum over all edges).
fn distance_to_boundary(cx: f32, cy: f32, poly: &[(f32, f32)]) -> f32 {
    let n = poly.len();
    let mut best = f32::INFINITY;
    let mut j = n - 1;
    for i in 0..n {
        let d = point_segment_distance(cx, cy, poly[j], poly[i]);
        if d < best {
            best = d;
        }
        j = i;
    }
    best
}

/// Distance from point `p` to segment `a`–`b`.
fn point_segment_distance(px: f32, py: f32, a: (f32, f32), b: (f32, f32)) -> f32 {
    let (ax, ay) = a;
    let (bx, by) = b;
    let abx = bx - ax;
    let aby = by - ay;
    let len2 = abx * abx + aby * aby;
    if len2 == 0.0 {
        let dx = px - ax;
        let dy = py - ay;
        return (dx * dx + dy * dy).sqrt();
    }
    let t = ((px - ax) * abx + (py - ay) * aby) / len2;
    let t = t.clamp(0.0, 1.0);
    let qx = ax + abx * t;
    let qy = ay + aby * t;
    let dx = px - qx;
    let dy = py - qy;
    (dx * dx + dy * dy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue, PixelData};

    fn solid_rgba8(w: u32, h: u32) -> Frame {
        Frame::new(
            w,
            h,
            PixelData::Rgba8(vec![255; (w * h * 4) as usize]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap()
    }

    fn alpha_at(data: &[u8], w: u32, x: u32, y: u32) -> u8 {
        data[((y * w + x) * 4 + 3) as usize]
    }

    #[test]
    fn empty_points_is_noop() {
        let m = AlphaPolygon;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        // Don't insert "points" — default is empty string.
        p.validate_and_fill(m.parameters()).unwrap();

        let f = solid_rgba8(4, 4);
        let out = m.apply(&mut ctx, f, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else {
            panic!()
        };
        for &a in px.iter().skip(3).step_by(4) {
            assert_eq!(a, 255, "expected unchanged alpha across the whole frame");
        }
    }

    #[test]
    fn triangle_upper_left_no_feather() {
        // Triangle covering the upper-left half of an 8x8 image:
        // (0,0) -> (8,0) -> (0,8). Interior pixel centers satisfy x+y<8.
        let m = AlphaPolygon;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("points", ParamValue::String("0,0; 8,0; 0,8".to_string()));
        p.insert("inside", ParamValue::Float(1.0));
        p.insert("outside", ParamValue::Float(0.0));
        p.insert("feather", ParamValue::Float(0.0));
        p.validate_and_fill(m.parameters()).unwrap();

        let f = solid_rgba8(8, 8);
        let out = m.apply(&mut ctx, f, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else {
            panic!()
        };

        for y in 0..8u32 {
            for x in 0..8u32 {
                let cx = x as f32 + 0.5;
                let cy = y as f32 + 0.5;
                let inside = cx + cy < 8.0;
                let a = alpha_at(&px, 8, x, y);
                if inside {
                    assert_eq!(a, 255, "expected 255 at ({x},{y})");
                } else {
                    assert_eq!(a, 0, "expected 0 at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn convex_quad_with_feather_ramps_at_edge() {
        // 16x16 image, square polygon centered: (4,4)–(12,4)–(12,12)–(4,12).
        // feather = 2 → ramp band 2px wide on each side of the edge.
        let m = AlphaPolygon;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert(
            "points",
            ParamValue::String("4,4; 12,4; 12,12; 4,12".to_string()),
        );
        p.insert("inside", ParamValue::Float(1.0));
        p.insert("outside", ParamValue::Float(0.0));
        p.insert("feather", ParamValue::Float(2.0));
        p.validate_and_fill(m.parameters()).unwrap();

        let f = solid_rgba8(16, 16);
        let out = m.apply(&mut ctx, f, &p).unwrap().into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = out.data else {
            panic!()
        };

        // Center pixel (8, 8) is well inside the polygon and beyond the
        // feather band → full alpha (255).
        assert_eq!(alpha_at(&px, 16, 8, 8), 255, "center should be full alpha");

        // A corner pixel (0, 0) is well outside the polygon and beyond
        // the feather band → zero alpha.
        assert_eq!(alpha_at(&px, 16, 0, 0), 0, "corner should be zero alpha");
        assert_eq!(alpha_at(&px, 16, 15, 15), 0);
        assert_eq!(alpha_at(&px, 16, 0, 15), 0);
        assert_eq!(alpha_at(&px, 16, 15, 0), 0);

        // An edge pixel just outside the polygon boundary (3, 8): center
        // at (3.5, 8.5), distance to nearest edge = 0.5 outside, so
        // signed dist = -0.5, t = (-0.5 + 2)/4 = 0.375 → alpha ≈ 0.375.
        let a = alpha_at(&px, 16, 3, 8);
        assert!(a > 0 && a < 255, "edge pixel should be partial, got {a}");
        // A pixel just inside the boundary (4, 8): center (4.5, 8.5),
        // distance to nearest edge = 0.5 inside, t = 0.625, alpha ≈ 0.625.
        let b = alpha_at(&px, 16, 4, 8);
        assert!(
            b > 0 && b < 255,
            "inside-edge pixel should be partial, got {b}"
        );
        assert!(b > a, "inside-edge alpha should exceed outside-edge alpha");
    }

    #[test]
    fn parse_points_robust_to_whitespace_and_bad_input() {
        assert_eq!(parse_points(""), Vec::<(f32, f32)>::new());
        assert_eq!(parse_points("   "), Vec::<(f32, f32)>::new());
        assert_eq!(parse_points("1,2"), vec![(1.0, 2.0)]);
        assert_eq!(
            parse_points(" 1 , 2 ; 3,4 ;5 , 6 "),
            vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        );
        // Trailing semicolons are tolerated.
        assert_eq!(parse_points("1,2;3,4;"), vec![(1.0, 2.0), (3.0, 4.0)]);
        // Bad input → empty.
        assert_eq!(parse_points("abc"), Vec::<(f32, f32)>::new());
        assert_eq!(parse_points("1,2,3"), Vec::<(f32, f32)>::new());
        assert_eq!(parse_points("1;2"), Vec::<(f32, f32)>::new());
    }
}
