//! 3D LUT effect — applies a Resolve / DaVinci-style `.cube` lookup table
//! to a frame using trilinear interpolation.
//!
//! The `.cube` text format we accept:
//!
//! - Comments start with `#`.
//! - A `LUT_3D_SIZE N` directive declares an N x N x N cube.
//! - Optional `DOMAIN_MIN r g b` / `DOMAIN_MAX r g b` directives bound
//!   the input range (default 0..1).
//! - Followed by N^3 rows of `r g b` floats. The fastest-varying axis
//!   is R, then G, then B (the cube convention).
//!
//! Behavior:
//!
//! - Empty `path` is a no-op pass-through.
//! - Strength of 0 returns the input untouched; strength of 1 returns
//!   the fully-graded LUT output; intermediate values are a linear blend.
//! - Parsed LUTs are cached keyed by `(path, mtime)` so repeated renders
//!   don't re-parse.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::SystemTime;

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Error, Frame, ParamKind, ParamSpec,
    ParamValues, Result,
};
use parking_lot::RwLock;
use tracing::instrument;

#[derive(Debug, Default)]
pub struct Lut3d;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-color.lut3d",
    display_name: "3D LUT",
    description: "Apply a Resolve-style .cube 3D LUT with trilinear interpolation.",
    category: Category::Color,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "path",
        display_name: "LUT Path",
        description: "Filesystem path to a .cube LUT file. Empty = pass-through.",
        kind: ParamKind::String { default: "" },
    },
    ParamSpec {
        id: "strength",
        display_name: "Strength",
        description: "Mix between input (0.0) and graded output (1.0).",
        kind: ParamKind::Float { default: 1.0, min: Some(0.0), max: Some(1.0) },
    },
];

/// A parsed `.cube` 3D LUT.
#[derive(Debug, Clone)]
pub struct ParsedLut {
    pub size: usize,
    pub domain_min: [f32; 3],
    pub domain_max: [f32; 3],
    /// Flat RGB samples, length = `size^3 * 3`.
    /// Indexing: `idx = ((b * size) + g) * size + r` for sample at (r, g, b).
    pub samples: Vec<f32>,
}

impl ParsedLut {
    /// Trilinearly sample the LUT at a normalized RGB coordinate.
    fn sample(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
        let n = self.size;
        if n < 2 {
            // Degenerate — single sample.
            return [self.samples[0], self.samples[1], self.samples[2]];
        }
        let span = (n - 1) as f32;

        // Normalize through declared domain.
        let nr = ((r - self.domain_min[0]) / (self.domain_max[0] - self.domain_min[0]))
            .clamp(0.0, 1.0);
        let ng = ((g - self.domain_min[1]) / (self.domain_max[1] - self.domain_min[1]))
            .clamp(0.0, 1.0);
        let nb = ((b - self.domain_min[2]) / (self.domain_max[2] - self.domain_min[2]))
            .clamp(0.0, 1.0);

        let fr = nr * span;
        let fg = ng * span;
        let fb = nb * span;

        let r0 = fr.floor() as usize;
        let g0 = fg.floor() as usize;
        let b0 = fb.floor() as usize;
        let r1 = (r0 + 1).min(n - 1);
        let g1 = (g0 + 1).min(n - 1);
        let b1 = (b0 + 1).min(n - 1);

        let dr = fr - r0 as f32;
        let dg = fg - g0 as f32;
        let db = fb - b0 as f32;

        // Sample the 8 cube corners and trilerp.
        let c000 = self.fetch(r0, g0, b0);
        let c100 = self.fetch(r1, g0, b0);
        let c010 = self.fetch(r0, g1, b0);
        let c110 = self.fetch(r1, g1, b0);
        let c001 = self.fetch(r0, g0, b1);
        let c101 = self.fetch(r1, g0, b1);
        let c011 = self.fetch(r0, g1, b1);
        let c111 = self.fetch(r1, g1, b1);

        let mut out = [0.0f32; 3];
        for i in 0..3 {
            let c00 = c000[i] * (1.0 - dr) + c100[i] * dr;
            let c10 = c010[i] * (1.0 - dr) + c110[i] * dr;
            let c01 = c001[i] * (1.0 - dr) + c101[i] * dr;
            let c11 = c011[i] * (1.0 - dr) + c111[i] * dr;
            let c0 = c00 * (1.0 - dg) + c10 * dg;
            let c1 = c01 * (1.0 - dg) + c11 * dg;
            out[i] = c0 * (1.0 - db) + c1 * db;
        }
        out
    }

    #[inline]
    fn fetch(&self, r: usize, g: usize, b: usize) -> [f32; 3] {
        let n = self.size;
        let idx = ((b * n) + g) * n + r;
        let base = idx * 3;
        [self.samples[base], self.samples[base + 1], self.samples[base + 2]]
    }
}

/// Parse a `.cube` text file into a [`ParsedLut`].
pub fn parse_cube(text: &str) -> Result<ParsedLut> {
    let mut size: Option<usize> = None;
    let mut domain_min = [0.0f32; 3];
    let mut domain_max = [1.0f32; 3];
    let mut samples: Vec<f32> = Vec::new();

    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut tok = line.split_whitespace();
        let head = tok.next().unwrap_or("");
        match head {
            "TITLE" | "LUT_1D_SIZE" => {
                // Ignored / unsupported — caller asked for a 3D LUT.
                if head == "LUT_1D_SIZE" {
                    return Err(Error::decode("expected 3D .cube LUT, got 1D"));
                }
            }
            "LUT_3D_SIZE" => {
                let n: usize = tok
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| Error::decode("LUT_3D_SIZE missing or invalid"))?;
                if n < 2 {
                    return Err(Error::decode("LUT_3D_SIZE must be >= 2"));
                }
                size = Some(n);
            }
            "DOMAIN_MIN" => {
                for slot in &mut domain_min {
                    *slot = tok
                        .next()
                        .and_then(|s| s.parse().ok())
                        .ok_or_else(|| Error::decode("DOMAIN_MIN expects 3 floats"))?;
                }
            }
            "DOMAIN_MAX" => {
                for slot in &mut domain_max {
                    *slot = tok
                        .next()
                        .and_then(|s| s.parse().ok())
                        .ok_or_else(|| Error::decode("DOMAIN_MAX expects 3 floats"))?;
                }
            }
            _ => {
                // Sample row: r g b
                let r: f32 = head
                    .parse()
                    .map_err(|_| Error::decode(format!("bad sample row: {raw:?}")))?;
                let g: f32 = tok
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| Error::decode(format!("bad sample row (G): {raw:?}")))?;
                let b: f32 = tok
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| Error::decode(format!("bad sample row (B): {raw:?}")))?;
                samples.push(r);
                samples.push(g);
                samples.push(b);
            }
        }
    }

    let size = size.ok_or_else(|| Error::decode("missing LUT_3D_SIZE directive"))?;
    let expected = size * size * size * 3;
    if samples.len() != expected {
        return Err(Error::decode(format!(
            "LUT sample count mismatch: expected {expected}, got {}",
            samples.len()
        )));
    }
    for axis in 0..3 {
        if domain_max[axis] <= domain_min[axis] {
            return Err(Error::decode("DOMAIN_MAX must exceed DOMAIN_MIN per channel"));
        }
    }
    Ok(ParsedLut { size, domain_min, domain_max, samples })
}

type CacheKey = PathBuf;
type CacheVal = (Option<SystemTime>, Arc<ParsedLut>);

fn cache() -> &'static RwLock<HashMap<CacheKey, CacheVal>> {
    static CELL: OnceLock<RwLock<HashMap<CacheKey, CacheVal>>> = OnceLock::new();
    CELL.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Load (or fetch from cache) a parsed LUT for the given path.
pub fn load_lut(path: &Path) -> Result<Arc<ParsedLut>> {
    let mtime = fs::metadata(path).ok().and_then(|m| m.modified().ok());

    {
        let r = cache().read();
        if let Some((cached_mtime, lut)) = r.get(path) {
            if *cached_mtime == mtime {
                return Ok(Arc::clone(lut));
            }
        }
    }

    let text = fs::read_to_string(path).map_err(|e| {
        Error::decode_at(path.to_path_buf(), format!("cannot read .cube file: {e}"))
    })?;
    let parsed = parse_cube(&text).map_err(|e| match e {
        Error::Decode { message, .. } => Error::decode_at(path.to_path_buf(), message),
        other => other,
    })?;
    let arc = Arc::new(parsed);

    {
        let mut w = cache().write();
        w.insert(path.to_path_buf(), (mtime, Arc::clone(&arc)));
    }
    Ok(arc)
}

impl Effect for Lut3d {
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
        let path = params.get_string("path").unwrap_or("").to_string();
        let strength = params.get_float("strength").unwrap_or(1.0) as f32;

        if path.is_empty() || strength <= 0.0 {
            return Ok(input);
        }

        let lut = load_lut(Path::new(&path))?;
        let s = strength.clamp(0.0, 1.0);

        let mut frame = input.into_rgba_f32_linear();
        let pixels = frame.as_f32_mut().expect("RgbaF32 after lift");
        for px in pixels.chunks_exact_mut(4) {
            let graded = lut.sample(px[0], px[1], px[2]);
            px[0] = px[0] * (1.0 - s) + graded[0] * s;
            px[1] = px[1] * (1.0 - s) + graded[1] * s;
            px[2] = px[2] * (1.0 - s) + graded[2] * s;
        }
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue, PixelData};
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Generate a `.cube` text body for an identity LUT of the requested
    /// size, where every sample equals its grid coordinate.
    fn identity_cube(size: usize) -> String {
        let mut s = format!("LUT_3D_SIZE {size}\n");
        let span = (size - 1) as f32;
        // R fastest, then G, then B.
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    let _ = std::fmt::Write::write_fmt(
                        &mut s,
                        format_args!(
                            "{:.6} {:.6} {:.6}\n",
                            r as f32 / span,
                            g as f32 / span,
                            b as f32 / span
                        ),
                    );
                }
            }
        }
        s
    }

    fn write_temp(text: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(text.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn identity_lut_round_trips() {
        let cube = identity_cube(2);
        let file = write_temp(&cube);

        let fx = Lut3d;
        let mut p = ParamValues::new();
        p.insert("path", ParamValue::String(file.path().to_string_lossy().to_string()));
        p.validate_and_fill(fx.parameters()).unwrap();

        let pixels: Vec<f32> = vec![
            0.10, 0.30, 0.70, 1.0,
            0.42, 0.58, 0.92, 1.0,
            0.00, 0.50, 1.00, 1.0,
            0.81, 0.13, 0.27, 1.0,
        ];
        let frame = Frame::new(
            2,
            2,
            PixelData::RgbaF32(pixels.clone()),
            ColorSpace::LinearSRgb,
            None,
        )
        .unwrap();

        let mut ctx = Context::for_still_srgb();
        let out = fx.apply(&mut ctx, frame, &p).unwrap();
        let out_px = out.as_f32().unwrap();
        for (a, b) in pixels.iter().zip(out_px.iter()) {
            assert!((a - b).abs() < 1e-5, "identity drift: {a} vs {b}");
        }
    }

    #[test]
    fn empty_path_is_passthrough() {
        let fx = Lut3d;
        let mut p = ParamValues::new();
        p.validate_and_fill(fx.parameters()).unwrap();

        let frame = Frame::new(
            1,
            1,
            PixelData::Rgba8(vec![10, 20, 30, 255]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();

        let mut ctx = Context::for_still_srgb();
        let out = fx.apply(&mut ctx, frame.clone(), &p).unwrap();
        // Path is empty -> input passes through untouched (still Rgba8).
        assert_eq!(out.data, frame.data);
    }

    #[test]
    fn bad_format_returns_err() {
        // Has LUT_3D_SIZE but garbage rows.
        let bad = "LUT_3D_SIZE 2\n0.0 0.0 0.0\nnot a row\n";
        let file = write_temp(bad);

        let fx = Lut3d;
        let mut p = ParamValues::new();
        p.insert("path", ParamValue::String(file.path().to_string_lossy().to_string()));
        p.validate_and_fill(fx.parameters()).unwrap();

        let frame = Frame::new(
            1,
            1,
            PixelData::Rgba8(vec![0, 0, 0, 255]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();

        let mut ctx = Context::for_still_srgb();
        let r = fx.apply(&mut ctx, frame, &p);
        assert!(r.is_err(), "expected parse error");
    }
}
