//! Camera RAW decode via [`rawloader`] (pure Rust).
//!
//! `rawloader` ships with built-in support for the major still-camera
//! RAW containers — Canon **CR2/CRW**, Nikon **NEF/NRW**, Sony **ARW**,
//! Adobe **DNG**, Fuji **RAF**, Olympus **ORF**, Panasonic **RW2**,
//! Pentax **PEF**, Samsung **SRW**, Sigma **X3F**, Hasselblad **3FR**,
//! Phase One **IIQ**, etc.
//!
//! What `rawloader` returns is *not* an RGB image: it's the raw sensor
//! buffer (a `Vec<u16>`) plus a CFA pattern (Bayer / X-Trans). To hand
//! the rest of Lumen something usable we run a deliberately minimal
//! pipeline:
//!
//! 1. Subtract per-channel black levels.
//! 2. Normalize to `[0.0, 1.0]` against the white level.
//! 3. Bilinear-interpolate the missing two channels at every pixel
//!    (a **basic** demosaic — fast, no edge-aware logic, good enough
//!    for thumbnails and "did this file decode?" smoke tests).
//! 4. Apply an sRGB transfer curve and pack to `u8` RGBA.
//!
//! This is intentionally *not* a finishing pipeline — there's no white
//! balance, no color matrix, no tone mapping. The output is a usable
//! preview suitable for round-tripping through Lumen's still-image
//! pipeline; serious RAW workflows will want a real RAW developer down
//! the line.
//!
//! For a lone-pixel "monochrome" file (i.e. `cpp == 1` but no valid
//! CFA) we just gray-replicate after black/white normalization.
//!
//! For a `cpp == 3` file (rare — some DNGs are pre-debayered) we skip
//! the demosaic and just pack the channels directly.

use std::path::Path;

use lumen_core::{AssetMetadata, ColorSpace, Error, Frame, PixelData, Result};
use rawloader::{CFA, RawImage, RawImageData};
use tracing::{debug, instrument};

/// Decode a camera-RAW file into a [`Frame`] (RGBA8 / sRGB).
///
/// The returned frame is a basic-demosaic preview: black/white
/// normalized, bilinear-interpolated where the CFA had a single
/// channel per pixel, sRGB-encoded. No white balance or color matrix
/// is applied. See module docs.
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub fn decode_raw<P: AsRef<Path>>(path: P) -> Result<Frame> {
    let path = path.as_ref();
    let raw = rawloader::decode_file(path).map_err(|e| {
        Error::decode_at(path.to_path_buf(), format!("rawloader: {e}"))
    })?;

    let (w, h) = (raw.width as u32, raw.height as u32);
    debug!(
        width = w,
        height = h,
        cpp = raw.cpp,
        make = %raw.make,
        model = %raw.model,
        "decoded RAW"
    );

    let rgba = match raw.cpp {
        1 => debayer_to_rgba8(&raw),
        3 => packed_rgb_to_rgba8(&raw),
        n => {
            return Err(Error::decode_at(
                path.to_path_buf(),
                format!("unexpected RAW cpp={n}"),
            ));
        }
    };

    Frame::new(w, h, PixelData::Rgba8(rgba), ColorSpace::SRgb, None)
}

/// Header-style probe for a RAW file. We *do* fully decode (rawloader
/// has no header-only API), but the cost is amortized — RAW files in a
/// project are typically probed once at import.
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub fn probe_raw<P: AsRef<Path>>(path: P) -> Result<AssetMetadata> {
    let path = path.as_ref();
    let raw = rawloader::decode_file(path).map_err(|e| {
        Error::decode_at(path.to_path_buf(), format!("rawloader: {e}"))
    })?;

    let container = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());
    let codec = Some(format!(
        "raw:{}/{}",
        raw.clean_make.trim(),
        raw.clean_model.trim()
    ));

    Ok(AssetMetadata {
        width: raw.width as u32,
        height: raw.height as u32,
        frame_count: Some(1),
        frame_rate: None,
        duration_secs: None,
        codec,
        container,
        // RAW sensors are typically 12–14 bit; rawloader normalizes to
        // u16 so 16 is the right bus width to advertise here.
        bit_depth: 16,
        channels: 4,
        color_space: Some(ColorSpace::SRgb),
        audio_sample_rate: None,
        audio_channels: None,
    })
}

// ---------------------------------------------------------------------
// Demosaic / packing helpers
// ---------------------------------------------------------------------

/// Pull u16 sensor values out of `RawImageData`, normalizing Float
/// payloads to the same range. Returns one `u16` per `cpp`-component.
fn raw_as_u16(raw: &RawImage) -> Vec<u16> {
    match &raw.data {
        RawImageData::Integer(v) => v.clone(),
        RawImageData::Float(v) => v
            .iter()
            .map(|f| (f.clamp(0.0, 1.0) * u16::MAX as f32) as u16)
            .collect(),
    }
}

/// Linear -> sRGB transfer.
#[inline]
fn linear_to_srgb_u8(x: f32) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let y = if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (y * 255.0 + 0.5) as u8
}

/// `cpp == 3` path: the file already has packed RGB, just sRGB-encode
/// after black/white normalization.
fn packed_rgb_to_rgba8(raw: &RawImage) -> Vec<u8> {
    let w = raw.width;
    let h = raw.height;
    let buf = raw_as_u16(raw);
    let black = raw.blacklevels[0] as f32;
    let white = (raw.whitelevels[0] as f32 - black).max(1.0);

    let mut out = Vec::with_capacity(w * h * 4);
    for px in buf.chunks_exact(3) {
        let r = ((px[0] as f32 - black) / white).clamp(0.0, 1.0);
        let g = ((px[1] as f32 - black) / white).clamp(0.0, 1.0);
        let b = ((px[2] as f32 - black) / white).clamp(0.0, 1.0);
        out.push(linear_to_srgb_u8(r));
        out.push(linear_to_srgb_u8(g));
        out.push(linear_to_srgb_u8(b));
        out.push(255);
    }
    out
}

/// `cpp == 1` path: bayer/X-trans CFA. Bilinear-interpolate the missing
/// two channels per pixel and convert to sRGB RGBA8.
fn debayer_to_rgba8(raw: &RawImage) -> Vec<u8> {
    let w = raw.width;
    let h = raw.height;
    let buf = raw_as_u16(raw);
    let cfa: &CFA = &raw.cfa;

    // Per-channel black/white normalization — RGBE order from rawloader.
    // We only use R(0), G(1), B(2); E(3) is folded into G if it shows up.
    let black = [
        raw.blacklevels[0] as f32,
        raw.blacklevels[1] as f32,
        raw.blacklevels[2] as f32,
        raw.blacklevels[3] as f32,
    ];
    let white = [
        (raw.whitelevels[0] as f32 - black[0]).max(1.0),
        (raw.whitelevels[1] as f32 - black[1]).max(1.0),
        (raw.whitelevels[2] as f32 - black[2]).max(1.0),
        (raw.whitelevels[3] as f32 - black[3]).max(1.0),
    ];

    // Monochrome short-circuit: cpp=1 + invalid CFA == genuine mono
    // sensor; just gray-replicate.
    if !cfa.is_valid() {
        let mut out = Vec::with_capacity(w * h * 4);
        for &v in &buf {
            let n = ((v as f32 - black[0]) / white[0]).clamp(0.0, 1.0);
            let g8 = linear_to_srgb_u8(n);
            out.push(g8);
            out.push(g8);
            out.push(g8);
            out.push(255);
        }
        return out;
    }

    // Map CFA color index (0..=3 from rawloader's RGBE convention) onto
    // an output channel: 0=R, 1=G, 2=B, 3 (E) -> green for the
    // bilinear average.
    #[inline]
    fn cfa_to_rgb(c: usize) -> usize {
        match c {
            0 => 0, // R
            2 => 2, // B
            _ => 1, // G or E -> G
        }
    }

    // Per-pixel normalized linear value, regardless of which channel
    // the CFA assigns to it.
    let normalize = |row: usize, col: usize| -> f32 {
        let raw_v = buf[row * w + col] as f32;
        let c = cfa.color_at(row, col);
        ((raw_v - black[c]) / white[c]).clamp(0.0, 1.0)
    };

    let mut out = vec![0u8; w * h * 4];

    for row in 0..h {
        for col in 0..w {
            // Channel value present at this pixel.
            let here_c = cfa_to_rgb(cfa.color_at(row, col));
            let here_v = normalize(row, col);

            // Average of each non-here channel from the immediate
            // neighborhood. We collect one bin per RGB channel and
            // count contributions for division.
            let mut sums = [0.0f32; 3];
            let mut counts = [0u32; 3];
            sums[here_c] = here_v;
            counts[here_c] = 1;

            // 3x3 neighborhood (skip the center, it's `here`).
            for dr in -1i32..=1 {
                for dc in -1i32..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let r = row as i32 + dr;
                    let c = col as i32 + dc;
                    if r < 0 || c < 0 || r >= h as i32 || c >= w as i32 {
                        continue;
                    }
                    let r = r as usize;
                    let c = c as usize;
                    let nb_c = cfa_to_rgb(cfa.color_at(r, c));
                    sums[nb_c] += normalize(r, c);
                    counts[nb_c] += 1;
                }
            }

            let mut rgb = [0.0f32; 3];
            for ch in 0..3 {
                rgb[ch] = if counts[ch] == 0 {
                    // Fallback: should be rare on Bayer/X-Trans
                    // patterns, but if a 3x3 happened to miss a
                    // channel just reuse `here`.
                    here_v
                } else {
                    sums[ch] / counts[ch] as f32
                };
            }

            let idx = (row * w + col) * 4;
            out[idx] = linear_to_srgb_u8(rgb[0]);
            out[idx + 1] = linear_to_srgb_u8(rgb[1]);
            out[idx + 2] = linear_to_srgb_u8(rgb[2]);
            out[idx + 3] = 255;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Garbage bytes -> rawloader rejects -> we surface a Decode error.
    #[test]
    fn decode_raw_rejects_non_raw_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-a-raw.cr2");
        std::fs::write(&path, b"this is definitely not a CR2 file").unwrap();
        let r = decode_raw(&path);
        assert!(matches!(r, Err(Error::Decode { .. })));
    }

    /// Same path through `probe_raw`.
    #[test]
    fn probe_raw_rejects_non_raw_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-a-raw.dng");
        std::fs::write(&path, b"\x00\x01\x02\x03").unwrap();
        let r = probe_raw(&path);
        assert!(matches!(r, Err(Error::Decode { .. })));
    }

    /// linear_to_srgb_u8 honors the boundary conditions we rely on
    /// elsewhere: 0.0 -> 0, 1.0 -> 255, monotonic in between.
    #[test]
    fn srgb_transfer_endpoints() {
        assert_eq!(linear_to_srgb_u8(0.0), 0);
        assert_eq!(linear_to_srgb_u8(1.0), 255);
        assert!(linear_to_srgb_u8(0.5) > linear_to_srgb_u8(0.25));
    }
}
