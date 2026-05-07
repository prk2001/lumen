//! HEIF / HEIC decode via [`libheif-rs`].
//!
//! Gated behind the `heif` Cargo feature. When the feature is **off**
//! (the default), both [`decode_heif`] and [`probe_heif`] still exist
//! as functions but return [`Error::UnsupportedFormat`] suggesting the
//! caller rebuild with `--features heif`. This keeps the call sites in
//! [`crate::probe`] and [`crate::still`] uniform — the fallback chain
//! doesn't have to be cfg-gated.
//!
//! When the feature is **on**, `libheif-rs` requires `libheif` to be
//! installed on the host (Homebrew: `brew install libheif`). On
//! decode we ask `libheif` for an interleaved RGBA plane and copy it
//! into a [`Frame`].

use std::path::Path;

use lumen_core::{AssetMetadata, Error, Frame, Result};
use tracing::instrument;

#[cfg(feature = "heif")]
mod imp {
    use super::*;
    use libheif_rs::{ColorSpace as HeifColorSpace, HeifContext, LibHeif, RgbChroma};
    use lumen_core::{ColorSpace, Frame, PixelData};

    fn path_to_str(p: &Path) -> Result<&str> {
        p.to_str().ok_or_else(|| {
            Error::decode_at(p.to_path_buf(), "non-UTF-8 path".to_string())
        })
    }

    pub(super) fn decode(path: &Path) -> Result<Frame> {
        let lib = LibHeif::new();
        let p = path_to_str(path)?;
        let ctx = HeifContext::read_from_file(p).map_err(|e| {
            Error::decode_at(path.to_path_buf(), format!("libheif read: {e}"))
        })?;
        let handle = ctx.primary_image_handle().map_err(|e| {
            Error::decode_at(path.to_path_buf(), format!("primary handle: {e}"))
        })?;

        let img = lib
            .decode(&handle, HeifColorSpace::Rgb(RgbChroma::Rgba), None)
            .map_err(|e| {
                Error::decode_at(path.to_path_buf(), format!("heif decode: {e}"))
            })?;

        let w = img.width();
        let h = img.height();
        let planes = img.planes();
        let interleaved = planes.interleaved.ok_or_else(|| {
            Error::decode_at(
                path.to_path_buf(),
                "libheif returned no interleaved plane".to_string(),
            )
        })?;

        // libheif rows may be wider than `width * 4` (stride padding).
        // Repack into a tight `width * 4 * height` buffer.
        let stride = interleaved.stride;
        let row_bytes = (w as usize) * 4;
        let mut out = Vec::with_capacity(row_bytes * (h as usize));
        for y in 0..(h as usize) {
            let start = y * stride;
            let end = start + row_bytes;
            if end > interleaved.data.len() {
                return Err(Error::decode_at(
                    path.to_path_buf(),
                    format!(
                        "heif row {y}: end {end} > buffer {}",
                        interleaved.data.len()
                    ),
                ));
            }
            out.extend_from_slice(&interleaved.data[start..end]);
        }

        Frame::new(w, h, PixelData::Rgba8(out), ColorSpace::SRgb, None)
    }

    pub(super) fn probe(path: &Path) -> Result<AssetMetadata> {
        let p = path_to_str(path)?;
        let ctx = HeifContext::read_from_file(p).map_err(|e| {
            Error::decode_at(path.to_path_buf(), format!("libheif read: {e}"))
        })?;
        let handle = ctx.primary_image_handle().map_err(|e| {
            Error::decode_at(path.to_path_buf(), format!("primary handle: {e}"))
        })?;
        let bit_depth = handle.luma_bits_per_pixel().max(8);
        let container = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());
        Ok(AssetMetadata {
            width: handle.width(),
            height: handle.height(),
            frame_count: Some(1),
            frame_rate: None,
            duration_secs: None,
            codec: Some("heif".to_string()),
            container,
            bit_depth,
            channels: if handle.has_alpha_channel() { 4 } else { 3 },
            color_space: Some(ColorSpace::SRgb),
            audio_sample_rate: None,
            audio_channels: None,
        })
    }
}

#[cfg(not(feature = "heif"))]
fn unavailable<P: AsRef<Path>>(path: P, what: &str) -> Error {
    Error::UnsupportedFormat(format!(
        "{}: HEIF/HEIC decoding is not built in (rebuild with --features heif): {}",
        what,
        path.as_ref().display()
    ))
}

/// Decode a HEIF/HEIC file into an RGBA8 / sRGB [`Frame`].
///
/// Without the `heif` feature this returns
/// [`Error::UnsupportedFormat`] — see module docs.
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub fn decode_heif<P: AsRef<Path>>(path: P) -> Result<Frame> {
    #[cfg(feature = "heif")]
    {
        imp::decode(path.as_ref())
    }
    #[cfg(not(feature = "heif"))]
    {
        Err(unavailable(path, "decode_heif"))
    }
}

/// Probe a HEIF/HEIC file for metadata.
///
/// Without the `heif` feature this returns
/// [`Error::UnsupportedFormat`] — see module docs.
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub fn probe_heif<P: AsRef<Path>>(path: P) -> Result<AssetMetadata> {
    #[cfg(feature = "heif")]
    {
        imp::probe(path.as_ref())
    }
    #[cfg(not(feature = "heif"))]
    {
        Err(unavailable(path, "probe_heif"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "heif"))]
    #[test]
    fn heif_off_returns_unsupported() {
        let r = decode_heif("/some/missing.heic");
        match r {
            Err(Error::UnsupportedFormat(msg)) => {
                assert!(
                    msg.contains("rebuild with --features heif"),
                    "expected feature-off hint, got: {msg}"
                );
            }
            other => panic!("expected UnsupportedFormat, got {other:?}"),
        }

        let r = probe_heif("/some/missing.heic");
        assert!(matches!(r, Err(Error::UnsupportedFormat(_))));
    }
}
