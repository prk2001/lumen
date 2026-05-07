//! JPEG XL decode via [`jpegxl-rs`].
//!
//! Gated behind the `jxl` Cargo feature. When the feature is **off**
//! (the default), [`decode_jxl`] and [`probe_jxl`] still exist as
//! functions but return [`Error::UnsupportedFormat`] suggesting the
//! caller rebuild with `--features jxl`. This keeps the fallback chain
//! in [`crate::probe`] uniform.
//!
//! When the feature is **on**, `jpegxl-rs` requires `libjxl` to be
//! installed on the host (Homebrew: `brew install jpeg-xl`).
//!
//! Note: `jpegxl-rs` is GPL-3.0-or-later. Lumen is Apache-2.0; only
//! enable the `jxl` feature for builds that are GPL-compatible (or
//! distributed without binary form). The feature is off by default
//! for exactly this reason.

use std::path::Path;

use lumen_core::{AssetMetadata, Error, Frame, Result};
use tracing::instrument;

#[cfg(feature = "jxl")]
mod imp {
    use super::*;
    use jpegxl_rs::decode::DecoderResult;
    use jpegxl_rs::decoder_builder;
    use lumen_core::{ColorSpace, PixelData};

    fn read_file(path: &Path) -> Result<Vec<u8>> {
        std::fs::read(path).map_err(|e| {
            Error::decode_at(path.to_path_buf(), format!("read jxl: {e}"))
        })
    }

    pub(super) fn decode(path: &Path) -> Result<Frame> {
        let bytes = read_file(path)?;
        let decoder = decoder_builder().build().map_err(|e| {
            Error::decode_at(path.to_path_buf(), format!("jxl builder: {e}"))
        })?;
        let DecoderResult { data, .. }: DecoderResult<u8> =
            decoder.decode_to::<u8>(&bytes).map_err(|e| {
                Error::decode_at(path.to_path_buf(), format!("jxl decode: {e}"))
            })?;

        // Re-probe to learn dimensions; cheaper than double-decoding
        // because libjxl already cached the frame.
        let m = probe_metadata(path, &bytes)?;
        let w = m.width;
        let h = m.height;

        // libjxl returns interleaved channels in its native ordering.
        // Coerce to RGBA8: handle both RGB and RGBA outputs.
        let expected_rgba = (w as usize) * (h as usize) * 4;
        let expected_rgb = (w as usize) * (h as usize) * 3;
        let rgba = if data.len() == expected_rgba {
            data
        } else if data.len() == expected_rgb {
            let mut out = Vec::with_capacity(expected_rgba);
            for px in data.chunks_exact(3) {
                out.extend_from_slice(px);
                out.push(255);
            }
            out
        } else {
            return Err(Error::decode_at(
                path.to_path_buf(),
                format!(
                    "jxl: unexpected pixel count {} for {}x{}",
                    data.len(),
                    w,
                    h
                ),
            ));
        };

        Frame::new(w, h, PixelData::Rgba8(rgba), ColorSpace::SRgb, None)
    }

    fn probe_metadata(path: &Path, bytes: &[u8]) -> Result<AssetMetadata> {
        let decoder = decoder_builder().build().map_err(|e| {
            Error::decode_at(path.to_path_buf(), format!("jxl builder: {e}"))
        })?;
        // decode_to returns metadata even though we throw the pixels away.
        let DecoderResult::<u8> { width, height, .. } =
            decoder.decode_to::<u8>(bytes).map_err(|e| {
                Error::decode_at(path.to_path_buf(), format!("jxl probe: {e}"))
            })?;
        let container = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());
        Ok(AssetMetadata {
            width,
            height,
            frame_count: Some(1),
            frame_rate: None,
            duration_secs: None,
            codec: Some("jxl".to_string()),
            container,
            bit_depth: 8,
            channels: 4,
            color_space: Some(ColorSpace::SRgb),
            audio_sample_rate: None,
            audio_channels: None,
        })
    }

    pub(super) fn probe(path: &Path) -> Result<AssetMetadata> {
        let bytes = read_file(path)?;
        probe_metadata(path, &bytes)
    }
}

#[cfg(not(feature = "jxl"))]
fn unavailable<P: AsRef<Path>>(path: P, what: &str) -> Error {
    Error::UnsupportedFormat(format!(
        "{}: JPEG XL decoding is not built in (rebuild with --features jxl): {}",
        what,
        path.as_ref().display()
    ))
}

/// Decode a JPEG XL file into an RGBA8 / sRGB [`Frame`].
///
/// Without the `jxl` feature this returns
/// [`Error::UnsupportedFormat`] — see module docs.
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub fn decode_jxl<P: AsRef<Path>>(path: P) -> Result<Frame> {
    #[cfg(feature = "jxl")]
    {
        imp::decode(path.as_ref())
    }
    #[cfg(not(feature = "jxl"))]
    {
        Err(unavailable(path, "decode_jxl"))
    }
}

/// Probe a JPEG XL file for metadata.
///
/// Without the `jxl` feature this returns
/// [`Error::UnsupportedFormat`] — see module docs.
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub fn probe_jxl<P: AsRef<Path>>(path: P) -> Result<AssetMetadata> {
    #[cfg(feature = "jxl")]
    {
        imp::probe(path.as_ref())
    }
    #[cfg(not(feature = "jxl"))]
    {
        Err(unavailable(path, "probe_jxl"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "jxl"))]
    #[test]
    fn jxl_off_returns_unsupported() {
        let r = decode_jxl("/some/missing.jxl");
        match r {
            Err(Error::UnsupportedFormat(msg)) => {
                assert!(
                    msg.contains("rebuild with --features jxl"),
                    "expected feature-off hint, got: {msg}"
                );
            }
            other => panic!("expected UnsupportedFormat, got {other:?}"),
        }
        let r = probe_jxl("/some/missing.jxl");
        assert!(matches!(r, Err(Error::UnsupportedFormat(_))));
    }
}
