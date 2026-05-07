//! AVIF decode through the [`image`] crate's `avif-native` decoder.
//!
//! Gated behind the `avif` Cargo feature (which forwards to
//! `image/avif-native`). When the feature is **off** (the default),
//! [`decode_avif`] and [`probe_avif`] still exist but return
//! [`Error::UnsupportedFormat`] — the `image` crate as configured for
//! Lumen's default build only handles PNG/JPEG/TIFF/WebP/BMP, so an
//! AVIF file would fail at `image::open` with a generic error and we
//! want a clearer message.
//!
//! When the feature is **on**, AVIF decoding requires `libdav1d` on
//! the host (Homebrew: `brew install dav1d`).

use std::path::Path;

use lumen_core::{AssetMetadata, Error, Frame, Result};
use tracing::instrument;

#[cfg(feature = "avif")]
mod imp {
    use super::*;
    use image::{ImageFormat, ImageReader};
    use lumen_core::{ColorSpace, PixelData};

    pub(super) fn decode(path: &Path) -> Result<Frame> {
        // Force the AVIF format — relying on extension would miss
        // .heic-extension'd AVIFs etc.
        let reader = ImageReader::open(path)
            .map_err(|e| {
                Error::decode_at(path.to_path_buf(), format!("avif open: {e}"))
            })?
            .with_guessed_format()
            .map_err(|e| {
                Error::decode_at(
                    path.to_path_buf(),
                    format!("avif guess fmt: {e}"),
                )
            })?;
        let mut reader = reader;
        if reader.format().is_none() {
            reader.set_format(ImageFormat::Avif);
        }
        let img = reader.decode().map_err(|e| {
            Error::decode_at(path.to_path_buf(), format!("avif decode: {e}"))
        })?;
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        Frame::new(w, h, PixelData::Rgba8(rgba.into_raw()), ColorSpace::SRgb, None)
    }

    pub(super) fn probe(path: &Path) -> Result<AssetMetadata> {
        let reader = ImageReader::open(path)
            .map_err(|e| {
                Error::decode_at(path.to_path_buf(), format!("avif open: {e}"))
            })?
            .with_guessed_format()
            .map_err(|e| {
                Error::decode_at(
                    path.to_path_buf(),
                    format!("avif guess fmt: {e}"),
                )
            })?;
        let mut reader = reader;
        if reader.format().is_none() {
            reader.set_format(ImageFormat::Avif);
        }
        let dims = reader.into_dimensions().map_err(|e| {
            Error::decode_at(path.to_path_buf(), format!("avif dims: {e}"))
        })?;
        let container = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());
        Ok(AssetMetadata {
            width: dims.0,
            height: dims.1,
            frame_count: Some(1),
            frame_rate: None,
            duration_secs: None,
            codec: Some("avif".to_string()),
            container,
            bit_depth: 8,
            channels: 4,
            color_space: Some(ColorSpace::SRgb),
            audio_sample_rate: None,
            audio_channels: None,
        })
    }
}

#[cfg(not(feature = "avif"))]
fn unavailable<P: AsRef<Path>>(path: P, what: &str) -> Error {
    Error::UnsupportedFormat(format!(
        "{}: AVIF decoding is not built in (rebuild with --features avif): {}",
        what,
        path.as_ref().display()
    ))
}

/// Decode an AVIF file into an RGBA8 / sRGB [`Frame`].
///
/// Without the `avif` feature this returns
/// [`Error::UnsupportedFormat`] — see module docs.
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub fn decode_avif<P: AsRef<Path>>(path: P) -> Result<Frame> {
    #[cfg(feature = "avif")]
    {
        imp::decode(path.as_ref())
    }
    #[cfg(not(feature = "avif"))]
    {
        Err(unavailable(path, "decode_avif"))
    }
}

/// Probe an AVIF file for metadata.
///
/// Without the `avif` feature this returns
/// [`Error::UnsupportedFormat`] — see module docs.
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub fn probe_avif<P: AsRef<Path>>(path: P) -> Result<AssetMetadata> {
    #[cfg(feature = "avif")]
    {
        imp::probe(path.as_ref())
    }
    #[cfg(not(feature = "avif"))]
    {
        Err(unavailable(path, "probe_avif"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "avif"))]
    #[test]
    fn avif_off_returns_unsupported() {
        let r = decode_avif("/some/missing.avif");
        match r {
            Err(Error::UnsupportedFormat(msg)) => {
                assert!(
                    msg.contains("rebuild with --features avif"),
                    "expected feature-off hint, got: {msg}"
                );
            }
            other => panic!("expected UnsupportedFormat, got {other:?}"),
        }
        let r = probe_avif("/some/missing.avif");
        assert!(matches!(r, Err(Error::UnsupportedFormat(_))));
    }
}
