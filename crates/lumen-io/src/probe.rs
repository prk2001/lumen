//! Asset probing — return [`AssetMetadata`] without decoding pixel data.
//!
//! For still images we still pay the cost of full decode (the `image`
//! crate does not expose lazy header probing for every format), but
//! once FFmpeg lands in Milestone 1.1.b we'll move to header-only
//! probes for video.

use std::path::Path;

use image::ImageReader;
use lumen_core::{Asset, AssetKind, AssetMetadata, ColorSpace, Error, Result};
use tracing::instrument;

/// Probe a single file, returning a populated [`AssetMetadata`].
///
/// We first try the still-image fast path via the `image` crate. ANY
/// failure on that path (unknown extension, no magic match, decoder
/// error) falls through to FFmpeg-backed video probing. The
/// still-image path is unchanged from Phase 1; existing
/// PNG/JPEG/TIFF/WebP/BMP behavior is preserved.
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub fn probe_path<P: AsRef<Path>>(path: P) -> Result<AssetMetadata> {
    let path = path.as_ref();
    match probe_via_image(path) {
        Ok(m) => Ok(m),
        Err(_) => probe_via_video(path),
    }
}

fn probe_via_image(path: &Path) -> Result<AssetMetadata> {
    let reader = ImageReader::open(path)
        .map_err(|e| Error::decode_at(path.to_path_buf(), format!("open: {e}")))?
        .with_guessed_format()
        .map_err(|e| Error::decode_at(path.to_path_buf(), format!("guess fmt: {e}")))?;

    if reader.format().is_none() {
        return Err(Error::UnsupportedFormat(format!(
            "not a recognized still image: {}",
            path.display()
        )));
    }

    let format = reader.format();
    let dims = reader
        .into_dimensions()
        .map_err(|e| Error::decode_at(path.to_path_buf(), format!("dims: {e}")))?;

    let container = format.map(|f| format!("{f:?}").to_lowercase());

    // Phase 1: still images only. Bit depth + channel detection require
    // a full decode for some formats; we use defaults for now and tighten
    // when we wire ICC profile parsing.
    Ok(AssetMetadata {
        width: dims.0,
        height: dims.1,
        frame_count: Some(1),
        frame_rate: None,
        duration_secs: None,
        codec: container.clone(),
        container,
        bit_depth: 8,
        channels: 4,
        color_space: Some(ColorSpace::SRgb),
        audio_sample_rate: None,
        audio_channels: None,
    })
}

/// Fallback: probe via FFmpeg for files the `image` crate doesn't grok.
fn probe_via_video(path: &Path) -> Result<AssetMetadata> {
    let probe = crate::video::probe_video(path)?;
    Ok(probe.into_asset_metadata())
}

/// Probe a file and return a populated [`Asset`] with an `AssetId`,
/// inferred kind, and BLAKE3 hash of the file bytes.
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub fn probe<P: AsRef<Path>>(path: P) -> Result<Asset> {
    let path = path.as_ref();
    let metadata = probe_path(path)?;
    let display_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    let uri = format!(
        "file://{}",
        path.canonicalize().unwrap_or_else(|_| path.to_path_buf()).display()
    );
    let hash = crate::hash::hash_file(path).ok();
    // Heuristic: anything with >1 frame or a frame rate is treated as video.
    let kind = if metadata.frame_count.map(|n| n > 1).unwrap_or(false)
        || metadata.frame_rate.is_some()
    {
        AssetKind::Video
    } else {
        AssetKind::StillImage
    };
    let mut asset = Asset::new(uri, display_name, kind);
    asset.metadata = metadata;
    asset.hash = hash;
    Ok(asset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::still::{encode_image, ImageEncodeOptions};
    use lumen_core::{Frame, PixelData};
    use tempfile::tempdir;

    fn write_test_png() -> std::path::PathBuf {
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.png");
        let f = Frame::new(
            10,
            6,
            PixelData::Rgba8(vec![0xAA; 10 * 6 * 4]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();
        encode_image(f, &path, ImageEncodeOptions::default()).unwrap();
        // Move into a stable location keyed by the dir lifetime — return
        // both so tempdir isn't dropped.
        let leaked = path.clone();
        std::mem::forget(dir); // intentional: test cleanup left to OS tmp
        leaked
    }

    #[test]
    fn probe_path_reports_dims() {
        let path = write_test_png();
        let m = probe_path(&path).unwrap();
        assert_eq!(m.width, 10);
        assert_eq!(m.height, 6);
        assert_eq!(m.channels, 4);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn probe_returns_asset_with_hash() {
        let path = write_test_png();
        let asset = probe(&path).unwrap();
        assert_eq!(asset.kind, AssetKind::StillImage);
        assert!(asset.hash.as_ref().unwrap().starts_with("blake3:"));
        let _ = std::fs::remove_file(&path);
    }
}
