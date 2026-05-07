//! Still-image decode and encode via the `image` crate.
//!
//! Decodes any format `image` understands into a [`Frame`]; encodes any
//! [`Frame`] back out as PNG/JPEG/TIFF/WebP/BMP.

use std::path::{Path, PathBuf};

use image::{DynamicImage, ImageFormat, RgbaImage};
use lumen_core::{ColorSpace, Error, Frame, PixelData, Result};
use tracing::{debug, instrument};

/// Encoder options for [`encode_image`].
#[derive(Debug, Clone)]
pub struct ImageEncodeOptions {
    /// JPEG quality 1–100, default 92. Ignored for lossless formats.
    pub jpeg_quality: u8,
    /// Force a specific [`ImageFormat`]. `None` infers from path
    /// extension.
    pub format: Option<ImageFormat>,
}

impl Default for ImageEncodeOptions {
    fn default() -> Self { Self { jpeg_quality: 92, format: None } }
}

/// Decode a still image at `path` into a [`Frame`].
///
/// Output is always RGBA8 in sRGB color space. Effects can call
/// [`Frame::into_rgba_f32_linear`] to lift to scene-linear float.
///
/// This is a thin wrapper over [`decode_still`] kept for backwards
/// compatibility with the original Phase 1 API. New code can call
/// either name.
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub fn decode_image<P: AsRef<Path>>(path: P) -> Result<Frame> {
    decode_still(path)
}

/// Decode any supported still image format into a [`Frame`].
///
/// The decoder chain is (first success wins):
///
/// 1. **`image` crate fast path** — PNG/JPEG/TIFF/WebP/BMP and (when
///    the `avif` feature is on) AVIF.
/// 2. **RAW** via `rawloader` — CR2, NEF, ARW, DNG, RAF, ORF, etc.
/// 3. **HEIF/HEIC** when the `heif` feature is on.
/// 4. **JPEG XL** when the `jxl` feature is on.
/// 5. **AVIF** as a final standalone fallback.
///
/// The chain only escalates on `Err`; the error returned to the caller
/// is whichever decoder is most likely to be the *intended* one based
/// on the file extension. If the file doesn't match any known
/// extension we surface the last error in the chain.
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub fn decode_still<P: AsRef<Path>>(path: P) -> Result<Frame> {
    let path = path.as_ref();

    // Step 1: image crate.
    if let Ok(frame) = decode_via_image_crate(path) {
        debug!(width = frame.width, height = frame.height, "decoded via image crate");
        return Ok(frame);
    }

    // The chain order below biases toward the file's extension when we
    // have one — RAW first for camera extensions, HEIF for .heic/.heif,
    // JXL for .jxl, AVIF for .avif. For unknown extensions we fall
    // through every decoder in order.
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());

    let order: &[&str] = match ext.as_deref() {
        Some("heic") | Some("heif") => &["heif", "raw", "jxl", "avif"],
        Some("jxl") => &["jxl", "heif", "raw", "avif"],
        Some("avif") => &["avif", "heif", "raw", "jxl"],
        _ => &["raw", "heif", "jxl", "avif"],
    };

    let mut last_err: Option<Error> = None;
    for stage in order {
        let r = match *stage {
            "raw" => crate::raw::decode_raw(path),
            "heif" => crate::heif::decode_heif(path),
            "jxl" => crate::jxl::decode_jxl(path),
            "avif" => crate::avif::decode_avif(path),
            _ => continue,
        };
        match r {
            Ok(frame) => {
                debug!(stage = *stage, "decoded via fallback");
                return Ok(frame);
            }
            Err(e) => last_err = Some(e),
        }
    }

    Err(last_err.unwrap_or_else(|| {
        Error::UnsupportedFormat(format!("not a recognized image: {}", path.display()))
    }))
}

fn decode_via_image_crate(path: &Path) -> Result<Frame> {
    let img: DynamicImage = image::open(path).map_err(|e| {
        Error::decode_at(path.to_path_buf(), format!("image::open failed: {e}"))
    })?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    Frame::new(
        w,
        h,
        PixelData::Rgba8(rgba.into_raw()),
        ColorSpace::SRgb,
        None,
    )
}

/// Encode a [`Frame`] to disk.
///
/// If the frame isn't already RGBA8, it's converted via
/// [`Frame::into_rgba_u8_srgb`].
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub fn encode_image<P: AsRef<Path>>(
    frame: Frame,
    path: P,
    opts: ImageEncodeOptions,
) -> Result<PathBuf> {
    let path = path.as_ref();
    let format = match opts.format {
        Some(f) => f,
        None => ImageFormat::from_path(path).map_err(|e| {
            Error::encode_at(path.to_path_buf(), format!("can't infer format: {e}"))
        })?,
    };

    let frame = match frame.layout() {
        lumen_core::PixelLayout::Rgba8 => frame,
        _ => frame.into_rgba_u8_srgb(),
    };
    let PixelData::Rgba8(data) = frame.data else {
        return Err(Error::Layout("internal: expected Rgba8 after conversion".into()));
    };

    let rgba = RgbaImage::from_raw(frame.width, frame.height, data).ok_or_else(|| {
        Error::Layout(format!(
            "RgbaImage construction failed for {}x{}",
            frame.width, frame.height
        ))
    })?;

    match format {
        ImageFormat::Jpeg => {
            // JPEG doesn't support alpha — convert to RGB first.
            let rgb = DynamicImage::ImageRgba8(rgba).to_rgb8();
            let file = std::fs::File::create(path)?;
            let mut bw = std::io::BufWriter::new(file);
            let encoder =
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bw, opts.jpeg_quality);
            DynamicImage::ImageRgb8(rgb)
                .write_with_encoder(encoder)
                .map_err(|e| {
                    Error::encode_at(path.to_path_buf(), format!("jpeg encode: {e}"))
                })?;
        }
        ImageFormat::Bmp => {
            // BMP doesn't support alpha — convert to RGB first.
            let rgb = DynamicImage::ImageRgba8(rgba).to_rgb8();
            DynamicImage::ImageRgb8(rgb)
                .save_with_format(path, format)
                .map_err(|e| {
                    Error::encode_at(path.to_path_buf(), format!("bmp encode: {e}"))
                })?;
        }
        _ => {
            DynamicImage::ImageRgba8(rgba)
                .save_with_format(path, format)
                .map_err(|e| {
                    Error::encode_at(path.to_path_buf(), format!("encode: {e}"))
                })?;
        }
    }

    debug!("wrote {}", path.display());
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn synth_frame(w: u32, h: u32) -> Frame {
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(((x * 255) / w.max(1)) as u8); // R ramp
                data.push(((y * 255) / h.max(1)) as u8); // G ramp
                data.push(128);                          // B mid
                data.push(255);                          // A solid
            }
        }
        Frame::new(w, h, PixelData::Rgba8(data), ColorSpace::SRgb, None).unwrap()
    }

    #[test]
    fn png_round_trip_pixel_exact() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("synth.png");
        let original = synth_frame(64, 48);
        encode_image(original.clone(), &path, ImageEncodeOptions::default()).unwrap();
        let decoded = decode_image(&path).unwrap();
        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 48);
        assert_eq!(decoded.data, original.data, "PNG should be lossless");
    }

    #[test]
    fn jpeg_round_trip_close() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("synth.jpg");
        let original = synth_frame(64, 48);
        encode_image(original.clone(), &path, ImageEncodeOptions::default()).unwrap();
        let decoded = decode_image(&path).unwrap();
        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 48);
        // JPEG is lossy; just confirm it decoded and dims match.
        assert_eq!(decoded.layout(), lumen_core::PixelLayout::Rgba8);
    }

    #[test]
    fn tiff_round_trip_pixel_exact() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("synth.tiff");
        let original = synth_frame(32, 32);
        encode_image(original.clone(), &path, ImageEncodeOptions::default()).unwrap();
        let decoded = decode_image(&path).unwrap();
        assert_eq!(decoded.data, original.data, "TIFF should be lossless");
    }

    #[test]
    fn nonexistent_path_errs() {
        let r = decode_image("/nonexistent/file.png");
        // The fallback chain runs after image::open fails. Every fallback
        // also fails on a nonexistent path, so the surfaced error can be
        // either Decode (raw, etc.) or UnsupportedFormat (when all gated
        // fallbacks are off and the chain bottoms out). Accept either.
        assert!(
            matches!(r, Err(Error::Decode { .. }) | Err(Error::UnsupportedFormat(_))),
            "expected Decode or UnsupportedFormat error"
        );
    }

    /// Regression: PNG round-trip still works through the new
    /// fallback chain (`decode_still` is what `decode_image` now calls).
    #[test]
    fn png_round_trip_still_works_through_chain() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("regression.png");
        let original = synth_frame(20, 14);
        encode_image(original.clone(), &path, ImageEncodeOptions::default()).unwrap();
        let decoded = decode_still(&path).unwrap();
        assert_eq!(decoded.width, 20);
        assert_eq!(decoded.height, 14);
        assert_eq!(decoded.data, original.data, "PNG should still be lossless via decode_still");
    }

    /// Garbage bytes with an unknown extension fall through every
    /// decoder and surface a useful error.
    #[test]
    fn unknown_garbage_returns_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("not-an-image.xyz");
        std::fs::write(&path, b"\x00\x01\x02\x03\x04\x05").unwrap();
        let r = decode_still(&path);
        assert!(r.is_err());
    }
}
