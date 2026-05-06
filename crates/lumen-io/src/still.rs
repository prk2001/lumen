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
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub fn decode_image<P: AsRef<Path>>(path: P) -> Result<Frame> {
    let path = path.as_ref();
    let img: DynamicImage = image::open(path).map_err(|e| {
        Error::decode_at(path.to_path_buf(), format!("image::open failed: {e}"))
    })?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    debug!(width = w, height = h, "decoded still image");
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
        assert!(matches!(r, Err(Error::Decode { .. })));
    }
}
