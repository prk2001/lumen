//! Image tensor — bridge between Lumen's [`Frame`] and the
//! `1×C×H×W` float tensors that almost every ONNX vision model expects.
//!
//! All Lumen-shipped vision models consume normalized float input in
//! **CHW order** (PyTorch convention). Alpha is stripped on the way in
//! and re-attached on the way out.

use lumen_core::frame::{Frame, PixelData};

use crate::error::{AiError, Result};

/// A `1 × C × H × W` float32 tensor.
///
/// Component values are typically in `[0.0, 1.0]` — the conversion
/// helpers in this module produce that range — but the struct itself is
/// just a flat `Vec<f32>` plus a 4-D shape, so callers can populate it
/// with any range they need.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageTensor {
    /// Row-major contiguous data, length `shape.iter().product()`.
    pub data: Vec<f32>,
    /// `[N, C, H, W]`. For Lumen's bridge helpers `N == 1` and
    /// `C == 3` (RGB without alpha).
    pub shape: [usize; 4],
}

impl ImageTensor {
    /// Number of elements in `data`. Equal to the product of `shape`.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// True if the tensor has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Batch size (axis 0).
    pub fn batch(&self) -> usize {
        self.shape[0]
    }

    /// Channel count (axis 1).
    pub fn channels(&self) -> usize {
        self.shape[1]
    }

    /// Height (axis 2).
    pub fn height(&self) -> usize {
        self.shape[2]
    }

    /// Width (axis 3).
    pub fn width(&self) -> usize {
        self.shape[3]
    }
}

/// Build a `1×3×H×W` CHW tensor from an RGBA frame.
///
/// Alpha is dropped. Component values are normalized to `[0, 1]` for
/// `Rgba8` / `Rgba16` inputs and copied as-is for `RgbaF32` (the
/// caller is responsible for any color-space conversion before
/// inference — typically the frame should already be in linear RGB).
pub fn from_frame_chw(frame: &Frame) -> ImageTensor {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let plane = h * w;
    let mut data = vec![0.0f32; 3 * plane];

    match &frame.data {
        PixelData::Rgba8(v) => {
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 4;
                    let p = y * w + x;
                    data[p] = (v[i] as f32) / 255.0;
                    data[plane + p] = (v[i + 1] as f32) / 255.0;
                    data[2 * plane + p] = (v[i + 2] as f32) / 255.0;
                }
            }
        }
        PixelData::Rgba16(v) => {
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 4;
                    let p = y * w + x;
                    data[p] = (v[i] as f32) / 65535.0;
                    data[plane + p] = (v[i + 1] as f32) / 65535.0;
                    data[2 * plane + p] = (v[i + 2] as f32) / 65535.0;
                }
            }
        }
        PixelData::RgbaF32(v) => {
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 4;
                    let p = y * w + x;
                    data[p] = v[i];
                    data[plane + p] = v[i + 1];
                    data[2 * plane + p] = v[i + 2];
                }
            }
        }
    }

    ImageTensor { data, shape: [1, 3, h, w] }
}

/// Re-build a [`Frame`] from a CHW `1×3×H×W` tensor, taking alpha and
/// metadata (color space, pts) from `original`.
///
/// If the tensor's spatial dimensions match `original`, the alpha
/// channel is copied verbatim from the original frame. If they differ
/// (e.g. an upscaler produced a larger output), alpha is filled with
/// the layout's "fully opaque" value (`255` / `65535` / `1.0`).
///
/// The output [`PixelData`] variant matches `original.layout()`. For
/// 8-/16-bit layouts, RGB values are clamped to `[0, 1]` and quantized.
///
/// Returns an error if the tensor shape isn't `1×3×H×W` (i.e. wrong
/// batch size or channel count).
pub fn to_frame(tensor: &ImageTensor, original: &Frame) -> Result<Frame> {
    if tensor.shape[0] != 1 {
        return Err(AiError::Shape(format!(
            "expected batch size 1, got {}",
            tensor.shape[0]
        )));
    }
    if tensor.shape[1] != 3 {
        return Err(AiError::Shape(format!(
            "expected 3 channels (RGB), got {}",
            tensor.shape[1]
        )));
    }
    let h = tensor.shape[2];
    let w = tensor.shape[3];
    let plane = h * w;
    if tensor.data.len() != 3 * plane {
        return Err(AiError::Shape(format!(
            "data length {} doesn't match shape {:?} ({} expected)",
            tensor.data.len(),
            tensor.shape,
            3 * plane
        )));
    }

    let same_dims = h == original.height as usize && w == original.width as usize;

    let pixel_data = match &original.data {
        PixelData::Rgba8(orig) => {
            let mut out = vec![0u8; w * h * 4];
            for y in 0..h {
                for x in 0..w {
                    let p = y * w + x;
                    let oi = p * 4;
                    out[oi] = quantize_u8(tensor.data[p]);
                    out[oi + 1] = quantize_u8(tensor.data[plane + p]);
                    out[oi + 2] = quantize_u8(tensor.data[2 * plane + p]);
                    out[oi + 3] = if same_dims { orig[oi + 3] } else { 255 };
                }
            }
            PixelData::Rgba8(out)
        }
        PixelData::Rgba16(orig) => {
            let mut out = vec![0u16; w * h * 4];
            for y in 0..h {
                for x in 0..w {
                    let p = y * w + x;
                    let oi = p * 4;
                    out[oi] = quantize_u16(tensor.data[p]);
                    out[oi + 1] = quantize_u16(tensor.data[plane + p]);
                    out[oi + 2] = quantize_u16(tensor.data[2 * plane + p]);
                    out[oi + 3] = if same_dims { orig[oi + 3] } else { 65535 };
                }
            }
            PixelData::Rgba16(out)
        }
        PixelData::RgbaF32(orig) => {
            let mut out = vec![0.0f32; w * h * 4];
            for y in 0..h {
                for x in 0..w {
                    let p = y * w + x;
                    let oi = p * 4;
                    out[oi] = tensor.data[p];
                    out[oi + 1] = tensor.data[plane + p];
                    out[oi + 2] = tensor.data[2 * plane + p];
                    out[oi + 3] = if same_dims { orig[oi + 3] } else { 1.0 };
                }
            }
            PixelData::RgbaF32(out)
        }
    };

    Ok(Frame {
        width: w as u32,
        height: h as u32,
        data: pixel_data,
        color_space: original.color_space.clone(),
        pts: original.pts,
    })
}

#[inline]
fn quantize_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[inline]
fn quantize_u16(v: f32) -> u16 {
    (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::color::ColorSpace;
    use lumen_core::frame::{Frame, PixelData};

    fn sample_f32_frame() -> Frame {
        // 2×2 image, RGBA float, deterministic gradient.
        let pixels: Vec<f32> = vec![
            // (0,0) R G B A
            0.10, 0.20, 0.30, 1.00,
            // (1,0)
            0.40, 0.50, 0.60, 0.75,
            // (0,1)
            0.70, 0.80, 0.90, 0.50,
            // (1,1)
            0.05, 0.15, 0.25, 0.25,
        ];
        Frame::new(2, 2, PixelData::RgbaF32(pixels), ColorSpace::LinearSRgb, None).unwrap()
    }

    #[test]
    fn shape_is_one_three_h_w() {
        let frame = sample_f32_frame();
        let tensor = from_frame_chw(&frame);
        assert_eq!(tensor.shape, [1, 3, 2, 2]);
        assert_eq!(tensor.data.len(), 12);
    }

    #[test]
    fn round_trip_f32_within_epsilon() {
        let frame = sample_f32_frame();
        let tensor = from_frame_chw(&frame);
        let back = to_frame(&tensor, &frame).expect("to_frame should succeed");
        let PixelData::RgbaF32(orig) = &frame.data else { unreachable!() };
        let PixelData::RgbaF32(out) = &back.data else { unreachable!() };
        assert_eq!(orig.len(), out.len());
        for (i, (&a, &b)) in orig.iter().zip(out.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6,
                "channel {i}: expected {a}, got {b}"
            );
        }
        assert_eq!(back.width, frame.width);
        assert_eq!(back.height, frame.height);
        assert_eq!(back.color_space, frame.color_space);
    }

    #[test]
    fn alpha_preserved_from_original() {
        let frame = sample_f32_frame();
        let tensor = from_frame_chw(&frame);
        let back = to_frame(&tensor, &frame).unwrap();
        let PixelData::RgbaF32(out) = &back.data else { unreachable!() };
        // Alpha values from the original: 1.00, 0.75, 0.50, 0.25
        assert!((out[3] - 1.00).abs() <= 1e-6);
        assert!((out[7] - 0.75).abs() <= 1e-6);
        assert!((out[11] - 0.50).abs() <= 1e-6);
        assert!((out[15] - 0.25).abs() <= 1e-6);
    }

    #[test]
    fn round_trip_u8_normalized() {
        let pixels: Vec<u8> = vec![
            10, 20, 30, 255,
            100, 110, 120, 200,
            200, 210, 220, 128,
            255, 0, 128, 64,
        ];
        let frame = Frame::new(
            2,
            2,
            PixelData::Rgba8(pixels.clone()),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();
        let tensor = from_frame_chw(&frame);
        let back = to_frame(&tensor, &frame).unwrap();
        let PixelData::Rgba8(out) = &back.data else { unreachable!() };
        // u8 round-trips exactly because 255 is the divisor and the
        // quantizer rounds half up.
        assert_eq!(out, &pixels);
    }

    #[test]
    fn rejects_wrong_channel_count() {
        let frame = sample_f32_frame();
        let bad = ImageTensor {
            data: vec![0.0; 4],
            shape: [1, 1, 2, 2],
        };
        let err = to_frame(&bad, &frame).unwrap_err();
        assert!(matches!(err, AiError::Shape(_)));
    }
}
