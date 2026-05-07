//! Image tensor — the bridge between Lumen's [`Frame`] and ONNX
//! Runtime's `1×3×H×W` float tensors.
//!
//! All vision models we ship for the Phase-2 effects expect normalized
//! float input in **CHW order** (PyTorch convention). Alpha is
//! stripped on the way in and re-applied from the original frame on
//! the way out.

use lumen_core::frame::{Frame, PixelData};
use ndarray::Array4;

/// A `1 × 3 × H × W` float32 tensor with values in `[0.0, 1.0]`.
///
/// Construct via [`ImageTensor::from_frame_chw_normalized`]; convert
/// back into a [`Frame`] via [`ImageTensor::to_frame`]. Everything in
/// between (running an inference session, scaling, padding) is up to
/// the caller.
#[derive(Debug, Clone)]
pub struct ImageTensor {
    /// Shape: `(1, 3, height, width)`.
    pub data: Array4<f32>,
}

impl ImageTensor {
    /// Build a tensor from an RGBA frame.
    ///
    /// The input frame is normalized to f32 in `[0, 1]` (any
    /// non-linear-light handling is the caller's job — pass a
    /// pre-converted frame if you need linear pixels). Alpha is
    /// dropped. Layout becomes `1×3×H×W`.
    pub fn from_frame_chw_normalized(frame: &Frame) -> Self {
        let (w, h) = (frame.width as usize, frame.height as usize);
        let mut arr = Array4::<f32>::zeros((1, 3, h, w));

        match &frame.data {
            PixelData::Rgba8(v) => {
                for y in 0..h {
                    for x in 0..w {
                        let i = (y * w + x) * 4;
                        arr[[0, 0, y, x]] = (v[i] as f32) / 255.0;
                        arr[[0, 1, y, x]] = (v[i + 1] as f32) / 255.0;
                        arr[[0, 2, y, x]] = (v[i + 2] as f32) / 255.0;
                    }
                }
            }
            PixelData::Rgba16(v) => {
                for y in 0..h {
                    for x in 0..w {
                        let i = (y * w + x) * 4;
                        arr[[0, 0, y, x]] = (v[i] as f32) / 65535.0;
                        arr[[0, 1, y, x]] = (v[i + 1] as f32) / 65535.0;
                        arr[[0, 2, y, x]] = (v[i + 2] as f32) / 65535.0;
                    }
                }
            }
            PixelData::RgbaF32(v) => {
                for y in 0..h {
                    for x in 0..w {
                        let i = (y * w + x) * 4;
                        arr[[0, 0, y, x]] = v[i];
                        arr[[0, 1, y, x]] = v[i + 1];
                        arr[[0, 2, y, x]] = v[i + 2];
                    }
                }
            }
        }
        ImageTensor { data: arr }
    }

    /// Height of the tensor (axis 2).
    pub fn height(&self) -> usize {
        self.data.shape()[2]
    }

    /// Width of the tensor (axis 3).
    pub fn width(&self) -> usize {
        self.data.shape()[3]
    }

    /// Build a [`Frame`] from this tensor, taking alpha and metadata
    /// (color space, pts) from `original`.
    ///
    /// If the tensor's spatial dimensions don't match the original
    /// frame, the output frame uses the tensor's dims and alpha is
    /// filled with `1.0` / `255` / `65535` (whichever matches the
    /// requested layout). The output layout matches `original.layout()`.
    pub fn to_frame(&self, original: &Frame) -> Frame {
        let h = self.height();
        let w = self.width();
        let same_dims = h == original.height as usize && w == original.width as usize;

        let pixel_data = match &original.data {
            PixelData::Rgba8(orig) => {
                let mut out = vec![0u8; w * h * 4];
                for y in 0..h {
                    for x in 0..w {
                        let oi = (y * w + x) * 4;
                        out[oi] = quantize_u8(self.data[[0, 0, y, x]]);
                        out[oi + 1] = quantize_u8(self.data[[0, 1, y, x]]);
                        out[oi + 2] = quantize_u8(self.data[[0, 2, y, x]]);
                        out[oi + 3] = if same_dims { orig[oi + 3] } else { 255 };
                    }
                }
                PixelData::Rgba8(out)
            }
            PixelData::Rgba16(orig) => {
                let mut out = vec![0u16; w * h * 4];
                for y in 0..h {
                    for x in 0..w {
                        let oi = (y * w + x) * 4;
                        out[oi] = quantize_u16(self.data[[0, 0, y, x]]);
                        out[oi + 1] = quantize_u16(self.data[[0, 1, y, x]]);
                        out[oi + 2] = quantize_u16(self.data[[0, 2, y, x]]);
                        out[oi + 3] = if same_dims { orig[oi + 3] } else { 65535 };
                    }
                }
                PixelData::Rgba16(out)
            }
            PixelData::RgbaF32(orig) => {
                let mut out = vec![0.0f32; w * h * 4];
                for y in 0..h {
                    for x in 0..w {
                        let oi = (y * w + x) * 4;
                        out[oi] = self.data[[0, 0, y, x]];
                        out[oi + 1] = self.data[[0, 1, y, x]];
                        out[oi + 2] = self.data[[0, 2, y, x]];
                        out[oi + 3] = if same_dims { orig[oi + 3] } else { 1.0 };
                    }
                }
                PixelData::RgbaF32(out)
            }
        };

        Frame {
            width: w as u32,
            height: h as u32,
            data: pixel_data,
            color_space: original.color_space,
            pts: original.pts,
        }
    }
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
            0.40, 0.50, 0.60, 1.00,
            // (0,1)
            0.70, 0.80, 0.90, 1.00,
            // (1,1)
            0.05, 0.15, 0.25, 1.00,
        ];
        Frame::new(2, 2, PixelData::RgbaF32(pixels), ColorSpace::LinearSRgb, None).unwrap()
    }

    #[test]
    fn round_trip_f32_within_epsilon() {
        let frame = sample_f32_frame();
        let tensor = ImageTensor::from_frame_chw_normalized(&frame);
        let back = tensor.to_frame(&frame);
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
    fn shape_is_one_three_h_w() {
        let frame = sample_f32_frame();
        let tensor = ImageTensor::from_frame_chw_normalized(&frame);
        assert_eq!(tensor.data.shape(), &[1, 3, 2, 2]);
    }
}
