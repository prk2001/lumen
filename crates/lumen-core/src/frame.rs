//! 2D image buffers — the fundamental unit of work.
//!
//! A [`Frame`] is *always* 4-channel RGBA (alpha defaults to 1.0). Effects
//! that don't care about alpha simply leave it untouched. The pixel
//! container is a tagged enum so we don't pay for f32 storage on the
//! ingest side — but every effect can normalize cheaply via
//! [`Frame::into_rgba_f32`].

use crate::color::{linear_to_srgb, srgb_to_linear, ColorSpace};
use crate::error::{Error, Result};
use crate::time::Pts;
use serde::{Deserialize, Serialize};

/// Per-pixel storage format. RGBA component order, tightly packed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PixelLayout {
    /// 8-bit unsigned, 4 bytes per pixel.
    Rgba8,
    /// 16-bit unsigned (big-endian on disk; native-endian in memory),
    /// 8 bytes per pixel.
    Rgba16,
    /// IEEE-754 binary32 float, 16 bytes per pixel.
    RgbaF32,
}

impl PixelLayout {
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            PixelLayout::Rgba8 => 4,
            PixelLayout::Rgba16 => 8,
            PixelLayout::RgbaF32 => 16,
        }
    }
}

/// Tagged pixel buffer. Length is always `width * height * 4 components`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PixelData {
    Rgba8(Vec<u8>),
    Rgba16(Vec<u16>),
    RgbaF32(Vec<f32>),
}

impl PixelData {
    pub fn layout(&self) -> PixelLayout {
        match self {
            PixelData::Rgba8(_) => PixelLayout::Rgba8,
            PixelData::Rgba16(_) => PixelLayout::Rgba16,
            PixelData::RgbaF32(_) => PixelLayout::RgbaF32,
        }
    }

    /// Number of components — pixels × 4. Use [`Self::pixel_count`] to
    /// get the pixel count.
    pub fn component_len(&self) -> usize {
        match self {
            PixelData::Rgba8(v) => v.len(),
            PixelData::Rgba16(v) => v.len(),
            PixelData::RgbaF32(v) => v.len(),
        }
    }

    pub fn pixel_count(&self) -> usize { self.component_len() / 4 }
}

/// A 2D RGBA image with associated color and timing metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub data: PixelData,
    pub color_space: ColorSpace,
    /// Presentation timestamp; `None` for stills.
    pub pts: Option<Pts>,
}

impl Frame {
    /// Construct a frame, validating that `data.len() == width*height*4`.
    pub fn new(
        width: u32,
        height: u32,
        data: PixelData,
        color_space: ColorSpace,
        pts: Option<Pts>,
    ) -> Result<Self> {
        let expected = (width as usize) * (height as usize) * 4;
        if data.component_len() != expected {
            return Err(Error::Layout(format!(
                "expected {expected} components for {width}x{height} RGBA, got {}",
                data.component_len()
            )));
        }
        Ok(Frame { width, height, data, color_space, pts })
    }

    /// Empty (transparent black) frame in the requested layout.
    pub fn zeros(width: u32, height: u32, layout: PixelLayout, cs: ColorSpace) -> Self {
        let n = (width as usize) * (height as usize) * 4;
        let data = match layout {
            PixelLayout::Rgba8 => PixelData::Rgba8(vec![0u8; n]),
            PixelLayout::Rgba16 => PixelData::Rgba16(vec![0u16; n]),
            PixelLayout::RgbaF32 => PixelData::RgbaF32(vec![0.0f32; n]),
        };
        Frame { width, height, data, color_space: cs, pts: None }
    }

    pub fn pixel_count(&self) -> usize { self.data.pixel_count() }

    pub fn layout(&self) -> PixelLayout { self.data.layout() }

    /// True if frame is empty (zero pixels).
    pub fn is_empty(&self) -> bool { self.pixel_count() == 0 }

    /// Convert into linearized RGBA f32. Component values are normalized to
    /// [0.0, 1.0] for u8/u16 inputs and the sRGB transfer is undone if
    /// `color_space == ColorSpace::SRgb`. The returned frame is tagged
    /// [`ColorSpace::LinearSRgb`] in that case.
    ///
    /// This is the ergonomic entry point most effects use — they can
    /// process float pixels and let the I/O layer worry about output
    /// formats.
    pub fn into_rgba_f32_linear(mut self) -> Self {
        // Pull data out and rebuild.
        let was_srgb = self.color_space == ColorSpace::SRgb;
        let f32_data: Vec<f32> = match std::mem::replace(
            &mut self.data,
            PixelData::RgbaF32(Vec::new()),
        ) {
            PixelData::Rgba8(v) => v
                .into_iter()
                .map(|b| (b as f32) / 255.0)
                .collect(),
            PixelData::Rgba16(v) => v
                .into_iter()
                .map(|b| (b as f32) / 65535.0)
                .collect(),
            PixelData::RgbaF32(v) => v,
        };

        let pixels = if was_srgb {
            f32_data
                .chunks_exact(4)
                .flat_map(|p| {
                    [srgb_to_linear(p[0]), srgb_to_linear(p[1]), srgb_to_linear(p[2]), p[3]]
                })
                .collect()
        } else {
            f32_data
        };

        Frame {
            width: self.width,
            height: self.height,
            data: PixelData::RgbaF32(pixels),
            color_space: if was_srgb { ColorSpace::LinearSRgb } else { self.color_space },
            pts: self.pts,
        }
    }

    /// Inverse of [`Self::into_rgba_f32_linear`] — convert linear-RGB f32
    /// pixels into 8-bit sRGB suitable for PNG/JPEG export. Other layouts
    /// are converted directly without transfer-function adjustment.
    pub fn into_rgba_u8_srgb(self) -> Self {
        let was_linear = self.color_space.is_linear();
        let pixels: Vec<u8> = match self.data {
            PixelData::Rgba8(v) => v,
            PixelData::Rgba16(v) => v.into_iter().map(|x| (x >> 8) as u8).collect(),
            PixelData::RgbaF32(v) => {
                if was_linear {
                    v.chunks_exact(4)
                        .flat_map(|p| {
                            [
                                f32_to_u8(linear_to_srgb(p[0])),
                                f32_to_u8(linear_to_srgb(p[1])),
                                f32_to_u8(linear_to_srgb(p[2])),
                                f32_to_u8(p[3]),
                            ]
                        })
                        .collect()
                } else {
                    v.into_iter().map(f32_to_u8).collect()
                }
            }
        };

        Frame {
            width: self.width,
            height: self.height,
            data: PixelData::Rgba8(pixels),
            color_space: ColorSpace::SRgb,
            pts: self.pts,
        }
    }

    /// View as a slice of `f32` if the layout matches; otherwise `None`.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.data {
            PixelData::RgbaF32(v) => Some(v),
            _ => None,
        }
    }

    /// Mutably view as a slice of `f32` if the layout matches.
    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        match &mut self.data {
            PixelData::RgbaF32(v) => Some(v),
            _ => None,
        }
    }
}

#[inline]
fn f32_to_u8(c: f32) -> u8 {
    (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_new_validates_layout() {
        // 2x2 RGBA = 16 components
        let ok = Frame::new(
            2,
            2,
            PixelData::Rgba8(vec![0u8; 16]),
            ColorSpace::SRgb,
            None,
        );
        assert!(ok.is_ok());

        // 2x2 with 8 components is wrong
        let bad = Frame::new(
            2,
            2,
            PixelData::Rgba8(vec![0u8; 8]),
            ColorSpace::SRgb,
            None,
        );
        assert!(bad.is_err());
    }

    #[test]
    fn srgb_to_linear_round_trip() {
        let f = Frame::new(
            1,
            1,
            PixelData::Rgba8(vec![128, 200, 64, 255]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();
        let lin = f.into_rgba_f32_linear();
        assert_eq!(lin.color_space, ColorSpace::LinearSRgb);
        let back = lin.into_rgba_u8_srgb();
        let PixelData::Rgba8(px) = back.data else { panic!("expected u8") };
        assert_eq!(px[0], 128);
        assert_eq!(px[1], 200);
        assert_eq!(px[2], 64);
        assert_eq!(px[3], 255);
    }

    #[test]
    fn zeros_has_correct_dims() {
        let f = Frame::zeros(4, 3, PixelLayout::RgbaF32, ColorSpace::AcesCg);
        assert_eq!(f.width, 4);
        assert_eq!(f.height, 3);
        assert_eq!(f.pixel_count(), 12);
        assert!(f.as_f32().unwrap().iter().all(|&x| x == 0.0));
    }
}
