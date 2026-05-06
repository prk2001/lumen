//! Rectangular crop.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Error, Frame, ParamKind, ParamSpec,
    ParamValues, PixelData, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct Crop;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-geometric.crop",
    display_name: "Crop",
    description: "Extract a rectangular region.",
    category: Category::Geometric,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "x",
        display_name: "X",
        description: "Left edge in pixels (0-based).",
        kind: ParamKind::Int { default: 0, min: Some(0), max: None },
    },
    ParamSpec {
        id: "y",
        display_name: "Y",
        description: "Top edge in pixels (0-based).",
        kind: ParamKind::Int { default: 0, min: Some(0), max: None },
    },
    ParamSpec {
        id: "width",
        display_name: "Width",
        description: "Crop width in pixels.",
        kind: ParamKind::Int { default: 256, min: Some(1), max: None },
    },
    ParamSpec {
        id: "height",
        display_name: "Height",
        description: "Crop height in pixels.",
        kind: ParamKind::Int { default: 256, min: Some(1), max: None },
    },
];

impl Effect for Crop {
    fn metadata(&self) -> &EffectMetadata { &META }
    fn parameters(&self) -> &[ParamSpec] { PARAMS }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            deterministic: true,
            gpu: false,
            streamable: false,
            temporal: false,
        }
    }

    #[instrument(skip_all, fields(effect = META.id))]
    fn apply(&self, _ctx: &mut Context, input: Frame, params: &ParamValues) -> Result<Frame> {
        let x = params.get_int("x").unwrap_or(0).max(0) as u32;
        let y = params.get_int("y").unwrap_or(0).max(0) as u32;
        let cw = params.get_int("width").unwrap_or(256).max(1) as u32;
        let ch = params.get_int("height").unwrap_or(256).max(1) as u32;

        if x + cw > input.width || y + ch > input.height {
            return Err(Error::InvalidParameter {
                name: "crop_rect".to_string(),
                reason: format!(
                    "crop rect {}x{}+{}+{} extends past image {}x{}",
                    cw, ch, x, y, input.width, input.height
                ),
            });
        }

        // Operate on the underlying RGBA8 buffer when possible to avoid
        // unnecessary float conversion. Fall back to f32 otherwise.
        let cs = input.color_space.clone();
        let pts = input.pts;
        let stride_src = input.width as usize * 4;
        let stride_dst = cw as usize * 4;

        let data = match input.data {
            PixelData::Rgba8(src) => {
                let mut dst = vec![0u8; cw as usize * ch as usize * 4];
                for ry in 0..ch as usize {
                    let so = (y as usize + ry) * stride_src + x as usize * 4;
                    let dst_o = ry * stride_dst;
                    dst[dst_o..dst_o + stride_dst].copy_from_slice(&src[so..so + stride_dst]);
                }
                PixelData::Rgba8(dst)
            }
            PixelData::Rgba16(src) => {
                let mut dst = vec![0u16; cw as usize * ch as usize * 4];
                for ry in 0..ch as usize {
                    let so = (y as usize + ry) * stride_src + x as usize * 4;
                    let dst_o = ry * stride_dst;
                    dst[dst_o..dst_o + stride_dst].copy_from_slice(&src[so..so + stride_dst]);
                }
                PixelData::Rgba16(dst)
            }
            PixelData::RgbaF32(src) => {
                let mut dst = vec![0f32; cw as usize * ch as usize * 4];
                for ry in 0..ch as usize {
                    let so = (y as usize + ry) * stride_src + x as usize * 4;
                    let dst_o = ry * stride_dst;
                    dst[dst_o..dst_o + stride_dst].copy_from_slice(&src[so..so + stride_dst]);
                }
                PixelData::RgbaF32(dst)
            }
        };

        Frame::new(cw, ch, data, cs, pts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue};

    #[test]
    fn crop_half() {
        let c = Crop;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("x", ParamValue::Int(2));
        p.insert("y", ParamValue::Int(2));
        p.insert("width", ParamValue::Int(2));
        p.insert("height", ParamValue::Int(2));
        p.validate_and_fill(c.parameters()).unwrap();

        let f = Frame::new(
            4,
            4,
            PixelData::Rgba8((0..64).map(|i| i as u8).collect()),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();
        let out = c.apply(&mut ctx, f, &p).unwrap();
        assert_eq!(out.width, 2);
        assert_eq!(out.height, 2);
    }

    #[test]
    fn crop_out_of_bounds_errs() {
        let c = Crop;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("x", ParamValue::Int(3));
        p.insert("y", ParamValue::Int(3));
        p.insert("width", ParamValue::Int(4));
        p.insert("height", ParamValue::Int(4));
        p.validate_and_fill(c.parameters()).unwrap();

        let f =
            Frame::new(4, 4, PixelData::Rgba8(vec![0; 64]), ColorSpace::SRgb, None).unwrap();
        let r = c.apply(&mut ctx, f, &p);
        assert!(matches!(r, Err(Error::InvalidParameter { .. })));
    }
}
