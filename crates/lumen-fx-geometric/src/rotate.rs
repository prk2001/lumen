//! Orthogonal rotation: 90 / 180 / 270 degrees clockwise. No
//! interpolation needed.
//!
//! Arbitrary-angle rotation will land in Phase 3 alongside affine
//! warps; for Phase 1 we cover the easy 90-multiples.

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Frame, ParamKind, ParamSpec,
    ParamValues, PixelData, Result,
};
use tracing::instrument;

#[derive(Debug, Default)]
pub struct RotateOrtho;

const META: EffectMetadata = EffectMetadata {
    id: "lumen-fx-geometric.rotate_ortho",
    display_name: "Rotate (90°)",
    description: "Rotate clockwise by a multiple of 90 degrees.",
    category: Category::Geometric,
    version: 1,
};

const PARAMS: &[ParamSpec] = &[ParamSpec {
    id: "turns",
    display_name: "Turns",
    description: "Number of 90° clockwise turns: 1 = 90°, 2 = 180°, 3 = 270°.",
    kind: ParamKind::Int { default: 1, min: Some(0), max: Some(3) },
}];

impl Effect for RotateOrtho {
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
        let turns = params.get_int("turns").unwrap_or(1).rem_euclid(4) as u32;

        if turns == 0 {
            return Ok(input);
        }

        let cs = input.color_space.clone();
        let pts = input.pts;
        let w = input.width as usize;
        let h = input.height as usize;

        match input.data {
            PixelData::Rgba8(src) => {
                let (nw, nh, dst) = rotate_u8(&src, w, h, turns);
                Frame::new(nw, nh, PixelData::Rgba8(dst), cs, pts)
            }
            PixelData::Rgba16(src) => {
                let (nw, nh, dst) = rotate_u16(&src, w, h, turns);
                Frame::new(nw, nh, PixelData::Rgba16(dst), cs, pts)
            }
            PixelData::RgbaF32(src) => {
                let (nw, nh, dst) = rotate_f32(&src, w, h, turns);
                Frame::new(nw, nh, PixelData::RgbaF32(dst), cs, pts)
            }
        }
    }
}

macro_rules! make_rotate {
    ($name:ident, $t:ty, $z:expr) => {
        fn $name(src: &[$t], w: usize, h: usize, turns: u32) -> (u32, u32, Vec<$t>) {
            let (nw, nh) = match turns {
                1 | 3 => (h as u32, w as u32),
                _ => (w as u32, h as u32),
            };
            let mut dst = vec![$z; src.len()];
            let stride_src = w * 4;
            let stride_dst = nw as usize * 4;
            for y in 0..h {
                for x in 0..w {
                    let (nx, ny) = match turns {
                        1 => (h - 1 - y, x),
                        2 => (w - 1 - x, h - 1 - y),
                        3 => (y, w - 1 - x),
                        _ => (x, y),
                    };
                    let so = y * stride_src + x * 4;
                    let do_ = ny * stride_dst + nx * 4;
                    dst[do_..do_ + 4].copy_from_slice(&src[so..so + 4]);
                }
            }
            (nw, nh, dst)
        }
    };
}

make_rotate!(rotate_u8, u8, 0u8);
make_rotate!(rotate_u16, u16, 0u16);
make_rotate!(rotate_f32, f32, 0.0f32);

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, ParamValue};

    #[test]
    fn rotate_90_swaps_dims() {
        let r = RotateOrtho;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("turns", ParamValue::Int(1));
        p.validate_and_fill(r.parameters()).unwrap();

        let f =
            Frame::new(4, 2, PixelData::Rgba8(vec![0; 4 * 2 * 4]), ColorSpace::SRgb, None)
                .unwrap();
        let out = r.apply(&mut ctx, f, &p).unwrap();
        assert_eq!(out.width, 2);
        assert_eq!(out.height, 4);
    }

    #[test]
    fn rotate_360_identity() {
        let r = RotateOrtho;
        let mut ctx = Context::for_still_srgb();
        let mut p = ParamValues::new();
        p.insert("turns", ParamValue::Int(0));
        p.validate_and_fill(r.parameters()).unwrap();

        let data: Vec<u8> = (0..32).map(|i| i as u8).collect();
        let f = Frame::new(4, 2, PixelData::Rgba8(data.clone()), ColorSpace::SRgb, None).unwrap();
        let out = r.apply(&mut ctx, f, &p).unwrap();
        assert_eq!(out.width, 4);
        assert_eq!(out.height, 2);
        let PixelData::Rgba8(out_d) = out.data else { panic!() };
        assert_eq!(out_d, data);
    }
}
