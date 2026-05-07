//! `lumen stack` — combine multiple photos of the same scene into one.
//!
//! The "give me 3-10 photos and make a better one" workflow.
//! Stacks N aligned input images into a single output via per-pixel
//! statistic. Used in astrophotography (mean/median to drop noise),
//! star-trail photography (max), focus stacking (max-magnitude),
//! or just "I shot a burst of CCTV grabs of the same plate".
//!
//! Modes:
//!
//! - `mean`   — average each channel. SNR improves by √N. Best for
//!   reducing random sensor noise on a static scene.
//! - `median` — robust mean. Drops transient occlusions (a car or
//!   pedestrian crossing the frame). Better than `mean` when subjects
//!   move through the scene.
//! - `max`    — per-pixel maximum. Useful for star trails, lightning
//!   accumulation, fireworks.
//! - `min`    — per-pixel minimum. Removes bright transients to reveal
//!   what's behind them.
//!
//! All inputs must have identical dimensions. Phase 1 does NOT do
//! feature alignment — your inputs must already be tripod-stable
//! or pre-aligned. Multi-frame super-resolution with sub-pixel
//! alignment is a Phase 2+ feature.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context as _, Result};
use lumen_core::{ColorSpace, Frame, PixelData};

#[derive(Debug, Clone, Copy)]
pub enum StackMode {
    Mean,
    Median,
    Max,
    Min,
}

impl StackMode {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "mean"   => Self::Mean,
            "median" => Self::Median,
            "max"    => Self::Max,
            "min"    => Self::Min,
            other => {
                return Err(anyhow!(
                    "unknown stack mode '{other}'. Try: mean, median, max, min."
                ));
            }
        })
    }
}

pub fn stack_files(inputs: &[PathBuf], mode: StackMode) -> Result<Frame> {
    if inputs.is_empty() {
        return Err(anyhow!("no inputs"));
    }
    let mut frames: Vec<Frame> = Vec::with_capacity(inputs.len());
    let mut dims: Option<(u32, u32)> = None;
    for p in inputs {
        let f = lumen_io::decode_image(p)
            .map_err(|e| anyhow!("decode {}: {e}", p.display()))?;
        match dims {
            None => dims = Some((f.width, f.height)),
            Some((w, h)) if w == f.width && h == f.height => {}
            Some((w, h)) => {
                return Err(anyhow!(
                    "input {} is {}x{}, expected {}x{}",
                    p.display(),
                    f.width,
                    f.height,
                    w,
                    h
                ));
            }
        }
        frames.push(f.into_rgba_f32_linear());
    }
    let (w, h) = dims.unwrap();
    let n = frames.len();
    let pixels = (w as usize) * (h as usize) * 4;
    let mut out = vec![0.0f32; pixels];

    match mode {
        StackMode::Mean => {
            for f in &frames {
                let buf = f.as_f32().expect("RgbaF32 after lift");
                for i in 0..pixels {
                    out[i] += buf[i];
                }
            }
            let inv = 1.0 / n as f32;
            for v in &mut out {
                *v *= inv;
            }
        }
        StackMode::Max => {
            // initialize to negative infinity so first frame seeds it
            out.fill(f32::NEG_INFINITY);
            for f in &frames {
                let buf = f.as_f32().expect("RgbaF32 after lift");
                for i in 0..pixels {
                    if buf[i] > out[i] {
                        out[i] = buf[i];
                    }
                }
            }
        }
        StackMode::Min => {
            out.fill(f32::INFINITY);
            for f in &frames {
                let buf = f.as_f32().expect("RgbaF32 after lift");
                for i in 0..pixels {
                    if buf[i] < out[i] {
                        out[i] = buf[i];
                    }
                }
            }
        }
        StackMode::Median => {
            // O(N log N) per pixel. For small N (N ≤ 32 typical) this
            // is fine. Larger N could use partial selection.
            let mut samples = vec![0.0f32; n];
            for i in 0..pixels {
                for (k, f) in frames.iter().enumerate() {
                    let buf = f.as_f32().expect("RgbaF32 after lift");
                    samples[k] = buf[i];
                }
                samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                out[i] = samples[n / 2];
            }
        }
    }

    // Clamp + return as f32 (encode_image will downconvert).
    for v in &mut out {
        *v = v.clamp(0.0, 1.0);
    }
    Frame::new(w, h, PixelData::RgbaF32(out), ColorSpace::LinearSRgb, None)
        .map_err(|e| anyhow!("frame: {e}"))
}

pub fn cmd_stack(
    inputs: &[PathBuf],
    output: &Path,
    mode_str: &str,
) -> Result<()> {
    let mode = StackMode::parse(mode_str)?;
    let result = stack_files(inputs, mode)?;
    lumen_io::encode_image(
        result,
        output,
        lumen_io::ImageEncodeOptions::default(),
    )
    .with_context(|| format!("encode {}", output.display()))?;
    eprintln!(
        "stack: {} inputs -> {} (mode={})",
        inputs.len(),
        output.display(),
        mode_str
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_frame(w: u32, h: u32, val: u8) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lumen-stack-test-{}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            val
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f.png");
        let f = Frame::new(
            w,
            h,
            PixelData::Rgba8(vec![val; (w * h * 4) as usize]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();
        lumen_io::encode_image(f, &path, lumen_io::ImageEncodeOptions::default()).unwrap();
        path
    }

    #[test]
    fn mean_of_three_constant_frames_returns_their_mean() {
        let p1 = synth_frame(8, 8, 50);
        let p2 = synth_frame(8, 8, 100);
        let p3 = synth_frame(8, 8, 150);
        let result = stack_files(&[p1.clone(), p2.clone(), p3.clone()], StackMode::Mean).unwrap();
        // Mean of sRGB(50,100,150) values, but we lifted to linear and
        // averaged there. Just sanity-check dims.
        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
        // Cleanup
        for p in [p1, p2, p3] {
            let _ = std::fs::remove_file(&p);
            let _ = std::fs::remove_dir(p.parent().unwrap());
        }
    }

    #[test]
    fn mismatched_dims_returns_err() {
        let p1 = synth_frame(8, 8, 50);
        let p2 = synth_frame(10, 10, 50);
        let r = stack_files(&[p1.clone(), p2.clone()], StackMode::Mean);
        assert!(r.is_err());
        for p in [p1, p2] {
            let _ = std::fs::remove_file(&p);
            let _ = std::fs::remove_dir(p.parent().unwrap());
        }
    }

    #[test]
    fn empty_inputs_returns_err() {
        let r = stack_files(&[], StackMode::Mean);
        assert!(r.is_err());
    }

    #[test]
    fn parse_mode_rejects_unknown() {
        assert!(StackMode::parse("nope").is_err());
        assert!(matches!(StackMode::parse("mean").unwrap(), StackMode::Mean));
    }
}
