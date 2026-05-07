//! Multi-frame super-resolution with sub-pixel registration.
//!
//! The technique federal image labs (FBI FAVIAU and similar) use to
//! recover license plates and small ROIs from CCTV: take 3-30 frames
//! of the same scene, register them sub-pixel against a reference,
//! upsample each, project onto a common high-resolution grid, and
//! fuse via robust mean.
//!
//! All steps are deterministic and reproducible from the audit log:
//!
//! 1. Decode every input to linear-light Float32 RGBA.
//! 2. Convert each to a luma surface (Rec. 709 weights) for shift
//!    estimation. Color is shifted using the same shift.
//! 3. For frame i (i > 0): phase-correlate luma_i against luma_0 to
//!    find the integer (dx, dy) shift; refine to sub-pixel via a
//!    parabolic fit on the cross-correlation peak's neighborhood.
//! 4. Upsample every frame by S× with Lanczos-3 (separable).
//! 5. Round each frame's sub-pixel shift to the nearest 1/S pixel,
//!    then translate the upsampled frame by S × shift in upsampled
//!    pixels (now integer-valued).
//! 6. Per-pixel: take the median of all registered upsampled samples.
//!    Median rejects outliers from registration failures, brief
//!    occlusions, and JPEG artifacts. (Mean is also available.)
//!
//! Why phase correlation: it's translation-invariant (matches well
//! across global lighting changes), sub-pixel via parabolic fit is
//! cited in Foroosh et al. (2002), and the whole pipeline is in
//! published peer-reviewed literature — a Daubert-passable witness
//! can defend every step.
//!
//! Limitations (intentional):
//!
//! - Only translation is estimated. Rotation / scale / perspective
//!   require feature-based registration (Phase 3+). For most CCTV
//!   the camera is fixed and the subject moves — a translational
//!   model is the right one.
//! - All inputs must have identical dimensions. Decode to RGBA
//!   first; this layer doesn't auto-rescale.
//! - Memory: holds N upsampled frames in RAM. For S=4 and 720x480
//!   inputs that's ~22 MB per frame. Fine up to ~30 frames.
//!
//! References:
//! - Foroosh, Zerubia, Berthod, "Extension of phase correlation to
//!   subpixel registration", IEEE TIP 2002.
//! - Fruchter & Hook, "Drizzle: a method for the linear
//!   reconstruction of undersampled images", PASP 2002.
//! - Keren, Peleg, Brada, "Image sequence enhancement using
//!   sub-pixel displacements", CVPR 1988.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context as _, Result};
use lumen_core::{ColorSpace, Frame, PixelData};
use rustfft::{num_complex::Complex32, FftPlanner};

/// Estimated sub-pixel shift (`dx`, `dy`) of a frame relative to the
/// reference frame, plus a quality score in [0, 1] (1 = perfect peak,
/// 0 = no detectable correspondence). Recorded per-frame so a reviewer
/// can spot misregistrations.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct FrameShift {
    pub dx: f32,
    pub dy: f32,
    pub peak_score: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum FuseMode {
    Mean,
    Median,
}

impl FuseMode {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "mean" => Self::Mean,
            "median" => Self::Median,
            other => {
                return Err(anyhow!(
                    "unknown SR fuse mode '{other}'. Try: mean, median."
                ));
            }
        })
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SrReport {
    pub input_paths: Vec<String>,
    pub n_frames: usize,
    pub input_w: u32,
    pub input_h: u32,
    pub scale: u32,
    pub output_w: u32,
    pub output_h: u32,
    pub fuse: &'static str,
    pub shifts: Vec<FrameShift>,
}

/// Run the full SR pipeline on a list of input image paths and produce
/// a single fused output frame. Returns the frame and a per-frame
/// shift report (for the audit log).
pub fn super_resolve_files(
    inputs: &[PathBuf],
    scale: u32,
    fuse: FuseMode,
) -> Result<(Frame, SrReport)> {
    if inputs.len() < 2 {
        return Err(anyhow!(
            "super-resolve needs at least 2 frames; got {}",
            inputs.len()
        ));
    }
    if !(2..=8).contains(&scale) {
        return Err(anyhow!("scale must be 2..=8; got {scale}"));
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
                    "input {} is {}x{}, expected {}x{} \
                     (super-resolve requires uniform input dimensions)",
                    p.display(), f.width, f.height, w, h
                ));
            }
        }
        frames.push(f.into_rgba_f32_linear());
    }
    let (w, h) = dims.unwrap();
    let n = frames.len();

    // 1. Build luma channels for shift estimation.
    let lumas: Vec<Vec<f32>> = frames
        .iter()
        .map(|f| {
            let buf = f.as_f32().expect("RgbaF32 after lift");
            extract_luma(buf, w as usize, h as usize)
        })
        .collect();

    // 2. Estimate per-frame shifts vs. frame 0.
    let mut shifts: Vec<FrameShift> = Vec::with_capacity(n);
    shifts.push(FrameShift { dx: 0.0, dy: 0.0, peak_score: 1.0 }); // reference
    for i in 1..n {
        let s = phase_correlate_subpixel(
            &lumas[0],
            &lumas[i],
            w as usize,
            h as usize,
        );
        shifts.push(s);
    }

    // 3. Upsample every frame Sx with Lanczos-3 (separable, on linear-light RGBA).
    let s = scale as usize;
    let uw = (w as usize) * s;
    let uh = (h as usize) * s;
    let mut up_frames: Vec<Vec<f32>> = Vec::with_capacity(n);
    for f in &frames {
        let src = f.as_f32().expect("RgbaF32");
        up_frames.push(lanczos_upsample_rgba(src, w as usize, h as usize, s));
    }

    // 4. Translate each upsampled frame so it registers onto the
    //    reference frame's grid. shifts[k] is the displacement
    //    "frame_k = shifted(frame_0, +S)"; to register frame_k onto
    //    frame_0 we translate it by -S. We round to the nearest
    //    1/S pixel, which after multiplication by S is an integer
    //    shift on the upsampled grid — fully deterministic, no
    //    resampling.
    let mut shifted: Vec<Vec<f32>> = Vec::with_capacity(n);
    for (k, up) in up_frames.iter().enumerate() {
        let dx_up = -(shifts[k].dx * s as f32).round() as i32;
        let dy_up = -(shifts[k].dy * s as f32).round() as i32;
        shifted.push(translate_rgba(up, uw, uh, dx_up, dy_up));
    }

    // 5. Fuse: per-pixel statistic across the N registered frames.
    let pixels = uw * uh * 4;
    let mut out = vec![0.0f32; pixels];
    match fuse {
        FuseMode::Mean => {
            let inv = 1.0 / n as f32;
            for buf in &shifted {
                for i in 0..pixels {
                    out[i] += buf[i];
                }
            }
            for v in &mut out { *v *= inv; }
        }
        FuseMode::Median => {
            let mut samples = vec![0.0f32; n];
            for i in 0..pixels {
                for k in 0..n { samples[k] = shifted[k][i]; }
                samples.sort_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                out[i] = if n % 2 == 1 {
                    samples[n / 2]
                } else {
                    0.5 * (samples[n / 2 - 1] + samples[n / 2])
                };
            }
        }
    }
    for v in &mut out { *v = v.clamp(0.0, 1.0); }

    let report = SrReport {
        input_paths: inputs.iter().map(|p| p.display().to_string()).collect(),
        n_frames: n,
        input_w: w,
        input_h: h,
        scale,
        output_w: uw as u32,
        output_h: uh as u32,
        fuse: match fuse { FuseMode::Mean => "mean", FuseMode::Median => "median" },
        shifts,
    };
    let frame = Frame::new(
        uw as u32, uh as u32,
        PixelData::RgbaF32(out),
        ColorSpace::LinearSRgb,
        None,
    ).map_err(|e| anyhow!("frame: {e}"))?;
    Ok((frame, report))
}

/// Rec. 709 luma extraction from interleaved linear-light RGBA.
fn extract_luma(rgba: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; w * h];
    for i in 0..w * h {
        let r = rgba[i * 4];
        let g = rgba[i * 4 + 1];
        let b = rgba[i * 4 + 2];
        y[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
    y
}

/// Phase correlation with sub-pixel parabolic refinement.
/// Returns the (dx, dy) shift such that translating `b` by (dx, dy)
/// best aligns it onto `a`. Sub-pixel resolution is Foroosh's
/// 2002 parabolic fit on the correlation peak's three-neighborhood.
fn phase_correlate_subpixel(
    a: &[f32], b: &[f32], w: usize, h: usize,
) -> FrameShift {
    // Pad to next pow-of-2 (rustfft handles non-pow-2 too, but pow-2
    // is faster and more cache-friendly).
    let fw = w.next_power_of_two().max(2);
    let fh = h.next_power_of_two().max(2);
    let n = fw * fh;
    let mut pa = vec![Complex32::new(0.0, 0.0); n];
    let mut pb = vec![Complex32::new(0.0, 0.0); n];
    // Apply Hann windowing to suppress boundary artifacts.
    let hx: Vec<f32> = (0..w)
        .map(|x| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * x as f32 / (w as f32 - 1.0)).cos())
        .collect();
    let hy: Vec<f32> = (0..h)
        .map(|y| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * y as f32 / (h as f32 - 1.0)).cos())
        .collect();
    for y in 0..h {
        for x in 0..w {
            let win = hx[x] * hy[y];
            pa[y * fw + x] = Complex32::new(a[y * w + x] * win, 0.0);
            pb[y * fw + x] = Complex32::new(b[y * w + x] * win, 0.0);
        }
    }

    // Forward FFT (row + column).
    fft2d(&mut pa, fw, fh, false);
    fft2d(&mut pb, fw, fh, false);
    // Cross-power spectrum: A * conj(B) / |A * conj(B)|.
    for i in 0..n {
        let cross = pa[i] * pb[i].conj();
        let mag = cross.norm() + 1e-12;
        pa[i] = cross / mag;
    }
    // Inverse FFT — peak is at the shift.
    fft2d(&mut pa, fw, fh, true);

    // Find the peak (real part), wrap-aware: shifts > N/2 are negative.
    let mut peak_x = 0usize;
    let mut peak_y = 0usize;
    let mut peak_v = f32::NEG_INFINITY;
    for y in 0..fh {
        for x in 0..fw {
            let v = pa[y * fw + x].re;
            if v > peak_v {
                peak_v = v;
                peak_x = x;
                peak_y = y;
            }
        }
    }
    // Sub-pixel parabolic refinement on the peak's neighborhood.
    let dx_sub = parabolic_subpixel(&pa, fw, fh, peak_x, peak_y, true);
    let dy_sub = parabolic_subpixel(&pa, fw, fh, peak_x, peak_y, false);
    // Wrap to signed shift, then negate. Sign convention:
    // PCM(A, B) with B(x) = A(x - x0) puts the inverse-FFT peak at
    // n = -x0 (not +x0). Verified analytically with the canonical
    // delta example: A=[1,0,0,0], B=[0,1,0,0] => peak at n=3 = -1
    // mod 4, not at n=1. We want the shift S such that
    // translating frame_i by (-S) registers it onto frame_0,
    // which means S equals the i→0 displacement = the value we
    // started with (+x0). So negate the recovered position.
    let mut dx = peak_x as f32 + dx_sub;
    let mut dy = peak_y as f32 + dy_sub;
    if dx > (fw as f32) / 2.0 { dx -= fw as f32; }
    if dy > (fh as f32) / 2.0 { dy -= fh as f32; }
    dx = -dx;
    dy = -dy;

    FrameShift { dx, dy, peak_score: peak_v.clamp(0.0, 1.0) }
}

fn parabolic_subpixel(
    grid: &[Complex32], fw: usize, fh: usize,
    px: usize, py: usize, x_axis: bool,
) -> f32 {
    let at = |x: usize, y: usize| grid[y * fw + x].re;
    let (a, b, c) = if x_axis {
        let xm = (px + fw - 1) % fw;
        let xp = (px + 1) % fw;
        (at(xm, py), at(px, py), at(xp, py))
    } else {
        let ym = (py + fh - 1) % fh;
        let yp = (py + 1) % fh;
        (at(px, ym), at(px, py), at(px, yp))
    };
    // Three-point parabolic fit; vertex offset relative to b.
    let denom = a - 2.0 * b + c;
    if denom.abs() < 1e-12 { 0.0 } else { 0.5 * (a - c) / denom }
}

/// 2D FFT in place. Uses rustfft's 1D primitive on rows then columns.
fn fft2d(buf: &mut [Complex32], w: usize, h: usize, inverse: bool) {
    let mut planner = FftPlanner::<f32>::new();
    let row_fft: Arc<dyn rustfft::Fft<f32>> = if inverse {
        planner.plan_fft_inverse(w)
    } else {
        planner.plan_fft_forward(w)
    };
    let col_fft: Arc<dyn rustfft::Fft<f32>> = if inverse {
        planner.plan_fft_inverse(h)
    } else {
        planner.plan_fft_forward(h)
    };
    // Row pass (in place per row).
    for y in 0..h {
        let row = &mut buf[y * w..(y + 1) * w];
        row_fft.process(row);
    }
    // Column pass: scratch per column.
    let mut col = vec![Complex32::new(0.0, 0.0); h];
    for x in 0..w {
        for y in 0..h {
            col[y] = buf[y * w + x];
        }
        col_fft.process(&mut col);
        for y in 0..h {
            buf[y * w + x] = col[y];
        }
    }
    if inverse {
        let inv = 1.0 / (w * h) as f32;
        for v in buf.iter_mut() { *v *= inv; }
    }
}

/// Lanczos-3 separable upsampler operating on interleaved RGBA float.
/// Output is `s` times wider and taller than the input.
fn lanczos_upsample_rgba(src: &[f32], w: usize, h: usize, s: usize) -> Vec<f32> {
    let uw = w * s;
    let uh = h * s;
    // Horizontal pass: src (w x h) -> interim (uw x h)
    let mut interim = vec![0.0f32; uw * h * 4];
    for y in 0..h {
        for ux in 0..uw {
            let sx = (ux as f32 + 0.5) / s as f32 - 0.5;
            let sx_floor = sx.floor() as i64;
            let mut sum = [0.0f32; 4];
            let mut wsum = 0.0f32;
            for k in -2..=3 {
                let xi = sx_floor + k;
                let xi_clamped = xi.clamp(0, w as i64 - 1) as usize;
                let t = sx - xi as f32;
                let wt = lanczos3(t);
                wsum += wt;
                let idx = (y * w + xi_clamped) * 4;
                sum[0] += src[idx] * wt;
                sum[1] += src[idx + 1] * wt;
                sum[2] += src[idx + 2] * wt;
                sum[3] += src[idx + 3] * wt;
            }
            let inv = if wsum.abs() > 1e-9 { 1.0 / wsum } else { 1.0 };
            let o = (y * uw + ux) * 4;
            interim[o] = sum[0] * inv;
            interim[o + 1] = sum[1] * inv;
            interim[o + 2] = sum[2] * inv;
            interim[o + 3] = sum[3] * inv;
        }
    }
    // Vertical pass: interim (uw x h) -> out (uw x uh)
    let mut out = vec![0.0f32; uw * uh * 4];
    for uy in 0..uh {
        let sy = (uy as f32 + 0.5) / s as f32 - 0.5;
        let sy_floor = sy.floor() as i64;
        for ux in 0..uw {
            let mut sum = [0.0f32; 4];
            let mut wsum = 0.0f32;
            for k in -2..=3 {
                let yi = sy_floor + k;
                let yi_clamped = yi.clamp(0, h as i64 - 1) as usize;
                let t = sy - yi as f32;
                let wt = lanczos3(t);
                wsum += wt;
                let idx = (yi_clamped * uw + ux) * 4;
                sum[0] += interim[idx] * wt;
                sum[1] += interim[idx + 1] * wt;
                sum[2] += interim[idx + 2] * wt;
                sum[3] += interim[idx + 3] * wt;
            }
            let inv = if wsum.abs() > 1e-9 { 1.0 / wsum } else { 1.0 };
            let o = (uy * uw + ux) * 4;
            out[o] = sum[0] * inv;
            out[o + 1] = sum[1] * inv;
            out[o + 2] = sum[2] * inv;
            out[o + 3] = sum[3] * inv;
        }
    }
    out
}

fn lanczos3(x: f32) -> f32 {
    let a = 3.0;
    let ax = x.abs();
    if ax < 1e-9 { return 1.0; }
    if ax >= a { return 0.0; }
    let pix = std::f32::consts::PI * x;
    let pix_a = pix / a;
    (a * pix.sin() * pix_a.sin()) / (pix * pix)
}

/// Translate an RGBA buffer by (dx, dy) integer pixels, padding with
/// repeat-edge for samples that fall outside.
fn translate_rgba(src: &[f32], w: usize, h: usize, dx: i32, dy: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h * 4];
    for y in 0..h {
        let sy = (y as i32 - dy).clamp(0, h as i32 - 1) as usize;
        for x in 0..w {
            let sx = (x as i32 - dx).clamp(0, w as i32 - 1) as usize;
            let so = (sy * w + sx) * 4;
            let oo = (y * w + x) * 4;
            out[oo]     = src[so];
            out[oo + 1] = src[so + 1];
            out[oo + 2] = src[so + 2];
            out[oo + 3] = src[so + 3];
        }
    }
    out
}

/// CLI entry point: run SR, write the output, print the report.
pub fn cmd_super_resolve(
    inputs: &[PathBuf],
    output: &Path,
    scale: u32,
    fuse_str: &str,
) -> Result<()> {
    let fuse = FuseMode::parse(fuse_str)?;
    let (frame, report) = super_resolve_files(inputs, scale, fuse)?;
    lumen_io::encode_image(
        frame,
        output,
        lumen_io::ImageEncodeOptions::default(),
    )
    .with_context(|| format!("encode {}", output.display()))?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    eprintln!(
        "super-resolve: {} frames, {}x scale, fuse={} -> {}x{} -> {}",
        report.n_frames, scale, fuse_str,
        report.output_w, report.output_h, output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lanczos3_kernel_basic_properties() {
        // L(0) = 1 (exact), L(±k) = 0 for nonzero integers in the
        // support, and L(x) = 0 for |x| >= 3 (compact support).
        assert!((lanczos3(0.0) - 1.0).abs() < 1e-6);
        for k in [-2i32, -1, 1, 2] {
            assert!(lanczos3(k as f32).abs() < 1e-5,
                    "L({k}) should vanish at nonzero integers, got {}",
                    lanczos3(k as f32));
        }
        assert_eq!(lanczos3(3.0), 0.0);
        assert_eq!(lanczos3(-5.0), 0.0);
    }

    #[test]
    fn translate_zero_is_identity() {
        let src: Vec<f32> = (0..16 * 4).map(|i| i as f32).collect();
        let out = translate_rgba(&src, 4, 4, 0, 0);
        assert_eq!(src, out);
    }

    #[test]
    fn phase_correlation_recovers_synthetic_shift() {
        // 128x128 luma with multi-frequency content (Hann windowing
        // can wipe out single-frequency 32x32 inputs). Shift it by
        // a known integer amount and verify recovery within 1 px.
        let n: usize = 128;
        let mut a = vec![0.0f32; n * n];
        for y in 0..n {
            for x in 0..n {
                let v = (x as f32 * 0.4).sin() * 0.3
                    + (y as f32 * 0.7).cos() * 0.3
                    + ((x as f32 + y as f32) * 0.13).sin() * 0.2
                    + 0.5;
                a[y * n + x] = v.clamp(0.0, 1.0);
            }
        }
        // Shift A by (+5, -3) into B.
        let dx_true = 5i32; let dy_true = -3i32;
        let mut b = vec![0.0f32; n * n];
        for y in 0..n {
            for x in 0..n {
                let sx = (x as i32 - dx_true).clamp(0, n as i32 - 1) as usize;
                let sy = (y as i32 - dy_true).clamp(0, n as i32 - 1) as usize;
                b[y * n + x] = a[sy * n + sx];
            }
        }
        let s = phase_correlate_subpixel(&a, &b, n, n);
        assert!(
            (s.dx - dx_true as f32).abs() <= 1.0
                && (s.dy - dy_true as f32).abs() <= 1.0,
            "expected ~({}, {}), got ({}, {}) score {}",
            dx_true, dy_true, s.dx, s.dy, s.peak_score,
        );
    }

    #[test]
    fn super_resolve_rejects_single_frame() {
        let r = super_resolve_files(&[PathBuf::from("/nope.png")], 2, FuseMode::Mean);
        assert!(r.is_err());
    }
}
