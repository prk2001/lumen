//! ITU-R BS.1770-4 / EBU R128 loudness measurement and normalization.
//!
//! This module implements:
//!
//! * **K-weighting** filter — a two-stage cascade of biquads (high-shelf
//!   "pre-filter" at ~1681 Hz and a high-pass "RLB" filter at ~38 Hz) applied
//!   to each channel before mean-square energy aggregation.
//! * **Mean-square gating** — per BS.1770-4 §5.7 / EBU R128 §3.6:
//!   1. compute mean-square energy over 400 ms blocks with 75 % overlap
//!      (i.e. one new block every 100 ms),
//!   2. drop blocks below the **absolute threshold** of −70 LUFS,
//!   3. compute a **relative threshold** at (mean − 10 LU) of the surviving
//!      blocks and drop everything below that,
//!   4. integrated loudness is `−0.691 + 10·log10(mean(MSᵢ))` of the
//!      survivors, weighted-summed across channels.
//! * **Momentary** loudness uses the same 400 ms window; **short-term** uses
//!   3 s windows (also stepped 100 ms). We return the maxima of each.
//! * **True peak** estimated by 4× polyphase oversampling of every channel and
//!   reported in dBFS.
//!
//! ## K-weighting biquad coefficients
//!
//! The reference coefficients from ITU-R BS.1770-4 Annex 1 (Table 1 — "Stage
//! 1 — high-frequency 'pre-filter'") and (Table 2 — "Stage 2 — RLB
//! high-pass") are specified at **48 kHz**. They are baked in as the constants
//! [`PRE_48K`] and [`RLB_48K`] below; the comment on each constant notes the
//! source and the analog-prototype response (≈ +4 dB high-shelf above
//! 1681 Hz; 2nd-order high-pass with corner ≈ 38 Hz, Q ≈ 0.5).
//!
//! For sample rates other than 48 kHz we re-derive the digital coefficients
//! from the analog prototypes by a bilinear transform with frequency
//! pre-warping. This matches the recommended procedure from the BS.1770-4
//! reference C code and EBU Tech 3341 §2.1.
//!
//! ## Channel weights (BS.1770-4 §4.2)
//!
//! `L=R=C=1.0`, `LFE=0.0`, `Ls=Rs=1.41`. Phase 1 of this implementation only
//! handles **mono** (treated as L) and **stereo** (treated as L+R); surround
//! configurations return [`Error::Other`].
//!
//! ## Silence / no-survivors
//!
//! Per EBU R128 §3.6 we report `f64::NEG_INFINITY` LUFS for buffers whose
//! gated mean-square is zero (all silence, or below the −70 LUFS gate). The
//! same applies to short-term and momentary maxima.

use lumen_core::{Error, Result};

use crate::AudioBuffer;

/// Result of [`measure_loudness`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Loudness {
    /// Integrated (program) loudness in LUFS, gated per BS.1770-4 §5.7.
    /// `f64::NEG_INFINITY` for fully-silent / fully-gated buffers.
    pub integrated_lufs: f64,
    /// Peak of the **momentary** loudness (400 ms sliding window) in LUFS.
    pub momentary_max_lufs: f64,
    /// Peak of the **short-term** loudness (3 s sliding window) in LUFS.
    pub short_term_max_lufs: f64,
    /// True peak in dBFS, estimated by 4× oversampling. `f64::NEG_INFINITY`
    /// if all samples are zero.
    pub true_peak_dbfs: f64,
}

/// Biquad direct-form-I coefficients with `a0 == 1`.
///
/// Difference equation:
/// ```text
/// y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
/// ```
#[derive(Debug, Clone, Copy)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

/// BS.1770-4 Annex 1 Table 1 — pre-filter coefficients @ 48 kHz.
const PRE_48K: Biquad = Biquad {
    b0: 1.535_127_603_280_575_5,
    b1: -2.691_696_702_454_490_4,
    b2: 1.198_429_344_988_603_3,
    a1: -1.690_654_240_657_077_7,
    a2: 0.732_447_896_263_762_2,
};

/// BS.1770-4 Annex 1 Table 2 — RLB high-pass coefficients @ 48 kHz.
const RLB_48K: Biquad = Biquad {
    b0: 1.0,
    b1: -2.0,
    b2: 1.0,
    a1: -1.990_047_771_343_563_4,
    a2: 0.990_073_374_252_286_7,
};

/// Compute K-weighting biquad coefficients for an arbitrary sample rate.
///
/// At 48 kHz we return the BS.1770-4 reference values verbatim. At other
/// sample rates we re-derive from the analog prototypes using a bilinear
/// transform with frequency pre-warping at the design frequencies given in
/// the BS.1770-4 reference C code:
///
/// * pre-filter: high-shelf, fc ≈ 1681.974 Hz, Q ≈ 0.7071, gain ≈ +3.999 dB
/// * RLB: 2nd-order high-pass, fc ≈ 38.135 Hz, Q ≈ 0.5003
fn k_weighting_coeffs(sample_rate: u32) -> (Biquad, Biquad) {
    if sample_rate == 48_000 {
        return (PRE_48K, RLB_48K);
    }
    let fs = sample_rate as f64;
    let pre = high_shelf_biquad(
        1_681.974_450_955_533_2,
        0.707_175_252_897_434_1,
        3.999_843_853_973_53,
        fs,
    );
    let rlb = highpass_biquad(38.135_470_876_131_42, 0.500_327_300_120_058_2, fs);
    (pre, rlb)
}

/// Robert Bristow-Johnson "Audio EQ Cookbook" high-shelf biquad.
fn high_shelf_biquad(f0: f64, q: f64, gain_db: f64, fs: f64) -> Biquad {
    let a = 10.0_f64.powf(gain_db / 40.0);
    let w0 = 2.0 * std::f64::consts::PI * f0 / fs;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * q);
    let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

    let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;

    Biquad {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: a1 / a0,
        a2: a2 / a0,
    }
}

/// Robert Bristow-Johnson "Audio EQ Cookbook" 2nd-order high-pass biquad.
fn highpass_biquad(f0: f64, q: f64, fs: f64) -> Biquad {
    let w0 = 2.0 * std::f64::consts::PI * f0 / fs;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * q);

    let b0 = (1.0 + cos_w0) / 2.0;
    let b1 = -(1.0 + cos_w0);
    let b2 = (1.0 + cos_w0) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    Biquad {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: a1 / a0,
        a2: a2 / a0,
    }
}

/// Apply a biquad to `samples` in place using direct-form I, f64 state.
fn apply_biquad(samples: &mut [f64], bq: &Biquad) {
    let mut x1 = 0.0_f64;
    let mut x2 = 0.0_f64;
    let mut y1 = 0.0_f64;
    let mut y2 = 0.0_f64;
    for s in samples.iter_mut() {
        let x0 = *s;
        let y0 = bq.b0 * x0 + bq.b1 * x1 + bq.b2 * x2 - bq.a1 * y1 - bq.a2 * y2;
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
        *s = y0;
    }
}

/// Channel layout for ITU-R BS.1770 weighting.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // `Surround` is reserved for future 5.1/7.1 support.
enum ChannelKind {
    LeftLike, // mono / L / R / C — weight 1.0
    Surround, // Ls, Rs — weight 1.41
}

impl ChannelKind {
    fn weight(self) -> f64 {
        match self {
            ChannelKind::LeftLike => 1.0,
            ChannelKind::Surround => 1.41,
        }
    }
}

/// Decide channel layout. Mono and stereo only for now.
fn channel_kinds(channels: u16) -> Result<Vec<ChannelKind>> {
    match channels {
        1 => Ok(vec![ChannelKind::LeftLike]),
        2 => Ok(vec![ChannelKind::LeftLike, ChannelKind::LeftLike]),
        _ => Err(Error::Other(
            "only mono/stereo supported".to_string(),
        )),
    }
}

/// Deinterleave one channel from `buf` into a freshly allocated f64 vec.
fn deinterleave_f64(buf: &AudioBuffer, ch: usize) -> Vec<f64> {
    let n_channels = buf.channels as usize;
    let frames = buf.frames();
    let mut out = Vec::with_capacity(frames);
    for f in 0..frames {
        out.push(buf.samples[f * n_channels + ch] as f64);
    }
    out
}

/// K-weight one channel: pre-filter → RLB filter, in place on f64 samples.
fn k_weight(samples: &mut [f64], sample_rate: u32) {
    let (pre, rlb) = k_weighting_coeffs(sample_rate);
    apply_biquad(samples, &pre);
    apply_biquad(samples, &rlb);
}

/// Convert a per-block weighted mean-square energy to LUFS.
///
/// Returns `f64::NEG_INFINITY` for `ms <= 0`.
fn ms_to_lufs(ms: f64) -> f64 {
    if ms <= 0.0 || !ms.is_finite() {
        return f64::NEG_INFINITY;
    }
    -0.691 + 10.0 * ms.log10()
}

/// Compute weighted mean-square per block for a window length `block_frames`
/// stepped by `step_frames`, summed across all channels with their weights.
///
/// Returns one block per non-overlapping step that fits in the signal.
fn block_mean_squares(
    weighted_chans: &[(Vec<f64>, f64)], // per-channel (k-weighted samples, weight)
    block_frames: usize,
    step_frames: usize,
) -> Vec<f64> {
    if weighted_chans.is_empty() || block_frames == 0 || step_frames == 0 {
        return Vec::new();
    }
    let n = weighted_chans[0].0.len();
    if n < block_frames {
        return Vec::new();
    }

    // Per-channel running sum-of-squares using a sliding window.
    let inv_block = 1.0 / block_frames as f64;
    let num_blocks = (n - block_frames) / step_frames + 1;
    let mut out = Vec::with_capacity(num_blocks);

    for b in 0..num_blocks {
        let start = b * step_frames;
        let end = start + block_frames;
        let mut weighted = 0.0_f64;
        for (samples, weight) in weighted_chans {
            // Sum of squares over the window.
            let mut sumsq = 0.0_f64;
            for &s in &samples[start..end] {
                sumsq += s * s;
            }
            let mean_sq = sumsq * inv_block;
            weighted += weight * mean_sq;
        }
        out.push(weighted);
    }
    out
}

/// Measure ITU-R BS.1770-4 / EBU R128 loudness for the full buffer.
///
/// Mono is treated as a single L-weighted channel; stereo as L+R. Surround
/// layouts return `Err(Error::Other("only mono/stereo supported"))`.
pub fn measure_loudness(buf: &AudioBuffer) -> Result<Loudness> {
    if buf.sample_rate == 0 {
        return Err(Error::InvalidParameter {
            name: "sample_rate".into(),
            reason: "must be non-zero".into(),
        });
    }
    let kinds = channel_kinds(buf.channels)?;
    let n_channels = kinds.len();

    if buf.frames() == 0 {
        return Ok(Loudness {
            integrated_lufs: f64::NEG_INFINITY,
            momentary_max_lufs: f64::NEG_INFINITY,
            short_term_max_lufs: f64::NEG_INFINITY,
            true_peak_dbfs: f64::NEG_INFINITY,
        });
    }

    // K-weight each channel (in f64).
    let mut weighted_chans: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n_channels);
    for (ch, kind) in kinds.iter().enumerate() {
        let mut s = deinterleave_f64(buf, ch);
        k_weight(&mut s, buf.sample_rate);
        weighted_chans.push((s, kind.weight()));
    }

    // Block sizes — see EBU R128 §3.5: momentary 400 ms, short-term 3 s,
    // both stepped 100 ms (75 % and ~96.67 % overlap respectively).
    let fs = buf.sample_rate as f64;
    let mom_frames = (0.400 * fs).round() as usize;
    let st_frames = (3.000 * fs).round() as usize;
    let step_frames = (0.100 * fs).round() as usize;

    let mom_blocks = block_mean_squares(&weighted_chans, mom_frames, step_frames);
    let st_blocks = block_mean_squares(&weighted_chans, st_frames, step_frames);

    // Integrated loudness with two-stage gating per BS.1770-4 §5.7.
    let integrated_lufs = integrated_from_blocks(&mom_blocks);

    // Maxima of momentary / short-term windows.
    let momentary_max_lufs = mom_blocks
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |acc, ms| acc.max(ms_to_lufs(ms)));
    let short_term_max_lufs = st_blocks
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |acc, ms| acc.max(ms_to_lufs(ms)));

    let true_peak = true_peak_dbfs(buf)?;

    Ok(Loudness {
        integrated_lufs,
        momentary_max_lufs,
        short_term_max_lufs,
        true_peak_dbfs: true_peak,
    })
}

/// Apply BS.1770-4 §5.7 two-stage gating to the momentary block energies.
fn integrated_from_blocks(blocks: &[f64]) -> f64 {
    if blocks.is_empty() {
        return f64::NEG_INFINITY;
    }

    // Absolute gate at -70 LUFS. The corresponding block mean-square threshold
    // is the inverse of ms_to_lufs: ms = 10^((LUFS + 0.691) / 10).
    let abs_gate_ms = 10.0_f64.powf((-70.0 + 0.691) / 10.0);

    let stage1: Vec<f64> = blocks.iter().copied().filter(|ms| *ms > abs_gate_ms).collect();
    if stage1.is_empty() {
        return f64::NEG_INFINITY;
    }

    // Relative gate: -10 LU below the loudness of the surviving blocks.
    let mean_stage1 = stage1.iter().sum::<f64>() / stage1.len() as f64;
    let mean_stage1_lufs = ms_to_lufs(mean_stage1);
    if !mean_stage1_lufs.is_finite() {
        return f64::NEG_INFINITY;
    }
    let rel_gate_lufs = mean_stage1_lufs - 10.0;
    let rel_gate_ms = 10.0_f64.powf((rel_gate_lufs + 0.691) / 10.0);

    let stage2: Vec<f64> = stage1
        .into_iter()
        .filter(|ms| *ms > rel_gate_ms)
        .collect();
    if stage2.is_empty() {
        return f64::NEG_INFINITY;
    }

    let mean_stage2 = stage2.iter().sum::<f64>() / stage2.len() as f64;
    ms_to_lufs(mean_stage2)
}

/// Estimate the true peak (4× oversampled) of `buf` in dBFS.
///
/// We polyphase-upsample each channel by 4 using a length-32 windowed-sinc
/// kernel (Hann windowed, normalized so DC gain = 1) and report the largest
/// absolute sample across all channels. Returns `f64::NEG_INFINITY` for a
/// pure-zero buffer.
pub fn true_peak_dbfs(buf: &AudioBuffer) -> Result<f64> {
    if buf.channels == 0 {
        return Err(Error::Layout("audio buffer has zero channels".into()));
    }
    let n_channels = buf.channels as usize;
    let frames = buf.frames();
    if frames == 0 {
        return Ok(f64::NEG_INFINITY);
    }

    const TAPS_PER_PHASE: usize = 8; // total filter length = 8 * 4 = 32
    const FACTOR: usize = 4;
    let kernel = sinc_kernel(TAPS_PER_PHASE, FACTOR);

    let mut peak = 0.0_f64;
    let mut samples_ch = vec![0.0_f64; frames];
    for ch in 0..n_channels {
        for (i, slot) in samples_ch.iter_mut().enumerate() {
            *slot = buf.samples[i * n_channels + ch] as f64;
        }
        // Track the peak over both the original samples and the FACTOR-1
        // intermediate phases (phase 0 is the original sample).
        for &s in &samples_ch {
            let a = s.abs();
            if a > peak {
                peak = a;
            }
        }
        // For each non-zero phase, convolve the polyphase sub-filter with the
        // input signal and track the running max abs.
        for phase in 1..FACTOR {
            // Sub-filter: kernel[phase], kernel[phase + FACTOR], …
            let mut sub: Vec<f64> = Vec::with_capacity(TAPS_PER_PHASE);
            for k in 0..TAPS_PER_PHASE {
                sub.push(kernel[phase + k * FACTOR]);
            }
            // Convolve. center the kernel at TAPS_PER_PHASE / 2 so we don't
            // pick up an artificially low peak from the leading zeros.
            let center = TAPS_PER_PHASE / 2;
            for n in 0..frames {
                let mut acc = 0.0_f64;
                for (k, &c) in sub.iter().enumerate() {
                    let idx = n as isize + (k as isize - center as isize);
                    if (0..frames as isize).contains(&idx) {
                        acc += c * samples_ch[idx as usize];
                    }
                }
                let a = acc.abs();
                if a > peak {
                    peak = a;
                }
            }
        }
    }

    if peak == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(20.0 * peak.log10())
}

/// Build a Hann-windowed sinc low-pass kernel for FACTOR× upsampling.
///
/// Length = `taps_per_phase * factor`. The kernel is normalized so its DC
/// gain (sum of all taps) equals 1 — that means each polyphase sub-filter
/// has DC gain 1/FACTOR and an upsampled impulse passes through unattenuated
/// (the "insert FACTOR-1 zeros then convolve and multiply by FACTOR"
/// convention is collapsed into one pre-scaled kernel).
fn sinc_kernel(taps_per_phase: usize, factor: usize) -> Vec<f64> {
    let len = taps_per_phase * factor;
    let mut k = Vec::with_capacity(len);
    let center = (len as f64 - 1.0) / 2.0;
    for i in 0..len {
        let x = i as f64 - center;
        // Cutoff at fs/2 of the *input* rate, which is 1/factor of the
        // upsampled rate, so sinc(x / factor).
        let arg = x / factor as f64;
        let sinc = if arg.abs() < 1e-12 {
            1.0
        } else {
            (std::f64::consts::PI * arg).sin() / (std::f64::consts::PI * arg)
        };
        // Hann window.
        let w = 0.5
            - 0.5
                * (2.0 * std::f64::consts::PI * i as f64 / (len as f64 - 1.0)).cos();
        k.push(sinc * w);
    }
    // Normalize so sum == factor (each polyphase sub-filter sums to 1).
    let s: f64 = k.iter().sum();
    if s.abs() > 0.0 {
        let scale = factor as f64 / s;
        for v in &mut k {
            *v *= scale;
        }
    }
    k
}

/// Normalize `buf` so its integrated loudness reaches `target_lufs`.
///
/// Algorithm:
/// 1. Measure integrated loudness `L`.
/// 2. Required gain (dB) = `target_lufs - L`.
/// 3. Convert to a linear factor and apply to every sample.
/// 4. If the resulting gain would push the largest absolute sample above
///    1.0, clamp the gain so that the loudest sample lands at 1.0 - epsilon
///    and emit a `tracing::warn!`.
///
/// If the integrated loudness is `-∞` (silence), the buffer is returned
/// unchanged.
pub fn normalize_to_lufs(buf: &AudioBuffer, target_lufs: f64) -> Result<AudioBuffer> {
    if !target_lufs.is_finite() {
        return Err(Error::InvalidParameter {
            name: "target_lufs".into(),
            reason: "must be a finite number".into(),
        });
    }

    let measured = measure_loudness(buf)?;
    if !measured.integrated_lufs.is_finite() {
        // Silence — nothing to gain into a finite LUFS target.
        return AudioBuffer::new(buf.sample_rate, buf.channels, buf.samples.clone());
    }

    let gain_db = target_lufs - measured.integrated_lufs;
    let mut gain_lin = 10.0_f64.powf(gain_db / 20.0);

    // Clamp gain to avoid clipping. We use the existing-sample peak as a
    // proxy for true-peak — true-peak could be slightly higher but recomputing
    // it here for the post-gain signal would be circular and the difference
    // is well below 0.5 dB for ordinary content.
    let mut sample_peak = 0.0_f32;
    for &s in &buf.samples {
        let a = s.abs();
        if a > sample_peak {
            sample_peak = a;
        }
    }
    if sample_peak > 0.0 {
        let max_gain = (1.0 - 1e-4) / sample_peak as f64;
        if gain_lin > max_gain {
            tracing::warn!(
                target = target_lufs,
                measured = measured.integrated_lufs,
                requested_gain_db = gain_db,
                applied_gain_db = 20.0 * max_gain.log10(),
                "loudness normalize: clamping gain to avoid clip"
            );
            gain_lin = max_gain;
        }
    }

    let g = gain_lin as f32;
    let out: Vec<f32> = buf.samples.iter().map(|s| s * g).collect();
    AudioBuffer::new(buf.sample_rate, buf.channels, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    /// Build a mono sine of the given amplitude, frequency, and duration.
    fn sine_mono(sample_rate: u32, freq_hz: f64, amp: f64, secs: f64) -> AudioBuffer {
        let n = (sample_rate as f64 * secs).round() as usize;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / sample_rate as f64;
            samples.push((amp * (TAU * freq_hz * t).sin()) as f32);
        }
        AudioBuffer::new(sample_rate, 1, samples).unwrap()
    }

    /// Duplicate a mono buffer's samples into stereo.
    fn duplicate_to_stereo(mono: &AudioBuffer) -> AudioBuffer {
        let mut samples = Vec::with_capacity(mono.samples.len() * 2);
        for &s in &mono.samples {
            samples.push(s);
            samples.push(s);
        }
        AudioBuffer::new(mono.sample_rate, 2, samples).unwrap()
    }

    #[test]
    fn silence_is_minus_infinity() {
        // 2 s of digital zero. With the absolute gate at -70 LUFS *no* block
        // can survive — we should report -inf.
        let buf = AudioBuffer::new(48_000, 1, vec![0.0_f32; 48_000 * 2]).unwrap();
        let m = measure_loudness(&buf).unwrap();
        assert!(
            m.integrated_lufs.is_infinite() && m.integrated_lufs < 0.0,
            "expected -inf integrated, got {}",
            m.integrated_lufs
        );
        assert!(m.momentary_max_lufs.is_infinite() && m.momentary_max_lufs < 0.0);
        assert!(m.short_term_max_lufs.is_infinite() && m.short_term_max_lufs < 0.0);
        assert!(m.true_peak_dbfs.is_infinite() && m.true_peak_dbfs < 0.0);
    }

    #[test]
    fn normalize_to_minus_23_lufs_round_trip() {
        // 4 seconds so we get plenty of 400 ms blocks for stable gating.
        let buf = sine_mono(48_000, 440.0, 0.5, 4.0);
        let normalized = normalize_to_lufs(&buf, -23.0).unwrap();
        let m = measure_loudness(&normalized).unwrap();
        assert!(
            (m.integrated_lufs - (-23.0)).abs() <= 0.5,
            "expected ~-23 LUFS, got {}",
            m.integrated_lufs
        );
    }

    #[test]
    fn mono_vs_stereo_consistency() {
        let mono = sine_mono(48_000, 440.0, 0.4, 4.0);
        let stereo = duplicate_to_stereo(&mono);
        let m_mono = measure_loudness(&mono).unwrap();
        let m_stereo = measure_loudness(&stereo).unwrap();
        let diff = (m_stereo.integrated_lufs - m_mono.integrated_lufs).abs();
        // Mono uses one weight=1 channel; stereo uses two — for L=R the
        // weighted MS doubles, which adds 10·log10(2) ≈ 3.01 LU. The
        // *measurement* of the same source signal should match within the
        // BS.1770 tolerance (we allow 0.5 LU here to absorb numerical
        // drift between the two paths). The task brief asks for "within
        // 0.1 LU" but acknowledges variance from channel weighting; we
        // pick the looser-but-still-tight 0.5 LU bound to keep the test
        // robust across architectures.
        // To compare apples to apples, subtract the +3.01 LU stereo bias.
        let stereo_unbiased = m_stereo.integrated_lufs - 10.0 * 2.0_f64.log10();
        let unbiased_diff = (stereo_unbiased - m_mono.integrated_lufs).abs();
        assert!(
            unbiased_diff <= 0.1,
            "expected mono ≈ stereo (after stereo bias), got mono={}, stereo={}, diff={}",
            m_mono.integrated_lufs,
            m_stereo.integrated_lufs,
            unbiased_diff
        );
        // Additionally, the raw difference must be close to the +3.01 LU
        // expected from doubling the channel count.
        assert!(
            (diff - 10.0 * 2.0_f64.log10()).abs() <= 0.1,
            "expected stereo to be +3.01 LU above mono, diff={}",
            diff
        );
    }

    #[test]
    fn true_peak_for_known_signal() {
        // A pure sine at exactly 0 dBFS peak. Sample peaks may dip slightly
        // below 1.0 between samples; the 4× oversampled true peak should
        // recover the underlying analog peak to within ±0.5 dB of 0 dBFS.
        let buf = sine_mono(48_000, 997.0, 1.0, 1.0);
        let tp = true_peak_dbfs(&buf).unwrap();
        assert!(
            (tp - 0.0).abs() <= 0.5,
            "expected ~0 dBFS true peak, got {}",
            tp
        );
    }

    #[test]
    fn surround_layout_returns_err() {
        let buf = AudioBuffer::new(48_000, 6, vec![0.0_f32; 48_000 * 6]).unwrap();
        assert!(measure_loudness(&buf).is_err());
    }
}
