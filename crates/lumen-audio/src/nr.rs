//! Classical spectral-subtraction noise reduction.
//!
//! Algorithm:
//! 1. Split the signal into frames of `frame_size` samples with 50% overlap.
//! 2. Apply a Hann window to each frame.
//! 3. Compute the magnitude spectrum via FFT.
//! 4. Estimate the noise spectrum by averaging the magnitudes of the first
//!    `noise_estimate_secs` worth of frames.
//! 5. For each frame:
//!    `clean_mag = max(mag - over_subtract * noise_mag, floor * noise_mag)`,
//!    preserving the original phase.
//! 6. Inverse FFT, re-window, and overlap-add.
//!
//! Each channel is processed independently.

use std::sync::Arc;

use lumen_core::{Error, Result};
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::AudioBuffer;

/// Tunable parameters for [`spectral_subtract`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpectralNrParams {
    /// FFT size / analysis window length in samples. Must be a power of two
    /// and at least 4. Default: `1024`.
    pub frame_size: usize,
    /// Aggressiveness of the noise subtraction. `1.0` subtracts the noise
    /// estimate exactly; values above 1 over-subtract for cleaner output at
    /// the cost of musical-noise artifacts. Default: `1.5`.
    pub over_subtract: f32,
    /// Spectral floor as a fraction of the noise magnitude. Bins are never
    /// reduced below `floor * noise_mag`. Default: `0.05`.
    pub floor: f32,
    /// Duration in seconds, taken from the start of the signal, used to
    /// estimate the noise spectrum. Default: `0.5`.
    pub noise_estimate_secs: f32,
}

impl Default for SpectralNrParams {
    fn default() -> Self {
        Self {
            frame_size: 1024,
            over_subtract: 1.5,
            floor: 0.05,
            noise_estimate_secs: 0.5,
        }
    }
}

/// Apply spectral-subtraction noise reduction to every channel of `buf`.
pub fn spectral_subtract(buf: &AudioBuffer, params: &SpectralNrParams) -> Result<AudioBuffer> {
    validate_params(params)?;
    if buf.channels == 0 {
        return Err(Error::Layout("audio buffer has zero channels".into()));
    }

    let channels = buf.channels as usize;
    let frames = buf.frames();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(params.frame_size);
    let ifft = planner.plan_fft_inverse(params.frame_size);
    let window = hann_window(params.frame_size);

    let mut out_samples = vec![0.0_f32; buf.samples.len()];
    let mut deinterleaved = vec![0.0_f32; frames];

    for ch in 0..channels {
        for (i, slot) in deinterleaved.iter_mut().enumerate().take(frames) {
            *slot = buf.samples[i * channels + ch];
        }
        let processed = process_channel(
            &deinterleaved,
            buf.sample_rate,
            params,
            &window,
            fft.clone(),
            ifft.clone(),
        )?;
        for (i, &s) in processed.iter().enumerate().take(frames) {
            out_samples[i * channels + ch] = s;
        }
    }

    AudioBuffer::new(buf.sample_rate, buf.channels, out_samples)
}

fn validate_params(p: &SpectralNrParams) -> Result<()> {
    if p.frame_size < 4 {
        return Err(Error::InvalidParameter {
            name: "frame_size".into(),
            reason: "must be at least 4".into(),
        });
    }
    if !p.frame_size.is_power_of_two() {
        return Err(Error::InvalidParameter {
            name: "frame_size".into(),
            reason: "must be a power of two".into(),
        });
    }
    if !(p.over_subtract.is_finite() && p.over_subtract >= 0.0) {
        return Err(Error::InvalidParameter {
            name: "over_subtract".into(),
            reason: "must be a finite, non-negative number".into(),
        });
    }
    if !(p.floor.is_finite() && p.floor >= 0.0) {
        return Err(Error::InvalidParameter {
            name: "floor".into(),
            reason: "must be a finite, non-negative number".into(),
        });
    }
    if !(p.noise_estimate_secs.is_finite() && p.noise_estimate_secs > 0.0) {
        return Err(Error::InvalidParameter {
            name: "noise_estimate_secs".into(),
            reason: "must be a finite, positive number".into(),
        });
    }
    Ok(())
}

fn hann_window(n: usize) -> Vec<f32> {
    let denom = (n as f32 - 1.0).max(1.0);
    (0..n)
        .map(|i| {
            let x = std::f32::consts::PI * 2.0 * i as f32 / denom;
            0.5 - 0.5 * x.cos()
        })
        .collect()
}

fn process_channel(
    samples: &[f32],
    sample_rate: u32,
    params: &SpectralNrParams,
    window: &[f32],
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
) -> Result<Vec<f32>> {
    let frame_size = params.frame_size;
    let hop = frame_size / 2;
    let n = samples.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let num_frames = if n <= frame_size {
        1
    } else {
        (n - frame_size).div_ceil(hop) + 1
    };
    let half_spec = frame_size / 2 + 1;

    // Number of leading frames to average for the noise estimate.
    let noise_samples = (params.noise_estimate_secs * sample_rate as f32).round() as usize;
    let noise_frames = ((noise_samples.saturating_sub(frame_size)) / hop + 1).max(1);
    let noise_frames = noise_frames.min(num_frames);

    // Per-frame magnitude and phase.
    let mut mags: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
    let mut phases: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
    let mut scratch = vec![Complex32::default(); frame_size];

    for f in 0..num_frames {
        let start = f * hop;
        for i in 0..frame_size {
            let s = if start + i < n { samples[start + i] } else { 0.0 };
            scratch[i] = Complex32::new(s * window[i], 0.0);
        }
        fft.process(&mut scratch);
        let mut mag = Vec::with_capacity(half_spec);
        let mut ph = Vec::with_capacity(half_spec);
        for c in scratch.iter().take(half_spec) {
            mag.push((c.re * c.re + c.im * c.im).sqrt());
            ph.push(c.im.atan2(c.re));
        }
        mags.push(mag);
        phases.push(ph);
    }

    // Average magnitudes from the leading frames to form the noise estimate.
    let mut noise_mag = vec![0.0_f32; half_spec];
    for mag in mags.iter().take(noise_frames) {
        for (k, m) in mag.iter().enumerate() {
            noise_mag[k] += *m;
        }
    }
    if noise_frames > 0 {
        let inv = 1.0 / noise_frames as f32;
        for v in &mut noise_mag {
            *v *= inv;
        }
    }

    // Synthesize cleaned frames and overlap-add.
    let synth_len = (num_frames - 1) * hop + frame_size;
    let mut output = vec![0.0_f32; synth_len];
    let mut norm = vec![0.0_f32; synth_len];
    let inv_n = 1.0 / frame_size as f32;

    for f in 0..num_frames {
        let mag = &mags[f];
        let phase = &phases[f];

        // Reconstruct full complex spectrum with Hermitian symmetry.
        for i in 0..frame_size {
            let (m, p) = if i < half_spec {
                let m_clean = (mag[i] - params.over_subtract * noise_mag[i])
                    .max(params.floor * noise_mag[i]);
                (m_clean, phase[i])
            } else {
                let mirror = frame_size - i;
                let m_clean = (mag[mirror] - params.over_subtract * noise_mag[mirror])
                    .max(params.floor * noise_mag[mirror]);
                (m_clean, -phase[mirror])
            };
            scratch[i] = Complex32::new(m * p.cos(), m * p.sin());
        }
        ifft.process(&mut scratch);

        let start = f * hop;
        for i in 0..frame_size {
            let s = scratch[i].re * inv_n * window[i];
            output[start + i] += s;
            norm[start + i] += window[i] * window[i];
        }
    }

    // Compensate for the squared-window overlap-add envelope. For Hann at
    // 50% overlap this sum is 0.5 across every fully-overlapped sample. At
    // the edges where only one window contributes, the sum dips to (and
    // through) zero, which would amplify near-silent edge samples
    // catastrophically — clamp the divisor to that interior value.
    let min_norm = 0.5_f32;
    for (o, w_sum) in output.iter_mut().zip(norm.iter()) {
        let denom = w_sum.max(min_norm);
        *o /= denom;
    }

    output.truncate(n);
    Ok(output)
}
