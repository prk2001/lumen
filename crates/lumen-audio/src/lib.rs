//! # lumen-audio
//!
//! Audio enhancement primitives for Lumen.
//!
//! This crate currently provides:
//!
//! * [`AudioBuffer`] — interleaved 32-bit float PCM in `[-1.0, 1.0]`.
//! * [`read_wav`] / [`write_wav`] — basic WAV I/O backed by [`hound`].
//! * [`spectral_subtract`] — classical spectral-subtraction noise reduction
//!   (STFT, Hann window, 50% overlap-add) configured via [`SpectralNrParams`].
//!
//! Audio is not yet wired into Lumen's [`Effect`](lumen_core::Effect) /
//! [`Frame`](lumen_core::Frame) pipeline; this crate is currently a plain
//! library that operates on owned buffers.

#![forbid(unsafe_op_in_unsafe_fn)]

mod io;
pub mod loudness;
mod nr;

pub use io::{read_wav, write_wav};
pub use loudness::{measure_loudness, normalize_to_lufs, true_peak_dbfs, Loudness};
pub use nr::{spectral_subtract, SpectralNrParams};

use lumen_core::{Error, Result};

/// Interleaved 32-bit float PCM audio in the range `[-1.0, 1.0]`.
///
/// Samples are interleaved by channel: for stereo with `samples = [L0, R0,
/// L1, R1, ...]`. The number of frames is `samples.len() / channels as usize`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioBuffer {
    /// Sample rate in Hz (e.g. `44_100`, `48_000`).
    pub sample_rate: u32,
    /// Channel count. `1` = mono, `2` = stereo, etc.
    pub channels: u16,
    /// Interleaved sample data.
    pub samples: Vec<f32>,
}

impl AudioBuffer {
    /// Build an [`AudioBuffer`], validating that the sample count is a
    /// multiple of the channel count.
    pub fn new(sample_rate: u32, channels: u16, samples: Vec<f32>) -> Result<Self> {
        if channels == 0 {
            return Err(Error::InvalidParameter {
                name: "channels".into(),
                reason: "must be non-zero".into(),
            });
        }
        if sample_rate == 0 {
            return Err(Error::InvalidParameter {
                name: "sample_rate".into(),
                reason: "must be non-zero".into(),
            });
        }
        if !samples.len().is_multiple_of(channels as usize) {
            return Err(Error::Layout(format!(
                "sample count {} is not a multiple of channel count {}",
                samples.len(),
                channels
            )));
        }
        Ok(Self { sample_rate, channels, samples })
    }

    /// Number of audio frames (per-channel sample groups).
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels as usize
        }
    }

    /// Total duration in seconds.
    pub fn duration_secs(&self) -> f32 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.frames() as f32 / self.sample_rate as f32
        }
    }
}

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lumen-audio-test-{}-{}-{}",
            std::process::id(),
            name,
            // crude unique suffix per call site
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        p
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
        (sum_sq / samples.len() as f32).sqrt()
    }

    #[test]
    fn wav_round_trip_preserves_samples_within_quantization() {
        // Build a mono sine wave at 440 Hz.
        let sample_rate: u32 = 44_100;
        let frames = 4096;
        let mut samples = Vec::with_capacity(frames);
        for n in 0..frames {
            let t = n as f32 / sample_rate as f32;
            samples.push(0.5 * (TAU * 440.0 * t).sin());
        }
        let buf = AudioBuffer::new(sample_rate, 1, samples.clone()).unwrap();

        let path = tmp_path("roundtrip.wav");
        write_wav(&buf, &path).unwrap();
        let back = read_wav(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(back.sample_rate, buf.sample_rate);
        assert_eq!(back.channels, buf.channels);
        assert_eq!(back.samples.len(), buf.samples.len());
        // 16-bit quantization step is ~ 1/32768; allow a few LSBs.
        let tol = 4.0 / 32_768.0;
        for (a, b) in buf.samples.iter().zip(back.samples.iter()) {
            assert!(
                (a - b).abs() <= tol,
                "round-trip diff {} exceeds tolerance {}",
                (a - b).abs(),
                tol
            );
        }
    }

    #[test]
    fn dc_signal_remains_low_energy_after_spectral_subtract() {
        // A constant-DC signal has all energy at bin 0; with windowing the
        // spectrum is dominated by a small set of bins. The processed output
        // should still have very low broadband energy because the noise
        // estimate captures that spectrum exactly.
        let sample_rate: u32 = 16_000;
        let samples = vec![0.25_f32; sample_rate as usize]; // 1 s of DC
        let buf = AudioBuffer::new(sample_rate, 1, samples).unwrap();

        let params = SpectralNrParams::default();
        let out = spectral_subtract(&buf, &params).unwrap();

        assert_eq!(out.samples.len(), buf.samples.len());
        // Output RMS should be far below the input RMS (which is 0.25).
        let r_in = rms(&buf.samples);
        let r_out = rms(&out.samples);
        assert!(r_out < r_in * 0.25, "expected r_out << r_in, got in={r_in} out={r_out}");
    }

    #[test]
    fn white_noise_rms_drops_after_spectral_subtract() {
        let sample_rate: u32 = 16_000;
        let n = sample_rate as usize; // 1 s
        // Deterministic pseudo-random white noise (xorshift) so the test is
        // hermetic without pulling in the `rand` crate.
        let mut state: u32 = 0x1234_5678;
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Map to [-0.5, 0.5).
            let f = (state as f32 / u32::MAX as f32) - 0.5;
            samples.push(f);
        }
        let buf = AudioBuffer::new(sample_rate, 1, samples).unwrap();

        let params = SpectralNrParams::default();
        let out = spectral_subtract(&buf, &params).unwrap();

        let r_in = rms(&buf.samples);
        let r_out = rms(&out.samples);
        assert!(
            r_out < r_in,
            "expected RMS to decrease after NR; in={r_in}, out={r_out}"
        );
    }

    #[test]
    fn invalid_frame_size_returns_err() {
        let buf = AudioBuffer::new(16_000, 1, vec![0.0; 16_000]).unwrap();
        let params = SpectralNrParams { frame_size: 0, ..Default::default() };
        assert!(spectral_subtract(&buf, &params).is_err());

        let params = SpectralNrParams { frame_size: 1023, ..Default::default() };
        assert!(spectral_subtract(&buf, &params).is_err());
    }

    #[test]
    fn mismatched_layout_returns_err() {
        // 5 samples for a 2-channel buffer is not a whole frame multiple.
        let res = AudioBuffer::new(16_000, 2, vec![0.0; 5]);
        assert!(res.is_err());

        // Zero channels is rejected.
        let res = AudioBuffer::new(16_000, 0, vec![0.0; 4]);
        assert!(res.is_err());
    }
}
