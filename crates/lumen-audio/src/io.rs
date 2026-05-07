//! WAV read/write helpers backed by [`hound`].
//!
//! Reading converts integer PCM (8/16/24/32-bit) and float WAVs to interleaved
//! `f32` samples in `[-1.0, 1.0]`. Writing always produces 16-bit PCM.

use std::path::{Path, PathBuf};

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use lumen_core::{Error, Result};

use crate::AudioBuffer;

/// Read a WAV file and return an [`AudioBuffer`] of interleaved `f32` samples.
///
/// Supports 8/16/24/32-bit integer PCM and 32-bit float WAVs.
pub fn read_wav<P: AsRef<Path>>(path: P) -> Result<AudioBuffer> {
    let path_ref = path.as_ref();
    let mut reader = WavReader::open(path_ref)
        .map_err(|e| decode_at(path_ref, format!("hound open failed: {e}")))?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => match spec.bits_per_sample {
            32 => reader
                .samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| decode_at(path_ref, format!("read float samples: {e}")))?,
            other => {
                return Err(Error::UnsupportedFormat(format!(
                    "WAV float sample width {other} not supported (only 32)"
                )))
            }
        },
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            // Maximum amplitude for an N-bit signed integer is 2^(N-1).
            // For 8-bit WAVs hound reports unsigned in spec but yields signed
            // i32 via the .samples::<i32>() iterator (it normalizes).
            let max = (1i64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| decode_at(path_ref, format!("read int samples: {e}")))?
        }
    };

    AudioBuffer::new(spec.sample_rate, spec.channels, samples)
}

/// Write an [`AudioBuffer`] to disk as a 16-bit PCM WAV.
pub fn write_wav<P: AsRef<Path>>(buf: &AudioBuffer, path: P) -> Result<()> {
    let path_ref = path.as_ref();
    let spec = WavSpec {
        channels: buf.channels,
        sample_rate: buf.sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path_ref, spec)
        .map_err(|e| encode_at(path_ref, format!("hound create failed: {e}")))?;

    for &s in &buf.samples {
        // Clamp to [-1, 1] then scale to i16 range.
        let clamped = s.clamp(-1.0, 1.0);
        let scaled = (clamped * i16::MAX as f32).round() as i32;
        let v = scaled.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        writer
            .write_sample(v)
            .map_err(|e| encode_at(path_ref, format!("write sample failed: {e}")))?;
    }
    writer
        .finalize()
        .map_err(|e| encode_at(path_ref, format!("finalize failed: {e}")))?;
    Ok(())
}

fn decode_at(path: &Path, msg: String) -> Error {
    Error::decode_at(PathBuf::from(path), msg)
}

fn encode_at(path: &Path, msg: String) -> Error {
    Error::encode_at(PathBuf::from(path), msg)
}
