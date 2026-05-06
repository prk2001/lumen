//! Inputs to the pipeline — files, sequences, streams.
//!
//! A [`Project`](crate::project::Project) holds a list of [`Asset`]s and
//! references them by [`AssetId`] from graph nodes. The actual decode
//! lives in `lumen-io`.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::color::ColorSpace;
use crate::time::Rational;

/// Stable identifier for an asset within a project.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AssetId(pub Uuid);

impl AssetId {
    pub fn new() -> Self { Self(Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext))) }
}

impl Default for AssetId {
    fn default() -> Self { Self::new() }
}

impl std::fmt::Display for AssetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// What kind of asset this is.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetKind {
    /// Single still image (PNG, JPEG, TIFF, RAW, HEIC, etc.).
    StillImage,
    /// Video file (MP4, MOV, MKV, WebM, etc.).
    Video,
    /// Numbered image sequence (DPX, EXR, RAW).
    ImageSequence,
    /// Audio-only file (WAV, FLAC, MP3, etc.).
    Audio,
}

/// Probed metadata for an asset. Populated by `lumen-io::probe`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssetMetadata {
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// Total frame count if known.
    pub frame_count: Option<u64>,
    /// Frame rate if known.
    pub frame_rate: Option<Rational>,
    /// Duration in seconds if known.
    pub duration_secs: Option<f64>,
    /// Codec name (`"h264"`, `"prores"`, …).
    pub codec: Option<String>,
    /// Container format (`"mp4"`, `"mov"`, …).
    pub container: Option<String>,
    /// Bits per channel (8, 10, 12, 16, 32 …).
    pub bit_depth: u8,
    /// Channel count (3 = RGB, 4 = RGBA, 1 = monochrome).
    pub channels: u8,
    /// Detected color space; `None` if unspecified.
    pub color_space: Option<ColorSpace>,
    /// Audio sample rate in Hz, if any.
    pub audio_sample_rate: Option<u32>,
    /// Audio channel count.
    pub audio_channels: Option<u8>,
}

impl AssetMetadata {
    pub const UNKNOWN: AssetMetadata = AssetMetadata {
        width: 0,
        height: 0,
        frame_count: None,
        frame_rate: None,
        duration_secs: None,
        codec: None,
        container: None,
        bit_depth: 8,
        channels: 4,
        color_space: None,
        audio_sample_rate: None,
        audio_channels: None,
    };
}

/// A resolved input source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Asset {
    pub id: AssetId,
    /// Source URI (`file:///…`, `http://…`, `s3://…`).
    pub uri: String,
    /// Display name shown in UI.
    pub display_name: String,
    pub kind: AssetKind,
    /// BLAKE3 hash of the source bytes, when known. Hex encoded with
    /// the `blake3:` prefix, e.g. `"blake3:af3…"`.
    pub hash: Option<String>,
    pub metadata: AssetMetadata,
}

impl Asset {
    pub fn new(uri: impl Into<String>, display_name: impl Into<String>, kind: AssetKind) -> Self {
        Self {
            id: AssetId::new(),
            uri: uri.into(),
            display_name: display_name.into(),
            kind,
            hash: None,
            metadata: AssetMetadata::UNKNOWN,
        }
    }
}
