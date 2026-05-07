//! Video decode via FFmpeg (`ffmpeg-next` 8.x).
//!
//! Phase 1 baseline:
//! * Header-only [`probe_video`] — opens the container, reads metadata,
//!   and returns without touching any compressed frames.
//! * [`decode_video_frame`] — seek to a single absolute frame index
//!   and return that frame as packed RGBA8 / sRGB.
//! * [`decode_video_range`] — iterate `[start, end)` and dispatch each
//!   frame through a caller callback. Returns the count delivered.
//!
//! All decoded frames are converted to packed RGBA8 with sRGB primaries
//! using `swscale`. Effects can lift to scene-linear via
//! [`Frame::into_rgba_f32_linear`] if they need float math.
//!
//! ## Build requirements
//!
//! `ffmpeg-next` is a thin wrapper over `ffmpeg-sys-next`, which uses
//! pkg-config to find the system FFmpeg shared libraries. On macOS with
//! Homebrew, ensure `pkg-config` is installed and the FFmpeg `.pc` files
//! are reachable, e.g.:
//!
//! ```sh
//! brew install pkg-config
//! export PKG_CONFIG_PATH=/usr/local/opt/ffmpeg/lib/pkgconfig:$PKG_CONFIG_PATH
//! ```
//!
//! Tested against system FFmpeg 8.1.

use std::path::Path;
use std::sync::Once;

use ffmpeg_next as ffmpeg;
use ffmpeg::format::Pixel;
use ffmpeg::media::Type as MediaType;
use ffmpeg::software::scaling::{context::Context as Scaler, flag::Flags as ScalerFlags};
use ffmpeg::util::frame::video::Video as AvVideo;
use ffmpeg::Rescale;
use lumen_core::{
    AssetMetadata, ColorSpace, Error, Frame, PixelData, Rational, Result,
};
use tracing::{debug, instrument};

/// Header-only probe result for a video container.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoProbe {
    /// Pixel `(width, height)`.
    pub dims: (u32, u32),
    /// Codec short name (e.g. `"h264"`).
    pub codec: String,
    /// Container format short name (e.g. `"mp4"`), if known.
    pub container: Option<String>,
    /// Frame rate as a rational (`Rational::new(0, 1)` if unknown).
    pub fps: Rational,
    /// Total frame count, if the demuxer reports it.
    pub frame_count: Option<u64>,
    /// Duration in seconds, if known.
    pub duration_secs: Option<f64>,
    /// Bits per component as best-effort from the pixel format descriptor.
    pub bit_depth: u8,
    /// Color space derived from primaries + transfer characteristic.
    pub color_space: Option<ColorSpace>,
}

impl VideoProbe {
    /// Lossy conversion to the workspace's [`AssetMetadata`].
    pub fn into_asset_metadata(self) -> AssetMetadata {
        AssetMetadata {
            width: self.dims.0,
            height: self.dims.1,
            frame_count: self.frame_count,
            frame_rate: Some(self.fps),
            duration_secs: self.duration_secs,
            codec: Some(self.codec),
            container: self.container,
            bit_depth: self.bit_depth,
            channels: 4,
            color_space: self.color_space,
            audio_sample_rate: None,
            audio_channels: None,
        }
    }
}

static FFMPEG_INIT: Once = Once::new();

/// Initialize FFmpeg lazily exactly once. `ffmpeg::init()` is idempotent
/// at the C level but the Rust wrapper does some re-registration we'd
/// rather avoid paying for repeatedly.
fn ensure_ffmpeg_init() {
    FFMPEG_INIT.call_once(|| {
        // Errors here are limited to extremely degenerate environments
        // (no allocator, etc.). We swallow but log — every later call
        // will just fail with a more specific decode error.
        if let Err(e) = ffmpeg::init() {
            tracing::warn!(error = %e, "ffmpeg::init() failed");
        }
    });
}

/// Map a Lumen path → an FFmpeg decode error with our error type.
fn decode_err(path: &Path, msg: impl Into<String>) -> Error {
    Error::decode_at(path.to_path_buf(), msg.into())
}

/// Map an `ffmpeg::Error` plus a path into our [`Error::Decode`].
fn from_ff(path: &Path, prefix: &str, e: ffmpeg::Error) -> Error {
    decode_err(path, format!("{prefix}: {e}"))
}

/// Translate a guessed FFmpeg primary + transfer pair into a workspace
/// [`ColorSpace`]. Falls back to `None` for the (very common in the
/// wild) "unspecified" case.
fn guess_color_space(
    primaries: ffmpeg::color::Primaries,
    transfer: ffmpeg::color::TransferCharacteristic,
) -> Option<ColorSpace> {
    use ffmpeg::color::Primaries as P;
    use ffmpeg::color::TransferCharacteristic as T;

    match (primaries, transfer) {
        (P::BT709, T::BT709)
        | (P::BT709, T::Unspecified)
        | (P::Unspecified, T::BT709) => Some(ColorSpace::Rec709),
        (P::BT2020, T::SMPTE2084) => Some(ColorSpace::Rec2020Pq),
        (P::BT2020, T::ARIB_STD_B67) => Some(ColorSpace::Rec2020Hlg),
        (P::BT2020, _) => Some(ColorSpace::LinearRec2020),
        (P::SMPTE432, _) => Some(ColorSpace::DisplayP3),
        (P::SMPTE431, _) => Some(ColorSpace::DciP3),
        _ => None,
    }
}

/// Best-effort bit depth from the pixel format descriptor.
fn bit_depth_for(fmt: Pixel) -> u8 {
    // The component-level bit depth lives in the AVPixFmtDescriptor's
    // first component; `nb_components > 0` guarantees at least one entry.
    // We avoid descending into `unsafe` here by approximating from the
    // pixel-format name, which encodes the depth for the formats we care
    // about (`yuv420p`, `yuv420p10le`, etc.).
    let name = fmt.descriptor().map(|d| d.name()).unwrap_or("");
    for depth in [16u8, 14, 12, 10] {
        if name.contains(&format!("p{depth}")) || name.contains(&format!("{depth}le")) {
            return depth;
        }
    }
    8
}

/// Probe a video container, returning header-only metadata.
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub fn probe_video<P: AsRef<Path>>(path: P) -> Result<VideoProbe> {
    let path = path.as_ref();
    ensure_ffmpeg_init();

    let ictx = ffmpeg::format::input(&path)
        .map_err(|e| from_ff(path, "open input", e))?;

    let stream = ictx
        .streams()
        .best(MediaType::Video)
        .ok_or_else(|| decode_err(path, "no video stream"))?;

    let codec_ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
        .map_err(|e| from_ff(path, "codec context", e))?;
    let decoder = codec_ctx
        .decoder()
        .video()
        .map_err(|e| from_ff(path, "open decoder", e))?;

    let codec_name = stream.parameters().id().name().to_string();
    let container = ictx.format().name().split(',').next().map(|s| s.to_string());

    let avg_rate = stream.avg_frame_rate();
    let fps = Rational::new(avg_rate.numerator() as i64, avg_rate.denominator() as i64);

    // `stream.frames()` is the demuxer's reported count (`nb_frames`),
    // 0 when the container hasn't recorded it.
    let raw_frames = stream.frames();
    let frame_count = if raw_frames > 0 { Some(raw_frames as u64) } else { None };

    // Container-level duration is in `AV_TIME_BASE` units (1e-6).
    let raw_duration = ictx.duration();
    let duration_secs = if raw_duration > 0 {
        Some(raw_duration as f64 / ffmpeg::ffi::AV_TIME_BASE as f64)
    } else {
        None
    };

    let pix_fmt = decoder.format();
    let bit_depth = bit_depth_for(pix_fmt);
    let color_space =
        guess_color_space(decoder.color_primaries(), decoder.color_transfer_characteristic());

    Ok(VideoProbe {
        dims: (decoder.width(), decoder.height()),
        codec: codec_name,
        container,
        fps,
        frame_count,
        duration_secs,
        bit_depth,
        color_space,
    })
}

/// Internal: build a sws scaler from the decoder's source format to RGBA.
fn build_rgba_scaler(decoder: &ffmpeg::decoder::Video) -> Result<Scaler> {
    Scaler::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        Pixel::RGBA,
        decoder.width(),
        decoder.height(),
        ScalerFlags::BILINEAR,
    )
    .map_err(|e| Error::decode(format!("sws context: {e}")))
}

/// Internal: convert an FFmpeg `AvVideo` (already scaled to RGBA) into a
/// Lumen [`Frame`]. Strips any row stride that swscale may have added.
fn av_rgba_to_frame(rgba: &AvVideo) -> Result<Frame> {
    let w = rgba.width();
    let h = rgba.height();
    let stride = rgba.stride(0);
    let row_bytes = (w as usize) * 4;
    let src = rgba.data(0);

    if stride == row_bytes {
        let mut data = Vec::with_capacity(row_bytes * h as usize);
        data.extend_from_slice(&src[..row_bytes * h as usize]);
        Frame::new(w, h, PixelData::Rgba8(data), ColorSpace::SRgb, None)
    } else {
        // Walk row-by-row, skipping padding.
        let mut data = Vec::with_capacity(row_bytes * h as usize);
        for row in 0..h as usize {
            let off = row * stride;
            data.extend_from_slice(&src[off..off + row_bytes]);
        }
        Frame::new(w, h, PixelData::Rgba8(data), ColorSpace::SRgb, None)
    }
}

/// Decode a single frame at absolute `frame_index` into RGBA8/sRGB.
///
/// "Frame index" is `floor(time * fps)` against the stream's average
/// frame rate. We seek by the rescaled timestamp on the video stream's
/// timebase, then walk packets forward until the decoder yields a frame.
/// Most inter-coded codecs require crossing back to the previous keyframe
/// during seek, which `avformat_seek_file` does for us with a backward
/// rounding bias.
#[instrument(skip_all, fields(path = %path.as_ref().display(), frame_index))]
pub fn decode_video_frame<P: AsRef<Path>>(path: P, frame_index: u64) -> Result<Frame> {
    let path = path.as_ref();
    ensure_ffmpeg_init();

    let mut ictx = ffmpeg::format::input(&path)
        .map_err(|e| from_ff(path, "open input", e))?;

    let stream_index;
    let mut decoder;
    let stream_time_base;
    let avg_rate;
    {
        let stream = ictx
            .streams()
            .best(MediaType::Video)
            .ok_or_else(|| decode_err(path, "no video stream"))?;
        stream_index = stream.index();
        stream_time_base = stream.time_base();
        avg_rate = stream.avg_frame_rate();

        let codec_ctx =
            ffmpeg::codec::context::Context::from_parameters(stream.parameters())
                .map_err(|e| from_ff(path, "codec context", e))?;
        decoder = codec_ctx
            .decoder()
            .video()
            .map_err(|e| from_ff(path, "open decoder", e))?;
    }

    let mut scaler = build_rgba_scaler(&decoder)?;

    // Compute target PTS in the stream's timebase: pts = idx / fps / tb.
    if avg_rate.numerator() <= 0 || avg_rate.denominator() <= 0 {
        return Err(decode_err(path, "stream has no average frame rate"));
    }
    let pts = (frame_index as i64).rescale(avg_rate.invert(), stream_time_base);

    if frame_index > 0 {
        ictx.seek(pts, ..pts)
            .map_err(|e| from_ff(path, "seek", e))?;
    }

    // Walk packets, decoding until we find a frame with PTS >= target.
    let mut decoded = AvVideo::empty();
    for (stream, packet) in ictx.packets() {
        if stream.index() != stream_index {
            continue;
        }
        decoder.send_packet(&packet)
            .map_err(|e| from_ff(path, "send_packet", e))?;
        while decoder.receive_frame(&mut decoded).is_ok() {
            let frame_pts = decoded.pts().unwrap_or(0);
            if frame_pts >= pts || frame_index == 0 {
                let mut rgba = AvVideo::empty();
                scaler
                    .run(&decoded, &mut rgba)
                    .map_err(|e| from_ff(path, "sws run", e))?;
                let mut out = av_rgba_to_frame(&rgba)?;
                debug!(width = out.width, height = out.height, "decoded video frame");
                // Stamp PTS on the output frame for downstream consumers.
                out.pts = Some(lumen_core::Pts::new(
                    Rational::new(
                        stream_time_base.numerator() as i64,
                        stream_time_base.denominator() as i64,
                    ),
                    frame_pts,
                ));
                return Ok(out);
            }
        }
    }

    // Drain.
    decoder.send_eof().ok();
    while decoder.receive_frame(&mut decoded).is_ok() {
        let frame_pts = decoded.pts().unwrap_or(0);
        if frame_pts >= pts || frame_index == 0 {
            let mut rgba = AvVideo::empty();
            scaler
                .run(&decoded, &mut rgba)
                .map_err(|e| from_ff(path, "sws run", e))?;
            return av_rgba_to_frame(&rgba);
        }
    }

    Err(decode_err(path, format!("frame index {frame_index} not produced")))
}

/// Iterate decoded frames in `[start, end_exclusive)`. The caller's
/// closure is invoked with `(frame_index, frame)`. Returns the number of
/// frames actually delivered (which can be less than `end - start` if
/// the stream ends sooner). The closure may abort iteration by
/// returning an `Err`.
#[instrument(
    skip_all,
    fields(path = %path.as_ref().display(), start, end_exclusive)
)]
pub fn decode_video_range<P, F>(
    path: P,
    start: u64,
    end_exclusive: u64,
    mut on_frame: F,
) -> Result<u64>
where
    P: AsRef<Path>,
    F: FnMut(u64, Frame) -> Result<()>,
{
    if end_exclusive <= start {
        return Ok(0);
    }
    let path = path.as_ref();
    ensure_ffmpeg_init();

    let mut ictx = ffmpeg::format::input(&path)
        .map_err(|e| from_ff(path, "open input", e))?;

    let stream_index;
    let mut decoder;
    let stream_time_base;
    let avg_rate;
    {
        let stream = ictx
            .streams()
            .best(MediaType::Video)
            .ok_or_else(|| decode_err(path, "no video stream"))?;
        stream_index = stream.index();
        stream_time_base = stream.time_base();
        avg_rate = stream.avg_frame_rate();

        let codec_ctx =
            ffmpeg::codec::context::Context::from_parameters(stream.parameters())
                .map_err(|e| from_ff(path, "codec context", e))?;
        decoder = codec_ctx
            .decoder()
            .video()
            .map_err(|e| from_ff(path, "open decoder", e))?;
    }

    if avg_rate.numerator() <= 0 || avg_rate.denominator() <= 0 {
        return Err(decode_err(path, "stream has no average frame rate"));
    }

    let mut scaler = build_rgba_scaler(&decoder)?;

    let start_pts = (start as i64).rescale(avg_rate.invert(), stream_time_base);
    if start > 0 {
        ictx.seek(start_pts, ..start_pts)
            .map_err(|e| from_ff(path, "seek", e))?;
    }

    let mut delivered: u64 = 0;
    let mut decoded = AvVideo::empty();

    let mut emit = |idx: u64, decoded: &AvVideo, scaler: &mut Scaler|
        -> Result<()>
    {
        let mut rgba = AvVideo::empty();
        scaler
            .run(decoded, &mut rgba)
            .map_err(|e| from_ff(path, "sws run", e))?;
        let frame = av_rgba_to_frame(&rgba)?;
        on_frame(idx, frame)
    };

    'outer: for (stream, packet) in ictx.packets() {
        if stream.index() != stream_index {
            continue;
        }
        decoder
            .send_packet(&packet)
            .map_err(|e| from_ff(path, "send_packet", e))?;
        while decoder.receive_frame(&mut decoded).is_ok() {
            let frame_pts = decoded.pts().unwrap_or(0);
            // Convert pts back to a frame index against the average rate.
            let idx = frame_pts.rescale(stream_time_base, avg_rate.invert()) as u64;
            if idx < start {
                continue;
            }
            if idx >= end_exclusive {
                break 'outer;
            }
            emit(idx, &decoded, &mut scaler)?;
            delivered += 1;
        }
    }

    decoder.send_eof().ok();
    while decoder.receive_frame(&mut decoded).is_ok() {
        let frame_pts = decoded.pts().unwrap_or(0);
        let idx = frame_pts.rescale(stream_time_base, avg_rate.invert()) as u64;
        if idx < start || idx >= end_exclusive {
            continue;
        }
        emit(idx, &decoded, &mut scaler)?;
        delivered += 1;
    }

    Ok(delivered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::process::Command;
    use tempfile::tempdir;

    /// Synthesize a 1-second, 24fps, 64x48 testsrc video on demand.
    /// Returns `None` if ffmpeg isn't on PATH (so CI without ffmpeg can
    /// gracefully skip rather than failing).
    fn synth_test_mp4() -> Option<(tempfile::TempDir, PathBuf)> {
        let dir = tempdir().ok()?;
        let path = dir.path().join("synth.mp4");
        let status = Command::new("ffmpeg")
            .args([
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x48:rate=24,trim=duration=1",
                "-pix_fmt",
                "yuv420p",
                "-y",
            ])
            .arg(&path)
            .status()
            .ok()?;
        if status.success() && path.exists() {
            Some((dir, path))
        } else {
            None
        }
    }

    #[test]
    fn probe_video_returns_sane_metadata() {
        let Some((_keep, path)) = synth_test_mp4() else {
            eprintln!("skipping probe_video test: ffmpeg CLI unavailable");
            return;
        };
        let p = probe_video(&path).expect("probe should succeed");
        assert_eq!(p.dims, (64, 48));
        // testsrc at 24fps for 1s is exactly 24 frames; some muxers
        // record `nb_frames=0` so accept either 24 or "unknown".
        if let Some(n) = p.frame_count {
            assert!((20..=28).contains(&n), "expected ~24 frames, got {n}");
        }
        assert_eq!(p.fps, Rational::new(24, 1));
        assert!(!p.codec.is_empty());
        // 1 second duration ± small slack.
        if let Some(d) = p.duration_secs {
            assert!((0.5..=1.5).contains(&d), "expected ~1s, got {d}");
        }
    }

    #[test]
    fn decode_video_frame_returns_expected_dims() {
        let Some((_keep, path)) = synth_test_mp4() else {
            eprintln!("skipping decode_video_frame test: ffmpeg CLI unavailable");
            return;
        };
        let frame = decode_video_frame(&path, 0).expect("decode frame 0");
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 48);
        assert_eq!(frame.layout(), lumen_core::PixelLayout::Rgba8);
        let PixelData::Rgba8(ref px) = frame.data else { panic!("expected rgba8") };
        assert_eq!(px.len(), 64 * 48 * 4);
    }

    #[test]
    fn probe_nonexistent_errs_gracefully() {
        let r = probe_video("/nonexistent/lumen-no-such-video.mp4");
        assert!(matches!(r, Err(Error::Decode { .. })));
    }

    #[test]
    fn decode_video_range_iterates_frames() {
        let Some((_keep, path)) = synth_test_mp4() else {
            eprintln!("skipping decode_video_range test: ffmpeg CLI unavailable");
            return;
        };
        let mut count = 0u64;
        let n = decode_video_range(&path, 0, 5, |_idx, frame| {
            assert_eq!(frame.width, 64);
            assert_eq!(frame.height, 48);
            count += 1;
            Ok(())
        })
        .expect("range decode");
        assert_eq!(n, count);
        assert!(n >= 1, "expected at least 1 frame, got {n}");
    }
}
