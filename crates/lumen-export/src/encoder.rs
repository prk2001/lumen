//! Frame-by-frame video encoder built on `ffmpeg-next` 8.x.
//!
//! Phase-1 codec set: H.264 (libx264), H.265 (libx265), ProRes 422
//! (prores_ks). All three are encoded into the container that the output
//! file extension implies (`.mp4`, `.mov`, `.mkv`, etc.) — `format::output`
//! resolves the muxer for us.
//!
//! The encoder is fed a [`lumen_core::Frame`] of arbitrary pixel layout
//! per call. We lift to packed RGBA u8 / sRGB internally and let swscale
//! convert to the encoder's planar YUV format (yuv420p for H.264/H.265,
//! yuv422p10le for ProRes).
//!
//! Pattern mirrors `lumen-io::video` (one-time `ffmpeg::init()`, swscale
//! contexts, error mapping). See that crate for the decode-side
//! equivalents.

use std::path::{Path, PathBuf};
use std::sync::Once;

use ffmpeg_next as ffmpeg;
use ffmpeg::format::Pixel;
use ffmpeg::software::scaling::{context::Context as Scaler, flag::Flags as ScalerFlags};
use ffmpeg::util::frame::video::Video as AvVideo;
use ffmpeg::Packet;
use lumen_core::{Error, Frame, PixelData, Rational, Result};
use tracing::{debug, instrument};

/// Phase-1 codec set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Codec {
    /// H.264 / AVC via libx264.
    H264,
    /// H.265 / HEVC via libx265.
    H265,
    /// Apple ProRes 422 via prores_ks.
    ProRes422,
}

impl Codec {
    /// Default CRF value when the caller specifies neither `bitrate_kbps`
    /// nor `crf`. Only meaningful for H.264 / H.265 — ProRes is a fixed-
    /// quality codec and uses an internal profile instead.
    pub fn default_crf(self) -> Option<u8> {
        match self {
            Codec::H264 => Some(23),
            Codec::H265 => Some(28),
            Codec::ProRes422 => None,
        }
    }

    /// Pixel format the encoder expects on its input plane.
    pub fn target_pixel_format(self) -> Pixel {
        match self {
            // libx264/libx265 universally accept 8-bit 4:2:0 chroma.
            Codec::H264 | Codec::H265 => Pixel::YUV420P,
            // prores_ks at the default profile (hq, 422) expects 10-bit
            // 4:2:2 planar — yuv422p10le is what every production tool
            // (FCP, Resolve, Premiere) writes.
            Codec::ProRes422 => Pixel::YUV422P10LE,
        }
    }

    /// FFmpeg encoder name used for the open-by-name lookup.
    pub fn encoder_name(self) -> &'static str {
        match self {
            Codec::H264 => "libx264",
            Codec::H265 => "libx265",
            Codec::ProRes422 => "prores_ks",
        }
    }
}

/// Configuration for [`VideoEncoder::open`].
#[derive(Debug, Clone)]
pub struct VideoEncoderOptions {
    pub codec: Codec,
    pub width: u32,
    pub height: u32,
    pub fps: Rational,
    /// Target average bitrate in kbps. If `None`, encoders that support
    /// CRF (H.264/H.265) fall back to that; ProRes uses its profile-based
    /// rate control.
    pub bitrate_kbps: Option<u32>,
    /// CRF target for H.264/H.265. Ignored for ProRes. If both this and
    /// `bitrate_kbps` are `None`, [`Codec::default_crf`] is used.
    pub crf: Option<u8>,
}

impl VideoEncoderOptions {
    /// Convenience constructor with sensible defaults.
    pub fn new(codec: Codec, width: u32, height: u32, fps: Rational) -> Self {
        Self {
            codec,
            width,
            height,
            fps,
            bitrate_kbps: None,
            crf: None,
        }
    }
}

static FFMPEG_INIT: Once = Once::new();

/// Initialize FFmpeg lazily exactly once. Mirrors the pattern in
/// `lumen-io`. Idempotent at the C level; the wrapper does some
/// re-registration we don't want to pay for on every encode.
fn ensure_ffmpeg_init() {
    FFMPEG_INIT.call_once(|| {
        if let Err(e) = ffmpeg::init() {
            tracing::warn!(error = %e, "ffmpeg::init() failed");
        }
    });
}

/// Map an `ffmpeg::Error` plus a path into our [`Error::Encode`].
fn from_ff(path: &Path, prefix: &str, e: ffmpeg::Error) -> Error {
    Error::encode_at(path.to_path_buf(), format!("{prefix}: {e}"))
}

fn encode_err(path: &Path, msg: impl Into<String>) -> Error {
    Error::encode_at(path.to_path_buf(), msg.into())
}

/// Frame-by-frame video encoder.
///
/// Construction opens the output container, allocates an encoder of the
/// requested codec, and writes the muxer header. Each [`write_frame`]
/// call accepts a Lumen [`Frame`], converts it to the encoder's pixel
/// format, and pushes it through the encode + mux pipeline. [`finish`]
/// drains any remaining packets, writes the trailer, and closes the file.
///
/// [`write_frame`]: VideoEncoder::write_frame
/// [`finish`]: VideoEncoder::finish
pub struct VideoEncoder {
    path: PathBuf,
    octx: ffmpeg::format::context::Output,
    /// Opened video encoder. `ffmpeg::encoder::Video` is the alias for
    /// the `video::Encoder` (i.e. the *opened* encoder, post-`avcodec_open2`).
    encoder: ffmpeg::encoder::Video,
    /// swscale: packed RGBA u8 -> encoder pixel format.
    scaler: Scaler,
    /// Output stream timebase (the muxer rescales packet PTS into this).
    stream_time_base: ffmpeg::Rational,
    /// Encoder timebase (1 / fps) — used to set frame PTS.
    encoder_time_base: ffmpeg::Rational,
    stream_index: usize,
    width: u32,
    height: u32,
    target_fmt: Pixel,
    /// Monotonic frame counter used to stamp PTS = frame_index.
    next_pts: i64,
    /// True after [`finish`] has consumed `self`. Since `finish` takes
    /// `self` by value the field is only meaningful during the function's
    /// own progression; left here for symmetry with future Drop work.
    finished: bool,
}

impl VideoEncoder {
    /// Open `path` for video encoding with the supplied options.
    ///
    /// The container format is inferred from the file extension by
    /// FFmpeg's `avformat_alloc_output_context2`. Typical pairings:
    /// `.mp4` for H.264/H.265, `.mov` for ProRes.
    #[instrument(
        skip_all,
        fields(path = %path.as_ref().display(), codec = ?opts.codec, dims = ?(opts.width, opts.height))
    )]
    pub fn open<P: AsRef<Path>>(path: P, opts: VideoEncoderOptions) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        ensure_ffmpeg_init();

        // --- fps validation ---
        if opts.fps.num <= 0 || opts.fps.den <= 0 {
            return Err(encode_err(&path, "fps must be positive"));
        }
        if opts.width == 0 || opts.height == 0 {
            return Err(encode_err(&path, "width/height must be > 0"));
        }

        // --- locate codec by name (the encoder may not be present in
        //     every FFmpeg build) ---
        let codec = ffmpeg::encoder::find_by_name(opts.codec.encoder_name())
            .ok_or_else(|| {
                Error::UnsupportedFormat(format!(
                    "encoder '{}' not present in this FFmpeg build",
                    opts.codec.encoder_name()
                ))
            })?;

        // --- output container ---
        let mut octx = ffmpeg::format::output(&path)
            .map_err(|e| from_ff(&path, "open output", e))?;
        let global_header = octx
            .format()
            .flags()
            .contains(ffmpeg::format::Flags::GLOBAL_HEADER);

        // --- encoder context ---
        // Build encoder timebase = 1/fps. ffmpeg::Rational uses i32, so
        // fold the fps-rational into that. For non-integer fps like
        // 24000/1001, recip becomes 1001/24000 which fits i32 fine.
        let fps_ff = ffmpeg::Rational::new(opts.fps.num as i32, opts.fps.den as i32);
        let encoder_tb = fps_ff.invert();

        let target_fmt = opts.codec.target_pixel_format();

        let mut stream = octx
            .add_stream(codec)
            .map_err(|e| from_ff(&path, "add stream", e))?;
        let stream_index = stream.index();
        // The output stream timebase is just a hint at this point; the
        // muxer reserves the right to override during write_header.
        stream.set_time_base(encoder_tb);

        let mut enc_ctx =
            ffmpeg::codec::context::Context::new_with_codec(codec)
                .encoder()
                .video()
                .map_err(|e| from_ff(&path, "encoder context", e))?;

        enc_ctx.set_width(opts.width);
        enc_ctx.set_height(opts.height);
        enc_ctx.set_format(target_fmt);
        enc_ctx.set_time_base(encoder_tb);
        enc_ctx.set_frame_rate(Some(fps_ff));

        if global_header {
            enc_ctx.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER);
        }

        // --- rate control ---
        let mut open_opts = ffmpeg::Dictionary::new();
        match opts.codec {
            Codec::H264 | Codec::H265 => {
                if let Some(kbps) = opts.bitrate_kbps {
                    enc_ctx.set_bit_rate((kbps as usize) * 1000);
                } else {
                    let crf = opts.crf.or_else(|| opts.codec.default_crf()).unwrap_or(23);
                    open_opts.set("crf", &crf.to_string());
                    // libx264 / libx265 both honour `preset`; "medium" is
                    // their default, set it explicitly for reproducibility.
                    open_opts.set("preset", "medium");
                }
            }
            Codec::ProRes422 => {
                // 3 = "HQ" (10-bit 422 ~ 220 Mb/s @ 1080p30). The encoder
                // also accepts 0..=5; default of 3 mirrors what Apple's
                // own tools pick.
                open_opts.set("profile", "3");
                if let Some(kbps) = opts.bitrate_kbps {
                    enc_ctx.set_bit_rate((kbps as usize) * 1000);
                }
            }
        }

        // --- stamp initial parameters on the stream so the muxer knows
        //     about the encoder's width/height/format before write_header.
        // open_as_with consumes the encoder context and returns the
        // opened Encoder we hand back.
        let opened = enc_ctx
            .open_as_with(codec, open_opts)
            .map_err(|e| from_ff(&path, "open encoder", e))?;

        // Re-fetch a mutable handle to the stream and copy parameters.
        {
            let mut stream = octx
                .stream_mut(stream_index)
                .ok_or_else(|| encode_err(&path, "stream went missing"))?;
            stream.set_parameters(&opened);
            stream.set_time_base(encoder_tb);
        }

        octx.write_header()
            .map_err(|e| from_ff(&path, "write header", e))?;

        // After write_header the muxer may have rewritten the stream
        // timebase (mp4, for instance, prefers a 1/12800-ish base).
        let stream_time_base = octx
            .stream(stream_index)
            .ok_or_else(|| encode_err(&path, "stream went missing post-header"))?
            .time_base();

        // --- swscale: RGBA u8 (any input we feed it) -> target YUV ---
        let scaler = Scaler::get(
            Pixel::RGBA,
            opts.width,
            opts.height,
            target_fmt,
            opts.width,
            opts.height,
            ScalerFlags::BILINEAR,
        )
        .map_err(|e| from_ff(&path, "sws context", e))?;

        debug!(
            ?target_fmt,
            stream_index,
            width = opts.width,
            height = opts.height,
            "video encoder opened"
        );

        Ok(Self {
            path,
            octx,
            encoder: opened,
            scaler,
            stream_time_base,
            encoder_time_base: encoder_tb,
            stream_index,
            width: opts.width,
            height: opts.height,
            target_fmt,
            next_pts: 0,
            finished: false,
        })
    }

    /// Encode a single Lumen [`Frame`] into the output container.
    ///
    /// The caller's frame must match `opts.width / opts.height`. Pixel
    /// layout is converted internally: u8/u16/f32 buffers are normalised
    /// to packed RGBA u8 (sRGB), then handed to swscale for conversion to
    /// the encoder's planar YUV format.
    #[instrument(skip_all, fields(pts = self.next_pts))]
    pub fn write_frame(&mut self, frame: &Frame) -> Result<()> {
        if self.finished {
            return Err(encode_err(
                &self.path,
                "write_frame called after finish()",
            ));
        }
        if frame.width != self.width || frame.height != self.height {
            return Err(Error::Layout(format!(
                "frame dims {}x{} do not match encoder {}x{}",
                frame.width, frame.height, self.width, self.height
            )));
        }

        // 1) lift to packed RGBA u8 / sRGB. We clone-by-need to avoid
        //    paying for the conversion when the caller already gave us
        //    Rgba8.
        let rgba_buf: Vec<u8> = match &frame.data {
            PixelData::Rgba8(v) => {
                if frame.color_space == lumen_core::ColorSpace::SRgb {
                    v.clone()
                } else {
                    // Reinterpret-as-sRGB for non-sRGB u8 inputs is the
                    // simplest defensible choice — same byte stream, the
                    // colour space tag is what changes.
                    v.clone()
                }
            }
            _ => {
                let f = frame.clone().into_rgba_f32_linear().into_rgba_u8_srgb();
                let PixelData::Rgba8(v) = f.data else {
                    return Err(encode_err(
                        &self.path,
                        "internal: into_rgba_u8_srgb did not yield Rgba8",
                    ));
                };
                v
            }
        };

        // 2) wrap as an AvVideo of Pixel::RGBA, then run swscale into a
        //    fresh AvVideo of the encoder's pixel format.
        let mut src = AvVideo::new(Pixel::RGBA, self.width, self.height);
        {
            let stride = src.stride(0);
            let row_bytes = (self.width as usize) * 4;
            let dst = src.data_mut(0);
            if stride == row_bytes {
                dst[..row_bytes * self.height as usize]
                    .copy_from_slice(&rgba_buf[..row_bytes * self.height as usize]);
            } else {
                for row in 0..self.height as usize {
                    let dst_off = row * stride;
                    let src_off = row * row_bytes;
                    dst[dst_off..dst_off + row_bytes]
                        .copy_from_slice(&rgba_buf[src_off..src_off + row_bytes]);
                }
            }
        }

        let mut dst = AvVideo::new(self.target_fmt, self.width, self.height);
        self.scaler
            .run(&src, &mut dst)
            .map_err(|e| from_ff(&self.path, "sws run", e))?;

        // 3) PTS in encoder timebase = monotonic frame index.
        dst.set_pts(Some(self.next_pts));
        self.next_pts += 1;

        // 4) push into encoder + drain any packets that pop out.
        self.encoder
            .send_frame(&dst)
            .map_err(|e| from_ff(&self.path, "send_frame", e))?;
        self.drain_packets()?;

        Ok(())
    }

    /// Pull encoded packets out of the encoder and write them into the
    /// muxer. Stops when the encoder reports EAGAIN.
    fn drain_packets(&mut self) -> Result<()> {
        loop {
            let mut packet = Packet::empty();
            match self.encoder.receive_packet(&mut packet) {
                Ok(()) => {
                    packet.set_stream(self.stream_index);
                    packet.rescale_ts(self.encoder_time_base, self.stream_time_base);
                    packet
                        .write_interleaved(&mut self.octx)
                        .map_err(|e| from_ff(&self.path, "write_interleaved", e))?;
                }
                // EAGAIN / EOF — both mean "no more packets right now".
                Err(ffmpeg::Error::Other { errno })
                    if errno == ffmpeg::error::EAGAIN =>
                {
                    return Ok(());
                }
                Err(ffmpeg::Error::Eof) => return Ok(()),
                Err(e) => {
                    return Err(from_ff(&self.path, "receive_packet", e));
                }
            }
        }
    }

    /// Flush the encoder and close the output container. Consumes the
    /// encoder so a second call is impossible at the type level.
    #[instrument(skip_all)]
    pub fn finish(mut self) -> Result<()> {
        // Send EOF to the encoder so it knows to flush its B-frame
        // reorder buffer.
        self.encoder
            .send_eof()
            .map_err(|e| from_ff(&self.path, "send_eof", e))?;
        self.drain_packets()?;

        self.octx
            .write_trailer()
            .map_err(|e| from_ff(&self.path, "write_trailer", e))?;
        self.finished = true;
        debug!(frames_written = self.next_pts, "video encoder finished");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, PixelLayout};
    use tempfile::tempdir;

    /// 64x48 RGBA8 sRGB gradient parametrised by `t` in [0, 1].
    fn synth_frame(t: f32) -> Frame {
        let w = 64u32;
        let h = 48u32;
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let r = ((x as f32 / w as f32) * 255.0) as u8;
                let g = ((y as f32 / h as f32) * 255.0) as u8;
                let b = (t * 255.0) as u8;
                data.extend_from_slice(&[r, g, b, 255]);
            }
        }
        Frame::new(w, h, PixelData::Rgba8(data), ColorSpace::SRgb, None).unwrap()
    }

    /// Detect whether the system FFmpeg has the named encoder. Tests
    /// that depend on a specific encoder (libx264, prores_ks) skip
    /// gracefully when it's absent rather than failing CI on builds
    /// where someone installed a stripped-down ffmpeg.
    fn encoder_present(name: &str) -> bool {
        ensure_ffmpeg_init();
        ffmpeg::encoder::find_by_name(name).is_some()
    }

    #[test]
    fn round_trip_h264_24fps_24_frames() {
        if !encoder_present("libx264") {
            eprintln!("skipping: libx264 not available in this FFmpeg build");
            return;
        }
        let dir = tempdir().unwrap();
        let path = dir.path().join("rt.mp4");

        let opts = VideoEncoderOptions::new(Codec::H264, 64, 48, Rational::FPS_24);
        let mut enc = VideoEncoder::open(&path, opts).expect("open encoder");
        for i in 0..24u32 {
            let t = i as f32 / 24.0;
            enc.write_frame(&synth_frame(t)).expect("write frame");
        }
        enc.finish().expect("finish");

        // Probe the result with the sibling crate.
        let probe = lumen_io::probe_video(&path).expect("probe");
        assert_eq!(probe.dims, (64, 48));
        assert_eq!(probe.codec, "h264");
        // 1 second ± some muxer slop.
        if let Some(d) = probe.duration_secs {
            assert!(
                (0.5..=1.5).contains(&d),
                "expected ~1s duration, got {d}"
            );
        }
        // mp4 records nb_frames in the moov 'stts'; some FFmpeg builds
        // count 24, others omit.
        if let Some(n) = probe.frame_count {
            assert!(
                (20..=28).contains(&n),
                "expected ~24 frames, got {n}"
            );
        }
    }

    #[test]
    fn write_frame_rejects_dim_mismatch() {
        if !encoder_present("libx264") {
            eprintln!("skipping: libx264 not available in this FFmpeg build");
            return;
        }
        let dir = tempdir().unwrap();
        let path = dir.path().join("dims.mp4");

        let opts = VideoEncoderOptions::new(Codec::H264, 64, 48, Rational::FPS_24);
        let mut enc = VideoEncoder::open(&path, opts).expect("open encoder");

        // Wrong dims: 32x24 instead of 64x48.
        let bad = Frame::zeros(32, 24, PixelLayout::Rgba8, ColorSpace::SRgb);
        let r = enc.write_frame(&bad);
        assert!(matches!(r, Err(Error::Layout(_))), "got {r:?}");

        // Encoder is still usable for matching frames after a rejection;
        // finish to clean up.
        enc.finish().ok();
    }

    #[test]
    fn no_use_after_finish() {
        if !encoder_present("libx264") {
            eprintln!("skipping: libx264 not available in this FFmpeg build");
            return;
        }
        let dir = tempdir().unwrap();
        let path = dir.path().join("uaf.mp4");

        let opts = VideoEncoderOptions::new(Codec::H264, 64, 48, Rational::FPS_24);
        let mut enc = VideoEncoder::open(&path, opts).expect("open encoder");
        enc.write_frame(&synth_frame(0.0)).expect("first frame");

        // finish() consumes self, which is what guarantees no UB at the
        // type level: a second call would fail to compile. Verify that
        // here with the canonical "first finish succeeds" assertion.
        enc.finish().expect("finish ok");

        // To exercise the runtime guard for write-after-finish, we open
        // a fresh encoder, force-mark it finished, and confirm a write
        // returns an error rather than panicking or producing UB.
        let path2 = dir.path().join("uaf2.mp4");
        let opts2 = VideoEncoderOptions::new(Codec::H264, 64, 48, Rational::FPS_24);
        let mut enc2 = VideoEncoder::open(&path2, opts2).expect("open encoder 2");
        enc2.finished = true;
        let r = enc2.write_frame(&synth_frame(0.0));
        assert!(matches!(r, Err(Error::Encode { .. })), "got {r:?}");
    }

    #[test]
    fn unsupported_codec_returns_clear_error_when_missing() {
        // This test is informational: it just exercises the
        // codec-not-found path. On every FFmpeg build we've targeted,
        // libx264/libx265/prores_ks are present, so this only triggers
        // if someone installs a stripped FFmpeg.
        for c in [Codec::H264, Codec::H265, Codec::ProRes422] {
            let name = c.encoder_name();
            if encoder_present(name) {
                continue;
            }
            let dir = tempdir().unwrap();
            let path = dir.path().join("missing.bin");
            let opts = VideoEncoderOptions::new(c, 64, 48, Rational::FPS_24);
            let r = VideoEncoder::open(&path, opts);
            assert!(
                matches!(r, Err(Error::UnsupportedFormat(_))),
                "expected UnsupportedFormat for missing {name}"
            );
        }
    }
}
