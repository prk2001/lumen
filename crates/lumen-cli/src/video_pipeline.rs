//! Per-frame video pipeline runner.
//!
//! Glue between [`lumen_io`] (decode), [`lumen_core`] (effect chain
//! scheduling), and [`lumen_export`] (encode). Given a [`Recipe`] whose
//! `input` is a video and whose `output` is a video, this runs the
//! recipe's chain over every frame of the input and writes the result
//! to the output container.
//!
//! The graph is built and validated once, outside the per-frame loop.
//! Each frame goes through a fresh [`Scheduler::run`] call with
//! closure-backed source/sink that just hand the already-decoded frame
//! to the chain and capture the final frame from the sink.
//!
//! See `cmd_video_pipeline` in `main.rs` for the CLI surface.
//!
//! Phase-1 scope:
//! * Linear chains only (same JSON shape as the still-image `pipeline`
//!   subcommand).
//! * Frame count comes from the demuxer's `nb_frames`; if missing we
//!   estimate from `duration_secs * fps` and round up.
//! * Output dims/fps mirror the input. No re-timing.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context as _, Result};
use lumen_core::{
    scheduler::special_effect_ids, Context, EffectRegistry, Frame, Graph, Node, NodeId,
    ParamValues, Scheduler,
};
use lumen_export::{Codec, VideoEncoder, VideoEncoderOptions};
use lumen_io::{decode_video_frame, probe_video};

use crate::{json_to_param, CliSink, CliSource, Recipe};

/// Aggregate statistics from a successful run. Returned by
/// [`run_video_pipeline`] and surfaced in the CLI's terminal output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunStats {
    pub frames_processed: u64,
    pub duration_ms: u128,
}

/// Run a recipe over every frame of a video input and write to a video
/// output. See module docs for the pipeline shape.
///
/// `base_dir` is the directory the recipe's relative paths resolve
/// against — typically the recipe file's own parent directory.
pub fn run_video_pipeline(
    recipe: &Recipe,
    base_dir: &Path,
    registry: &EffectRegistry,
    codec: Codec,
    crf: Option<u8>,
) -> Result<RunStats> {
    let resolve = |p: &Path| -> PathBuf {
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            base_dir.join(p)
        }
    };
    let input_path = resolve(&recipe.input);
    let output_path = resolve(&recipe.output);

    // Reject still-image inputs early with a user-actionable message.
    // The extension check is best-effort; a video container with an
    // unusual extension still falls through to the probe below, which
    // is the source of truth.
    if is_still_image_extension(&input_path) {
        anyhow::bail!(
            "input {} is a still image — use `lumen pipeline` for still images",
            input_path.display()
        );
    }

    // ── 1. probe the input ────────────────────────────────────────────
    let probe = probe_video(&input_path)
        .with_context(|| format!("probing input {}", input_path.display()))?;
    let (width, height) = probe.dims;
    if width == 0 || height == 0 {
        anyhow::bail!(
            "input {} has zero-sized dims {}x{}",
            input_path.display(),
            width,
            height
        );
    }
    if probe.fps.num <= 0 || probe.fps.den <= 0 {
        anyhow::bail!(
            "input {} has no usable frame rate (got {}/{})",
            input_path.display(),
            probe.fps.num,
            probe.fps.den
        );
    }
    let fps = probe.fps;
    let frame_count = match probe.frame_count {
        Some(n) => n,
        None => match probe.duration_secs {
            Some(d) if d > 0.0 && fps.den > 0 => {
                let est = (d * (fps.num as f64) / (fps.den as f64)).round() as i64;
                if est <= 0 {
                    anyhow::bail!(
                        "could not determine frame count for {}",
                        input_path.display()
                    );
                }
                est as u64
            }
            _ => anyhow::bail!(
                "could not determine frame count for {}",
                input_path.display()
            ),
        },
    };

    // ── 2. open the output encoder ────────────────────────────────────
    let mut enc_opts = VideoEncoderOptions::new(codec, width, height, fps);
    if let Some(c) = crf {
        enc_opts.crf = Some(c);
    }
    let mut encoder = VideoEncoder::open(&output_path, enc_opts)
        .map_err(|e| anyhow!("encoder open ({}): {e}", output_path.display()))?;

    // ── 3. build the chain graph once ─────────────────────────────────
    let graph = build_chain_graph(recipe)?;
    // Validate now so a bad recipe fails before we touch any frames.
    graph
        .validate()
        .map_err(|e| anyhow!("recipe graph invalid: {e}"))?;

    // Phase 1: still-image color-management defaults (sRGB, linear-
    // internal) are sufficient for per-frame effects. When
    // `lumen-fx-temporal` lands we'll thread fps + the running frame
    // index into a richer Context.
    let _ = fps; // fps is captured by the encoder; ctx ignores it for now.
    let mut ctx = Context::for_still_srgb();

    // ── 4. for each frame: decode → run chain → encode ────────────────
    let started = Instant::now();
    let mut frames_processed: u64 = 0;
    for idx in 0..frame_count {
        let decoded = decode_video_frame(&input_path, idx)
            .with_context(|| format!("decoding frame {idx} of {}", input_path.display()))?;

        let frame_cell: RefCell<Option<Frame>> = RefCell::new(Some(decoded));
        let captured: RefCell<Option<Frame>> = RefCell::new(None);

        // The scheduler is generic over SourceLoader / SinkWriter. We use
        // the existing CliSource / CliSink newtypes (declared in main.rs)
        // so closures can stand in for traits without colliding with the
        // orphan rule.
        let source = CliSource(|_id: NodeId, _p: &ParamValues| -> lumen_core::Result<Frame> {
            frame_cell.borrow_mut().take().ok_or_else(|| {
                lumen_core::Error::Graph(
                    "video_pipeline: source loader called more than once for one frame"
                        .into(),
                )
            })
        });
        let sink = CliSink(
            |_id: NodeId, _p: &ParamValues, f: Frame| -> lumen_core::Result<()> {
                *captured.borrow_mut() = Some(f);
                Ok(())
            },
        );
        let mut sched = Scheduler {
            registry,
            ctx: &mut ctx,
            source_loader: source,
            sink_writer: sink,
        };
        sched
            .run(&graph)
            .map_err(|e| anyhow!("frame {idx}: chain run failed: {e}"))?;

        let final_frame = captured.into_inner().ok_or_else(|| {
            anyhow!("frame {idx}: chain produced no sink output (empty chain or routing bug)")
        })?;

        encoder
            .write_frame(&final_frame)
            .map_err(|e| anyhow!("frame {idx}: encoder write_frame: {e}"))?;
        frames_processed += 1;
    }

    encoder
        .finish()
        .map_err(|e| anyhow!("encoder finish ({}): {e}", output_path.display()))?;

    Ok(RunStats {
        frames_processed,
        duration_ms: started.elapsed().as_millis(),
    })
}

/// Build the linear-chain graph for the recipe. Mirrors `cmd_pipeline`
/// in main.rs but factored out so the per-frame loop can build it once.
fn build_chain_graph(recipe: &Recipe) -> Result<Graph> {
    let mut graph = Graph::new();
    let src_node = graph.insert(Node::new(special_effect_ids::SOURCE, "source"));
    let mut prev = src_node;
    for (i, step) in recipe.chain.iter().enumerate() {
        let mut params = ParamValues::new();
        if let serde_json::Value::Object(map) = &step.params {
            for (k, v) in map {
                params.insert(
                    k.clone(),
                    json_to_param(v).ok_or_else(|| {
                        anyhow!("step {i}: param '{k}' has unsupported JSON type")
                    })?,
                );
            }
        }
        let label = step
            .label
            .clone()
            .unwrap_or_else(|| format!("step{i:02}"));
        let node = graph.insert(
            Node::new(step.effect.clone(), label)
                .with_input(prev)
                .with_params(params),
        );
        prev = node;
    }
    let sink_node = graph.insert(Node::new(special_effect_ids::SINK, "sink").with_input(prev));
    graph.add_sink(sink_node);
    Ok(graph)
}

/// True iff `path`'s extension matches a still-image format we know how
/// to decode via `lumen-io`. Used to reject still inputs before paying
/// for an FFmpeg probe.
fn is_still_image_extension(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "png"
            | "jpg"
            | "jpeg"
            | "tiff"
            | "tif"
            | "bmp"
            | "webp"
            | "heic"
            | "heif"
            | "jxl"
            | "avif"
            | "dng"
            | "cr2"
            | "cr3"
            | "nef"
            | "arw"
            | "raf"
            | "orf"
            | "rw2"
            | "pef"
            | "srw"
            | "iiq"
            | "3fr"
            | "x3f"
    )
}

/// Helper: best-effort detection of "is this path a video?" based on
/// extension first, then a probe fallback. Used by the CLI to redirect
/// users with still-image inputs back to the `pipeline` subcommand.
pub fn is_video_input(path: &Path) -> bool {
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        let lower = ext.to_ascii_lowercase();
        match lower.as_str() {
            "mp4" | "mov" | "mkv" | "webm" | "avi" | "m4v" | "mpg" | "mpeg" | "ts" | "wmv"
            | "flv" => return true,
            "png" | "jpg" | "jpeg" | "tiff" | "tif" | "bmp" | "webp" | "heic" | "heif"
            | "jxl" | "avif" | "dng" | "cr2" | "cr3" | "nef" | "arw" | "raf" | "orf"
            | "rw2" | "pef" | "srw" | "iiq" | "3fr" | "x3f" => return false,
            _ => {}
        }
    }
    // Unknown extension — try a real probe.
    probe_video(path).is_ok()
}

/// Parse the CLI's `--codec` string to a [`Codec`] enum. Same matrix as
/// `cmd_export_video` so users get one consistent error message.
pub fn parse_codec(name: &str) -> Result<Codec> {
    match name.to_ascii_lowercase().as_str() {
        "h264" => Ok(Codec::H264),
        "h265" | "hevc" => Ok(Codec::H265),
        "prores" | "prores422" => Ok(Codec::ProRes422),
        other => anyhow::bail!("unknown codec: {other} (h264|h265|prores)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_registry, Recipe, RecipeStep};
    use lumen_core::{PixelData, Rational};
    use std::path::PathBuf;
    use std::process::Command;
    use tempfile::tempdir;

    /// Synthesize an `n`-frame, 24fps, 64x48 testsrc video. Returns
    /// `None` if ffmpeg isn't on PATH so CI without it skips cleanly.
    fn synth_testsrc_mp4(n: u32) -> Option<(tempfile::TempDir, PathBuf)> {
        let dir = tempdir().ok()?;
        let path = dir.path().join("input.mp4");
        // duration = n / 24 seconds
        let secs = n as f32 / 24.0;
        let filter = format!("testsrc=size=64x48:rate=24,trim=duration={secs:.5}");
        let status = Command::new("ffmpeg")
            .args(["-loglevel", "error", "-f", "lavfi", "-i", &filter])
            .args(["-pix_fmt", "yuv420p", "-y"])
            .arg(&path)
            .status()
            .ok()?;
        if status.success() && path.exists() {
            Some((dir, path))
        } else {
            None
        }
    }

    fn libx264_present() -> bool {
        // We don't take a direct ffmpeg-next dep in lumen-cli, so probe
        // by attempting to open a tiny encoder. Anything other than the
        // workspace-typed `UnsupportedFormat` (the codec-not-found path)
        // means the encoder is registered.
        let probe_dir = match tempdir() {
            Ok(d) => d,
            Err(_) => return false,
        };
        let probe_path = probe_dir.path().join("probe.mp4");
        let opts = VideoEncoderOptions::new(Codec::H264, 16, 16, Rational::new(24, 1));
        match VideoEncoder::open(&probe_path, opts) {
            Ok(enc) => {
                let _ = enc.finish();
                true
            }
            Err(lumen_core::Error::UnsupportedFormat(_)) => false,
            // Any other error class — including write_header issues on
            // the throwaway path — implies the encoder *was* found.
            Err(_) => true,
        }
    }

    #[test]
    fn run_video_pipeline_smoke() {
        if !libx264_present() {
            eprintln!("skipping: libx264 not available in this FFmpeg build");
            return;
        }
        let Some((_keep, in_path)) = synth_testsrc_mp4(12) else {
            eprintln!("skipping run_video_pipeline_smoke: ffmpeg CLI unavailable");
            return;
        };
        let dir = tempdir().unwrap();
        let out_path = dir.path().join("out.mp4");
        let recipe = Recipe {
            input: in_path.clone(),
            output: out_path.clone(),
            chain: vec![],
        };
        let registry = build_registry().expect("build registry");

        let stats = run_video_pipeline(
            &recipe,
            std::path::Path::new("."),
            &registry,
            Codec::H264,
            None,
        )
        .expect("video pipeline should run");

        assert_eq!(stats.frames_processed, 12);

        // Round-trip via probe_video.
        let probe = probe_video(&out_path).expect("probe output");
        assert_eq!(probe.dims, (64, 48));
        if let Some(n) = probe.frame_count {
            assert!(
                (10..=14).contains(&n),
                "expected ~12 frames, got {n}"
            );
        }
        // FPS round-trip is muxer-dependent for very short clips —
        // mp4's avg_frame_rate is computed as `nb_frames / duration`
        // and the moov's coarse timing can produce e.g. 288/1 or
        // 24000/1001 instead of an exact 24/1. Just check it's sane.
        assert!(probe.fps.num > 0 && probe.fps.den > 0);
        let fps_f =
            probe.fps.num as f64 / probe.fps.den as f64;
        assert!(
            (1.0..=600.0).contains(&fps_f),
            "fps round-trip out of plausible range: {fps_f}"
        );
    }

    #[test]
    fn pipeline_applies_effect_per_frame() {
        if !libx264_present() {
            eprintln!("skipping: libx264 not available in this FFmpeg build");
            return;
        }
        let Some((_keep, in_path)) = synth_testsrc_mp4(12) else {
            eprintln!("skipping pipeline_applies_effect_per_frame: ffmpeg CLI unavailable");
            return;
        };
        let dir = tempdir().unwrap();
        let out_path = dir.path().join("gray.mp4");
        let recipe = Recipe {
            input: in_path.clone(),
            output: out_path.clone(),
            // amount=0.0 fully desaturates → R==G==B per pixel.
            chain: vec![RecipeStep {
                effect: "lumen-fx-color.saturation".to_string(),
                label: Some("desat".into()),
                params: serde_json::json!({ "amount": 0.0 }),
            }],
        };
        let registry = build_registry().expect("build registry");

        let stats = run_video_pipeline(
            &recipe,
            std::path::Path::new("."),
            &registry,
            Codec::H264,
            None,
        )
        .expect("video pipeline should run");
        assert_eq!(stats.frames_processed, 12);

        // Decode frames 0 and 11 from the output and assert R == G == B.
        // YUV420p round-trip introduces ±1-LSB chroma noise even for true
        // gray inputs, so allow a small tolerance.
        for idx in [0u64, 11u64] {
            let f = decode_video_frame(&out_path, idx)
                .unwrap_or_else(|e| panic!("decode frame {idx}: {e}"));
            let PixelData::Rgba8(ref px) = f.data else {
                panic!("expected rgba8")
            };
            assert_eq!(px.len(), 64 * 48 * 4);
            let mut max_dev: i32 = 0;
            for chunk in px.chunks_exact(4) {
                let r = chunk[0] as i32;
                let g = chunk[1] as i32;
                let b = chunk[2] as i32;
                let d = (r - g).abs().max((g - b).abs()).max((r - b).abs());
                if d > max_dev {
                    max_dev = d;
                }
            }
            assert!(
                max_dev <= 6,
                "frame {idx} not grayscale: max channel deviation = {max_dev}"
            );
        }
    }

    #[test]
    fn errors_on_still_input() {
        // Create a tiny PNG.
        let dir = tempdir().unwrap();
        let png = dir.path().join("still.png");
        let frame = lumen_core::Frame::new(
            8,
            8,
            PixelData::Rgba8(vec![0x55; 8 * 8 * 4]),
            lumen_core::ColorSpace::SRgb,
            None,
        )
        .unwrap();
        lumen_io::encode_image(frame, &png, lumen_io::ImageEncodeOptions::default())
            .expect("encode test png");

        // The pre-flight `is_video_input` should reject by extension. The
        // runner should also reject if invoked directly — `probe_video`
        // on a PNG fails with a Decode error, which surfaces as our
        // "probing input" anyhow context.
        assert!(!is_video_input(&png));

        let registry = build_registry().expect("build registry");
        let recipe = Recipe {
            input: png.clone(),
            output: dir.path().join("out.mp4"),
            chain: vec![],
        };
        let r = run_video_pipeline(
            &recipe,
            std::path::Path::new("."),
            &registry,
            Codec::H264,
            None,
        );
        let err = r.expect_err("expected still-image input to error");
        let msg = format!("{:#}", err);
        // The runner's still-image gate explicitly redirects the user
        // to the `pipeline` subcommand. That phrasing is the contract.
        assert!(
            msg.contains("pipeline") && msg.contains("still image"),
            "expected error to redirect to `pipeline` for still image, got: {msg}"
        );
    }
}

