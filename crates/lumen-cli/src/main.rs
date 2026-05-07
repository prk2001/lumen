//! Lumen command-line interface.
//!
//! Subcommands:
//!
//! - `probe <file>` — print metadata about an input file as JSON.
//! - `list-effects` — list every effect in the registry.
//! - `apply --input X --output Y --effect ID [--param k=v]…`
//!   decode → effect → encode for one still image.
//! - `pipeline --recipe R.json` — run a multi-effect chain from JSON.
//! - `serve --recipe R.json [--port 8080]` — live HTTP preview that
//!   re-renders on file changes and shows side-by-side input / output.

mod serve;
mod video_pipeline;

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{anyhow, Context as _, Result};
use clap::{Parser, Subcommand};
use lumen_core::{
    scheduler::special_effect_ids, Context, EffectRegistry, Frame, Graph, Node, NodeId,
    ParamValue, ParamValues, Scheduler, SinkWriter, SourceLoader,
};
use lumen_io::{decode_image, encode_image, probe, ImageEncodeOptions};
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Parser, Debug)]
#[command(
    name = "lumen",
    version,
    about = "Lumen — photo & video enhancement (CLI)",
    long_about = None
)]
struct Cli {
    /// Increase log verbosity (-v info, -vv debug, -vvv trace).
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Print metadata about a file as JSON.
    Probe {
        /// Input file path.
        path: PathBuf,
    },
    /// List every effect registered in this build.
    ListEffects,
    /// Decode an image, apply one effect, and write the result.
    Apply {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Effect id (see `lumen list-effects`).
        #[arg(long)]
        effect: String,
        /// Parameters as `key=value`. Repeatable. Values that parse as
        /// bool/int/float are coerced; otherwise they're strings.
        #[arg(long = "param", value_name = "KEY=VALUE")]
        params: Vec<String>,
        /// JPEG quality (1–100), used only when output is JPEG.
        #[arg(long, default_value_t = 92)]
        jpeg_quality: u8,
    },
    /// Run a multi-effect chain from a JSON recipe.
    Pipeline {
        /// Path to a JSON recipe (see `docs/PIPELINE.md`).
        #[arg(long)]
        recipe: PathBuf,
        /// JPEG quality (1–100), used only when output is JPEG.
        #[arg(long, default_value_t = 92)]
        jpeg_quality: u8,
    },
    /// Run a live-preview HTTP server that re-renders the recipe on
    /// file changes and shows input/output side-by-side at /.
    Serve {
        /// Path to a JSON recipe.
        #[arg(long)]
        recipe: PathBuf,
        /// TCP port to bind on 127.0.0.1.
        #[arg(long, default_value_t = 8723)]
        port: u16,
        /// JPEG quality if the output is a JPEG.
        #[arg(long, default_value_t = 92)]
        jpeg_quality: u8,
    },
    /// Compute reference-based image-quality metrics between two files.
    Measure {
        /// Reference / "A" file.
        #[arg(long)]
        a: PathBuf,
        /// Test / "B" file. Must match A's dimensions.
        #[arg(long)]
        b: PathBuf,
    },
    /// Spectral-subtraction noise reduction on a WAV file.
    AudioNr {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 1024)]
        frame_size: usize,
        #[arg(long, default_value_t = 1.5)]
        over_subtract: f32,
        #[arg(long, default_value_t = 0.05)]
        floor: f32,
        #[arg(long, default_value_t = 0.5)]
        noise_estimate_secs: f32,
    },
    /// Generate an Ed25519 keypair, write hex-encoded keys to stdout as JSON.
    Keygen,
    /// Build a chain-of-custody manifest, sign it, and write
    /// `<output>.lumen-cco.json` next to the output file.
    Sign {
        #[arg(long)] input: PathBuf,
        #[arg(long)] output: PathBuf,
        #[arg(long)] recipe: PathBuf,
        /// Hex-encoded Ed25519 signing key (private). Use `lumen keygen` to mint.
        #[arg(long)] secret_key_hex: String,
    },
    /// Verify a `*.lumen-cco.json` signed manifest against a public key.
    Verify {
        #[arg(long)] signed: PathBuf,
        /// Hex-encoded Ed25519 public key. If omitted, the public key
        /// embedded in the signed manifest is used (sanity check only).
        #[arg(long)] public_key_hex: Option<String>,
    },
    /// Generate a self-contained HTML render report.
    Report {
        #[arg(long)] input: PathBuf,
        #[arg(long)] output: PathBuf,
        #[arg(long)] recipe: PathBuf,
        #[arg(long, default_value = "Lumen Render Report")] title: String,
        /// Output HTML path.
        #[arg(long = "html-out")] html_out: PathBuf,
    },
    /// Encode a sequence of frames (one PNG per frame, sorted by filename)
    /// into a video file via ffmpeg.
    ExportVideo {
        /// Directory of numbered PNG frames (e.g. frame_00001.png …).
        #[arg(long)] frames_dir: PathBuf,
        #[arg(long)] output: PathBuf,
        #[arg(long, default_value = "h264")] codec: String,
        #[arg(long, default_value_t = 24)] fps: u32,
        #[arg(long)] crf: Option<u8>,
    },
    /// Run a recipe over every frame of a video input -> video output.
    VideoPipeline {
        #[arg(long)] recipe: PathBuf,
        #[arg(long, default_value = "h264")] codec: String,
        #[arg(long)] crf: Option<u8>,
    },
    /// Run a Lua plugin against a still image.
    Plugin {
        #[arg(long)] plugin: PathBuf,
        #[arg(long)] input: PathBuf,
        #[arg(long)] output: PathBuf,
        /// Repeated --param key=value (typed coercion).
        #[arg(long = "param", value_name = "KEY=VALUE")]
        params: Vec<String>,
    },
    /// Run a directory of golden-frame regression cases.
    Qa {
        /// Directory of `*.case.json` files.
        #[arg(long)] cases: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    init_tracing(cli.verbose);
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Probe { path } => cmd_probe(&path),
        Command::ListEffects => cmd_list_effects(),
        Command::Apply { input, output, effect, params, jpeg_quality } => {
            cmd_apply(&input, &output, &effect, &params, jpeg_quality)
        }
        Command::Pipeline { recipe, jpeg_quality } => cmd_pipeline(&recipe, jpeg_quality),
        Command::Serve { recipe, port, jpeg_quality } => {
            let registry = build_registry()?;
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("tokio runtime")?;
            rt.block_on(serve::run(recipe, port, jpeg_quality, registry))
        }
        Command::Measure { a, b } => cmd_measure(&a, &b),
        Command::AudioNr { input, output, frame_size, over_subtract, floor, noise_estimate_secs } => {
            cmd_audio_nr(&input, &output, frame_size, over_subtract, floor, noise_estimate_secs)
        }
        Command::Keygen => cmd_keygen(),
        Command::Sign { input, output, recipe, secret_key_hex } => {
            cmd_sign(&input, &output, &recipe, &secret_key_hex)
        }
        Command::Verify { signed, public_key_hex } => cmd_verify(&signed, public_key_hex.as_deref()),
        Command::Report { input, output, recipe, title, html_out } => {
            cmd_report(&input, &output, &recipe, &title, &html_out)
        }
        Command::ExportVideo { frames_dir, output, codec, fps, crf } => {
            cmd_export_video(&frames_dir, &output, &codec, fps, crf)
        }
        Command::VideoPipeline { recipe, codec, crf } => {
            cmd_video_pipeline(&recipe, &codec, crf)
        }
        Command::Plugin { plugin, input, output, params } => {
            cmd_plugin(&plugin, &input, &output, &params)
        }
        Command::Qa { cases } => cmd_qa(&cases),
    }
}

fn cmd_video_pipeline(
    recipe_path: &std::path::Path,
    codec_name: &str,
    crf: Option<u8>,
) -> Result<()> {
    let recipe_str = std::fs::read_to_string(recipe_path)
        .with_context(|| format!("reading recipe {}", recipe_path.display()))?;
    let recipe: Recipe = serde_json::from_str(&recipe_str)
        .with_context(|| format!("parsing recipe {} as JSON", recipe_path.display()))?;

    let base = recipe_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let resolve = |p: &std::path::Path| -> PathBuf {
        if p.is_absolute() { p.to_path_buf() } else { base.join(p) }
    };
    let input_path = resolve(&recipe.input);
    let output_path = resolve(&recipe.output);

    if !video_pipeline::is_video_input(&input_path) {
        anyhow::bail!(
            "input {} is not a video — use `lumen pipeline` for still images",
            input_path.display()
        );
    }

    let codec = video_pipeline::parse_codec(codec_name)?;
    let registry = build_registry()?;
    let stats = video_pipeline::run_video_pipeline(&recipe, &base, &registry, codec, crf)?;
    println!(
        "wrote {} ({} frames in {} ms)",
        output_path.display(),
        stats.frames_processed,
        stats.duration_ms
    );
    Ok(())
}

// ─── New subcommands ─────────────────────────────────────────────────────

fn cmd_audio_nr(
    input: &std::path::Path,
    output: &std::path::Path,
    frame_size: usize,
    over_subtract: f32,
    floor: f32,
    noise_estimate_secs: f32,
) -> Result<()> {
    use lumen_audio::{read_wav, spectral_subtract, write_wav, SpectralNrParams};
    let buf = read_wav(input).map_err(|e| anyhow!("read wav: {e}"))?;
    let params = SpectralNrParams { frame_size, over_subtract, floor, noise_estimate_secs };
    let cleaned = spectral_subtract(&buf, &params).map_err(|e| anyhow!("spectral_subtract: {e}"))?;
    write_wav(&cleaned, output).map_err(|e| anyhow!("write wav: {e}"))?;
    println!("wrote {}", output.display());
    Ok(())
}

fn cmd_keygen() -> Result<()> {
    let (sk, vk) = lumen_auth::keypair_generate();
    let json = serde_json::json!({
        "secret_key_hex": hex_encode(sk.to_bytes().as_ref()),
        "public_key_hex": hex_encode(vk.to_bytes().as_ref()),
    });
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}

fn cmd_sign(
    input: &std::path::Path,
    output: &std::path::Path,
    recipe: &std::path::Path,
    secret_key_hex: &str,
) -> Result<()> {
    let manifest = lumen_auth::build_manifest(input, output, recipe)
        .map_err(|e| anyhow!("build_manifest: {e}"))?;
    let bytes = hex_decode(secret_key_hex)
        .ok_or_else(|| anyhow!("--secret-key-hex must be 64 hex chars (32 bytes)"))?;
    if bytes.len() != 32 {
        anyhow::bail!("expected 32-byte signing key, got {} bytes", bytes.len());
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    let signing_key = lumen_auth::SigningKey::from_bytes(&arr);
    let signature = lumen_auth::sign(&manifest, &signing_key);
    let signed = lumen_auth::SignedManifest {
        manifest,
        signature_hex: hex_encode(signature.to_bytes().as_ref()),
        public_key_hex: hex_encode(signing_key.verifying_key().to_bytes().as_ref()),
    };
    let mut sidecar = output.to_path_buf();
    sidecar.set_extension("lumen-cco.json");
    lumen_auth::save_signed(&signed, &sidecar).map_err(|e| anyhow!("save_signed: {e}"))?;
    println!("wrote {}", sidecar.display());
    Ok(())
}

fn cmd_verify(signed: &std::path::Path, public_key_hex: Option<&str>) -> Result<()> {
    let pkg = lumen_auth::load_signed(signed).map_err(|e| anyhow!("load_signed: {e}"))?;
    let pk_hex = public_key_hex.unwrap_or(&pkg.public_key_hex);
    let pk_bytes = hex_decode(pk_hex)
        .ok_or_else(|| anyhow!("--public-key-hex must be 64 hex chars (32 bytes)"))?;
    if pk_bytes.len() != 32 {
        anyhow::bail!("expected 32-byte public key, got {} bytes", pk_bytes.len());
    }
    let mut pk_arr = [0u8; 32];
    pk_arr.copy_from_slice(&pk_bytes);
    let verifying_key = lumen_auth::VerifyingKey::from_bytes(&pk_arr)
        .map_err(|e| anyhow!("invalid public key: {e}"))?;
    let sig_bytes = hex_decode(&pkg.signature_hex)
        .ok_or_else(|| anyhow!("malformed signature hex in signed manifest"))?;
    if sig_bytes.len() != 64 {
        anyhow::bail!("expected 64-byte signature, got {} bytes", sig_bytes.len());
    }
    let mut sig_arr = [0u8; 64];
    sig_arr.copy_from_slice(&sig_bytes);
    let signature = lumen_auth::Signature::from_bytes(&sig_arr);
    let ok = lumen_auth::verify(&pkg.manifest, &signature, &verifying_key);
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "verified": ok,
        "manifest": pkg.manifest,
    }))?);
    if !ok { anyhow::bail!("signature verification failed"); }
    Ok(())
}

fn cmd_report(
    input: &std::path::Path,
    output: &std::path::Path,
    recipe: &std::path::Path,
    title: &str,
    html_out: &std::path::Path,
) -> Result<()> {
    let recipe_str = std::fs::read_to_string(recipe).context("read recipe")?;
    let frame_a = decode_image(input).map_err(|e| anyhow!("decode input: {e}"))?;
    let frame_b = decode_image(output).map_err(|e| anyhow!("decode output: {e}"))?;
    let metrics = lumen_measure::all_metrics(&frame_a, &frame_b)
        .map_err(|e| anyhow!("metrics: {e}"))?;
    let in_hash = lumen_io::hash_file(input).ok();
    let out_hash = lumen_io::hash_file(output).ok();
    let report = lumen_report::ReportInput {
        title,
        generated_at: time::OffsetDateTime::now_utc(),
        input_path: input,
        output_path: output,
        recipe_json: Some(recipe_str.as_str()),
        mse: Some(metrics.mse),
        psnr: if metrics.psnr.is_finite() { Some(metrics.psnr) } else { None },
        ssim: Some(metrics.ssim),
        input_hash: in_hash.as_deref(),
        output_hash: out_hash.as_deref(),
        render_ms: None,
        software_version: env!("CARGO_PKG_VERSION"),
    };
    lumen_report::save_html(&report, html_out).map_err(|e| anyhow!("save_html: {e}"))?;
    println!("wrote {}", html_out.display());
    Ok(())
}

fn cmd_export_video(
    frames_dir: &std::path::Path,
    output: &std::path::Path,
    codec: &str,
    fps: u32,
    crf: Option<u8>,
) -> Result<()> {
    use lumen_export::{Codec, VideoEncoder, VideoEncoderOptions};
    let codec = match codec.to_ascii_lowercase().as_str() {
        "h264" => Codec::H264,
        "h265" | "hevc" => Codec::H265,
        "prores" | "prores422" => Codec::ProRes422,
        other => anyhow::bail!("unknown codec: {other} (h264|h265|prores)"),
    };
    let mut entries: Vec<PathBuf> = std::fs::read_dir(frames_dir)
        .with_context(|| format!("read_dir {}", frames_dir.display()))?
        .filter_map(|r| r.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    entries.sort();
    if entries.is_empty() { anyhow::bail!("no frames found in {}", frames_dir.display()); }

    let first = decode_image(&entries[0]).map_err(|e| anyhow!("decode first frame: {e}"))?;
    let opts = VideoEncoderOptions::new(
        codec,
        first.width,
        first.height,
        lumen_core::Rational::new(fps as i64, 1),
    );
    let opts = match crf {
        Some(c) => VideoEncoderOptions { crf: Some(c), ..opts },
        None => opts,
    };
    let mut enc = VideoEncoder::open(output, opts).map_err(|e| anyhow!("encoder open: {e}"))?;
    enc.write_frame(&first).map_err(|e| anyhow!("encode frame 0: {e}"))?;
    for (i, p) in entries.iter().enumerate().skip(1) {
        let f = decode_image(p).map_err(|e| anyhow!("decode {}: {e}", p.display()))?;
        enc.write_frame(&f).map_err(|e| anyhow!("encode frame {i}: {e}"))?;
    }
    enc.finish().map_err(|e| anyhow!("encoder finish: {e}"))?;
    println!("wrote {} ({} frames)", output.display(), entries.len());
    Ok(())
}

fn cmd_plugin(
    plugin: &std::path::Path,
    input: &std::path::Path,
    output: &std::path::Path,
    raw_params: &[String],
) -> Result<()> {
    use lumen_core::Effect;
    let p = lumen_api::load_lua_plugin(plugin).map_err(|e| anyhow!("load plugin: {e}"))?;
    let mut params = ParamValues::new();
    for kv in raw_params {
        let (k, v) = kv.split_once('=').ok_or_else(|| anyhow!("--param '{kv}' missing '='"))?;
        params.insert(k, parse_param_value(v));
    }
    params.validate_and_fill(p.parameters())
        .map_err(|e| anyhow!("parameter validation: {e}"))?;
    let frame = decode_image(input).map_err(|e| anyhow!("decode: {e}"))?;
    let mut ctx = Context::for_still_srgb();
    let out = p.apply(&mut ctx, frame, &params).map_err(|e| anyhow!("plugin apply: {e}"))?;
    encode_image(out, output, ImageEncodeOptions::default())
        .map_err(|e| anyhow!("encode: {e}"))?;
    println!("wrote {}", output.display());
    Ok(())
}

fn cmd_qa(cases_dir: &std::path::Path) -> Result<()> {
    let set = lumen_qa::GoldenSet::from_dir(cases_dir)
        .map_err(|e| anyhow!("from_dir: {e}"))?;
    let registry = build_registry()?;
    let results = lumen_qa::GoldenSet::run_all(&set, &registry);
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    println!("{}", serde_json::to_string_pretty(&results)?);
    if passed == total {
        println!("OK: {passed}/{total} cases passed");
        Ok(())
    } else {
        anyhow::bail!("FAIL: {passed}/{total} cases passed")
    }
}

// ─── Hex helpers ─────────────────────────────────────────────────────────

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) { return None; }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

fn cmd_measure(a: &std::path::Path, b: &std::path::Path) -> Result<()> {
    let frame_a = decode_image(a).map_err(|e| anyhow!("decode a: {e}"))?;
    let frame_b = decode_image(b).map_err(|e| anyhow!("decode b: {e}"))?;
    let m = lumen_measure::all_metrics(&frame_a, &frame_b)
        .map_err(|e| anyhow!("metrics: {e}"))?;
    let json = serde_json::json!({
        "mse":  m.mse,
        "psnr": m.psnr,
        "ssim": m.ssim,
        "a":    a.display().to_string(),
        "b":    b.display().to_string(),
    });
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}

fn init_tracing(level: u8) {
    let filter = match level {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
        )
        .with_target(false)
        .compact()
        .init();
}

fn cmd_probe(path: &std::path::Path) -> Result<()> {
    let asset = probe(path).with_context(|| format!("probing {}", path.display()))?;
    println!("{}", serde_json::to_string_pretty(&asset)?);
    Ok(())
}

fn build_registry() -> Result<EffectRegistry> {
    let r = EffectRegistry::new();
    lumen_fx_exposure::register_all(&r).map_err(|e| anyhow!("register exposure: {e}"))?;
    lumen_fx_color::register_all(&r).map_err(|e| anyhow!("register color: {e}"))?;
    lumen_fx_sharpen::register_all(&r).map_err(|e| anyhow!("register sharpen: {e}"))?;
    lumen_fx_denoise::register_all(&r).map_err(|e| anyhow!("register denoise: {e}"))?;
    lumen_fx_geometric::register_all(&r).map_err(|e| anyhow!("register geometric: {e}"))?;
    lumen_fx_upscale::register_all(&r).map_err(|e| anyhow!("register upscale: {e}"))?;
    lumen_fx_mask::register_all(&r).map_err(|e| anyhow!("register mask: {e}"))?;
    lumen_fx_stabilize::register_all(&r).map_err(|e| anyhow!("register stabilize: {e}"))?;
    lumen_fx_deblur::register_all(&r).map_err(|e| anyhow!("register deblur: {e}"))?;
    lumen_fx_temporal::register_all(&r).map_err(|e| anyhow!("register temporal: {e}"))?;
    lumen_fx_weather::register_all(&r).map_err(|e| anyhow!("register weather: {e}"))?;
    lumen_fx_compression::register_all(&r).map_err(|e| anyhow!("register compression: {e}"))?;
    lumen_fx_face::register_all(&r).map_err(|e| anyhow!("register face: {e}"))?;
    lumen_fx_text::register_all(&r).map_err(|e| anyhow!("register text: {e}"))?;
    lumen_fx_modalities::register_all(&r).map_err(|e| anyhow!("register modalities: {e}"))?;
    Ok(r)
}

fn cmd_list_effects() -> Result<()> {
    let r = build_registry()?;
    for id in r.ids() {
        let e = r.get(&id).expect("registered above");
        let m = e.metadata();
        println!("{:<48} {}", m.id, m.display_name);
        for spec in e.parameters() {
            println!("    {:<16} {}", spec.id, spec.description);
        }
    }
    Ok(())
}

fn cmd_apply(
    input: &std::path::Path,
    output: &std::path::Path,
    effect_id: &str,
    raw_params: &[String],
    jpeg_quality: u8,
) -> Result<()> {
    let registry = build_registry()?;
    let effect = registry
        .get(effect_id)
        .ok_or_else(|| anyhow!("unknown effect: {effect_id}\n\nrun `lumen list-effects` to see all"))?;

    let mut params = ParamValues::new();
    for kv in raw_params {
        let (k, v) =
            kv.split_once('=').ok_or_else(|| anyhow!("--param '{kv}' missing '='"))?;
        params.insert(k, parse_param_value(v));
    }
    params
        .validate_and_fill(effect.parameters())
        .map_err(|e| anyhow!("parameter validation: {e}"))?;

    info!("decoding {}", input.display());
    let frame = decode_image(input).map_err(|e| anyhow!("decode: {e}"))?;
    info!(width = frame.width, height = frame.height, "applying {}", effect_id);

    let mut ctx = Context::for_still_srgb();
    let out = effect.apply(&mut ctx, frame, &params).map_err(|e| anyhow!("apply: {e}"))?;

    info!("encoding {}", output.display());
    let written = encode_image(
        out,
        output,
        ImageEncodeOptions { jpeg_quality, format: None },
    )
    .map_err(|e| anyhow!("encode: {e}"))?;
    println!("wrote {}", written.display());
    Ok(())
}

// ─── Pipeline recipes ─────────────────────────────────────────────────────

/// Linear-chain recipe format. Phase 1 supports linear chains only;
/// branched DAGs land alongside multi-input effects in Phase 3.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Recipe {
    /// Input file path (relative to the recipe file or absolute).
    pub input: PathBuf,
    /// Output file path.
    pub output: PathBuf,
    /// Ordered list of effects to apply.
    pub chain: Vec<RecipeStep>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RecipeStep {
    /// Effect id (e.g. `"lumen-fx-denoise.gaussian"`).
    pub effect: String,
    /// Optional human label.
    #[serde(default)]
    pub label: Option<String>,
    /// Parameter values keyed by parameter id.
    #[serde(default)]
    pub params: serde_json::Value,
}

fn cmd_pipeline(recipe_path: &std::path::Path, jpeg_quality: u8) -> Result<()> {
    let recipe_str = std::fs::read_to_string(recipe_path)
        .with_context(|| format!("reading recipe {}", recipe_path.display()))?;
    let recipe: Recipe = serde_json::from_str(&recipe_str).with_context(|| {
        format!("parsing recipe {} as JSON", recipe_path.display())
    })?;

    // Resolve relative paths against the recipe's directory.
    let base = recipe_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let resolve = |p: &std::path::Path| {
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            base.join(p)
        }
    };
    let input_path = resolve(&recipe.input);
    let output_path = resolve(&recipe.output);

    let registry = build_registry()?;

    // Build the linear graph.
    let mut graph = Graph::new();
    let src_node = graph.insert(Node::new(special_effect_ids::SOURCE, "source"));
    let mut prev = src_node;
    for (i, step) in recipe.chain.iter().enumerate() {
        let mut params = ParamValues::new();
        if let serde_json::Value::Object(map) = &step.params {
            for (k, v) in map {
                params.insert(k.clone(), json_to_param(v).ok_or_else(|| {
                    anyhow!("step {i}: param '{k}' has unsupported JSON type")
                })?);
            }
        }
        let label = step
            .label
            .clone()
            .unwrap_or_else(|| format!("step{i:02}"));
        let node = graph.insert(
            Node::new(step.effect.clone(), label).with_input(prev).with_params(params),
        );
        prev = node;
    }
    let sink_node = graph.insert(
        Node::new(special_effect_ids::SINK, "sink").with_input(prev),
    );
    graph.add_sink(sink_node);

    // Execute via Scheduler.
    let mut ctx = Context::for_still_srgb();
    let written = std::cell::RefCell::new(None::<PathBuf>);

    let source = CliSource(|_id: NodeId, _params: &ParamValues| -> lumen_core::Result<Frame> {
        decode_image(&input_path)
    });
    let sink = CliSink(|_id: NodeId, _params: &ParamValues, frame: Frame| -> lumen_core::Result<()> {
        let p = encode_image(
            frame,
            &output_path,
            ImageEncodeOptions { jpeg_quality, format: None },
        )?;
        *written.borrow_mut() = Some(p);
        Ok(())
    });
    let mut sched = Scheduler {
        registry: &registry,
        ctx: &mut ctx,
        source_loader: source,
        sink_writer: sink,
    };

    sched
        .run(&graph)
        .map_err(|e| anyhow!("pipeline run failed: {e}"))?;

    if let Some(p) = written.into_inner() {
        println!("wrote {}", p.display());
    }
    Ok(())
}

pub(crate) fn json_to_param(v: &serde_json::Value) -> Option<ParamValue> {
    match v {
        serde_json::Value::Bool(b) => Some(ParamValue::Bool(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(ParamValue::Int(i))
            } else {
                n.as_f64().map(ParamValue::Float)
            }
        }
        serde_json::Value::String(s) => Some(ParamValue::String(s.clone())),
        _ => None,
    }
}

fn parse_param_value(s: &str) -> ParamValue {
    if let Ok(b) = s.parse::<bool>() {
        return ParamValue::Bool(b);
    }
    if let Ok(i) = s.parse::<i64>() {
        return ParamValue::Int(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        return ParamValue::Float(f);
    }
    ParamValue::String(s.to_string())
}

// Newtype wrappers so we can implement the foreign SourceLoader / SinkWriter
// traits for closures. The orphan rule forbids `impl<F> Trait for F`.
pub(crate) struct CliSource<F>(pub(crate) F);
pub(crate) struct CliSink<F>(pub(crate) F);

impl<F> SourceLoader for CliSource<F>
where
    F: FnMut(NodeId, &ParamValues) -> lumen_core::Result<Frame>,
{
    fn load(&mut self, id: NodeId, p: &ParamValues) -> lumen_core::Result<Frame> {
        (self.0)(id, p)
    }
}

impl<F> SinkWriter for CliSink<F>
where
    F: FnMut(NodeId, &ParamValues, Frame) -> lumen_core::Result<()>,
{
    fn write(&mut self, id: NodeId, p: &ParamValues, f: Frame) -> lumen_core::Result<()> {
        (self.0)(id, p, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bool_int_float_string() {
        assert!(matches!(parse_param_value("true"), ParamValue::Bool(true)));
        assert!(matches!(parse_param_value("42"), ParamValue::Int(42)));
        assert!(matches!(parse_param_value("0.5"), ParamValue::Float(_)));
        assert!(matches!(parse_param_value("hi"), ParamValue::String(_)));
    }
}
