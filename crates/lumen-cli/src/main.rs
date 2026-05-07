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

mod auto;
mod batch;
mod case;
mod clarify;
mod colorize;
mod operator;
mod presets;
mod project;
mod report;
mod serve;
mod smart;
mod stack;
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
enum OperatorCommand {
    /// Mint a new Ed25519 keypair + identity at ~/.lumen/operator.json.
    Init {
        #[arg(long)] name: String,
        #[arg(long)] agency: String,
        #[arg(long, default_value = "")] identifier: String,
        #[arg(long, default_value_t = false)] force: bool,
    },
    /// Print the local operator identity (public key + metadata).
    Show,
}

#[derive(Subcommand, Debug)]
enum CaseCommand {
    /// Open a new case folder. Records the original input's BLAKE3
    /// hash and writes the first audit entry (case-init).
    Init {
        /// Path of the case folder to create.
        #[arg(long)] dir: PathBuf,
        #[arg(long)] case_id: String,
        #[arg(long)] evidence_id: String,
        #[arg(long)] case_name: String,
        #[arg(long)] agency: String,
        /// Optional original input file. Copied into <dir>/inputs/
        /// and its hash is recorded in case.json.
        #[arg(long)] input: Option<PathBuf>,
    },
    /// Run a recipe within a case — auto-records every artifact and
    /// appends a signed `render` entry to the audit log.
    Render {
        #[arg(long)] dir: PathBuf,
        #[arg(long)] recipe: PathBuf,
        #[arg(long)] input: PathBuf,
        #[arg(long)] output: PathBuf,
        /// Note attached to the audit entry (e.g. "applied clarify aggressive").
        #[arg(long, default_value = "")] note: String,
    },
    /// Append a free-form note to the audit log (e.g. reviewer comments).
    Note {
        #[arg(long)] dir: PathBuf,
        #[arg(long)] note: String,
    },
    /// Verify the audit log: every signature valid, chain unbroken.
    /// With `--strict`, also re-hash every artifact referenced by any
    /// audit entry and confirm a matching file still exists in the
    /// case folder. This catches post-hoc tampering of `inputs/`,
    /// `outputs/`, `recipes/`, or `stages/` files that the regular
    /// audit doesn't cover.
    Audit {
        #[arg(long)] dir: PathBuf,
        /// Re-hash every referenced artifact and verify it still
        /// matches the hash recorded in the audit log.
        #[arg(long, default_value_t = false)] strict: bool,
        /// Require at least one `sign-off` entry signed by a pubkey
        /// different from the operator who opened the case
        /// (analyst-vs-reviewer separation). Exits non-zero if no
        /// such sign-off exists.
        #[arg(long, default_value_t = false)] require_signoff: bool,
    },
    /// Append a reviewer's sign-off to the audit log. The reviewer
    /// must be running under a different operator identity (different
    /// `~/.lumen/operator.json` or `LUMEN_OPERATOR=...`) than the
    /// analyst who opened the case — `lumen case audit --require-signoff`
    /// rejects same-operator sign-offs to prevent self-approval.
    SignOff {
        #[arg(long)] dir: PathBuf,
        /// `approve` or `reject`. Recorded verbatim in the entry note.
        #[arg(long)] decision: String,
        /// Reviewer's free-form note explaining the decision.
        #[arg(long, default_value = "")] note: String,
    },
    /// Export the case folder as a tamper-evident zip.
    Export {
        #[arg(long)] dir: PathBuf,
        #[arg(long)] output: PathBuf,
    },
    /// Render a self-contained HTML report into `reports/` showing the
    /// case metadata, full audit timeline, and embedded thumbnails of
    /// the original input, final output, and stage frames. Useful for
    /// reviewer inspection without running the CLI.
    Report {
        #[arg(long)] dir: PathBuf,
        /// Output filename (relative to `reports/`). Default: case-report.html
        #[arg(long, default_value = "case-report.html")] output: String,
    },
}

#[derive(Subcommand, Debug)]
enum ProjectCommand {
    /// Pretty-print a `.lumenproj` file's metadata.
    Show {
        /// Path to the `.lumenproj` file.
        path: PathBuf,
    },
    /// Run a project's graph against the first still-image asset.
    Run {
        /// Path to the `.lumenproj` file.
        #[arg(long)] project: PathBuf,
        /// Output file path.
        #[arg(long)] output: PathBuf,
        /// JPEG quality (1-100), used only when output is JPEG.
        #[arg(long, default_value_t = 92)] jpeg_quality: u8,
    },
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
        /// Optional directory to write a PNG snapshot after each
        /// effect runs. Useful for diagnosing "the pipeline went crazy
        /// at step 4". Files are named `NN-effect_id.png`.
        #[arg(long)] save_stages: Option<PathBuf>,
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
    /// Auto-enhance: analyze the input, pick optimal effect parameters,
    /// build a 4-step chain (BC -> gamma -> saturation -> sharpen),
    /// and run it. Mirrors the in-browser demo's auto-enhance button.
    AutoEnhance {
        #[arg(long)] input: PathBuf,
        #[arg(long)] output: PathBuf,
        /// If set, print the analyzed recipe JSON to stdout instead
        /// of running it.
        #[arg(long)] print_recipe: bool,
    },
    /// Forensic operator identity (`~/.lumen/operator.json`).
    Operator {
        #[command(subcommand)]
        sub: OperatorCommand,
    },
    /// Forensic case management — case folder + signed audit log.
    Case {
        #[command(subcommand)]
        sub: CaseCommand,
    },
    /// Combine multiple photos of the same scene into one.
    ///
    /// All inputs must have identical dimensions. Phase 1 stacking is
    /// unaligned — feed tripod-stable or already-aligned inputs.
    /// Modes: mean (average — best for noise reduction), median
    /// (drops transient occlusions), max (star trails), min.
    Stack {
        /// Two or more input image paths.
        #[arg(long = "input", value_name = "PATH", num_args = 1..)]
        inputs: Vec<PathBuf>,
        #[arg(long)] output: PathBuf,
        /// Stacking mode: mean | median | max | min.
        #[arg(long, default_value = "mean")] mode: String,
    },
    /// Apply a recipe to every image in a folder.
    ///
    /// Walks --input-dir for image files (png/jpg/tif/webp/bmp/raw),
    /// runs the recipe against each, and writes outputs to --output-dir
    /// with the same filename. Existing outputs are skipped unless
    /// --force is passed. Parallel via rayon.
    Batch {
        #[arg(long)] input_dir: PathBuf,
        #[arg(long)] output_dir: PathBuf,
        /// Recipe JSON file. The `input` and `output` fields are
        /// overridden per-image.
        #[arg(long)] recipe: PathBuf,
        /// Re-run even if an output already exists.
        #[arg(long, default_value_t = false)] force: bool,
        #[arg(long, default_value_t = 92)] jpeg_quality: u8,
    },
    /// Operate on a `.lumenproj` project file.
    Project {
        #[command(subcommand)]
        sub: ProjectCommand,
    },
    /// Heuristic colorization — channel_isolate(luma) -> duotone with
    /// a chosen palette, plus CLAHE + sharpen. No model required.
    ///
    /// Palettes: night, day, sepia, cyan-orange, noir.
    Colorize {
        #[arg(long)] input: PathBuf,
        #[arg(long)] output: PathBuf,
        /// Palette name. Default: night.
        #[arg(long, default_value = "night")] palette: String,
        #[arg(long)] print_recipe: bool,
    },
    /// Apply a stylistic preset chain (no input analysis).
    ///
    /// Available: pop, bw, vintage, sharpen, restore.
    Style {
        #[arg(long)] input: PathBuf,
        #[arg(long)] output: PathBuf,
        /// Preset name (pop / bw / vintage / sharpen / restore).
        #[arg(long)] name: String,
        #[arg(long)] print_recipe: bool,
    },
    /// Smart Auto — analyze the input and pick auto-enhance OR
    /// clarify automatically. The one-button entry point.
    Smart {
        #[arg(long)] input: PathBuf,
        #[arg(long)] output: PathBuf,
        /// 2x bicubic upscale at the end (clarify path only).
        #[arg(long, default_value_t = false)] upscale: bool,
        #[arg(long)] print_recipe: bool,
    },
    /// Surveillance / forensic clarification preset — denoise + deblock
    /// + dehaze + CLAHE + laplacian deblur + sharpen + tone stretch.
    ///
    /// Optimized for CCTV / cell-phone / low-quality stills.
    Clarify {
        #[arg(long)] input: PathBuf,
        #[arg(long)] output: PathBuf,
        /// Strength of the chain: light / standard / aggressive / plate /
        /// forensic. Forensic is the maximum-fidelity tier (bilateral +
        /// Wiener + Richardson-Lucy + 4x Lanczos) for evidence work.
        #[arg(long, default_value = "standard")] strength: String,
        /// 2x bicubic upscale at the end (default true).
        #[arg(long, default_value_t = true)] upscale: bool,
        /// Print the recipe JSON to stdout instead of running it.
        #[arg(long)] print_recipe: bool,
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
        Command::Pipeline { recipe, jpeg_quality, save_stages } => {
            cmd_pipeline(&recipe, jpeg_quality, save_stages.as_deref())
        }
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
        Command::AutoEnhance { input, output, print_recipe } => {
            cmd_auto_enhance(&input, &output, print_recipe)
        }
        Command::Clarify { input, output, strength, upscale, print_recipe } => {
            cmd_clarify(&input, &output, &strength, upscale, print_recipe)
        }
        Command::Smart { input, output, upscale, print_recipe } => {
            cmd_smart(&input, &output, upscale, print_recipe)
        }
        Command::Style { input, output, name, print_recipe } => {
            cmd_style(&input, &output, &name, print_recipe)
        }
        Command::Colorize { input, output, palette, print_recipe } => {
            cmd_colorize(&input, &output, &palette, print_recipe)
        }
        Command::Batch { input_dir, output_dir, recipe, force, jpeg_quality } => {
            cmd_batch(&input_dir, &output_dir, &recipe, force, jpeg_quality)
        }
        Command::Stack { inputs, output, mode } => {
            stack::cmd_stack(&inputs, &output, &mode)
        }
        Command::Operator { sub } => match sub {
            OperatorCommand::Init { name, agency, identifier, force } => {
                cmd_operator_init(&name, &agency, &identifier, force)
            }
            OperatorCommand::Show => cmd_operator_show(),
        },
        Command::Case { sub } => match sub {
            CaseCommand::Init { dir, case_id, evidence_id, case_name, agency, input } => {
                cmd_case_init(&dir, &case_id, &evidence_id, &case_name, &agency, input.as_deref())
            }
            CaseCommand::Render { dir, recipe, input, output, note } => {
                cmd_case_render(&dir, &recipe, &input, &output, &note)
            }
            CaseCommand::Note { dir, note } => cmd_case_note(&dir, &note),
            CaseCommand::Audit { dir, strict, require_signoff } => {
                cmd_case_audit(&dir, strict, require_signoff)
            }
            CaseCommand::SignOff { dir, decision, note } => {
                cmd_case_signoff(&dir, &decision, &note)
            }
            CaseCommand::Export { dir, output } => cmd_case_export(&dir, &output),
            CaseCommand::Report { dir, output } => cmd_case_report(&dir, &output),
        },
        Command::Project { sub } => match sub {
            ProjectCommand::Show { path } => project::cmd_project_show(&path),
            ProjectCommand::Run { project: p, output, jpeg_quality } => {
                project::cmd_project_run(&p, &output, jpeg_quality)
            }
        },
    }
}

// ─── Operator + case (forensic) dispatchers ──────────────────────────────

fn cmd_operator_init(
    name: &str,
    agency: &str,
    identifier: &str,
    force: bool,
) -> Result<()> {
    let id = operator::init(name, agency, identifier, force)?;
    println!("{}", serde_json::to_string_pretty(&id)?);
    eprintln!(
        "operator file written to {}",
        operator::operator_path().display()
    );
    Ok(())
}

fn cmd_operator_show() -> Result<()> {
    let id = operator::current_identity()
        .context("no operator. Run `lumen operator init --name … --agency …` first.")?;
    println!("{}", serde_json::to_string_pretty(&id)?);
    Ok(())
}

fn cmd_case_init(
    dir: &std::path::Path,
    case_id: &str,
    evidence_id: &str,
    case_name: &str,
    agency: &str,
    input: Option<&std::path::Path>,
) -> Result<()> {
    let m = case::init(dir, case_id, evidence_id, case_name, agency, input)?;
    println!("{}", serde_json::to_string_pretty(&m)?);
    eprintln!("case opened at {}", dir.display());
    Ok(())
}

fn cmd_case_render(
    dir: &std::path::Path,
    recipe_path: &std::path::Path,
    input: &std::path::Path,
    output: &std::path::Path,
    note: &str,
) -> Result<()> {
    // Verify the case is well-formed before adding to it.
    let _meta = case::load_metadata(dir)?;
    case::verify_audit_log(dir)
        .map_err(|e| anyhow!("audit log was already broken before this render: {e}"))?;

    // Stage the recipe + input into the case so they're permanently captured.
    let recipe_dst = dir.join("recipes").join(
        recipe_path.file_name().ok_or_else(|| anyhow!("recipe has no filename"))?,
    );
    std::fs::copy(recipe_path, &recipe_dst)?;
    let input_dst = dir.join("inputs").join(
        input.file_name().ok_or_else(|| anyhow!("input has no filename"))?,
    );
    if input_dst != input.to_path_buf() {
        std::fs::copy(input, &input_dst)?;
    }
    let recipe_hash = lumen_io::hash_file(&recipe_dst).ok();
    let input_hash = lumen_io::hash_file(&input_dst).ok();

    // Read recipe, override input/output to live inside the case folder.
    let recipe_str = std::fs::read_to_string(&recipe_dst)?;
    let mut recipe: Recipe = serde_json::from_str(&recipe_str)?;
    let output_dst = dir.join("outputs").join(
        output.file_name().ok_or_else(|| anyhow!("output has no filename"))?,
    );
    recipe.input = input_dst.clone();
    recipe.output = output_dst.clone();

    // Run, with stages captured to <dir>/stages/<output_stem>/.
    let stages_dir = dir
        .join("stages")
        .join(output.file_stem().unwrap_or_else(|| std::ffi::OsStr::new("render")));
    run_chain_in_memory_with_stages(&recipe, Some(&stages_dir))?;

    let output_hash = lumen_io::hash_file(&output_dst).ok();

    // Append signed audit entry.
    let entry = case::append_entry(
        dir,
        case::AuditEntryDraft {
            action: "render".into(),
            note: if note.is_empty() {
                format!("Recipe applied: {}", recipe_dst.file_name().unwrap().to_string_lossy())
            } else {
                note.to_string()
            },
            input_hash,
            output_hash,
            recipe_hash,
        },
    )?;

    println!("wrote {}", output_dst.display());
    eprintln!(
        "audit entry seq {} signed by operator {}",
        entry.seq, entry.operator_public_key_hex
    );
    Ok(())
}

fn cmd_case_note(dir: &std::path::Path, note: &str) -> Result<()> {
    let entry = case::append_entry(
        dir,
        case::AuditEntryDraft {
            action: "note".into(),
            note: note.to_string(),
            ..Default::default()
        },
    )?;
    println!("{}", serde_json::to_string_pretty(&entry)?);
    Ok(())
}

fn cmd_case_signoff(dir: &std::path::Path, decision: &str, note: &str) -> Result<()> {
    let decision_norm = decision.to_lowercase();
    if !matches!(decision_norm.as_str(), "approve" | "reject") {
        return Err(anyhow!(
            "decision must be 'approve' or 'reject', got '{}'",
            decision
        ));
    }
    // Self-signoff sanity check: warn if the current operator pubkey
    // matches the operator who opened the case. The audit step's
    // --require-signoff also enforces this, but a CLI-time warning
    // helps catch accidents earlier.
    if let (Ok(metadata), Ok(op)) = (case::load_metadata(dir), operator::current_identity()) {
        if op.public_key_hex == metadata.created_by.public_key_hex {
            eprintln!(
                "warning: signing off as the same operator who opened the case. \
                 For analyst-vs-reviewer separation, run sign-off under a different \
                 LUMEN_OPERATOR identity."
            );
        }
    }
    let formatted_note = if note.is_empty() {
        format!("decision: {}", decision_norm)
    } else {
        format!("decision: {} — {}", decision_norm, note)
    };
    let entry = case::append_entry(
        dir,
        case::AuditEntryDraft {
            action: "sign-off".into(),
            note: formatted_note,
            ..Default::default()
        },
    )?;
    println!("{}", serde_json::to_string_pretty(&entry)?);
    Ok(())
}

fn cmd_case_audit(dir: &std::path::Path, strict: bool, require_signoff: bool) -> Result<()> {
    let m = case::load_metadata(dir)?;
    let signoff_status = case::signoff_status(dir, &m)?;
    if strict {
        let report = case::verify_audit_log_strict(dir)?;
        let any_missing = report.artifacts.iter().any(|a| !a.ok);
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "case": m,
                "audit_chain_verified": true,
                "all_artifacts_match": report.all_artifacts_match,
                "signoff": signoff_status,
                "entry_count": report.entries.len(),
                "artifact_count": report.artifacts.len(),
                "artifacts": report.artifacts,
                "entries": report.entries,
            }))?
        );
        if any_missing {
            return Err(anyhow!(
                "strict audit failed: one or more referenced artifacts \
                 do not match their recorded BLAKE3 (see `artifacts` array)"
            ));
        }
    } else {
        let entries = case::verify_audit_log(dir)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "case": m,
                "audit_chain_verified": true,
                "signoff": signoff_status,
                "entry_count": entries.len(),
                "entries": entries,
            }))?
        );
    }
    if require_signoff && !signoff_status.has_independent_approval {
        return Err(anyhow!(
            "audit requires sign-off, but no independent reviewer has approved \
             this case (analyst's own pubkey doesn't count). Run \
             `lumen case sign-off --decision approve --dir <case>` under a \
             different operator identity."
        ));
    }
    Ok(())
}

fn cmd_case_report(dir: &std::path::Path, output_filename: &str) -> Result<()> {
    let out = report::render_html_report(dir, output_filename)?;
    let size = std::fs::metadata(&out)?.len();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "report_path": out.display().to_string(),
            "size_bytes": size,
        }))?
    );
    Ok(())
}

fn cmd_case_export(dir: &std::path::Path, output: &std::path::Path) -> Result<()> {
    // Verify before export so we never bundle a broken chain.
    case::verify_audit_log(dir)?;
    case::export_zip(dir, output)?;
    let h = lumen_io::hash_file(output).ok();
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "exported_zip": output.display().to_string(),
        "size_bytes": std::fs::metadata(output)?.len(),
        "blake3_hash": h,
    }))?);
    Ok(())
}

fn cmd_batch(
    input_dir: &std::path::Path,
    output_dir: &std::path::Path,
    recipe_path: &std::path::Path,
    force: bool,
    jpeg_quality: u8,
) -> Result<()> {
    let s = std::fs::read_to_string(recipe_path)
        .with_context(|| format!("reading recipe {}", recipe_path.display()))?;
    let recipe: Recipe = serde_json::from_str(&s)
        .with_context(|| format!("parsing recipe {}", recipe_path.display()))?;
    let stats = batch::run_batch(
        input_dir,
        output_dir,
        &recipe,
        batch::BatchOptions { force, jpeg_quality, out_ext: None },
    )?;
    eprintln!(
        "batch: {} processed, {} skipped, {} failed",
        stats.processed, stats.skipped, stats.failed
    );
    if stats.failed > 0 {
        anyhow::bail!("batch had {} failures", stats.failed);
    }
    Ok(())
}

fn cmd_colorize(
    input: &std::path::Path,
    output: &std::path::Path,
    palette: &str,
    print_recipe: bool,
) -> Result<()> {
    let recipe = colorize::build_heuristic_recipe(input, output, palette)?;
    if print_recipe {
        println!("{}", serde_json::to_string_pretty(&recipe)?);
        return Ok(());
    }
    eprintln!(
        "colorize (heuristic, palette={}, {} steps):",
        palette,
        recipe.chain.len()
    );
    for s in &recipe.chain {
        eprintln!("  {} {}", s.effect, s.params);
    }
    run_chain_in_memory(&recipe)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn cmd_style(
    input: &std::path::Path,
    output: &std::path::Path,
    name: &str,
    print_recipe: bool,
) -> Result<()> {
    let recipe = presets::build_style_recipe(input, output, name)?;
    if print_recipe {
        println!("{}", serde_json::to_string_pretty(&recipe)?);
        return Ok(());
    }
    eprintln!("style preset '{}' ({} steps):", name, recipe.chain.len());
    for s in &recipe.chain {
        eprintln!("  {} {}", s.effect, s.params);
    }
    run_chain_in_memory(&recipe)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn cmd_smart(
    input: &std::path::Path,
    output: &std::path::Path,
    upscale: bool,
    print_recipe: bool,
) -> Result<()> {
    let (stats, strategy, recipe) = smart::build_smart_recipe(input, output, upscale)?;
    if print_recipe {
        println!("{}", serde_json::to_string_pretty(&recipe)?);
        return Ok(());
    }
    eprintln!(
        "stats: p01={:.3} p50={:.3} p99={:.3} chroma̅={:.3} edges̅={:.3}",
        stats.p01, stats.p50, stats.p99, stats.chroma_mean, stats.edge_mean,
    );
    eprintln!("verdict: {} ({} steps)", strategy.label(), recipe.chain.len());
    for s in &recipe.chain {
        eprintln!("  {} {}", s.effect, s.params);
    }
    run_chain_in_memory(&recipe)?;
    println!("wrote {}", output.display());
    Ok(())
}

// ─── Auto-enhance + clarify dispatch ─────────────────────────────────────

pub(crate) fn run_chain_in_memory(recipe: &Recipe) -> Result<()> {
    run_chain_in_memory_with_stages(recipe, None)
}

/// Run the chain. If `stages_dir` is `Some`, also write a PNG snapshot
/// to that directory after every effect: `00-input.png`, `01-<label>.png`,
/// `02-<label>.png`, …, `NN-output.png`. Lets users see exactly where a
/// pipeline goes wrong.
pub(crate) fn run_chain_in_memory_with_stages(
    recipe: &Recipe,
    stages_dir: Option<&std::path::Path>,
) -> Result<()> {
    let registry = build_registry()?;

    if let Some(dir) = stages_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating stages dir {}", dir.display()))?;
    }

    // If we want stages, we run effects sequentially ourselves so we can
    // capture each intermediate frame. Otherwise use the Scheduler.
    if let Some(dir) = stages_dir {
        // Sequential execution path with stage capture.
        let mut frame = decode_image(&recipe.input).map_err(|e| anyhow!("decode: {e}"))?;
        let stage_path = dir.join("00-input.png");
        encode_image(
            frame.clone(),
            &stage_path,
            ImageEncodeOptions::default(),
        )
        .map_err(|e| anyhow!("write stage 00: {e}"))?;
        eprintln!("stage 00 input -> {}", stage_path.display());

        let mut ctx = Context::for_still_srgb();
        for (i, step) in recipe.chain.iter().enumerate() {
            let effect = registry
                .get(&step.effect)
                .ok_or_else(|| anyhow!("effect '{}' not in registry", step.effect))?;
            let mut params = ParamValues::new();
            if let serde_json::Value::Object(map) = &step.params {
                for (k, v) in map {
                    let pv = json_to_param(v).ok_or_else(|| {
                        anyhow!("step {i}: param '{k}' has unsupported JSON type")
                    })?;
                    params.insert(k.clone(), pv);
                }
            }
            params
                .validate_and_fill(effect.parameters())
                .map_err(|e| anyhow!("step {i} params: {e}"))?;
            frame = effect
                .apply(&mut ctx, frame, &params)
                .map_err(|e| anyhow!("step {i} apply: {e}"))?;
            let safe_label = step
                .effect
                .replace('.', "_")
                .replace(['/', '\\', ' '], "_");
            let stage_path = dir.join(format!("{:02}-{}.png", i + 1, safe_label));
            encode_image(
                frame.clone(),
                &stage_path,
                ImageEncodeOptions::default(),
            )
            .map_err(|e| anyhow!("write stage {}: {e}", i + 1))?;
            eprintln!("stage {:02} {} -> {}", i + 1, step.effect, stage_path.display());
        }

        // Final output
        encode_image(
            frame,
            &recipe.output,
            ImageEncodeOptions { jpeg_quality: 92, format: None },
        )
        .map_err(|e| anyhow!("encode final: {e}"))?;
        return Ok(());
    }

    // No stages requested — use the Scheduler.
    let mut graph = Graph::new();
    let src_node = graph.insert(Node::new(special_effect_ids::SOURCE, "source"));
    let mut prev = src_node;
    for (i, step) in recipe.chain.iter().enumerate() {
        let mut params = ParamValues::new();
        if let serde_json::Value::Object(map) = &step.params {
            for (k, v) in map {
                let pv = json_to_param(v).ok_or_else(|| {
                    anyhow!("step {i}: param '{k}' has unsupported JSON type")
                })?;
                params.insert(k.clone(), pv);
            }
        }
        let label = step.label.clone().unwrap_or_else(|| format!("step{i:02}"));
        let node = graph.insert(
            Node::new(step.effect.clone(), label).with_input(prev).with_params(params),
        );
        prev = node;
    }
    let sink_node = graph.insert(Node::new(special_effect_ids::SINK, "sink").with_input(prev));
    graph.add_sink(sink_node);

    let mut ctx = Context::for_still_srgb();
    let written = std::cell::RefCell::new(None::<PathBuf>);
    let input_path = recipe.input.clone();
    let output_path = recipe.output.clone();
    let source = CliSource(move |_id: NodeId, _params: &ParamValues| -> lumen_core::Result<Frame> {
        decode_image(&input_path)
    });
    let sink = CliSink(move |_id: NodeId, _params: &ParamValues, frame: Frame| -> lumen_core::Result<()> {
        let p = encode_image(
            frame,
            &output_path,
            ImageEncodeOptions { jpeg_quality: 92, format: None },
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
    sched.run(&graph).map_err(|e| anyhow!("scheduler: {e}"))?;
    Ok(())
}

fn cmd_auto_enhance(
    input: &std::path::Path,
    output: &std::path::Path,
    print_recipe: bool,
) -> Result<()> {
    let (stats, recipe) = auto::analyze_path(input, output)?;
    if print_recipe {
        println!("{}", serde_json::to_string_pretty(&recipe)?);
        return Ok(());
    }
    eprintln!(
        "stats: p01={:.3} p50={:.3} p99={:.3} chroma̅={:.3} edges̅={:.3} luma̅={:.3}",
        stats.p01, stats.p50, stats.p99,
        stats.chroma_mean, stats.edge_mean, stats.luminance_mean,
    );
    eprintln!(
        "auto chain ({} steps):",
        recipe.chain.len()
    );
    for s in &recipe.chain {
        eprintln!("  {} {}", s.effect, s.params);
    }
    run_chain_in_memory(&recipe)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn cmd_clarify(
    input: &std::path::Path,
    output: &std::path::Path,
    strength: &str,
    upscale: bool,
    print_recipe: bool,
) -> Result<()> {
    let recipe = clarify::build_clarify_recipe(input, output, strength, upscale)?;
    if print_recipe {
        println!("{}", serde_json::to_string_pretty(&recipe)?);
        return Ok(());
    }
    eprintln!(
        "clarify ({} strength, upscale={}, {} steps):",
        strength,
        upscale,
        recipe.chain.len()
    );
    for s in &recipe.chain {
        eprintln!("  {} {}", s.effect, s.params);
    }
    run_chain_in_memory(&recipe)?;
    println!("wrote {}", output.display());
    Ok(())
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Recipe {
    /// Input file path (relative to the recipe file or absolute).
    pub input: PathBuf,
    /// Output file path.
    pub output: PathBuf,
    /// Ordered list of effects to apply.
    pub chain: Vec<RecipeStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

fn cmd_pipeline(
    recipe_path: &std::path::Path,
    jpeg_quality: u8,
    save_stages: Option<&std::path::Path>,
) -> Result<()> {
    let recipe_str = std::fs::read_to_string(recipe_path)
        .with_context(|| format!("reading recipe {}", recipe_path.display()))?;
    let mut recipe: Recipe = serde_json::from_str(&recipe_str).with_context(|| {
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
    recipe.input = resolve(&recipe.input);
    recipe.output = resolve(&recipe.output);
    let _ = jpeg_quality; // wired through recipe defaults; future flag work

    run_chain_in_memory_with_stages(&recipe, save_stages)
        .map_err(|e| anyhow!("pipeline run failed: {e}"))?;
    println!("wrote {}", recipe.output.display());
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
