//! Lumen command-line interface.
//!
//! Subcommands:
//!
//! - `probe <file>` — print metadata about an input file as JSON.
//! - `list-effects` — list every effect in the registry.
//! - `apply --input X --output Y --effect ID [--param k=v]…`
//!   decode → effect → encode for one still image.
//! - `pipeline --recipe R.json` — run a multi-effect chain from JSON.

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
    }
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
struct Recipe {
    /// Input file path (relative to the recipe file or absolute).
    input: PathBuf,
    /// Output file path.
    output: PathBuf,
    /// Ordered list of effects to apply.
    chain: Vec<RecipeStep>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RecipeStep {
    /// Effect id (e.g. `"lumen-fx-denoise.gaussian"`).
    effect: String,
    /// Optional human label.
    #[serde(default)]
    label: Option<String>,
    /// Parameter values keyed by parameter id.
    #[serde(default)]
    params: serde_json::Value,
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

fn json_to_param(v: &serde_json::Value) -> Option<ParamValue> {
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
struct CliSource<F>(F);
struct CliSink<F>(F);

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
