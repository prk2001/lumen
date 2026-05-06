//! Lumen command-line interface.
//!
//! Subcommands:
//!
//! - `probe <file>` — print metadata about an input file as JSON.
//! - `list-effects` — list every effect in the registry.
//! - `apply --input X --output Y --effect ID [--param k=v]…`
//!   decode → effect → encode for one still image.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{anyhow, Context as _, Result};
use clap::{Parser, Subcommand};
use lumen_core::{Context, EffectRegistry, ParamValue, ParamValues};
use lumen_io::{decode_image, encode_image, probe, ImageEncodeOptions};
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
        /// Input file path.
        #[arg(long)]
        input: PathBuf,
        /// Output file path. Format is inferred from the extension.
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
    println!(
        "{}",
        serde_json::to_string_pretty(&asset).context("serializing asset")?
    );
    Ok(())
}

fn build_registry() -> Result<EffectRegistry> {
    let r = EffectRegistry::new();
    lumen_fx_exposure::register_all(&r).map_err(|e| anyhow!("register exposure: {e}"))?;
    lumen_fx_color::register_all(&r).map_err(|e| anyhow!("register color: {e}"))?;
    lumen_fx_sharpen::register_all(&r).map_err(|e| anyhow!("register sharpen: {e}"))?;
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

    // Build ParamValues by coercing each `key=value` string.
    let mut params = ParamValues::new();
    for kv in raw_params {
        let (k, v) = kv
            .split_once('=')
            .ok_or_else(|| anyhow!("--param '{kv}' missing '='"))?;
        params.insert(k, parse_param_value(v));
    }
    params
        .validate_and_fill(effect.parameters())
        .map_err(|e| anyhow!("parameter validation: {e}"))?;

    info!("decoding {}", input.display());
    let frame = decode_image(input).map_err(|e| anyhow!("decode: {e}"))?;
    info!(
        width = frame.width,
        height = frame.height,
        "applying {}",
        effect_id
    );

    let mut ctx = Context::for_still_srgb();
    let out = effect
        .apply(&mut ctx, frame, &params)
        .map_err(|e| anyhow!("apply: {e}"))?;

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
