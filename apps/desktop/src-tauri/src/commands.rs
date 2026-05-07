//! Tauri IPC commands that expose the Lumen pipeline to the React UI.
//!
//! All commands here return `Result<T, String>` so that errors flow
//! cleanly through Tauri's JSON IPC bridge. The Rust crate is linked
//! directly into the desktop binary, so there is no `lumen serve` HTTP
//! hop — the React side calls `invoke()` and the work happens in-process.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use lumen_core::{
    scheduler::special_effect_ids, Context, EffectRegistry, Frame, Graph, Node, NodeId, ParamValue,
    ParamValues, Scheduler, SinkWriter, SourceLoader,
};
use lumen_io::{decode_image, encode_image, ImageEncodeOptions};
use serde::{Deserialize, Serialize};

// ─── Registry singleton ──────────────────────────────────────────────────

/// Build the effect registry once on first use. Mirrors the CLI's
/// `build_registry()` but only with the starter set of fx crates wired
/// into this Tauri binary.
fn registry() -> &'static EffectRegistry {
    static REG: OnceLock<EffectRegistry> = OnceLock::new();
    REG.get_or_init(|| {
        let r = EffectRegistry::new();
        // Each register_all returns a Result; we expect them all to
        // succeed at startup. If they don't, surface a panic since the
        // app is unusable anyway.
        lumen_fx_exposure::register_all(&r).expect("register exposure");
        lumen_fx_color::register_all(&r).expect("register color");
        lumen_fx_sharpen::register_all(&r).expect("register sharpen");
        lumen_fx_denoise::register_all(&r).expect("register denoise");
        lumen_fx_geometric::register_all(&r).expect("register geometric");
        r
    })
}

// ─── DTOs (Tauri-friendly, fully owned, Serde-able) ──────────────────────

/// Description of one parameter, simplified for the UI.
#[derive(Debug, Clone, Serialize)]
pub struct ParamSpecDto {
    pub id: String,
    pub display_name: String,
    pub description: String,
    /// One of: `bool`, `int`, `float`, `choice`, `string`.
    pub kind: String,
    /// Default value rendered as JSON so the UI can hydrate inputs.
    pub default: serde_json::Value,
    pub min: Option<f64>,
    pub max: Option<f64>,
    /// Populated for `choice` parameters.
    pub options: Option<Vec<String>>,
}

/// Description of one effect, returned by `list_effects`.
#[derive(Debug, Clone, Serialize)]
pub struct EffectInfo {
    pub id: String,
    pub display_name: String,
    pub description: String,
    pub category: String,
    pub parameters: Vec<ParamSpecDto>,
}

/// Mirror of `lumen_core::AssetMetadata` with all fields owned. The
/// upstream type is already `Serialize` but lives in a path-only crate;
/// re-exposing it through this DTO keeps the IPC surface stable.
#[derive(Debug, Clone, Serialize)]
pub struct AssetMetadataDto {
    pub width: u32,
    pub height: u32,
    pub frame_count: Option<u64>,
    pub frame_rate_num: Option<i64>,
    pub frame_rate_den: Option<i64>,
    pub duration_secs: Option<f64>,
    pub codec: Option<String>,
    pub container: Option<String>,
    pub bit_depth: u8,
    pub channels: u8,
    pub kind: String,
    pub uri: String,
    pub display_name: String,
    pub hash: Option<String>,
}

/// Result returned by `run_pipeline`.
#[derive(Debug, Clone, Serialize)]
pub struct RenderStats {
    pub duration_ms: u64,
    pub output_bytes: u64,
}

/// Recipe schema used by `run_pipeline`. Mirrors the CLI's `Recipe` but
/// duplicated here so `apps/desktop/` doesn't depend on `lumen-cli`.
#[derive(Debug, Deserialize)]
struct Recipe {
    input: PathBuf,
    output: PathBuf,
    chain: Vec<RecipeStep>,
}

#[derive(Debug, Deserialize)]
struct RecipeStep {
    effect: String,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    params: serde_json::Value,
}

// ─── Helpers ─────────────────────────────────────────────────────────────

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

fn param_to_dto(spec: &lumen_core::ParamSpec) -> ParamSpecDto {
    use lumen_core::ParamKind;
    match &spec.kind {
        ParamKind::Bool { default } => ParamSpecDto {
            id: spec.id.to_string(),
            display_name: spec.display_name.to_string(),
            description: spec.description.to_string(),
            kind: "bool".into(),
            default: serde_json::json!(default),
            min: None,
            max: None,
            options: None,
        },
        ParamKind::Int { default, min, max } => ParamSpecDto {
            id: spec.id.to_string(),
            display_name: spec.display_name.to_string(),
            description: spec.description.to_string(),
            kind: "int".into(),
            default: serde_json::json!(default),
            min: min.map(|m| m as f64),
            max: max.map(|m| m as f64),
            options: None,
        },
        ParamKind::Float { default, min, max } => ParamSpecDto {
            id: spec.id.to_string(),
            display_name: spec.display_name.to_string(),
            description: spec.description.to_string(),
            kind: "float".into(),
            default: serde_json::json!(default),
            min: *min,
            max: *max,
            options: None,
        },
        ParamKind::Choice { default, options } => ParamSpecDto {
            id: spec.id.to_string(),
            display_name: spec.display_name.to_string(),
            description: spec.description.to_string(),
            kind: "choice".into(),
            default: serde_json::json!(default),
            min: None,
            max: None,
            options: Some(options.iter().map(|s| s.to_string()).collect()),
        },
        ParamKind::String { default } => ParamSpecDto {
            id: spec.id.to_string(),
            display_name: spec.display_name.to_string(),
            description: spec.description.to_string(),
            kind: "string".into(),
            default: serde_json::json!(default),
            min: None,
            max: None,
            options: None,
        },
    }
}

fn build_param_values(
    spec: &[lumen_core::ParamSpec],
    params: &serde_json::Value,
) -> Result<ParamValues, String> {
    let mut pv = ParamValues::new();
    if let serde_json::Value::Object(map) = params {
        for (k, v) in map {
            let value = json_to_param(v)
                .ok_or_else(|| format!("param '{k}' has unsupported JSON type"))?;
            pv.insert(k.clone(), value);
        }
    }
    pv.validate_and_fill(spec)
        .map_err(|e| format!("parameter validation: {e}"))?;
    Ok(pv)
}

// ─── Source/sink wrappers (orphan-rule workaround) ───────────────────────

struct ClosureSource<F>(F);
struct ClosureSink<F>(F);

impl<F> SourceLoader for ClosureSource<F>
where
    F: FnMut(NodeId, &ParamValues) -> lumen_core::Result<Frame>,
{
    fn load(&mut self, id: NodeId, p: &ParamValues) -> lumen_core::Result<Frame> {
        (self.0)(id, p)
    }
}

impl<F> SinkWriter for ClosureSink<F>
where
    F: FnMut(NodeId, &ParamValues, Frame) -> lumen_core::Result<()>,
{
    fn write(&mut self, id: NodeId, p: &ParamValues, f: Frame) -> lumen_core::Result<()> {
        (self.0)(id, p, f)
    }
}

// ─── Commands ────────────────────────────────────────────────────────────

#[tauri::command]
pub fn list_effects() -> Vec<EffectInfo> {
    let r = registry();
    r.ids()
        .into_iter()
        .filter_map(|id| {
            let e = r.get(&id)?;
            let m = e.metadata();
            Some(EffectInfo {
                id: m.id.to_string(),
                display_name: m.display_name.to_string(),
                description: m.description.to_string(),
                category: format!("{:?}", m.category),
                parameters: e.parameters().iter().map(param_to_dto).collect(),
            })
        })
        .collect()
}

#[tauri::command]
pub fn probe(path: String) -> Result<AssetMetadataDto, String> {
    let p = PathBuf::from(&path);
    let asset = lumen_io::probe(&p).map_err(|e| format!("probe: {e}"))?;
    let m = &asset.metadata;
    Ok(AssetMetadataDto {
        width: m.width,
        height: m.height,
        frame_count: m.frame_count,
        frame_rate_num: m.frame_rate.map(|r| r.num),
        frame_rate_den: m.frame_rate.map(|r| r.den),
        duration_secs: m.duration_secs,
        codec: m.codec.clone(),
        container: m.container.clone(),
        bit_depth: m.bit_depth,
        channels: m.channels,
        kind: format!("{:?}", asset.kind).to_lowercase(),
        uri: asset.uri,
        display_name: asset.display_name,
        hash: asset.hash,
    })
}

#[tauri::command]
pub fn apply_effect(
    input_path: String,
    output_path: String,
    effect_id: String,
    params: serde_json::Value,
) -> Result<(), String> {
    let r = registry();
    let effect = r
        .get(&effect_id)
        .ok_or_else(|| format!("unknown effect: {effect_id}"))?;
    let pv = build_param_values(effect.parameters(), &params)?;

    let frame = decode_image(Path::new(&input_path)).map_err(|e| format!("decode: {e}"))?;
    let mut ctx = Context::for_still_srgb();
    let out = effect
        .apply(&mut ctx, frame, &pv)
        .map_err(|e| format!("apply: {e}"))?;
    encode_image(out, Path::new(&output_path), ImageEncodeOptions::default())
        .map_err(|e| format!("encode: {e}"))?;
    Ok(())
}

#[tauri::command]
pub fn run_pipeline(
    recipe_json: String,
    input_path: String,
    output_path: String,
) -> Result<RenderStats, String> {
    let recipe: Recipe =
        serde_json::from_str(&recipe_json).map_err(|e| format!("parse recipe: {e}"))?;

    // CLI semantics: paths inside the recipe are resolved relative to
    // the recipe file's directory. Here we accept the input/output
    // paths from the caller as authoritative — they may override the
    // ones embedded in the recipe so the UI can pick its own files.
    let input_p = if input_path.is_empty() {
        recipe.input.clone()
    } else {
        PathBuf::from(&input_path)
    };
    let output_p = if output_path.is_empty() {
        recipe.output.clone()
    } else {
        PathBuf::from(&output_path)
    };

    let started = Instant::now();
    let r = registry();

    let mut graph = Graph::new();
    let src = graph.insert(Node::new(special_effect_ids::SOURCE, "source"));
    let mut prev = src;
    for (i, step) in recipe.chain.iter().enumerate() {
        let effect = r
            .get(&step.effect)
            .ok_or_else(|| format!("step {i}: unknown effect '{}'", step.effect))?;
        let params = build_param_values(effect.parameters(), &step.params)
            .map_err(|e| format!("step {i}: {e}"))?;
        let label = step.label.clone().unwrap_or_else(|| format!("step{i:02}"));
        let node = graph.insert(
            Node::new(step.effect.clone(), label)
                .with_input(prev)
                .with_params(params),
        );
        prev = node;
    }
    let sink = graph.insert(Node::new(special_effect_ids::SINK, "sink").with_input(prev));
    graph.add_sink(sink);

    let mut ctx = Context::for_still_srgb();
    let input_for_loader = input_p.clone();
    let output_for_writer = output_p.clone();
    let source = ClosureSource(move |_id: NodeId, _p: &ParamValues| decode_image(&input_for_loader));
    let sink_w = ClosureSink(move |_id: NodeId, _p: &ParamValues, frame: Frame| {
        encode_image(frame, &output_for_writer, ImageEncodeOptions::default())?;
        Ok(())
    });
    let mut sched = Scheduler {
        registry: r,
        ctx: &mut ctx,
        source_loader: source,
        sink_writer: sink_w,
    };
    sched
        .run(&graph)
        .map_err(|e| format!("scheduler: {e}"))?;

    let duration_ms = started.elapsed().as_millis() as u64;
    let output_bytes = std::fs::metadata(&output_p)
        .map(|m| m.len())
        .unwrap_or(0);
    Ok(RenderStats {
        duration_ms,
        output_bytes,
    })
}
