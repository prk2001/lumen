//! # lumen-qa
//!
//! QA & monitoring: golden frames, fuzz, regression, telemetry.
//!
//! This crate provides a small **golden-frame regression harness**.  A
//! `GoldenCase` pairs an input image with a recipe (linear-chain JSON,
//! identical to the format used by `lumen-cli`) and a previously-saved
//! reference (the "golden") image.  [`run_case`] decodes the input, runs
//! the recipe, decodes the golden, computes MSE / PSNR / SSIM via
//! [`lumen_measure::all_metrics`], and compares against per-case
//! thresholds.
//!
//! When an algorithm change is intentional and the golden has gone
//! stale, [`update_golden`] regenerates it by re-running the recipe and
//! saving the result.  This is a developer convenience; **never call it
//! from CI**.

#![forbid(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use lumen_core::scheduler::special_effect_ids;
use lumen_core::{
    Context, EffectRegistry, Frame, Graph, Node, NodeId, ParamValue, ParamValues, Result,
    Scheduler, SinkWriter, SourceLoader,
};
use lumen_io::{decode_image, encode_image, ImageEncodeOptions};
use serde::{Deserialize, Serialize};

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

const DEFAULT_SSIM_THRESHOLD: f64 = 0.99;
const DEFAULT_PSNR_THRESHOLD_DB: f64 = 35.0;

// ─── Recipe (mirrors lumen-cli's format) ──────────────────────────────────

/// Linear-chain recipe — same shape `lumen-cli` uses.
///
/// Defined locally so this crate doesn't pull in `lumen-cli`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recipe {
    /// Input file path (relative to the recipe file, or absolute).
    pub input: PathBuf,
    /// Output file path (relative to the recipe file, or absolute).
    pub output: PathBuf,
    /// Ordered list of effects to apply.
    pub chain: Vec<RecipeStep>,
}

/// One step in a [`Recipe`] chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipeStep {
    /// Effect id (e.g. `"lumen-fx-exposure.brightness_contrast"`).
    pub effect: String,
    /// Optional human label.
    #[serde(default)]
    pub label: Option<String>,
    /// Parameter values keyed by parameter id.
    #[serde(default)]
    pub params: serde_json::Value,
}

// ─── Public types ─────────────────────────────────────────────────────────

/// Default SSIM pass threshold (`0.99`) used when a case omits it.
pub const fn default_ssim_threshold() -> f64 {
    DEFAULT_SSIM_THRESHOLD
}

/// Default PSNR pass threshold in dB (`35.0`) used when a case omits it.
pub const fn default_psnr_threshold_db() -> f64 {
    DEFAULT_PSNR_THRESHOLD_DB
}

/// One golden-frame regression case.
///
/// The two threshold fields are inclusive lower bounds: a case passes
/// only if **both** `ssim ≥ ssim_threshold` and `psnr ≥
/// psnr_threshold_db` hold for the rendered-vs-golden pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenCase {
    /// Human-readable case name (used in result messages).
    pub name: String,
    /// Input image path.
    pub input_path: PathBuf,
    /// Recipe JSON path.
    pub recipe_path: PathBuf,
    /// Golden reference image path.
    pub golden_path: PathBuf,
    /// SSIM pass threshold. Defaults to [`default_ssim_threshold`].
    #[serde(default = "default_ssim_threshold")]
    pub ssim_threshold: f64,
    /// PSNR pass threshold in dB. Defaults to [`default_psnr_threshold_db`].
    #[serde(default = "default_psnr_threshold_db")]
    pub psnr_threshold_db: f64,
}

impl GoldenCase {
    /// Build a case with the default thresholds.
    pub fn new(
        name: impl Into<String>,
        input_path: impl Into<PathBuf>,
        recipe_path: impl Into<PathBuf>,
        golden_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            name: name.into(),
            input_path: input_path.into(),
            recipe_path: recipe_path.into(),
            golden_path: golden_path.into(),
            ssim_threshold: DEFAULT_SSIM_THRESHOLD,
            psnr_threshold_db: DEFAULT_PSNR_THRESHOLD_DB,
        }
    }
}

/// Outcome of running a single [`GoldenCase`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenResult {
    /// Echo of the case name.
    pub name: String,
    /// Whether the case met both thresholds.
    pub passed: bool,
    /// Mean-squared error in `[0, 1]^2`.
    pub mse: f64,
    /// Peak signal-to-noise ratio in dB. May be `f64::INFINITY`.
    pub psnr: f64,
    /// Structural similarity index in `[-1, 1]`.
    pub ssim: f64,
    /// Human-readable summary line.
    pub message: String,
}

/// A collection of regression cases.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GoldenSet {
    /// All cases in the set.
    pub cases: Vec<GoldenCase>,
}

impl GoldenSet {
    /// Construct a set from a vector of cases.
    pub fn new(cases: Vec<GoldenCase>) -> Self {
        Self { cases }
    }

    /// Run every case sequentially.
    pub fn run_all(set: &GoldenSet, registry: &EffectRegistry) -> Vec<GoldenResult> {
        set.cases
            .iter()
            .map(|c| match run_case(c, registry) {
                Ok(r) => r,
                Err(e) => GoldenResult {
                    name: c.name.clone(),
                    passed: false,
                    mse: f64::NAN,
                    psnr: f64::NAN,
                    ssim: f64::NAN,
                    message: format!("error: {e}"),
                },
            })
            .collect()
    }

    /// Discover cases from a directory of `<name>.case.json` files.
    ///
    /// Each file is parsed as a [`GoldenCase`].  Files whose name does
    /// not end in `.case.json` are ignored.  The discovered cases are
    /// returned in lexicographic filename order for determinism.
    pub fn from_dir<P: AsRef<Path>>(dir: P) -> Result<GoldenSet> {
        let dir = dir.as_ref();
        let read = std::fs::read_dir(dir)?;
        let mut entries: Vec<PathBuf> = read
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.ends_with(".case.json"))
                    .unwrap_or(false)
            })
            .collect();
        entries.sort();

        let mut cases = Vec::with_capacity(entries.len());
        for path in entries {
            let raw = std::fs::read_to_string(&path)?;
            let case: GoldenCase = serde_json::from_str(&raw)?;
            cases.push(case);
        }
        Ok(GoldenSet { cases })
    }
}

/// Run one case end-to-end. See module docs for the full pipeline.
pub fn run_case(case: &GoldenCase, registry: &EffectRegistry) -> Result<GoldenResult> {
    let rendered = render_recipe(case, registry)?;
    let golden = decode_image(&case.golden_path).map_err(|e| {
        lumen_core::Error::Other(format!(
            "case '{}': decoding golden '{}': {e}",
            case.name,
            case.golden_path.display()
        ))
    })?;

    if rendered.width != golden.width || rendered.height != golden.height {
        let message = format!(
            "case '{}': dimension mismatch — rendered={}x{}, golden={}x{}",
            case.name, rendered.width, rendered.height, golden.width, golden.height
        );
        return Ok(GoldenResult {
            name: case.name.clone(),
            passed: false,
            mse: f64::NAN,
            psnr: f64::NAN,
            ssim: f64::NAN,
            message,
        });
    }

    let metrics = lumen_measure::all_metrics(&rendered, &golden)?;
    let passed =
        metrics.ssim >= case.ssim_threshold && metrics.psnr >= case.psnr_threshold_db;
    let message = if passed {
        format!(
            "case '{}' passed: ssim={:.6} (>= {:.6}), psnr={:.3} dB (>= {:.3} dB)",
            case.name,
            metrics.ssim,
            case.ssim_threshold,
            metrics.psnr,
            case.psnr_threshold_db
        )
    } else {
        format!(
            "case '{}' FAILED: ssim={:.6} (need >= {:.6}), psnr={:.3} dB (need >= {:.3} dB), mse={:.6}",
            case.name,
            metrics.ssim,
            case.ssim_threshold,
            metrics.psnr,
            case.psnr_threshold_db,
            metrics.mse
        )
    };
    Ok(GoldenResult {
        name: case.name.clone(),
        passed,
        mse: metrics.mse,
        psnr: metrics.psnr,
        ssim: metrics.ssim,
        message,
    })
}

/// Regenerate `case.golden_path` by re-running the recipe and writing
/// the rendered output as a PNG.
///
/// Use this when an intentional algorithm change is expected and the
/// existing golden is known to be stale.  **Never call from CI** — by
/// definition this hides regressions.
pub fn update_golden(case: &GoldenCase, registry: &EffectRegistry) -> Result<()> {
    let rendered = render_recipe(case, registry)?;
    if let Some(parent) = case.golden_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    encode_image(
        rendered,
        &case.golden_path,
        ImageEncodeOptions { jpeg_quality: 92, format: None },
    )?;
    Ok(())
}

// ─── Internals ────────────────────────────────────────────────────────────

/// Decode the case's input, run its recipe through the [`Scheduler`],
/// and return the rendered frame.
fn render_recipe(case: &GoldenCase, registry: &EffectRegistry) -> Result<Frame> {
    let recipe_str = std::fs::read_to_string(&case.recipe_path)?;
    let recipe: Recipe = serde_json::from_str(&recipe_str)?;

    // Resolve the case's `input_path` directly (it's already case-relative
    // by convention). The recipe's own `input`/`output` fields are
    // ignored — we plug the case's input into the source and capture
    // the sink output in-memory.
    let _ = recipe.input; // silence unused-field warning intentionally
    let _ = recipe.output;

    let mut graph = Graph::new();
    let src_node = graph.insert(Node::new(special_effect_ids::SOURCE, "source"));
    let mut prev = src_node;
    for (i, step) in recipe.chain.iter().enumerate() {
        let mut params = ParamValues::new();
        if let serde_json::Value::Object(map) = &step.params {
            for (k, v) in map {
                let Some(pv) = json_to_param(v) else {
                    return Err(lumen_core::Error::Other(format!(
                        "case '{}' step {i}: param '{k}' has unsupported JSON type",
                        case.name
                    )));
                };
                params.insert(k.clone(), pv);
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

    let input_path = case.input_path.clone();
    let captured: RefCell<Option<Frame>> = RefCell::new(None);

    let mut ctx = Context::for_still_srgb();
    let source = QaSource(|_id: NodeId, _params: &ParamValues| -> Result<Frame> {
        decode_image(&input_path)
    });
    let sink = QaSink(|_id: NodeId, _params: &ParamValues, frame: Frame| -> Result<()> {
        *captured.borrow_mut() = Some(frame);
        Ok(())
    });
    let mut sched = Scheduler {
        registry,
        ctx: &mut ctx,
        source_loader: source,
        sink_writer: sink,
    };
    sched.run(&graph)?;

    captured
        .into_inner()
        .ok_or_else(|| lumen_core::Error::Other(format!("case '{}': sink produced no frame", case.name)))
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

// Newtype wrappers so we can implement the foreign SourceLoader / SinkWriter
// traits for closures (orphan rule forbids the blanket impl).
struct QaSource<F>(F);
struct QaSink<F>(F);

impl<F> SourceLoader for QaSource<F>
where
    F: FnMut(NodeId, &ParamValues) -> Result<Frame>,
{
    fn load(&mut self, id: NodeId, p: &ParamValues) -> Result<Frame> {
        (self.0)(id, p)
    }
}

impl<F> SinkWriter for QaSink<F>
where
    F: FnMut(NodeId, &ParamValues, Frame) -> Result<()>,
{
    fn write(&mut self, id: NodeId, p: &ParamValues, f: Frame) -> Result<()> {
        (self.0)(id, p, f)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, PixelData};

    /// Build a tiny synthetic input frame and write it as a PNG. Returns
    /// (`png_path`, `frame`).
    fn write_synthetic_input(dir: &Path) -> (PathBuf, Frame) {
        let w: u32 = 24;
        let h: u32 = 24;
        let mut bytes = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                // A simple gradient so SSIM / PSNR have something to chew on.
                let r = ((x * 255) / (w - 1)) as u8;
                let g = ((y * 255) / (h - 1)) as u8;
                let b = (((x + y) * 255) / (w + h - 2)) as u8;
                bytes.extend_from_slice(&[r, g, b, 255]);
            }
        }
        let frame = Frame::new(w, h, PixelData::Rgba8(bytes), ColorSpace::SRgb, None).unwrap();
        let path = dir.join("input.png");
        encode_image(
            frame.clone(),
            &path,
            ImageEncodeOptions { jpeg_quality: 92, format: None },
        )
        .unwrap();
        (path, frame)
    }

    /// Write a recipe that brightens by 0.1 — non-trivial enough that
    /// the rendered output meaningfully differs from the input.
    fn write_recipe(dir: &Path, input: &Path, output: &Path) -> PathBuf {
        let recipe = Recipe {
            input: input.to_path_buf(),
            output: output.to_path_buf(),
            chain: vec![RecipeStep {
                effect: "lumen-fx-exposure.brightness_contrast".to_string(),
                label: Some("bright".to_string()),
                params: serde_json::json!({ "brightness": 0.1, "contrast": 1.0 }),
            }],
        };
        let path = dir.join("case.recipe.json");
        std::fs::write(&path, serde_json::to_string_pretty(&recipe).unwrap()).unwrap();
        path
    }

    fn build_registry() -> EffectRegistry {
        let r = EffectRegistry::new();
        lumen_fx_exposure::register_all(&r).unwrap();
        r
    }

    /// Per-test temp dir that isolates one fixture from another and is
    /// cleaned up on drop. We avoid pulling in `tempfile` to keep
    /// dev-deps minimal.
    struct TempDir {
        path: PathBuf,
    }
    impl TempDir {
        fn new(label: &str) -> Self {
            let mut path = std::env::temp_dir();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            path.push(format!("lumen-qa-{label}-{nanos}"));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn round_trip_passes_with_unit_ssim() {
        let tmp = TempDir::new("roundtrip");
        let dir = &tmp.path;

        let (input_path, _input_frame) = write_synthetic_input(dir);
        let golden_path = dir.join("golden.png");
        let recipe_path = write_recipe(dir, &input_path, &golden_path);

        let registry = build_registry();

        // Generate the golden once via update_golden.
        let case = GoldenCase::new("roundtrip", &input_path, &recipe_path, &golden_path);
        update_golden(&case, &registry).unwrap();

        // Now run the case. Rendered output should be byte-identical to
        // the just-saved golden, so SSIM ≈ 1.0 and PSNR is huge / inf.
        let result = run_case(&case, &registry).unwrap();
        assert!(
            result.passed,
            "expected pass, got: {} (ssim={}, psnr={})",
            result.message, result.ssim, result.psnr
        );
        // Encode/decode round-trip introduces sub-1e-3 FP noise; accept
        // anything close to 1.0.
        assert!(
            (result.ssim - 1.0).abs() < 1e-3,
            "ssim should be ~1.0, got {}",
            result.ssim
        );
    }

    #[test]
    fn tampered_golden_fails_with_message() {
        let tmp = TempDir::new("tamper");
        let dir = &tmp.path;

        let (input_path, _) = write_synthetic_input(dir);
        let golden_path = dir.join("golden.png");
        let recipe_path = write_recipe(dir, &input_path, &golden_path);

        let registry = build_registry();
        let case = GoldenCase {
            name: "tamper".to_string(),
            input_path: input_path.clone(),
            recipe_path: recipe_path.clone(),
            golden_path: golden_path.clone(),
            // Tighten thresholds so the tamper is obvious.
            ssim_threshold: 0.999,
            psnr_threshold_db: 50.0,
        };
        update_golden(&case, &registry).unwrap();

        // Corrupt the golden by overwriting it with a totally different
        // image (a solid red, same dimensions).
        let bad = {
            let w = 24;
            let h = 24;
            let mut bytes = Vec::with_capacity((w * h * 4) as usize);
            for _ in 0..(w * h) {
                bytes.extend_from_slice(&[255, 0, 0, 255]);
            }
            Frame::new(w, h, PixelData::Rgba8(bytes), ColorSpace::SRgb, None).unwrap()
        };
        encode_image(
            bad,
            &golden_path,
            ImageEncodeOptions { jpeg_quality: 92, format: None },
        )
        .unwrap();

        let result = run_case(&case, &registry).unwrap();
        assert!(!result.passed, "expected FAIL after tampering, got pass: {}", result.message);
        assert!(
            result.message.contains("FAILED"),
            "message should mention failure, got: {}",
            result.message
        );
        // Sanity-check: a near-uniform red replacement vs a gradient
        // brightened a touch should not be SSIM ≈ 1.
        assert!(result.ssim < 0.99, "ssim was suspiciously high: {}", result.ssim);
    }

    #[test]
    fn update_golden_refreshes_stale_reference() {
        let tmp = TempDir::new("update");
        let dir = &tmp.path;

        let (input_path, _) = write_synthetic_input(dir);
        let golden_path = dir.join("golden.png");
        let recipe_path = write_recipe(dir, &input_path, &golden_path);

        // Seed a *stale* golden — solid blue, nothing like what the
        // recipe will produce.
        let stale = {
            let w = 24;
            let h = 24;
            let mut bytes = Vec::with_capacity((w * h * 4) as usize);
            for _ in 0..(w * h) {
                bytes.extend_from_slice(&[0, 0, 255, 255]);
            }
            Frame::new(w, h, PixelData::Rgba8(bytes), ColorSpace::SRgb, None).unwrap()
        };
        encode_image(
            stale,
            &golden_path,
            ImageEncodeOptions { jpeg_quality: 92, format: None },
        )
        .unwrap();

        let registry = build_registry();
        let case = GoldenCase::new("update", &input_path, &recipe_path, &golden_path);

        // With the stale golden the case should fail.
        let stale_result = run_case(&case, &registry).unwrap();
        assert!(
            !stale_result.passed,
            "stale golden unexpectedly passed: {}",
            stale_result.message
        );

        // Refresh the golden, then the same case should pass.
        update_golden(&case, &registry).unwrap();
        let fresh_result = run_case(&case, &registry).unwrap();
        assert!(
            fresh_result.passed,
            "after update_golden, case should pass; got: {}",
            fresh_result.message
        );
    }

    #[test]
    fn from_dir_discovers_case_json_files() {
        let tmp = TempDir::new("discover");
        let dir = &tmp.path;

        // Write two cases plus an unrelated file (should be ignored).
        let case1 = GoldenCase::new(
            "alpha",
            dir.join("a.png"),
            dir.join("a.recipe.json"),
            dir.join("a.golden.png"),
        );
        let case2 = GoldenCase::new(
            "beta",
            dir.join("b.png"),
            dir.join("b.recipe.json"),
            dir.join("b.golden.png"),
        );
        std::fs::write(
            dir.join("alpha.case.json"),
            serde_json::to_string_pretty(&case1).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("beta.case.json"),
            serde_json::to_string_pretty(&case2).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join("README.txt"), "ignore me").unwrap();

        let set = GoldenSet::from_dir(dir).unwrap();
        assert_eq!(set.cases.len(), 2);
        let names: Vec<&str> = set.cases.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "beta"]);
    }
}
