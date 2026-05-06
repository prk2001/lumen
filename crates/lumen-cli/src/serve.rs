//! `lumen serve` — live preview HTTP server.
//!
//! Watches a recipe + its referenced input for changes and re-runs the
//! pipeline, serving a side-by-side input/output preview at
//! `http://127.0.0.1:<port>/`. The HTML page auto-reloads images when
//! the server reports a newer mtime.

use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, SystemTime};

use anyhow::{anyhow, Context as _, Result};
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{Html, IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use lumen_core::{
    scheduler::special_effect_ids, Context, EffectRegistry, Frame, Graph, Node, NodeId,
    ParamValues, Scheduler,
};
use lumen_io::{decode_image, encode_image, ImageEncodeOptions};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

const INDEX_HTML: &str = include_str!("serve_index.html");

/// Per-request shared state.
struct AppState {
    recipe_path: PathBuf,
    base_dir: PathBuf,
    /// Last successful render's wall-clock at unix-millis, for cache busting.
    output_version: AtomicU64,
    /// Last error message, surfaced to the page.
    last_error: Mutex<Option<String>>,
    /// Last render duration in ms.
    last_render_ms: AtomicU64,
    /// Resolved input/output paths cached from the last successful recipe parse.
    cached_paths: Mutex<Option<CachedPaths>>,
    /// Effect registry built once.
    registry: EffectRegistry,
    jpeg_quality: u8,
}

#[derive(Debug, Clone)]
struct CachedPaths {
    input: PathBuf,
    output: PathBuf,
}

#[derive(serde::Serialize)]
struct StatusJson {
    output_version: u64,
    last_render_ms: u64,
    input_path: Option<String>,
    output_path: Option<String>,
    error: Option<String>,
}

pub async fn run(
    recipe_path: PathBuf,
    port: u16,
    jpeg_quality: u8,
    registry: EffectRegistry,
) -> Result<()> {
    let recipe_path = recipe_path
        .canonicalize()
        .context("canonicalizing recipe path")?;
    let base_dir = recipe_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let state = Arc::new(AppState {
        recipe_path: recipe_path.clone(),
        base_dir,
        output_version: AtomicU64::new(0),
        last_error: Mutex::new(None),
        last_render_ms: AtomicU64::new(0),
        cached_paths: Mutex::new(None),
        registry,
        jpeg_quality,
    });

    // Initial render so the first page load has something to display.
    if let Err(e) = render_once(&state).await {
        warn!("initial render failed: {e:#}");
        *state.last_error.lock().await = Some(format!("{e:#}"));
    }

    // Background watcher: poll mtimes every 400ms; re-render on change.
    let watcher_state = Arc::clone(&state);
    tokio::spawn(async move { watch_loop(watcher_state).await });

    let app = Router::new()
        .route("/", get(index))
        .route("/input", get(serve_input))
        .route("/output", get(serve_output))
        .route("/status", get(status))
        .route("/render", post(render_now))
        .with_state(state);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let listener = TcpListener::bind(addr).await.context("bind")?;
    info!("lumen serve listening on http://{addr}/");
    println!("lumen serve listening on http://{addr}/");
    println!("watching {} (re-renders on change)", recipe_path.display());

    axum::serve(listener, app).await.context("axum serve")?;
    Ok(())
}

async fn watch_loop(state: Arc<AppState>) {
    let mut prev_input = None::<SystemTime>;
    let mut prev_recipe = None::<SystemTime>;
    loop {
        tokio::time::sleep(Duration::from_millis(400)).await;
        let recipe_mt = mtime_of(&state.recipe_path).ok();
        let input_path = state.cached_paths.lock().await.as_ref().map(|p| p.input.clone());
        let input_mt = input_path.as_deref().and_then(|p| mtime_of(p).ok());

        let recipe_changed = recipe_mt != prev_recipe;
        let input_changed = input_mt.is_some() && input_mt != prev_input;
        if recipe_changed || input_changed {
            if let Err(e) = render_once(&state).await {
                warn!("re-render failed: {e:#}");
                *state.last_error.lock().await = Some(format!("{e:#}"));
            }
            prev_recipe = recipe_mt;
            prev_input = input_mt;
        }
    }
}

fn mtime_of(p: &Path) -> std::io::Result<SystemTime> {
    std::fs::metadata(p)?.modified()
}

async fn render_once(state: &Arc<AppState>) -> Result<()> {
    let started = std::time::Instant::now();
    let recipe_str =
        std::fs::read_to_string(&state.recipe_path).context("reading recipe")?;
    let recipe: super::Recipe =
        serde_json::from_str(&recipe_str).context("parsing recipe")?;

    let resolve = |p: &std::path::Path| {
        if p.is_absolute() { p.to_path_buf() } else { state.base_dir.join(p) }
    };
    let input_path = resolve(&recipe.input);
    let output_path = resolve(&recipe.output);
    *state.cached_paths.lock().await = Some(CachedPaths {
        input: input_path.clone(),
        output: output_path.clone(),
    });

    // Build graph from chain.
    let mut graph = Graph::new();
    let src_node = graph.insert(Node::new(special_effect_ids::SOURCE, "source"));
    let mut prev = src_node;
    for (i, step) in recipe.chain.iter().enumerate() {
        let mut params = ParamValues::new();
        if let serde_json::Value::Object(map) = &step.params {
            for (k, v) in map {
                let pv = super::json_to_param(v).ok_or_else(|| {
                    anyhow!("step {i}: param '{k}' has unsupported JSON type")
                })?;
                params.insert(k.clone(), pv);
            }
        }
        let label = step.label.clone().unwrap_or_else(|| format!("step{i:02}"));
        let node = graph
            .insert(Node::new(step.effect.clone(), label).with_input(prev).with_params(params));
        prev = node;
    }
    let sink_node = graph.insert(Node::new(special_effect_ids::SINK, "sink").with_input(prev));
    graph.add_sink(sink_node);

    // Render.
    let mut ctx = Context::for_still_srgb();
    let written = std::cell::RefCell::new(None::<()>);
    let jpeg_quality = state.jpeg_quality;
    let input_path_for_loader = input_path.clone();
    let output_path_for_writer = output_path.clone();
    let source = super::CliSource(move |_id: NodeId, _params: &ParamValues| {
        decode_image(&input_path_for_loader)
    });
    let sink = super::CliSink(move |_id: NodeId, _params: &ParamValues, frame: Frame| {
        encode_image(
            frame,
            &output_path_for_writer,
            ImageEncodeOptions { jpeg_quality, format: None },
        )?;
        *written.borrow_mut() = Some(());
        Ok(())
    });
    let mut sched = Scheduler {
        registry: &state.registry,
        ctx: &mut ctx,
        source_loader: source,
        sink_writer: sink,
    };
    sched.run(&graph).map_err(|e| anyhow!("scheduler: {e}"))?;

    let ms = started.elapsed().as_millis() as u64;
    state.last_render_ms.store(ms, Ordering::Relaxed);
    state.output_version.fetch_add(1, Ordering::Relaxed);
    *state.last_error.lock().await = None;
    info!("rendered {} in {ms} ms", output_path.display());
    Ok(())
}

async fn index() -> Html<&'static str> { Html(INDEX_HTML) }

async fn serve_input(State(state): State<Arc<AppState>>) -> Response {
    let path = state.cached_paths.lock().await.as_ref().map(|p| p.input.clone());
    serve_file(path, "image/png").await
}

async fn serve_output(State(state): State<Arc<AppState>>) -> Response {
    let path = state.cached_paths.lock().await.as_ref().map(|p| p.output.clone());
    serve_file(path, "image/png").await
}

async fn serve_file(path: Option<PathBuf>, content_type: &'static str) -> Response {
    match path {
        Some(p) => match tokio::fs::read(&p).await {
            Ok(bytes) => {
                let mime = mime_for(&p).unwrap_or(content_type);
                ([(header::CONTENT_TYPE, mime)], bytes).into_response()
            }
            Err(e) => (StatusCode::NOT_FOUND, format!("not found: {e}")).into_response(),
        },
        None => (StatusCode::SERVICE_UNAVAILABLE, "no render yet").into_response(),
    }
}

fn mime_for(p: &Path) -> Option<&'static str> {
    let ext = p.extension().and_then(|e| e.to_str())?;
    Some(match ext.to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "tif" | "tiff" => "image/tiff",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        _ => return None,
    })
}

async fn status(State(state): State<Arc<AppState>>) -> Json<StatusJson> {
    let cached = state.cached_paths.lock().await;
    let err = state.last_error.lock().await.clone();
    Json(StatusJson {
        output_version: state.output_version.load(Ordering::Relaxed),
        last_render_ms: state.last_render_ms.load(Ordering::Relaxed),
        input_path: cached.as_ref().map(|p| p.input.display().to_string()),
        output_path: cached.as_ref().map(|p| p.output.display().to_string()),
        error: err,
    })
}

async fn render_now(State(state): State<Arc<AppState>>) -> Response {
    match render_once(&state).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => {
            error!("forced render failed: {e:#}");
            (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response()
        }
    }
}
