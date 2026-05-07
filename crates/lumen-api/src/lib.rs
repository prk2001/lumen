//! # lumen-api
//!
//! Extensibility: plugin host, scripting (Lua/Python/JS), REST/GraphQL.
//!
//! ## Lua plugin host (Phase 1)
//!
//! [`LuaPlugin`] loads a `.lua` script that declares an effect's
//! identity, parameters, and a per-frame `apply` function. Once loaded,
//! the plugin implements [`lumen_core::Effect`] and is indistinguishable
//! from a built-in effect — host code can register it in the same
//! [`lumen_core::EffectRegistry`] and run it through the normal
//! scheduler.
//!
//! ### Plugin contract
//!
//! Every plugin script must define two globals:
//!
//! * `effect_metadata()` — returns a Lua table describing identity and
//!   parameters. Called exactly once at load time.
//! * `apply(width, height, pixels, params)` — called for every frame.
//!   `pixels` is a Lua table of `f32`s in interleaved RGBA order
//!   (length = `width * height * 4`). The function mutates `pixels` in
//!   place and returns it (or returns a new table of the same length).
//!
//! ```lua
//! function effect_metadata()
//!   return {
//!     id = "plugin.invert",
//!     display_name = "Invert",
//!     description = "Inverts RGB",
//!     category = "Color",          -- one of Category's variants as a string
//!     version = 1,
//!     params = {
//!       {
//!         id = "amount",
//!         kind = "float",          -- "bool" | "int" | "float" | "string" | "choice"
//!         default = 1.0,
//!         min = 0.0,                 -- optional, numeric kinds only
//!         max = 1.0,                 -- optional, numeric kinds only
//!         display_name = "Amount",   -- optional
//!         description = "...",       -- optional
//!         options = { "a", "b" },    -- required for "choice"
//!       },
//!     },
//!   }
//! end
//!
//! function apply(width, height, pixels, params)
//!   for i = 1, #pixels, 4 do
//!     pixels[i]     = 1.0 - pixels[i]
//!     pixels[i + 1] = 1.0 - pixels[i + 1]
//!     pixels[i + 2] = 1.0 - pixels[i + 2]
//!   end
//!   return pixels
//! end
//! ```
//!
//! Phase 1 exchanges pixels via a Lua table of f32s. That is correct
//! and demonstrates the API but is slow for large frames; tile-based
//! and zero-copy buffer interop are scheduled for Phase 4.
//!
//! ### Example
//!
//! ```no_run
//! use lumen_api::load_lua_plugin;
//! use lumen_core::Effect;
//!
//! let plugin = load_lua_plugin("plugins/invert.lua").unwrap();
//! assert_eq!(plugin.metadata().id, "plugin.invert");
//! ```

#![forbid(unsafe_op_in_unsafe_fn)]

use std::path::Path;
use std::sync::Arc;

use mlua::{Function, Lua, Table};
use parking_lot::Mutex;
use tracing::instrument;

use lumen_core::{
    Capabilities, Category, Context, Effect, EffectMetadata, Error, Frame, ParamKind, ParamSpec,
    ParamValue, ParamValues, PixelData, Result,
};

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// A Lua script loaded as a Lumen [`Effect`].
///
/// The Lua state is owned by the plugin and mutex-guarded so the impl
/// can satisfy `Effect: Send + Sync`. Each `apply` call calls the Lua
/// `apply` global and round-trips pixel data through a Lua table.
pub struct LuaPlugin {
    metadata: EffectMetadata,
    parameters: Vec<ParamSpec>,
    capabilities: Capabilities,
    lua: Arc<Mutex<Lua>>,
}

impl std::fmt::Debug for LuaPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LuaPlugin")
            .field("metadata", &self.metadata)
            .field("parameters", &self.parameters.len())
            .finish()
    }
}

impl LuaPlugin {
    /// Borrow the parsed metadata.
    pub fn metadata(&self) -> &EffectMetadata { &self.metadata }
}

impl Effect for LuaPlugin {
    fn metadata(&self) -> &EffectMetadata { &self.metadata }
    fn parameters(&self) -> &[ParamSpec] { &self.parameters }
    fn capabilities(&self) -> Capabilities { self.capabilities }

    #[instrument(skip_all, fields(effect = self.metadata.id))]
    fn apply(&self, _ctx: &mut Context, input: Frame, params: &ParamValues) -> Result<Frame> {
        // Lift to RGBA f32 linear so Lua only ever sees floats.
        let mut frame = input.into_rgba_f32_linear();
        let width = frame.width;
        let height = frame.height;
        let pixels = match &mut frame.data {
            PixelData::RgbaF32(v) => v,
            // Unreachable — into_rgba_f32_linear always produces RgbaF32.
            _ => return Err(Error::Layout("LuaPlugin expected RgbaF32 buffer".into())),
        };

        let lua = self.lua.lock();
        let apply_fn: Function = lua
            .globals()
            .get::<Function>("apply")
            .map_err(|e| Error::Other(format!("plugin missing `apply`: {e}")))?;

        // Build a Lua table from the pixel buffer.
        let pixel_tbl = lua
            .create_sequence_from(pixels.iter().copied())
            .map_err(|e| Error::Other(format!("plugin: pixel table alloc failed: {e}")))?;

        // Build the params table from ParamValues.
        let param_tbl = build_params_table(&lua, &self.parameters, params)?;

        let result: Table = apply_fn
            .call::<Table>((width, height, pixel_tbl, param_tbl))
            .map_err(|e| Error::Other(format!("plugin `apply` failed: {e}")))?;

        // Read back into the existing buffer. This avoids a second alloc
        // and validates the returned table length.
        let expected = pixels.len();
        let returned_len = result.raw_len();
        if returned_len as usize != expected {
            return Err(Error::Layout(format!(
                "plugin `apply` returned table of length {returned_len}, expected {expected}",
            )));
        }
        for (i, slot) in pixels.iter_mut().enumerate() {
            // Lua tables are 1-indexed.
            let v: f32 = result
                .get::<f32>((i + 1) as i64)
                .map_err(|e| Error::Other(format!("plugin: bad pixel at {i}: {e}")))?;
            *slot = v;
        }

        Ok(frame)
    }
}

/// Load a Lua plugin from disk.
///
/// Reads the file, executes it in a fresh Lua state, then calls the
/// global `effect_metadata()` to extract identity and parameters. The
/// returned [`LuaPlugin`] implements [`lumen_core::Effect`].
pub fn load_lua_plugin<P: AsRef<Path>>(path: P) -> Result<LuaPlugin> {
    let path = path.as_ref();
    let source = std::fs::read_to_string(path).map_err(Error::Io)?;

    let lua = Lua::new();
    lua.load(&source)
        .set_name(path.to_string_lossy())
        .exec()
        .map_err(|e| Error::Other(format!("lua load error ({}): {e}", path.display())))?;

    let metadata_fn: Function = lua
        .globals()
        .get::<Function>("effect_metadata")
        .map_err(|e| Error::Other(format!("plugin missing `effect_metadata`: {e}")))?;

    let meta_tbl: Table = metadata_fn
        .call::<Table>(())
        .map_err(|e| Error::Other(format!("`effect_metadata` failed: {e}")))?;

    let metadata = parse_metadata(&meta_tbl)?;
    let parameters = parse_parameters(&meta_tbl)?;

    // Plugins are CPU-only and treated as deterministic by default;
    // authors can revise this surface in later phases.
    let capabilities = Capabilities {
        deterministic: true,
        gpu: false,
        streamable: false,
        temporal: false,
    };

    // Sanity check: `apply` must exist.
    let _: Function = lua
        .globals()
        .get::<Function>("apply")
        .map_err(|e| Error::Other(format!("plugin missing `apply`: {e}")))?;

    Ok(LuaPlugin {
        metadata,
        parameters,
        capabilities,
        lua: Arc::new(Mutex::new(lua)),
    })
}

fn parse_metadata(tbl: &Table) -> Result<EffectMetadata> {
    let id: String = tbl
        .get("id")
        .map_err(|e| Error::Other(format!("plugin metadata missing `id`: {e}")))?;
    let display_name: String = tbl.get("display_name").unwrap_or_else(|_| id.clone());
    let description: String = tbl.get("description").unwrap_or_default();
    let category_s: String = tbl.get("category").unwrap_or_else(|_| "Api".to_string());
    let version: u32 = tbl.get("version").unwrap_or(1);

    let category = parse_category(&category_s)?;

    Ok(EffectMetadata {
        id: leak_str(id),
        display_name: leak_str(display_name),
        description: leak_str(description),
        category,
        version,
    })
}

fn parse_parameters(tbl: &Table) -> Result<Vec<ParamSpec>> {
    let params: Table = match tbl.get::<Table>("params") {
        Ok(t) => t,
        // Missing params is fine — effect just has no inputs.
        Err(_) => return Ok(Vec::new()),
    };

    let mut out = Vec::new();
    for pair in params.sequence_values::<Table>() {
        let p = pair.map_err(|e| Error::Other(format!("plugin: bad param entry: {e}")))?;
        out.push(parse_param_spec(&p)?);
    }
    Ok(out)
}

fn parse_param_spec(p: &Table) -> Result<ParamSpec> {
    let id: String = p
        .get("id")
        .map_err(|e| Error::Other(format!("plugin: param missing `id`: {e}")))?;
    let display_name: String = p.get("display_name").unwrap_or_else(|_| id.clone());
    let description: String = p.get("description").unwrap_or_default();
    let kind_s: String = p
        .get("kind")
        .map_err(|e| Error::Other(format!("plugin: param `{id}` missing `kind`: {e}")))?;

    let kind = match kind_s.as_str() {
        "bool" => ParamKind::Bool {
            default: p.get("default").unwrap_or(false),
        },
        "int" => ParamKind::Int {
            default: p.get("default").unwrap_or(0),
            min: p.get::<Option<i64>>("min").unwrap_or(None),
            max: p.get::<Option<i64>>("max").unwrap_or(None),
        },
        "float" => ParamKind::Float {
            default: p.get("default").unwrap_or(0.0),
            min: p.get::<Option<f64>>("min").unwrap_or(None),
            max: p.get::<Option<f64>>("max").unwrap_or(None),
        },
        "string" => {
            let default: String = p.get("default").unwrap_or_default();
            ParamKind::String { default: leak_str(default) }
        }
        "choice" => {
            let default: String = p.get("default").unwrap_or_default();
            let opts_tbl: Table = p.get("options").map_err(|e| {
                Error::Other(format!("plugin: choice param `{id}` missing `options`: {e}"))
            })?;
            let mut opts: Vec<&'static str> = Vec::new();
            for entry in opts_tbl.sequence_values::<String>() {
                let s = entry
                    .map_err(|e| Error::Other(format!("plugin: bad choice option: {e}")))?;
                opts.push(leak_str(s));
            }
            let opts_static: &'static [&'static str] = Box::leak(opts.into_boxed_slice());
            ParamKind::Choice { default: leak_str(default), options: opts_static }
        }
        other => {
            return Err(Error::Other(format!(
                "plugin: param `{id}` has unknown kind `{other}`",
            )));
        }
    };

    Ok(ParamSpec {
        id: leak_str(id),
        display_name: leak_str(display_name),
        description: leak_str(description),
        kind,
    })
}

fn parse_category(s: &str) -> Result<Category> {
    Ok(match s {
        "Input" => Category::Input,
        "Playback" => Category::Playback,
        "Ui" => Category::Ui,
        "Exposure" => Category::Exposure,
        "Color" => Category::Color,
        "Sharpen" => Category::Sharpen,
        "Denoise" => Category::Denoise,
        "Compression" => Category::Compression,
        "Geometric" => Category::Geometric,
        "Stabilize" => Category::Stabilize,
        "Deblur" => Category::Deblur,
        "Upscale" => Category::Upscale,
        "Temporal" => Category::Temporal,
        "Ai" => Category::Ai,
        "Face" => Category::Face,
        "Text" => Category::Text,
        "Mask" => Category::Mask,
        "Weather" => Category::Weather,
        "Modalities" => Category::Modalities,
        "Measure" => Category::Measure,
        "Audio" => Category::Audio,
        "Auth" => Category::Auth,
        "Workflow" => Category::Workflow,
        "Collaboration" => Category::Collaboration,
        "Report" => Category::Report,
        "Export" => Category::Export,
        "Performance" => Category::Performance,
        "Api" => Category::Api,
        "Platform" => Category::Platform,
        "Qa" => Category::Qa,
        other => {
            return Err(Error::Other(format!(
                "plugin: unknown category `{other}`",
            )));
        }
    })
}

fn build_params_table(
    lua: &Lua,
    specs: &[ParamSpec],
    values: &ParamValues,
) -> Result<Table> {
    let tbl = lua
        .create_table()
        .map_err(|e| Error::Other(format!("plugin: param table alloc failed: {e}")))?;

    for spec in specs {
        let v = values.get(spec.id);
        match v {
            Some(ParamValue::Bool(b)) => tbl.set(spec.id, *b),
            Some(ParamValue::Int(i)) => tbl.set(spec.id, *i),
            Some(ParamValue::Float(f)) => tbl.set(spec.id, *f),
            Some(ParamValue::String(s)) => tbl.set(spec.id, s.clone()),
            None => {
                // Fall back to spec default so plugins always see a value.
                match &spec.kind {
                    ParamKind::Bool { default } => tbl.set(spec.id, *default),
                    ParamKind::Int { default, .. } => tbl.set(spec.id, *default),
                    ParamKind::Float { default, .. } => tbl.set(spec.id, *default),
                    ParamKind::Choice { default, .. } => tbl.set(spec.id, *default),
                    ParamKind::String { default } => tbl.set(spec.id, *default),
                }
            }
        }
        .map_err(|e| Error::Other(format!("plugin: param `{}` set failed: {e}", spec.id)))?;
    }

    Ok(tbl)
}

/// Leak a `String` to obtain a `&'static str`. Plugin metadata is loaded
/// once per process and lives for the program's lifetime, so a small
/// per-plugin leak is acceptable in exchange for fitting into the
/// `&'static str` shape that built-in effects use.
fn leak_str(s: String) -> &'static str { Box::leak(s.into_boxed_str()) }

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, PixelData};

    fn example_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join("invert.lua")
    }

    #[test]
    fn missing_file_errors() {
        let r = load_lua_plugin("/this/path/does/not/exist.lua");
        assert!(r.is_err(), "expected error for missing file");
    }

    #[test]
    fn loads_bundled_invert_plugin() {
        let plugin = load_lua_plugin(example_path()).expect("load invert plugin");
        let meta = Effect::metadata(&plugin);
        assert_eq!(meta.id, "plugin.invert");
        assert_eq!(meta.display_name, "Invert");
        assert_eq!(meta.category, Category::Color);
        assert_eq!(plugin.parameters().len(), 1);
        assert_eq!(plugin.parameters()[0].id, "amount");
    }

    #[test]
    fn red_inverts_to_cyan() {
        let plugin = load_lua_plugin(example_path()).expect("load invert plugin");

        // Build a 2x2 solid-red linear-RGB frame.
        let pixels = (0..4)
            .flat_map(|_| [1.0_f32, 0.0, 0.0, 1.0])
            .collect::<Vec<_>>();
        let frame = Frame::new(
            2,
            2,
            PixelData::RgbaF32(pixels),
            ColorSpace::LinearSRgb,
            None,
        )
        .expect("build red frame");

        let mut params = ParamValues::new();
        params
            .validate_and_fill(plugin.parameters())
            .expect("fill params");

        let mut ctx = Context::for_still_srgb();
        let out = Effect::apply(&plugin, &mut ctx, frame, &params).expect("apply plugin");

        let buf = out.as_f32().expect("RgbaF32 output");
        for px in buf.chunks_exact(4) {
            // Red (1,0,0) inverted with amount=1 -> cyan (0,1,1).
            assert!((px[0] - 0.0).abs() < 1e-5, "R should be ~0, got {}", px[0]);
            assert!((px[1] - 1.0).abs() < 1e-5, "G should be ~1, got {}", px[1]);
            assert!((px[2] - 1.0).abs() < 1e-5, "B should be ~1, got {}", px[2]);
            assert!((px[3] - 1.0).abs() < 1e-5, "A should be ~1, got {}", px[3]);
        }
    }
}
