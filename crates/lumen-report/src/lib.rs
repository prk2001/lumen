//! # lumen-report
//!
//! Reporting / visualization / presentation builders.
//!
//! This crate produces self-contained HTML reports describing a single
//! Lumen render — the input image, the output image, the recipe used,
//! quality metrics (MSE / PSNR / SSIM), and integrity hashes. The
//! produced HTML is a single file with all images inlined as `data:`
//! URIs, suitable for delivering to a client or attaching to a
//! forensic case file. No external CDN references are emitted.
//!
//! ## Example
//!
//! ```no_run
//! use std::path::Path;
//! use lumen_report::{ReportInput, save_html};
//!
//! let input = ReportInput {
//!     title: "Case 0001 — denoise pass",
//!     generated_at: time::OffsetDateTime::now_utc(),
//!     input_path: Path::new("in.png"),
//!     output_path: Path::new("out.png"),
//!     recipe_json: Some(r#"{"steps":[]}"#),
//!     mse: Some(0.0),
//!     psnr: None,
//!     ssim: Some(1.0),
//!     input_hash: Some("blake3:abc"),
//!     output_hash: Some("blake3:def"),
//!     render_ms: Some(120),
//!     software_version: env!("CARGO_PKG_VERSION"),
//! };
//! save_html(&input, "report.html").unwrap();
//! ```

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, rust_2018_idioms)]

use std::fmt::Write as _;
use std::path::Path;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use lumen_core::{Error, Result};
use time::OffsetDateTime;

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// All inputs needed to render a single-render HTML report.
///
/// All fields except the few required ones are optional — pass `None`
/// when a metric or a hash is unavailable and the corresponding row in
/// the report will render as an em-dash.
#[derive(Debug, Clone)]
pub struct ReportInput<'a> {
    /// Report title. Shown in the document `<title>` and the page header.
    pub title: &'a str,
    /// Wall-clock time the render finished. Rendered next to the title.
    pub generated_at: OffsetDateTime,
    /// On-disk path to the input image. Used for the row label and to
    /// determine the MIME type of the inlined image.
    pub input_path: &'a Path,
    /// On-disk path to the output image. Used for the row label and to
    /// determine the MIME type of the inlined image.
    pub output_path: &'a Path,
    /// Pretty-printable recipe JSON (the entire pipeline). When `Some`,
    /// it is emitted verbatim inside a `<pre>` block, with `<>&`
    /// escaped. The caller is responsible for pretty-printing — this
    /// crate does not parse the JSON.
    pub recipe_json: Option<&'a str>,
    /// Mean Squared Error between input and output, if known.
    pub mse: Option<f64>,
    /// Peak Signal-to-Noise Ratio in dB, if known. `None`, NaN, and
    /// infinite values render as the symbol `∞ dB` (a perfect match).
    pub psnr: Option<f64>,
    /// Structural Similarity Index Measure ∈ [-1, 1], if known.
    pub ssim: Option<f64>,
    /// Hex-encoded hash of the input file (any algorithm — string is
    /// rendered verbatim, so callers typically prefix `"blake3:"` or
    /// `"sha256:"`).
    pub input_hash: Option<&'a str>,
    /// Hex-encoded hash of the output file.
    pub output_hash: Option<&'a str>,
    /// Render wall-time in milliseconds, if known.
    pub render_ms: Option<u64>,
    /// Lumen build version string (typically `env!("CARGO_PKG_VERSION")`
    /// from the calling binary).
    pub software_version: &'a str,
}

/// Build the HTML report as a self-contained string.
///
/// Reads `input.input_path` and `input.output_path` from disk and
/// inlines them as base64 `data:` URIs — the resulting string can be
/// written to a file and opened in any browser without further
/// resources.
///
/// # Errors
///
/// Returns [`Error::Io`] if either image file cannot be read, and
/// [`Error::UnsupportedFormat`] if the file extension is not one of
/// the recognized image formats (`png`, `jpg`/`jpeg`, `webp`, `gif`,
/// `bmp`, `tiff`/`tif`).
pub fn build_html(input: &ReportInput<'_>) -> Result<String> {
    let in_mime = mime_for(input.input_path)?;
    let out_mime = mime_for(input.output_path)?;
    let in_bytes = std::fs::read(input.input_path)?;
    let out_bytes = std::fs::read(input.output_path)?;
    let in_data = BASE64.encode(&in_bytes);
    let out_data = BASE64.encode(&out_bytes);

    let title_e = escape_html(input.title);
    let version_e = escape_html(input.software_version);
    let generated_e = escape_html(&input.generated_at.to_string());
    let input_path_e = escape_html(&input.input_path.display().to_string());
    let output_path_e = escape_html(&input.output_path.display().to_string());

    let recipe_block = match input.recipe_json {
        Some(j) => format!("<pre class=\"recipe\">{}</pre>", escape_html(j)),
        None => "<p class=\"dim\">No recipe attached.</p>".to_string(),
    };

    let mse_cell = fmt_metric(input.mse, 6);
    let psnr_cell = fmt_psnr(input.psnr);
    let ssim_cell = fmt_metric(input.ssim, 4);
    let render_ms_cell = match input.render_ms {
        Some(ms) => format!("{ms} ms"),
        None => "—".to_string(),
    };

    let in_hash_cell = match input.input_hash {
        Some(h) => escape_html(h),
        None => "—".to_string(),
    };
    let out_hash_cell = match input.output_hash {
        Some(h) => escape_html(h),
        None => "—".to_string(),
    };

    let mut out = String::with_capacity(in_data.len() + out_data.len() + 8 * 1024);

    write!(
        &mut out,
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{
    --bg: #0e0f12;
    --fg: #e7e9ee;
    --dim: #8b8f99;
    --accent: #d6a854;
    --warn: #d65454;
    --border: #1f2128;
    --panel: #16181d;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; background: var(--bg); color: var(--fg);
    font: 13px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }}
  header {{
    padding: 14px 22px; display: flex; align-items: baseline; gap: 14px;
    border-bottom: 1px solid var(--border); background: var(--panel);
  }}
  header .brand {{ font-weight: 600; letter-spacing: 0.04em; color: var(--accent); }}
  header h1 {{ margin: 0; font-size: 16px; font-weight: 500; }}
  header .meta {{ color: var(--dim); margin-left: auto; }}
  main {{ padding: 22px; max-width: 1400px; margin: 0 auto; }}
  section {{ margin-bottom: 24px; }}
  section h2 {{
    margin: 0 0 10px 0; font-size: 11px; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.08em; color: var(--dim);
  }}
  .images {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .frame {{
    background: repeating-conic-gradient(#1d2027 0% 25%, #161920 0% 50%) 0 0 / 16px 16px;
    border: 1px solid var(--border);
    padding: 12px; display: flex; flex-direction: column; min-height: 180px;
  }}
  .frame .label {{
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--dim); margin-bottom: 8px;
  }}
  .frame img {{ max-width: 100%; max-height: 70vh; display: block; margin: auto; image-rendering: pixelated; }}
  table.kv {{ width: 100%; border-collapse: collapse; }}
  table.kv td, table.kv th {{
    border-top: 1px solid var(--border); padding: 6px 10px; text-align: left;
    font-weight: 400;
  }}
  table.kv th {{ color: var(--dim); width: 18ch; font-weight: 500; }}
  table.kv td.mono, table.kv td.hash {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  pre.recipe {{
    background: var(--panel); border: 1px solid var(--border); padding: 12px;
    margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px; line-height: 1.5; white-space: pre-wrap; word-break: break-word;
    max-height: 50vh; overflow: auto;
  }}
  .dim {{ color: var(--dim); }}
  footer {{
    margin-top: 32px; padding: 12px 22px; border-top: 1px solid var(--border);
    color: var(--dim); font-size: 11px;
  }}
</style>
</head>
<body>
<header>
  <span class="brand">LUMEN</span>
  <h1>{title}</h1>
  <span class="meta">generated {generated} · v{version}</span>
</header>
<main>
  <section>
    <h2>Images</h2>
    <div class="images">
      <div class="frame">
        <span class="label">Input · {input_path}</span>
        <img src="data:{in_mime};base64,{in_data}" alt="input">
      </div>
      <div class="frame">
        <span class="label">Output · {output_path}</span>
        <img src="data:{out_mime};base64,{out_data}" alt="output">
      </div>
    </div>
  </section>
  <section>
    <h2>Metrics</h2>
    <table class="kv">
      <tr><th>MSE</th><td class="mono">{mse}</td></tr>
      <tr><th>PSNR</th><td class="mono">{psnr}</td></tr>
      <tr><th>SSIM</th><td class="mono">{ssim}</td></tr>
      <tr><th>Render time</th><td class="mono">{render_ms}</td></tr>
    </table>
  </section>
  <section>
    <h2>Hashes</h2>
    <table class="kv">
      <tr><th>Input</th><td class="hash">{in_hash}</td></tr>
      <tr><th>Output</th><td class="hash">{out_hash}</td></tr>
    </table>
  </section>
  <section>
    <h2>Recipe</h2>
    {recipe}
  </section>
</main>
<footer>
  Generated by lumen-report v{report_version} — Lumen v{version}
</footer>
</body>
</html>
"##,
        title = title_e,
        generated = generated_e,
        version = version_e,
        input_path = input_path_e,
        output_path = output_path_e,
        in_mime = in_mime,
        out_mime = out_mime,
        in_data = in_data,
        out_data = out_data,
        mse = mse_cell,
        psnr = psnr_cell,
        ssim = ssim_cell,
        render_ms = render_ms_cell,
        in_hash = in_hash_cell,
        out_hash = out_hash_cell,
        recipe = recipe_block,
        report_version = CRATE_VERSION,
    )
    .map_err(|e| Error::Other(format!("html format error: {e}")))?;

    Ok(out)
}

/// Build the HTML report and write it atomically to `path`.
///
/// Writes to a sibling temp file (`<path>.tmp`) and renames into place
/// when the bytes are flushed, matching the same pattern used by
/// `lumen_core::Project::save`.
///
/// # Errors
///
/// Any error from [`build_html`], plus [`Error::Io`] for write
/// failures.
pub fn save_html<P: AsRef<Path>>(input: &ReportInput<'_>, path: P) -> Result<()> {
    let path = path.as_ref();
    let html = build_html(input)?;
    let tmp = path.with_extension("html.tmp");
    std::fs::write(&tmp, html.as_bytes())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn mime_for(path: &Path) -> Result<&'static str> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase());
    match ext.as_deref() {
        Some("png") => Ok("image/png"),
        Some("jpg" | "jpeg") => Ok("image/jpeg"),
        Some("webp") => Ok("image/webp"),
        Some("gif") => Ok("image/gif"),
        Some("bmp") => Ok("image/bmp"),
        Some("tif" | "tiff") => Ok("image/tiff"),
        Some(other) => Err(Error::UnsupportedFormat(format!(
            "image extension '.{other}' is not supported by lumen-report"
        ))),
        None => Err(Error::UnsupportedFormat(
            "image path has no extension".to_string(),
        )),
    }
}

fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

fn fmt_metric(v: Option<f64>, precision: usize) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:.precision$}"),
        Some(x) if x.is_nan() => "NaN".to_string(),
        Some(_) => "∞".to_string(),
        None => "—".to_string(),
    }
}

fn fmt_psnr(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:.3} dB"),
        // None / NaN / ±Inf → perfect match: ∞ dB.
        _ => "∞ dB".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests");
        p.push("fixtures");
        p.push(name);
        p
    }

    fn sample_input<'a>(
        title: &'a str,
        in_p: &'a Path,
        out_p: &'a Path,
        recipe: Option<&'a str>,
        version: &'a str,
    ) -> ReportInput<'a> {
        ReportInput {
            title,
            generated_at: OffsetDateTime::UNIX_EPOCH,
            input_path: in_p,
            output_path: out_p,
            recipe_json: recipe,
            mse: Some(0.012_345),
            psnr: Some(38.421),
            ssim: Some(0.9876),
            input_hash: Some("blake3:0xdead"),
            output_hash: Some("blake3:0xbeef"),
            render_ms: Some(123),
            software_version: version,
        }
    }

    #[test]
    fn build_html_contains_required_pieces() {
        let in_p = fixture("input.png");
        let out_p = fixture("output.png");
        let recipe = r#"{"steps":[{"effect":"<denoise>"}]}"#;
        let input = sample_input(
            "Case 42 — primary render",
            &in_p,
            &out_p,
            Some(recipe),
            "0.1.0-test",
        );

        let html = build_html(&input).unwrap();

        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(html.contains("Case 42 — primary render"));
        assert!(html.contains("blake3:0xdead"));
        assert!(html.contains("blake3:0xbeef"));
        assert!(html.contains("data:image/png;base64,"));
        // Both PNGs are inlined.
        let png_uri_count = html.matches("data:image/png;base64,").count();
        assert_eq!(png_uri_count, 2);
        // Recipe JSON is HTML-escaped.
        assert!(html.contains("&lt;denoise&gt;"));
        assert!(!html.contains("<denoise>"));
        // Metric formatting.
        assert!(html.contains("38.421 dB"));
    }

    #[test]
    fn psnr_renders_as_infinity_when_unknown_or_non_finite() {
        let in_p = fixture("input.png");
        let out_p = fixture("output.png");

        let mut input = sample_input("t", &in_p, &out_p, None, "v");
        // None
        input.psnr = None;
        let html = build_html(&input).unwrap();
        assert!(html.contains("∞ dB"));

        // NaN
        input.psnr = Some(f64::NAN);
        let html = build_html(&input).unwrap();
        assert!(html.contains("∞ dB"));

        // +Inf (perfect match)
        input.psnr = Some(f64::INFINITY);
        let html = build_html(&input).unwrap();
        assert!(html.contains("∞ dB"));

        // Finite values must NOT render as ∞ dB.
        input.psnr = Some(40.0);
        let html = build_html(&input).unwrap();
        assert!(!html.contains("∞ dB"));
        assert!(html.contains("40.000 dB"));
    }

    #[test]
    fn save_html_writes_valid_html_file() {
        let dir = tempfile::tempdir().unwrap();
        let report_path = dir.path().join("report.html");

        let in_p = fixture("input.png");
        let out_p = fixture("output.png");
        let input = sample_input("save test", &in_p, &out_p, Some("{}"), "0.0.0");

        save_html(&input, &report_path).unwrap();

        let bytes = std::fs::read(&report_path).unwrap();
        let s = std::str::from_utf8(&bytes).expect("output is valid UTF-8");
        assert!(s.starts_with("<!DOCTYPE html>"));
        assert!(s.contains("save test"));
        assert!(s.contains("</html>"));
        // Atomic-write tmp file should not be left behind.
        let tmp_leftover = report_path.with_extension("html.tmp");
        assert!(!tmp_leftover.exists());
    }

    #[test]
    fn unsupported_extension_is_rejected() {
        let in_p = PathBuf::from("nope.xyz");
        let out_p = fixture("output.png");
        let input = sample_input("t", &in_p, &out_p, None, "v");
        let err = build_html(&input).unwrap_err();
        assert_eq!(err.code(), "UNSUPPORTED_FORMAT");
    }
}
