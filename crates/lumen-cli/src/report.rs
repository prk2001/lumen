//! Self-contained HTML case report.
//!
//! Renders a single HTML file under `reports/` that captures everything
//! a reviewer needs without running the CLI:
//!
//! - case metadata (id, evidence id, case name, agency, operator)
//! - audit timeline: every signed entry, in order, with action + note
//! - artifact thumbnails: original input, final output, and every stage
//!   frame (`stages/<output-stem>/NN-<effect>.png`), inlined as
//!   data: URIs so the HTML is portable and survives offline review
//!
//! The report is generated AFTER a chain audit, so a malformed log
//! (broken signatures, reordered entries) errors before any HTML hits
//! disk. We intentionally don't sign the HTML itself — it's a
//! presentation layer; the audit log + strict artifact check remain
//! the cryptographic source of truth.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use base64::Engine as _;

use crate::case::{self, AuditEntry, CaseMetadata};

/// Render a self-contained HTML report into `reports/<output_filename>`.
/// Returns the path written.
pub fn render_html_report(case_dir: &Path, output_filename: &str) -> Result<PathBuf> {
    // Audit must be valid before we hand the reviewer a friendly UI.
    let entries = case::verify_audit_log(case_dir)?;
    let metadata = case::load_metadata(case_dir)?;

    // Collect inline images. Each tile is (label, mime, data-uri-payload).
    let mut tiles: Vec<Tile> = Vec::new();

    // 1. Original inputs (every file under inputs/).
    for path in dir_files(&case_dir.join("inputs"))? {
        if let Some(tile) = tile_from_path(&path, "Input")? {
            tiles.push(tile);
        }
    }
    // 2. Final outputs.
    for path in dir_files(&case_dir.join("outputs"))? {
        if path.is_file() {
            if let Some(tile) = tile_from_path(&path, "Output")? {
                tiles.push(tile);
            }
        }
    }
    // 3. Stage frames — flatten one level deep (stages/<output_stem>/...).
    let stages_root = case_dir.join("stages");
    if stages_root.is_dir() {
        for child in std::fs::read_dir(&stages_root)? {
            let child = child?;
            if child.file_type()?.is_dir() {
                let label_root = child.file_name().to_string_lossy().to_string();
                for path in dir_files(&child.path())? {
                    if let Some(mut tile) =
                        tile_from_path(&path, &format!("Stage · {label_root}"))?
                    {
                        // Bump the file-stem into the tile caption so
                        // viewers can tell stage 03 from stage 06.
                        tile.caption = format!(
                            "{} · {}",
                            tile.caption,
                            path.file_stem().unwrap_or_default().to_string_lossy()
                        );
                        tiles.push(tile);
                    }
                }
            }
        }
    }

    // 4. Recipes — inline as <details><pre>... so reviewers can read
    //    the recipe that drove each render.
    let mut recipes: Vec<(String, String)> = Vec::new();
    for path in dir_files(&case_dir.join("recipes"))? {
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "(unknown)".into());
        let body = std::fs::read_to_string(&path).unwrap_or_default();
        recipes.push((name, body));
    }

    let html = build_html(&metadata, &entries, &tiles, &recipes);

    let reports_dir = case_dir.join("reports");
    std::fs::create_dir_all(&reports_dir)?;
    let out_path = reports_dir.join(output_filename);
    std::fs::write(&out_path, html.as_bytes())
        .with_context(|| format!("writing {}", out_path.display()))?;
    Ok(out_path)
}

struct Tile {
    caption: String,
    data_uri: String,
}

fn dir_files(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().ok().is_some_and(|t| t.is_file()))
        .map(|e| e.path())
        .collect();
    out.sort();
    Ok(out)
}

fn mime_for(path: &Path) -> Option<&'static str> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());
    match ext.as_deref() {
        Some("png") => Some("image/png"),
        Some("jpg" | "jpeg") => Some("image/jpeg"),
        Some("webp") => Some("image/webp"),
        Some("bmp") => Some("image/bmp"),
        Some("tif" | "tiff") => Some("image/tiff"),
        Some("gif") => Some("image/gif"),
        _ => None,
    }
}

fn tile_from_path(path: &Path, caption_kind: &str) -> Result<Option<Tile>> {
    let Some(mime) = mime_for(path) else {
        return Ok(None);
    };
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "(unnamed)".into());
    Ok(Some(Tile {
        caption: format!("{caption_kind} · {name}"),
        data_uri: format!("data:{mime};base64,{b64}"),
    }))
}

fn build_html(
    m: &CaseMetadata,
    entries: &[AuditEntry],
    tiles: &[Tile],
    recipes: &[(String, String)],
) -> String {
    let title = format!(
        "Lumen case report · {} ({})",
        html_escape(&m.case_name),
        html_escape(&m.case_id),
    );

    let mut body = String::new();
    body.push_str("<header><h1>");
    body.push_str(&html_escape(&m.case_name));
    body.push_str("</h1><p class=meta>Case ");
    body.push_str(&html_escape(&m.case_id));
    body.push_str(" · Evidence ");
    body.push_str(&html_escape(&m.evidence_id));
    body.push_str(" · Agency ");
    body.push_str(&html_escape(&m.agency));
    body.push_str("</p>");
    body.push_str("<p class=meta>Operator <b>");
    body.push_str(&html_escape(&m.created_by.display_name));
    body.push_str("</b> · ");
    body.push_str(&html_escape(&m.created_by.identifier));
    body.push_str(" · pubkey <code>");
    body.push_str(&m.created_by.public_key_hex[..16.min(m.created_by.public_key_hex.len())]);
    body.push_str("…</code></p>");
    if let Some(h) = &m.original_input_hash {
        body.push_str("<p class=meta>Original input · <code>");
        body.push_str(&html_escape(h));
        body.push_str("</code></p>");
    }
    body.push_str("</header>");

    // Audit timeline.
    body.push_str("<section><h2>Audit timeline</h2><ol class=timeline>");
    for e in entries {
        body.push_str("<li><div class=row><span class=seq>#");
        body.push_str(&e.seq.to_string());
        body.push_str("</span><span class=action>");
        body.push_str(&html_escape(&e.action));
        body.push_str("</span><span class=at>");
        body.push_str(&html_escape(&format!("{}", e.at)));
        body.push_str("</span></div>");
        if !e.note.is_empty() {
            body.push_str("<div class=note>");
            body.push_str(&html_escape(&e.note));
            body.push_str("</div>");
        }
        body.push_str("<div class=hashes>");
        for (label, h) in [
            ("input", e.input_hash.as_deref()),
            ("output", e.output_hash.as_deref()),
            ("recipe", e.recipe_hash.as_deref()),
        ] {
            if let Some(hash) = h {
                body.push_str("<span class=hash><b>");
                body.push_str(label);
                body.push_str("</b> ");
                body.push_str(&html_escape(hash));
                body.push_str("</span>");
            }
        }
        body.push_str("</div>");
        body.push_str("<div class=sig>sig <code>");
        body.push_str(&e.entry_signature_hex[..16.min(e.entry_signature_hex.len())]);
        body.push_str("…</code></div></li>");
    }
    body.push_str("</ol></section>");

    // Image gallery.
    if !tiles.is_empty() {
        body.push_str("<section><h2>Artifacts</h2><div class=gallery>");
        for t in tiles {
            body.push_str("<figure><img loading=lazy src=\"");
            body.push_str(&t.data_uri);
            body.push_str("\" alt=\"");
            body.push_str(&html_escape(&t.caption));
            body.push_str("\"><figcaption>");
            body.push_str(&html_escape(&t.caption));
            body.push_str("</figcaption></figure>");
        }
        body.push_str("</div></section>");
    }

    // Recipes.
    if !recipes.is_empty() {
        body.push_str("<section><h2>Recipes</h2>");
        for (name, json) in recipes {
            body.push_str("<details><summary>");
            body.push_str(&html_escape(name));
            body.push_str("</summary><pre>");
            body.push_str(&html_escape(json));
            body.push_str("</pre></details>");
        }
        body.push_str("</section>");
    }

    body.push_str("<footer><p>Generated by <code>lumen case report</code>. \
        Audit signatures are the cryptographic source of truth — re-run \
        <code>lumen case audit --strict --dir .</code> to re-verify.</p></footer>");

    format!(
        "<!doctype html>\n<html lang=en><head><meta charset=utf-8>\
         <meta name=viewport content=\"width=device-width,initial-scale=1\">\
         <title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>",
        title = title,
        CSS = REPORT_CSS,
        body = body,
    )
}

fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            other => out.push(other),
        }
    }
    out
}

const REPORT_CSS: &str = r#"
:root { color-scheme: light dark; }
body { font: 15px/1.5 -apple-system, system-ui, "Segoe UI", sans-serif;
       max-width: 1100px; margin: 32px auto; padding: 0 18px; color: #1a1a1a; }
header h1 { margin-bottom: 4px; }
.meta { color: #555; margin: 4px 0; }
section { margin: 28px 0; }
h2 { border-bottom: 1px solid #ddd; padding-bottom: 6px; }
.timeline { list-style: none; padding-left: 0; }
.timeline li { border-left: 3px solid #3da06a; padding: 10px 14px; margin: 12px 0;
               background: #f6f8f6; border-radius: 0 6px 6px 0; }
.row { display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap; }
.seq { font-weight: 700; color: #3da06a; }
.action { font-weight: 600; }
.at { color: #777; font-size: 13px; }
.note { margin-top: 6px; }
.hashes { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 6px; font-size: 12px; color: #555; }
.hash code, .sig code, code { font-family: "SF Mono", Menlo, monospace; }
.sig { color: #888; font-size: 12px; margin-top: 4px; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
           gap: 12px; }
figure { margin: 0; background: #f1f1f1; border-radius: 6px; padding: 6px; }
figure img { width: 100%; height: auto; display: block; border-radius: 4px; }
figcaption { font-size: 12px; color: #555; margin-top: 4px; word-break: break-all; }
details { background: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px;
          padding: 6px 10px; margin: 8px 0; }
summary { cursor: pointer; font-weight: 600; }
pre { white-space: pre-wrap; word-break: break-word; font-size: 12px;
      background: #f1f1f1; padding: 8px; border-radius: 4px; }
footer { color: #777; font-size: 13px; margin-top: 40px; border-top: 1px solid #ddd;
         padding-top: 14px; }
@media (prefers-color-scheme: dark) {
  body { background: #161616; color: #e8e8e8; }
  .meta, .at, .sig, figcaption, footer { color: #aaa; }
  h2 { border-color: #333; }
  .timeline li { background: #1d2620; }
  figure { background: #232323; }
  pre, details { background: #1d1d1d; border-color: #2a2a2a; }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_escape_handles_basic_entities() {
        assert_eq!(html_escape("a&b<c>\"d'"), "a&amp;b&lt;c&gt;&quot;d&#39;");
    }

    #[test]
    fn report_renders_with_only_metadata() -> Result<()> {
        // Shared with case::tests — both touch LUMEN_OPERATOR and must
        // serialize across the whole test binary.
        let _guard = crate::case::TEST_OPERATOR_SERIALIZER
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!(
            "lumen-report-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir)?;
        let op_path = dir.join("operator.json");
        std::env::set_var("LUMEN_OPERATOR", &op_path);
        crate::operator::init("Reporter", "Test PD", "R-1", false)?;
        crate::case::init(&dir, "C-R", "EVD-R", "Report Test", "Test PD", None)?;
        let out = render_html_report(&dir, "case-report.html")?;
        assert!(out.exists());
        let html = std::fs::read_to_string(&out)?;
        assert!(html.contains("Audit timeline"));
        assert!(html.contains("Report Test"));
        std::env::remove_var("LUMEN_OPERATOR");
        Ok(())
    }
}
