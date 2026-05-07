//! `lumen batch` — apply a recipe to every image in a folder.
//!
//! The folder workflow most users actually want: "I have a directory of
//! CCTV grabs, run clarify aggressive on every one of them, write the
//! results to /out keeping filenames." Skips already-processed outputs
//! by default so it's safe to re-run.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{anyhow, Context as _, Result};
use rayon::prelude::*;

use crate::Recipe;

#[derive(Debug, Clone, Copy)]
pub struct BatchOptions {
    pub force: bool,
    /// Forwarded to encode_image for JPEG outputs (currently unused
    /// because run_chain_in_memory uses its own default).
    #[allow(dead_code)]
    pub jpeg_quality: u8,
    /// Output extension (".png" / ".jpg" / etc.). If `None`, mirrors input.
    pub out_ext: Option<&'static str>,
}

#[derive(Debug, Default)]
pub struct BatchStats {
    pub processed: usize,
    pub skipped:   usize,
    pub failed:    usize,
}

/// Apply `recipe_template` to every image under `input_dir`, writing
/// to `output_dir` with the same filename. The recipe's `input` and
/// `output` fields are overwritten per-file.
pub fn run_batch(
    input_dir: &Path,
    output_dir: &Path,
    recipe_template: &Recipe,
    opts: BatchOptions,
) -> Result<BatchStats> {
    if !input_dir.is_dir() {
        return Err(anyhow!("--input-dir is not a directory: {}", input_dir.display()));
    }
    std::fs::create_dir_all(output_dir).with_context(|| {
        format!("creating --output-dir {}", output_dir.display())
    })?;

    let mut entries: Vec<PathBuf> = std::fs::read_dir(input_dir)
        .with_context(|| format!("reading {}", input_dir.display()))?
        .filter_map(|r| r.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && is_image(p))
        .collect();
    entries.sort();

    let processed = AtomicUsize::new(0);
    let skipped   = AtomicUsize::new(0);
    let failed    = AtomicUsize::new(0);

    entries.par_iter().for_each(|src| {
        let file_name = match src.file_name() {
            Some(n) => n.to_string_lossy().into_owned(),
            None => { failed.fetch_add(1, Ordering::Relaxed); return; }
        };
        let out_name = match opts.out_ext {
            Some(ext) => {
                let stem = src.file_stem().map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| file_name.clone());
                format!("{stem}{ext}")
            }
            None => file_name.clone(),
        };
        let dst = output_dir.join(&out_name);
        if dst.exists() && !opts.force {
            skipped.fetch_add(1, Ordering::Relaxed);
            return;
        }

        let mut recipe: Recipe = (*recipe_template).clone();
        recipe.input = src.clone();
        recipe.output = dst.clone();

        match crate::run_chain_in_memory(&recipe) {
            Ok(()) => {
                processed.fetch_add(1, Ordering::Relaxed);
                eprintln!("ok    {}", file_name);
            }
            Err(e) => {
                failed.fetch_add(1, Ordering::Relaxed);
                eprintln!("FAIL  {}: {e:#}", file_name);
            }
        }
    });

    Ok(BatchStats {
        processed: processed.into_inner(),
        skipped:   skipped.into_inner(),
        failed:    failed.into_inner(),
    })
}

fn is_image(p: &Path) -> bool {
    let Some(ext) = p.extension().and_then(|e| e.to_str()) else { return false; };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "png" | "jpg" | "jpeg" | "tif" | "tiff" | "webp" | "bmp"
            | "cr2" | "cr3" | "nef" | "nrw" | "arw" | "dng" | "raf" | "orf"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn is_image_recognizes_common_extensions() {
        assert!(is_image(Path::new("a.png")));
        assert!(is_image(Path::new("b.JPG")));
        assert!(is_image(Path::new("c.cr2")));
        assert!(!is_image(Path::new("d.txt")));
        assert!(!is_image(Path::new("e")));
    }

    #[test]
    fn rejects_non_directory_input() {
        // Any path that isn't a real directory should err.
        let recipe = Recipe {
            input: PathBuf::new(),
            output: PathBuf::new(),
            chain: vec![crate::RecipeStep {
                effect: "lumen-fx-color.saturation".into(),
                label: None,
                params: json!({ "amount": 1.0 }),
            }],
        };
        let r = run_batch(
            Path::new("/this/does/not/exist/zzz"),
            Path::new("/tmp/lumen-batch-out"),
            &recipe,
            BatchOptions { force: false, jpeg_quality: 92, out_ext: None },
        );
        assert!(r.is_err());
    }
}
