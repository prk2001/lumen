//! Model registry — Phase-1 stub.
//!
//! Real model distribution lands in Phase 2. For Phase 1 the registry
//! just gives us a stable [`Manifest`] type and a `verify_or_download`
//! that **only** verifies an already-present file's SHA-256 (no
//! network). Effects can plumb the type through their config now and
//! the implementation will be filled in later without an API churn.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::{AiError, Result};

/// What sort of model this is. Effects pick a manifest by `kind` so
/// they don't accidentally load an upscaler when they wanted a
/// denoiser.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelKind {
    /// Image denoising (e.g. SCUNet, NAFNet).
    Denoise,
    /// Super-resolution / upscaling (e.g. Real-ESRGAN, SwinIR).
    Upscale,
    /// Face restoration (e.g. CodeFormer, GFPGAN).
    FaceRestore,
    /// Catch-all for experimental / one-off models.
    Other,
}

/// Metadata describing where to fetch a model and how to verify it.
///
/// Future phases will extend this with size hints, signature support,
/// and per-EP optimizer hints. The fields below are the minimum
/// surface we are willing to commit to long-term.
#[derive(Debug, Clone)]
pub struct Manifest {
    /// Stable identifier used by effects, e.g. `"scunet-color-real-psnr"`.
    pub id: String,
    /// HTTPS URL the model can be fetched from. Phase 1 does not use
    /// this; it is here so the manifest format is stable.
    pub url: String,
    /// Lower-hex SHA-256 of the model bytes on disk after download.
    pub sha256: String,
    /// Which family of effects this model serves.
    pub model_kind: ModelKind,
}

impl Manifest {
    /// Construct a manifest from owned strings.
    pub fn new(
        id: impl Into<String>,
        url: impl Into<String>,
        sha256: impl Into<String>,
        model_kind: ModelKind,
    ) -> Self {
        Self {
            id: id.into(),
            url: url.into(),
            sha256: sha256.into(),
            model_kind,
        }
    }
}

/// Compute the lower-hex SHA-256 of a file.
pub fn sha256_file(path: &Path) -> Result<String> {
    let f = File::open(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => AiError::ModelNotFound(path.to_path_buf()),
        _ => AiError::Io(e),
    })?;
    let mut reader = BufReader::new(f);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Phase-1 stub: verify that the file at `path` matches `manifest.sha256`.
///
/// In Phase 2 this will additionally fetch the file from
/// `manifest.url` if it's missing. For now, a missing file is an
/// error — callers are responsible for placing weights on disk
/// themselves.
pub fn verify_or_download(manifest: &Manifest, path: &Path) -> Result<PathBuf> {
    if !path.exists() {
        // Phase 2 will perform the download here. Phase 1 surfaces
        // the missing file so the caller can fail fast.
        return Err(AiError::ModelNotFound(path.to_path_buf()));
    }

    let actual = sha256_file(path)?;
    let expected = manifest.sha256.to_ascii_lowercase();
    if actual != expected {
        return Err(AiError::Sha256Mismatch {
            path: path.to_path_buf(),
            expected,
            actual,
        });
    }
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(content: &[u8]) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob.bin");
        let mut f = File::create(&path).unwrap();
        f.write_all(content).unwrap();
        (dir, path)
    }

    #[test]
    fn sha256_of_known_string() {
        // sha256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        let (_dir, path) = write_temp(b"hello");
        let got = sha256_file(&path).unwrap();
        assert_eq!(
            got,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn verify_ok_when_hash_matches() {
        let (_dir, path) = write_temp(b"hello");
        let manifest = Manifest::new(
            "test",
            "https://example.invalid/model.onnx",
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
            ModelKind::Other,
        );
        assert!(verify_or_download(&manifest, &path).is_ok());
    }

    #[test]
    fn verify_fails_on_mismatch() {
        let (_dir, path) = write_temp(b"hello");
        let manifest = Manifest::new("test", "https://example.invalid/m", "deadbeef", ModelKind::Other);
        let err = verify_or_download(&manifest, &path).unwrap_err();
        assert!(matches!(err, AiError::Sha256Mismatch { .. }));
    }

    #[test]
    fn verify_fails_when_missing() {
        let manifest = Manifest::new("test", "https://example.invalid/m", "deadbeef", ModelKind::Other);
        let err = verify_or_download(&manifest, Path::new("/nope/nope/nope.onnx")).unwrap_err();
        assert!(matches!(err, AiError::ModelNotFound(_)));
    }
}
