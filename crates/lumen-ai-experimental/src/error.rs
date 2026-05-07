//! Crate-local error type.
//!
//! Wraps both [`ort`] errors and the workspace-wide
//! [`lumen_core::Error`] so callers get one type to match on, while
//! still being able to convert into the workspace error via the
//! `From` impl below.

use std::path::PathBuf;

/// All failures the AI layer can produce.
#[derive(Debug, thiserror::Error)]
pub enum AiError {
    /// I/O while reading a model file or a temp scratch file.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// ONNX Runtime returned an error (load, run, EP setup, …).
    #[error("ort error: {0}")]
    Ort(#[from] ort::Error),

    /// Model file does not exist or is not readable.
    #[error("model not found: {0}")]
    ModelNotFound(PathBuf),

    /// SHA-256 mismatch when verifying a model file against a manifest.
    #[error("sha256 mismatch for {path:?}: expected {expected}, got {actual}")]
    Sha256Mismatch {
        /// Model file that was checked.
        path: PathBuf,
        /// Expected SHA-256 (lower hex).
        expected: String,
        /// Actual SHA-256 of the file (lower hex).
        actual: String,
    },

    /// Tensor shape didn't match what the API expected.
    #[error("tensor shape error: {0}")]
    Shape(String),

    /// Anything else — used sparingly for messages we don't model.
    #[error("{0}")]
    Other(String),
}

impl From<AiError> for lumen_core::Error {
    fn from(value: AiError) -> Self {
        // Collapse into the workspace's free-form variant so callers in
        // other crates can use `?` without depending on `AiError`
        // directly.
        match value {
            AiError::Io(e) => lumen_core::Error::Io(e),
            other => lumen_core::Error::Other(other.to_string()),
        }
    }
}

/// Convenience alias.
pub type Result<T, E = AiError> = std::result::Result<T, E>;
