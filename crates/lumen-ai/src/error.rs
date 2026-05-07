//! Crate-local error type.
//!
//! Wraps tract failures and the workspace-wide [`lumen_core::Error`] so
//! callers get one type to match on, while still being able to convert
//! into the workspace error via the `From` impl below.

use std::path::PathBuf;

/// All failures the AI layer can produce.
#[derive(Debug, thiserror::Error)]
pub enum AiError {
    /// I/O while reading a model file or a temp scratch file.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// tract returned an error while loading, optimizing, or running a
    /// model. tract uses [`anyhow::Error`] internally; we capture it as
    /// a string so this enum stays `Send + Sync` and easy to display.
    #[error("tract error: {0}")]
    Tract(String),

    /// Model file does not exist or is not readable.
    #[error("model not found: {0}")]
    ModelNotFound(PathBuf),

    /// Tensor shape didn't match what the API expected.
    #[error("tensor shape error: {0}")]
    Shape(String),

    /// Anything else — used sparingly for messages we don't model.
    #[error("{0}")]
    Other(String),
}

impl From<tract_onnx::prelude::TractError> for AiError {
    fn from(err: tract_onnx::prelude::TractError) -> Self {
        AiError::Tract(format!("{err:#}"))
    }
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
