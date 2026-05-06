//! Workspace-wide error type.
//!
//! Every fallible Lumen API returns [`Result<T>`](crate::Result), aliased
//! over [`Error`]. Using one error enum across crates keeps `?` ergonomic
//! and lets the UI surface stable error codes.

use std::path::PathBuf;

/// The single Lumen error type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// I/O error from `std::io`.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Failed to decode a media file.
    #[error("decode error ({path:?}): {message}")]
    Decode { path: Option<PathBuf>, message: String },

    /// Failed to encode a media file.
    #[error("encode error ({path:?}): {message}")]
    Encode { path: Option<PathBuf>, message: String },

    /// File format isn't supported.
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Invalid effect parameter.
    #[error("invalid parameter '{name}': {reason}")]
    InvalidParameter { name: String, reason: String },

    /// Required parameter missing.
    #[error("missing required parameter '{0}'")]
    MissingParameter(String),

    /// Effect not in registry.
    #[error("effect '{0}' not found in registry")]
    EffectNotFound(String),

    /// Pipeline graph problem (cycle, dangling edge, etc.).
    #[error("graph error: {0}")]
    Graph(String),

    /// Color space conversion failed.
    #[error("color error: {0}")]
    Color(String),

    /// Frame layout mismatch (wrong size, wrong channel count, etc.).
    #[error("layout error: {0}")]
    Layout(String),

    /// JSON ser/de failure.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Project file schema mismatch.
    #[error("project schema mismatch: expected {expected}, found {found}")]
    SchemaMismatch { expected: String, found: String },

    /// A feature is recognized but not yet implemented.
    /// Phase markers in the codebase return this.
    #[error("not yet implemented: {0}")]
    NotImplemented(&'static str),

    /// Catch-all for anyhow-style application errors.
    #[error("{0}")]
    Other(String),
}

impl Error {
    /// Construct a [`Error::Decode`] with no path.
    pub fn decode<S: Into<String>>(message: S) -> Self {
        Self::Decode { path: None, message: message.into() }
    }

    /// Construct a [`Error::Decode`] with a path.
    pub fn decode_at<S: Into<String>>(path: PathBuf, message: S) -> Self {
        Self::Decode { path: Some(path), message: message.into() }
    }

    /// Construct an [`Error::Encode`] with no path.
    pub fn encode<S: Into<String>>(message: S) -> Self {
        Self::Encode { path: None, message: message.into() }
    }

    /// Construct an [`Error::Encode`] with a path.
    pub fn encode_at<S: Into<String>>(path: PathBuf, message: S) -> Self {
        Self::Encode { path: Some(path), message: message.into() }
    }

    /// Stable string code for telemetry / UI.
    pub fn code(&self) -> &'static str {
        match self {
            Error::Io(_) => "IO",
            Error::Decode { .. } => "DECODE",
            Error::Encode { .. } => "ENCODE",
            Error::UnsupportedFormat(_) => "UNSUPPORTED_FORMAT",
            Error::InvalidParameter { .. } => "INVALID_PARAMETER",
            Error::MissingParameter(_) => "MISSING_PARAMETER",
            Error::EffectNotFound(_) => "EFFECT_NOT_FOUND",
            Error::Graph(_) => "GRAPH",
            Error::Color(_) => "COLOR",
            Error::Layout(_) => "LAYOUT",
            Error::Json(_) => "JSON",
            Error::SchemaMismatch { .. } => "SCHEMA_MISMATCH",
            Error::NotImplemented(_) => "NOT_IMPLEMENTED",
            Error::Other(_) => "OTHER",
        }
    }
}

/// Workspace `Result` alias.
pub type Result<T, E = Error> = std::result::Result<T, E>;
