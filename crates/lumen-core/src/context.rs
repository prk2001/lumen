//! Per-render execution context.
//!
//! Effects receive a `&mut Context` and can stash per-render state on it
//! (e.g. lazily computed LUTs). The host also uses it to surface global
//! information like the working color space and the active frame cache.

use std::sync::Arc;

use crate::color::ColorSpace;
use crate::frame::Frame;
use crate::time::Pts;

/// Frame cache trait — implementations live in higher-level crates.
/// Phase 1 ships only an in-memory LRU; later phases add disk caches.
pub trait FrameCache: Send + Sync + std::fmt::Debug {
    /// Look up a frame by content-addressed key (BLAKE3 hex).
    fn get(&self, key: &str) -> Option<Frame>;
    /// Store a frame under a content-addressed key.
    fn put(&self, key: &str, frame: Frame);
}

/// No-op cache used when the host doesn't supply one.
#[derive(Debug, Default)]
pub struct NullCache;

impl FrameCache for NullCache {
    fn get(&self, _key: &str) -> Option<Frame> { None }
    fn put(&self, _key: &str, _frame: Frame) {}
}

/// Per-render execution context.
#[derive(Debug)]
pub struct Context {
    /// Working color space — every effect should produce frames in this
    /// space unless explicitly converting.
    pub working_color_space: ColorSpace,
    /// Frame cache. Use [`NullCache`] for one-shot renders.
    pub cache: Arc<dyn FrameCache>,
    /// The PTS we're currently rendering, when relevant. `None` for
    /// stills.
    pub current_pts: Option<Pts>,
}

impl Context {
    /// Create a context with sensible defaults: ACEScg working space and
    /// a null cache.
    pub fn new_default() -> Self {
        Self {
            working_color_space: ColorSpace::AcesCg,
            cache: Arc::new(NullCache),
            current_pts: None,
        }
    }

    /// Context configured for sRGB single-image work — the default for
    /// the CLI's `apply` subcommand.
    pub fn for_still_srgb() -> Self {
        Self {
            working_color_space: ColorSpace::LinearSRgb,
            cache: Arc::new(NullCache),
            current_pts: None,
        }
    }
}

impl Default for Context {
    fn default() -> Self { Self::new_default() }
}
