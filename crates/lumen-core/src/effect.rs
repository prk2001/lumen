//! The [`Effect`] trait — the unit of image processing.
//!
//! Built-in effects in `lumen-fx-*` and third-party plugins via
//! `lumen-api` implement the same trait. There is no separate "internal"
//! API.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::context::Context;
use crate::error::Result;
use crate::frame::Frame;
use crate::params::{ParamSpec, ParamValues};

/// A semantic grouping that mirrors the source spec's 30 categories.
/// Used by UIs to bucket effects into menus / palettes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Category {
    Input,
    Playback,
    Ui,
    Exposure,
    Color,
    Sharpen,
    Denoise,
    Compression,
    Geometric,
    Stabilize,
    Deblur,
    Upscale,
    Temporal,
    Ai,
    Face,
    Text,
    Mask,
    Weather,
    Modalities,
    Measure,
    Audio,
    Auth,
    Workflow,
    Collaboration,
    Report,
    Export,
    Performance,
    Api,
    Platform,
    Qa,
}

/// What an effect can and can't do. Used by the scheduler to plan work.
#[derive(Debug, Clone, Copy, Default)]
pub struct Capabilities {
    /// True if `apply` produces deterministic output for fixed input
    /// + params (allows result caching).
    pub deterministic: bool,
    /// True if a GPU implementation exists. Falls back to CPU otherwise.
    pub gpu: bool,
    /// True if effect can stream pixels (no full-frame buffer needed).
    pub streamable: bool,
    /// True if effect needs more than one input frame
    /// (e.g. temporal denoise looking at neighbors).
    pub temporal: bool,
}

impl Capabilities {
    pub const fn cpu_only_deterministic() -> Self {
        Self { deterministic: true, gpu: false, streamable: false, temporal: false }
    }
}

/// Static metadata identifying an effect.
///
/// Defined in code by each effect — not loaded from JSON — so it carries
/// `&'static str` references and only needs `Serialize` for UI/API
/// introspection.
#[derive(Debug, Clone, Serialize)]
pub struct EffectMetadata {
    /// Globally unique stable id, e.g. `"lumen-fx-exposure.brightness_contrast"`.
    /// Used in project files; never localized.
    pub id: &'static str,
    /// Human-readable name (English; localization elsewhere).
    pub display_name: &'static str,
    /// One-line description shown in UI tooltips.
    pub description: &'static str,
    /// Spec category mapping.
    pub category: Category,
    /// Implementation version; bump on parameter or behavior changes
    /// to invalidate caches.
    pub version: u32,
}

/// The unit of image processing.
pub trait Effect: Send + Sync + std::fmt::Debug {
    /// Identity + categorization.
    fn metadata(&self) -> &EffectMetadata;

    /// Parameter specifications. Order is significant for UI layout.
    fn parameters(&self) -> &[ParamSpec];

    /// Capabilities — drives scheduler routing.
    fn capabilities(&self) -> Capabilities;

    /// Apply the effect to one frame. Implementations may consume the
    /// input (no need to clone if they can repurpose its buffer).
    fn apply(&self, ctx: &mut Context, input: Frame, params: &ParamValues) -> Result<Frame>;
}

/// Convenience type alias.
pub type EffectRef = Arc<dyn Effect>;
