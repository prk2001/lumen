//! # lumen-core
//!
//! Core types, pipeline DAG, error model, and project file format for
//! Lumen — the photo/video enhancement suite.
//!
//! Every other crate in the workspace depends on this one. It is
//! deliberately small and dependency-light: pure Rust, no FFmpeg,
//! no GPU, no AI.
//!
//! See `docs/ARCHITECTURE.md` at the repo root for the wider picture.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, rust_2018_idioms)]
#![allow(missing_docs)] // Phase 1 — re-tighten before v0.1 release

pub mod asset;
pub mod color;
pub mod context;
pub mod effect;
pub mod error;
pub mod frame;
pub mod graph;
pub mod params;
pub mod project;
pub mod registry;
pub mod time;

pub use asset::{Asset, AssetId, AssetKind, AssetMetadata};
pub use color::{linear_to_srgb, srgb_to_linear, ColorSpace};
pub use context::{Context, FrameCache, NullCache};
pub use effect::{Capabilities, Category, Effect, EffectMetadata, EffectRef};
pub use error::{Error, Result};
pub use frame::{Frame, PixelData, PixelLayout};
pub use graph::{Graph, Node, NodeId};
pub use params::{ParamKind, ParamSpec, ParamValue, ParamValues};
pub use project::{Project, SCHEMA};
pub use registry::EffectRegistry;
pub use time::{Pts, Rational};

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
