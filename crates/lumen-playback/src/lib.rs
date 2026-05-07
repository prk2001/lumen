//! # lumen-playback
//!
//! Playback engine: scrubbing, frame cache, A/B compare, timecode.
//!
//! This crate exposes:
//!
//! * [`FrameKey`] — a content-addressed cache key combining an asset URI
//!   and an absolute frame index.
//! * [`LruFrameCache`] — a thread-safe in-memory LRU cache of decoded
//!   frames, capped by either entry count or total byte budget. Also
//!   implements [`lumen_core::FrameCache`] so existing pipeline code
//!   (e.g. `lumen_core::Scheduler`) can reuse it.
//! * [`FrameSource`] — the trait the engine reaches through when a frame
//!   isn't cached. A `lumen-io`-backed implementation is trivial: open a
//!   video and call `decode_video_frame`.
//! * [`PlaybackEngine`] — drives frame retrieval at a target index, with
//!   opportunistic look-ahead prefetching for smooth scrubbing.
//!
//! The engine does **not** render to a screen. It produces frames on
//! demand — UI layers (Tauri / wgpu / etc.) call into it from their own
//! display loop.

#![forbid(unsafe_op_in_unsafe_fn)]

mod cache;
mod engine;

pub use cache::{FrameKey, LruFrameCache};
pub use engine::{CacheStats, FrameSource, PlaybackEngine};

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
