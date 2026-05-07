//! Playback engine — drives frame retrieval against a [`FrameSource`]
//! with an [`LruFrameCache`] in front for cheap scrubbing.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use lumen_core::{Frame, Result};
use tracing::trace;

use crate::cache::{FrameKey, LruFrameCache};

/// Source of frames the engine reaches through on a cache miss.
///
/// A real-world implementation will typically wrap a `lumen-io` video
/// decoder: `frame_at` calls `decode_video_frame(&path, frame_index)`
/// and `frame_count` returns the demuxer-reported frame count.
///
/// Implementations must be `Send + Sync` because the engine offloads
/// prefetch work to a background thread pool.
pub trait FrameSource: Send + Sync {
    /// Decode the frame at `frame_index` from `asset_uri`.
    fn frame_at(&self, asset_uri: &str, frame_index: u64) -> Result<Frame>;
    /// Total frame count for the asset, if known. Used to clamp prefetch.
    fn frame_count(&self, asset_uri: &str) -> Result<Option<u64>>;
}

/// Snapshot of cache hit/miss telemetry. Cheap to fetch — just three
/// atomic loads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of `frame()` calls that were satisfied from the cache.
    pub hits: u64,
    /// Number of `frame()` calls that fell through to the underlying
    /// [`FrameSource`].
    pub misses: u64,
    /// Live entry count in the backing cache at observation time.
    pub entries: usize,
}

/// Default look-ahead distance for [`PlaybackEngine::scrub`]: enough to
/// keep a 24-fps display loop a few frames ahead of where the user just
/// landed.
const DEFAULT_PREFETCH_WINDOW: usize = 4;

/// Frame retrieval engine.
///
/// `PlaybackEngine` is the public driver for video playback / scrubbing.
/// On `frame()`, it first checks the cache; on miss it asks the source
/// and caches the result. `scrub()` does the same and *additionally*
/// kicks a fire-and-forget prefetch of the next `prefetch_window`
/// frames into the cache so the next display tick is fast.
pub struct PlaybackEngine<S: FrameSource> {
    source: Arc<S>,
    cache: Arc<LruFrameCache>,
    /// How many frames ahead of the current target to opportunistically
    /// prefetch on `scrub`.
    pub prefetch_window: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl<S: FrameSource + 'static> PlaybackEngine<S> {
    /// Build an engine around `source` with the given shared cache.
    /// The default look-ahead window is 4 frames.
    pub fn new(source: S, cache: Arc<LruFrameCache>) -> Self {
        Self {
            source: Arc::new(source),
            cache,
            prefetch_window: DEFAULT_PREFETCH_WINDOW,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Override the prefetch look-ahead window.
    pub fn with_prefetch_window(mut self, window: usize) -> Self {
        self.prefetch_window = window;
        self
    }

    /// Borrow the underlying source. Useful when the source carries
    /// configuration callers need (e.g. a path or decoder handle).
    pub fn source(&self) -> &S { &self.source }

    /// Borrow the shared cache.
    pub fn cache(&self) -> &Arc<LruFrameCache> { &self.cache }

    /// Fetch a frame at `frame_index`. Hits the cache first; on miss
    /// asks the source, then caches the result before returning.
    pub fn frame(&self, asset_uri: &str, frame_index: u64) -> Result<Frame> {
        let key = FrameKey::new(asset_uri.to_string(), frame_index);
        if let Some(frame) = self.cache.get(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            trace!(asset_uri, frame_index, "cache hit");
            return Ok(frame);
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        trace!(asset_uri, frame_index, "cache miss");
        let frame = self.source.frame_at(asset_uri, frame_index)?;
        self.cache.put(key, frame.clone());
        Ok(frame)
    }

    /// Fetch a frame at `target_index` and kick a fire-and-forget
    /// prefetch of the next `prefetch_window` frames into the cache.
    ///
    /// The prefetch runs on the global rayon thread pool — failures are
    /// swallowed (we just won't have those frames cached). The `scrub`
    /// call itself returns synchronously with the requested frame.
    pub fn scrub(&self, asset_uri: &str, target_index: u64) -> Result<Frame> {
        let frame = self.frame(asset_uri, target_index)?;

        let window = self.prefetch_window;
        if window == 0 {
            return Ok(frame);
        }

        // Optional upper clamp so we don't ask for frames past EOF.
        let upper = self.source.frame_count(asset_uri).ok().flatten();

        // Capture handles for the background tasks.
        let asset_uri_owned = asset_uri.to_string();
        let cache = Arc::clone(&self.cache);
        let source = Arc::clone(&self.source);

        rayon::spawn(move || {
            for offset in 1..=window as u64 {
                let idx = target_index.saturating_add(offset);
                if let Some(max) = upper {
                    if idx >= max {
                        break;
                    }
                }
                let key = FrameKey::new(asset_uri_owned.clone(), idx);
                if cache.get(&key).is_some() {
                    continue;
                }
                match source.frame_at(&asset_uri_owned, idx) {
                    Ok(f) => cache.put(key, f),
                    Err(e) => {
                        trace!(error = %e, idx, "prefetch failed");
                        // Don't abort the whole prefetch on one bad frame.
                        continue;
                    }
                }
            }
        });

        Ok(frame)
    }

    /// Snapshot the hit/miss counters and current cache size.
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            entries: self.cache.len(),
        }
    }
}

impl<S: FrameSource> std::fmt::Debug for PlaybackEngine<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlaybackEngine")
            .field("prefetch_window", &self.prefetch_window)
            .field("hits", &self.hits.load(Ordering::Relaxed))
            .field("misses", &self.misses.load(Ordering::Relaxed))
            .field("cache_len", &self.cache.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, PixelLayout};
    use std::sync::atomic::AtomicU64 as Counter;
    use std::thread;
    use std::time::{Duration, Instant};

    /// Test source that fabricates a [`Frame`] for any index and counts
    /// how many times it has been hit. Optional `total` lets the test
    /// expose a finite frame count.
    struct MockSource {
        calls: Counter,
        total: Option<u64>,
    }

    impl MockSource {
        fn new(total: Option<u64>) -> Self {
            Self { calls: Counter::new(0), total }
        }

        fn call_count(&self) -> u64 { self.calls.load(Ordering::Relaxed) }
    }

    impl FrameSource for MockSource {
        fn frame_at(&self, _asset_uri: &str, _frame_index: u64) -> Result<Frame> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(Frame::zeros(2, 2, PixelLayout::Rgba8, ColorSpace::SRgb))
        }

        fn frame_count(&self, _asset_uri: &str) -> Result<Option<u64>> {
            Ok(self.total)
        }
    }

    /// Wait for a predicate to become true, with a timeout. Returns
    /// `true` if the predicate held within the deadline.
    fn wait_until(deadline: Duration, mut pred: impl FnMut() -> bool) -> bool {
        let start = Instant::now();
        while start.elapsed() < deadline {
            if pred() {
                return true;
            }
            thread::sleep(Duration::from_millis(5));
        }
        pred()
    }

    #[test]
    fn second_frame_call_is_cache_hit() {
        let cache = Arc::new(LruFrameCache::new_by_count(8));
        let engine = PlaybackEngine::new(MockSource::new(None), Arc::clone(&cache));

        let _ = engine.frame("video://test", 0).unwrap();
        let _ = engine.frame("video://test", 0).unwrap();

        let stats = engine.cache_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.entries, 1);
        assert_eq!(engine.source().call_count(), 1, "source consulted exactly once");
    }

    #[test]
    fn distinct_indices_each_miss_then_hit() {
        let cache = Arc::new(LruFrameCache::new_by_count(8));
        let engine = PlaybackEngine::new(MockSource::new(None), Arc::clone(&cache));

        for i in 0..4u64 {
            let _ = engine.frame("v", i).unwrap();
        }
        for i in 0..4u64 {
            let _ = engine.frame("v", i).unwrap();
        }

        let stats = engine.cache_stats();
        assert_eq!(stats.misses, 4);
        assert_eq!(stats.hits, 4);
        assert_eq!(stats.entries, 4);
    }

    #[test]
    fn scrub_prefetches_next_window() {
        let cache = Arc::new(LruFrameCache::new_by_count(32));
        // No upper clamp.
        let engine =
            PlaybackEngine::new(MockSource::new(None), Arc::clone(&cache))
                .with_prefetch_window(3);

        let _ = engine.scrub("v", 10).unwrap();

        // Wait for the prefetch tasks to populate frames 11..=13.
        let ok = wait_until(Duration::from_secs(2), || {
            cache.get(&FrameKey::new("v", 11)).is_some()
                && cache.get(&FrameKey::new("v", 12)).is_some()
                && cache.get(&FrameKey::new("v", 13)).is_some()
        });
        assert!(ok, "prefetch never populated cache");
    }

    #[test]
    fn scrub_respects_frame_count_clamp() {
        let cache = Arc::new(LruFrameCache::new_by_count(32));
        // Total of 12 frames — scrubbing to 10 with a window of 4 should
        // only prefetch 11 (12 is past EOF since indices are 0..total).
        let engine =
            PlaybackEngine::new(MockSource::new(Some(12)), Arc::clone(&cache))
                .with_prefetch_window(4);

        let _ = engine.scrub("v", 10).unwrap();

        let ok = wait_until(Duration::from_secs(2), || {
            cache.get(&FrameKey::new("v", 11)).is_some()
        });
        assert!(ok, "frame 11 should have been prefetched");
        // Frame 12 is at-or-past EOF; ensure prefetch did NOT fetch it.
        // Wait a small grace window for any in-flight prefetches.
        thread::sleep(Duration::from_millis(50));
        assert!(
            cache.get(&FrameKey::new("v", 12)).is_none(),
            "frame 12 should not be prefetched (clamped at total=12)"
        );
        assert!(cache.get(&FrameKey::new("v", 13)).is_none());
    }

    #[test]
    fn concurrent_reads_produce_correct_hit_miss_counts() {
        let cache = Arc::new(LruFrameCache::new_by_count(64));
        // Disable prefetch so we have a clean accounting picture.
        let engine = Arc::new(
            PlaybackEngine::new(MockSource::new(None), Arc::clone(&cache))
                .with_prefetch_window(0),
        );

        // Pre-warm: each thread will request frames 0..N. The very
        // first request per index is a miss, every subsequent one is a
        // hit. With T threads × N indices, total calls = T*N, misses
        // = N (one per unique index), hits = T*N - N.
        let threads = 4;
        let indices: u64 = 8;

        let mut handles = Vec::new();
        for _ in 0..threads {
            let engine = Arc::clone(&engine);
            handles.push(thread::spawn(move || {
                for i in 0..indices {
                    engine.frame("v", i).expect("frame fetch");
                }
            }));
        }
        for h in handles {
            h.join().expect("thread panicked");
        }

        let stats = engine.cache_stats();
        let total = (threads as u64) * indices;
        assert_eq!(stats.hits + stats.misses, total);
        // Every unique frame index is missed at least once. Because
        // multiple threads can race on the very first access to the
        // same key, misses can be slightly > N — but never more than
        // total, and never zero.
        assert!(stats.misses >= indices, "misses={} indices={}", stats.misses, indices);
        assert!(
            stats.misses <= total,
            "misses={} total={}",
            stats.misses,
            total
        );
        assert_eq!(stats.entries as u64, indices);
        // Source should have been called once per *miss*.
        assert_eq!(engine.source().call_count(), stats.misses);
    }
}
