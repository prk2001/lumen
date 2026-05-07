//! Thread-safe LRU frame cache.
//!
//! Backed by a hand-rolled doubly-linked list so eviction and
//! most-recently-used promotion are both O(1). The list is stored in a
//! `Vec<Slot>` arena with prev/next indices, and a `HashMap<FrameKey,
//! usize>` indexes live entries to their slot.
//!
//! Two budgeting modes are supported:
//!
//! * **By entry count** — cap on number of frames in the cache.
//! * **By byte budget** — cap on the sum of `frame_bytes(...)` across
//!   live entries. Frames are evicted from the LRU end until the
//!   inserted frame fits.
//!
//! Both modes share the same arena/list code; only [`Inner::evict_until_fits`]
//! differs. A single mutex guards the whole structure — the operations
//! are short and the protected data is in-process, so the simplicity of
//! `parking_lot::Mutex` outweighs the extra concurrency a `RwLock` could
//! provide for read-heavy workloads.

use std::collections::HashMap;

use lumen_core::{Frame, FrameCache, PixelData};
use parking_lot::Mutex;

/// Cache key combining the asset URI (e.g. file path or content hash)
/// and an absolute frame index within that asset.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FrameKey {
    /// Stable identifier for the asset — file path, content hash, or
    /// any other URI the caller chooses.
    pub asset_uri: String,
    /// Zero-based absolute frame index within the asset.
    pub frame_index: u64,
}

impl FrameKey {
    /// Convenience constructor.
    pub fn new(asset_uri: impl Into<String>, frame_index: u64) -> Self {
        Self { asset_uri: asset_uri.into(), frame_index }
    }
}

/// Approximate the heap footprint of a [`Frame`]. Returns the size of
/// the pixel buffer in bytes — metadata (color space, PTS) is small and
/// constant so we don't bother accounting for it.
fn frame_bytes(frame: &Frame) -> usize {
    match &frame.data {
        PixelData::Rgba8(v) => v.len(),
        PixelData::Rgba16(v) => v.len() * std::mem::size_of::<u16>(),
        PixelData::RgbaF32(v) => v.len() * std::mem::size_of::<f32>(),
    }
}

/// Cache budgeting strategy.
#[derive(Debug, Clone, Copy)]
enum Budget {
    /// Hold at most `cap` entries.
    ByCount { cap: usize },
    /// Hold up to `max` bytes across all entries.
    ByBytes { max: usize },
}

/// One arena slot. `None` means free; a `Some` slot is part of the LRU
/// doubly-linked list.
#[derive(Debug)]
struct Slot {
    entry: Option<Entry>,
    prev: Option<usize>,
    next: Option<usize>,
}

#[derive(Debug)]
struct Entry {
    key: FrameKey,
    frame: Frame,
    bytes: usize,
}

#[derive(Debug)]
struct Inner {
    budget: Budget,
    /// Slot arena. Free slots have `entry == None`.
    slots: Vec<Slot>,
    /// Stack of free slot indices, popped on insert and pushed on remove.
    free: Vec<usize>,
    /// Live entries indexed by key.
    index: HashMap<FrameKey, usize>,
    /// Most-recently-used (head) slot index.
    head: Option<usize>,
    /// Least-recently-used (tail) slot index — evicted first.
    tail: Option<usize>,
    /// Sum of `bytes` across live entries.
    total_bytes: usize,
}

impl Inner {
    fn new(budget: Budget) -> Self {
        Self {
            budget,
            slots: Vec::new(),
            free: Vec::new(),
            index: HashMap::new(),
            head: None,
            tail: None,
            total_bytes: 0,
        }
    }

    fn allocate_slot(&mut self) -> usize {
        if let Some(idx) = self.free.pop() {
            self.slots[idx].prev = None;
            self.slots[idx].next = None;
            self.slots[idx].entry = None;
            idx
        } else {
            self.slots.push(Slot { entry: None, prev: None, next: None });
            self.slots.len() - 1
        }
    }

    /// Detach the slot at `idx` from the linked list (does not free it).
    fn unlink(&mut self, idx: usize) {
        let prev = self.slots[idx].prev;
        let next = self.slots[idx].next;
        match prev {
            Some(p) => self.slots[p].next = next,
            None => self.head = next,
        }
        match next {
            Some(n) => self.slots[n].prev = prev,
            None => self.tail = prev,
        }
        self.slots[idx].prev = None;
        self.slots[idx].next = None;
    }

    /// Insert `idx` at the head of the linked list.
    fn link_at_head(&mut self, idx: usize) {
        self.slots[idx].prev = None;
        self.slots[idx].next = self.head;
        if let Some(h) = self.head {
            self.slots[h].prev = Some(idx);
        }
        self.head = Some(idx);
        if self.tail.is_none() {
            self.tail = Some(idx);
        }
    }

    fn promote(&mut self, idx: usize) {
        if self.head == Some(idx) {
            return;
        }
        self.unlink(idx);
        self.link_at_head(idx);
    }

    /// Pop the LRU (tail) entry. Returns the removed key+bytes if any.
    fn evict_one(&mut self) -> Option<(FrameKey, usize)> {
        let idx = self.tail?;
        self.unlink(idx);
        let entry = self.slots[idx].entry.take()?;
        self.index.remove(&entry.key);
        self.total_bytes = self.total_bytes.saturating_sub(entry.bytes);
        self.free.push(idx);
        Some((entry.key, entry.bytes))
    }

    fn get(&mut self, key: &FrameKey) -> Option<Frame> {
        let idx = *self.index.get(key)?;
        self.promote(idx);
        // Clone the frame (Frame is Clone). Cache values are immutable
        // from the caller's POV.
        self.slots[idx].entry.as_ref().map(|e| e.frame.clone())
    }

    /// Evict from the tail until `incoming_bytes` of additional payload
    /// fits within the budget, or the cache is empty. (For by-count this
    /// is "until len < cap".)
    fn evict_until_fits(&mut self, incoming_bytes: usize) {
        match self.budget {
            Budget::ByCount { cap } => {
                if cap == 0 {
                    return;
                }
                while self.index.len() >= cap {
                    if self.evict_one().is_none() {
                        break;
                    }
                }
            }
            Budget::ByBytes { max } => {
                if max == 0 {
                    return;
                }
                while self.total_bytes + incoming_bytes > max && !self.index.is_empty() {
                    if self.evict_one().is_none() {
                        break;
                    }
                }
            }
        }
    }

    fn put(&mut self, key: FrameKey, frame: Frame) {
        // If a value already exists for this key, drop it first so the
        // new insert is treated as a fresh MRU entry.
        if let Some(&existing_idx) = self.index.get(&key) {
            self.unlink(existing_idx);
            if let Some(entry) = self.slots[existing_idx].entry.take() {
                self.index.remove(&entry.key);
                self.total_bytes = self.total_bytes.saturating_sub(entry.bytes);
            }
            self.free.push(existing_idx);
        }

        let bytes = frame_bytes(&frame);

        // For by-bytes mode: if the single frame exceeds the entire
        // budget, refuse to cache it rather than blowing past the cap.
        if let Budget::ByBytes { max } = self.budget {
            if max == 0 || bytes > max {
                return;
            }
        }
        if let Budget::ByCount { cap } = self.budget {
            if cap == 0 {
                return;
            }
        }

        self.evict_until_fits(bytes);

        let idx = self.allocate_slot();
        self.slots[idx].entry = Some(Entry { key: key.clone(), frame, bytes });
        self.index.insert(key, idx);
        self.total_bytes += bytes;
        self.link_at_head(idx);
    }

    fn len(&self) -> usize { self.index.len() }

    fn clear(&mut self) {
        self.slots.clear();
        self.free.clear();
        self.index.clear();
        self.head = None;
        self.tail = None;
        self.total_bytes = 0;
    }
}

/// Thread-safe LRU cache of decoded frames.
///
/// All operations take `&self` — the underlying mutex is internal so
/// callers can freely share an `Arc<LruFrameCache>` across threads.
#[derive(Debug)]
pub struct LruFrameCache {
    inner: Mutex<Inner>,
}

impl LruFrameCache {
    /// Build a cache that holds at most `capacity` frames. A capacity
    /// of 0 disables caching (every put is a no-op).
    pub fn new_by_count(capacity: usize) -> Self {
        Self { inner: Mutex::new(Inner::new(Budget::ByCount { cap: capacity })) }
    }

    /// Build a cache with a byte-budget cap. Frames are evicted in LRU
    /// order until the new entry fits. Frames larger than the budget
    /// are silently dropped on insert.
    pub fn new_by_bytes(max_bytes: usize) -> Self {
        Self { inner: Mutex::new(Inner::new(Budget::ByBytes { max: max_bytes })) }
    }

    /// Look up a frame by key. A successful get promotes the entry to
    /// the most-recently-used position.
    pub fn get(&self, key: &FrameKey) -> Option<Frame> {
        self.inner.lock().get(key)
    }

    /// Insert a frame under `key`, evicting older entries as needed.
    pub fn put(&self, key: FrameKey, frame: Frame) {
        self.inner.lock().put(key, frame);
    }

    /// Number of live entries.
    pub fn len(&self) -> usize { self.inner.lock().len() }

    /// True if the cache holds zero entries.
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Drop every entry. The cache remains usable.
    pub fn clear(&self) { self.inner.lock().clear(); }

    /// Total byte footprint of cached pixel buffers.
    pub fn total_bytes(&self) -> usize { self.inner.lock().total_bytes }
}

// ---------------------------------------------------------------------------
// `lumen_core::FrameCache` adapter.
//
// The trait keys frames by a content-addressed string (BLAKE3 hex). We
// reuse the same arena by storing those keys with `frame_index = 0` and
// the hash string as the asset URI.
// ---------------------------------------------------------------------------

impl FrameCache for LruFrameCache {
    fn get(&self, key: &str) -> Option<Frame> {
        let key = FrameKey { asset_uri: key.to_string(), frame_index: 0 };
        LruFrameCache::get(self, &key)
    }

    fn put(&self, key: &str, frame: Frame) {
        let key = FrameKey { asset_uri: key.to_string(), frame_index: 0 };
        LruFrameCache::put(self, key, frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{ColorSpace, PixelLayout};

    fn dummy_frame(side: u32) -> Frame {
        Frame::zeros(side, side, PixelLayout::Rgba8, ColorSpace::SRgb)
    }

    fn key(uri: &str, idx: u64) -> FrameKey {
        FrameKey::new(uri, idx)
    }

    #[test]
    fn by_count_evicts_oldest() {
        let cache = LruFrameCache::new_by_count(3);
        for i in 0..3u64 {
            cache.put(key("a", i), dummy_frame(2));
        }
        assert_eq!(cache.len(), 3);

        // Insert a 4th — the oldest (index 0) should be evicted.
        cache.put(key("a", 3), dummy_frame(2));
        assert_eq!(cache.len(), 3);

        assert!(cache.get(&key("a", 0)).is_none(), "frame 0 should have been evicted");
        assert!(cache.get(&key("a", 1)).is_some());
        assert!(cache.get(&key("a", 2)).is_some());
        assert!(cache.get(&key("a", 3)).is_some());
    }

    #[test]
    fn by_count_get_promotes_to_mru() {
        let cache = LruFrameCache::new_by_count(3);
        cache.put(key("a", 0), dummy_frame(2));
        cache.put(key("a", 1), dummy_frame(2));
        cache.put(key("a", 2), dummy_frame(2));

        // Touch frame 0 — that should make it MRU, so frame 1 becomes
        // the next eviction candidate.
        let _ = cache.get(&key("a", 0));
        cache.put(key("a", 3), dummy_frame(2));

        assert!(cache.get(&key("a", 0)).is_some(), "frame 0 was promoted, not evicted");
        assert!(cache.get(&key("a", 1)).is_none(), "frame 1 should be evicted");
        assert!(cache.get(&key("a", 2)).is_some());
        assert!(cache.get(&key("a", 3)).is_some());
    }

    #[test]
    fn by_bytes_respects_budget_with_varying_sizes() {
        // Each 4x4 RGBA8 frame = 64 bytes; each 2x2 = 16 bytes.
        // Set a 100-byte budget. We can hold either one 64+ a couple
        // 16-byte frames, etc. — and should evict the LRU when needed.
        let cache = LruFrameCache::new_by_bytes(100);

        cache.put(key("a", 0), dummy_frame(4)); // 64 bytes
        cache.put(key("a", 1), dummy_frame(2)); // +16 = 80 bytes
        cache.put(key("a", 2), dummy_frame(2)); // +16 = 96 bytes
        assert_eq!(cache.len(), 3);
        assert!(cache.total_bytes() <= 100);

        // Adding another 16-byte frame brings us to 112 — needs to evict.
        cache.put(key("a", 3), dummy_frame(2));
        assert!(cache.total_bytes() <= 100, "budget exceeded: {}", cache.total_bytes());
        // The 64-byte frame at index 0 is the LRU; it should have been
        // dropped to make room.
        assert!(cache.get(&key("a", 0)).is_none());

        // Adding a frame larger than the entire budget is a no-op.
        cache.put(key("a", 999), dummy_frame(8)); // 256 bytes >> 100
        assert!(cache.get(&key("a", 999)).is_none());
    }

    #[test]
    fn put_overwrite_keeps_one_entry() {
        let cache = LruFrameCache::new_by_count(4);
        cache.put(key("a", 0), dummy_frame(2));
        cache.put(key("a", 0), dummy_frame(4)); // same key, replace
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.total_bytes(), 64);
    }

    #[test]
    fn clear_empties_cache() {
        let cache = LruFrameCache::new_by_count(8);
        cache.put(key("a", 0), dummy_frame(2));
        cache.put(key("a", 1), dummy_frame(2));
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.total_bytes(), 0);
    }

    #[test]
    fn lumen_core_framecache_impl_round_trips() {
        let cache = LruFrameCache::new_by_count(4);
        let trait_obj: &dyn FrameCache = &cache;
        trait_obj.put("hash-deadbeef", dummy_frame(2));
        let got = trait_obj.get("hash-deadbeef");
        assert!(got.is_some());
        assert!(trait_obj.get("nope").is_none());
    }
}
