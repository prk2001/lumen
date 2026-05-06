//! Project file: `.lumenproj`.
//!
//! Schema-versioned JSON. The current version is `lumenproj/v1`.
//! Round-trips through `serde_json`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::asset::Asset;
use crate::error::{Error, Result};
use crate::graph::Graph;

/// Schema identifier baked into project files.
pub const SCHEMA: &str = "lumenproj/v1";

/// One entry in the immutable history log. Phase 1 keeps this minimal;
/// later phases extend it with structured diffs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// Wall-clock time the entry was recorded.
    #[serde(with = "time::serde::rfc3339")]
    pub at: OffsetDateTime,
    /// Free-form description.
    pub note: String,
}

/// A user-saved bundle of parameter values. Phase 1 stores only the JSON
/// payload; richer modelling lands with `lumen-workflow`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preset {
    pub id: Uuid,
    pub name: String,
    pub effect_id: String,
    pub params: serde_json::Value,
}

/// Top-level project document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    /// Schema identifier — checked on load.
    pub schema: String,
    /// Stable project id (ULID-like UUID v7).
    pub id: Uuid,
    #[serde(with = "time::serde::rfc3339")]
    pub created: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub modified: OffsetDateTime,
    /// All assets referenced by the graph.
    pub assets: Vec<Asset>,
    /// Pipeline graph.
    pub graph: Graph,
    /// Append-only history log.
    #[serde(default)]
    pub history: Vec<HistoryEntry>,
    /// User-saved presets.
    #[serde(default)]
    pub presets: Vec<Preset>,
    /// Pinned model hashes — `model_id -> "blake3:…"`. Reproducibility
    /// only holds if every model in the graph is listed here.
    #[serde(default)]
    pub models: BTreeMap<String, String>,
}

impl Project {
    /// Empty project, ready to be populated.
    pub fn empty() -> Self {
        let now = OffsetDateTime::now_utc();
        Self {
            schema: SCHEMA.to_string(),
            id: Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)),
            created: now,
            modified: now,
            assets: Vec::new(),
            graph: Graph::default(),
            history: Vec::new(),
            presets: Vec::new(),
            models: BTreeMap::new(),
        }
    }

    /// Append a history entry and bump `modified`.
    pub fn record(&mut self, note: impl Into<String>) {
        let at = OffsetDateTime::now_utc();
        self.history.push(HistoryEntry { at, note: note.into() });
        self.modified = at;
    }

    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize from JSON, validating the schema field.
    pub fn from_json(s: &str) -> Result<Self> {
        let p: Project = serde_json::from_str(s)?;
        if p.schema != SCHEMA {
            return Err(Error::SchemaMismatch {
                expected: SCHEMA.to_string(),
                found: p.schema,
            });
        }
        Ok(p)
    }

    /// Save to disk atomically (write to `.tmp`, then rename).
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let tmp = path.with_extension("lumenproj.tmp");
        std::fs::write(&tmp, self.to_json()?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load from disk.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Self::from_json(&s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_empty() {
        let p = Project::empty();
        let s = p.to_json().unwrap();
        let p2 = Project::from_json(&s).unwrap();
        assert_eq!(p2.schema, SCHEMA);
        assert_eq!(p2.id, p.id);
    }

    #[test]
    fn schema_mismatch_caught() {
        let bad = serde_json::json!({
            "schema": "lumenproj/v999",
            "id": "00000000-0000-0000-0000-000000000000",
            "created": "2026-01-01T00:00:00Z",
            "modified": "2026-01-01T00:00:00Z",
            "assets": [],
            "graph": { "nodes": {}, "sinks": [] }
        });
        let r = Project::from_json(&bad.to_string());
        assert!(matches!(r, Err(Error::SchemaMismatch { .. })));
    }

    #[test]
    fn record_appends_history() {
        let mut p = Project::empty();
        let before = p.modified;
        std::thread::sleep(std::time::Duration::from_millis(2));
        p.record("opened project");
        assert_eq!(p.history.len(), 1);
        assert_eq!(p.history[0].note, "opened project");
        assert!(p.modified >= before);
    }
}
