//! # lumen-workflow
//!
//! Non-destructive editing primitives: parameter and graph diffs, plus a
//! preset store. These are the building blocks for higher-level history,
//! branches, and merge tooling that follow in later phases.
//!
//! ## Overview
//!
//! - [`ParamDiff`] / [`diff_params`] / [`apply_param_diff`] — compute and
//!   apply minimal change sets between two [`ParamValues`].
//! - [`GraphDiff`] / [`diff_graphs`] / [`apply_graph_diff`] — compute and
//!   apply node/edge deltas between two [`Graph`]s.
//! - [`PresetStore`] — in-memory, JSON-serializable bag of named
//!   [`Preset`]s. The underlying preset type is the
//!   [`lumen_core::Project::Preset`](lumen_core::project::Preset) struct,
//!   re-exported here as [`Preset`].
//!
//! All public diff structures use [`BTreeMap`] / [`BTreeSet`] so iteration
//! order — and therefore JSON output — is deterministic.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, rust_2018_idioms)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use lumen_core::error::{Error, Result};
use lumen_core::graph::{Graph, Node, NodeId};
use lumen_core::params::{ParamValue, ParamValues};
use lumen_core::project::Preset;

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

// -----------------------------------------------------------------------
// Parameter diffs
// -----------------------------------------------------------------------

/// Minimal change set between two [`ParamValues`] bundles.
///
/// Applying the diff to the `a` side produces the `b` side. Iteration is
/// deterministic thanks to the [`BTreeMap`] / [`BTreeSet`] backing.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ParamDiff {
    /// Keys present in `b` but not in `a`, with their `b` values.
    pub additions: BTreeMap<String, ParamValue>,
    /// Keys present in `a` but not in `b`.
    pub removals: BTreeSet<String>,
    /// Keys present in both sides whose values differ; carries the `b` value.
    pub changes: BTreeMap<String, ParamValue>,
}

impl ParamDiff {
    /// `true` when the diff would produce no changes if applied.
    pub fn is_empty(&self) -> bool {
        self.additions.is_empty() && self.removals.is_empty() && self.changes.is_empty()
    }
}

/// `ParamValues` doesn't expose an iterator publicly, so we round-trip via
/// JSON to peek at the internal map. The shape of `ParamValues` is a
/// `{ "values": { ... } }` object thanks to its `#[derive(Serialize)]`.
fn as_map(p: &ParamValues) -> BTreeMap<String, ParamValue> {
    let value = serde_json::to_value(p).expect("ParamValues serializes infallibly");
    let inner = value
        .get("values")
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
    serde_json::from_value(inner).unwrap_or_default()
}

/// Rebuild a [`ParamValues`] from a flat key/value map.
fn from_map(map: BTreeMap<String, ParamValue>) -> ParamValues {
    let mut pv = ParamValues::new();
    for (k, v) in map {
        pv.insert(k, v);
    }
    pv
}

/// Compute the minimal diff such that `apply_param_diff(&mut a, &diff)`
/// produces `b`.
pub fn diff_params(a: &ParamValues, b: &ParamValues) -> ParamDiff {
    let am = as_map(a);
    let bm = as_map(b);
    let mut diff = ParamDiff::default();

    for (k, bv) in &bm {
        match am.get(k) {
            None => {
                diff.additions.insert(k.clone(), bv.clone());
            }
            Some(av) if av != bv => {
                diff.changes.insert(k.clone(), bv.clone());
            }
            Some(_) => {}
        }
    }
    for k in am.keys() {
        if !bm.contains_key(k) {
            diff.removals.insert(k.clone());
        }
    }
    diff
}

/// Apply `diff` to `target` in place.
///
/// Removals for missing keys are silently ignored — diff application is
/// idempotent in that direction.
pub fn apply_param_diff(target: &mut ParamValues, diff: &ParamDiff) {
    let mut map = as_map(target);
    for k in &diff.removals {
        map.remove(k);
    }
    for (k, v) in &diff.additions {
        map.insert(k.clone(), v.clone());
    }
    for (k, v) in &diff.changes {
        map.insert(k.clone(), v.clone());
    }
    *target = from_map(map);
}

// -----------------------------------------------------------------------
// Graph diffs
// -----------------------------------------------------------------------

/// One directed edge in a graph (from `src` into `dst`'s input list).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EdgeRef {
    /// Upstream node id.
    pub src: NodeId,
    /// Downstream node id.
    pub dst: NodeId,
}

/// Delta between two [`Graph`]s.
///
/// Nodes added on the `b` side carry their full payload. Nodes removed on
/// the `b` side appear as id-only entries. Nodes whose effect/label/inputs
/// match but whose params differ are emitted as a `ParamDiff` keyed by
/// node id under [`GraphDiff::node_changed`].
///
/// The struct intentionally treats a node whose `effect_id`, `label`, or
/// `inputs` changed as both a removal and an addition; this keeps the
/// applier simple and still round-trips losslessly.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphDiff {
    /// Nodes that exist in `b` but not in `a`.
    pub node_added: BTreeMap<NodeId, Node>,
    /// Node ids that exist in `a` but not in `b`.
    pub node_removed: BTreeSet<NodeId>,
    /// Per-node parameter diffs for nodes that exist on both sides.
    pub node_changed: BTreeMap<NodeId, ParamDiff>,
    /// Sink edges (or input edges) added in `b`.
    pub edge_added: BTreeSet<EdgeRef>,
    /// Sink/input edges removed in `b`.
    pub edge_removed: BTreeSet<EdgeRef>,
}

impl GraphDiff {
    /// `true` when the diff would produce no changes if applied.
    pub fn is_empty(&self) -> bool {
        self.node_added.is_empty()
            && self.node_removed.is_empty()
            && self.node_changed.is_empty()
            && self.edge_added.is_empty()
            && self.edge_removed.is_empty()
    }
}

/// Collect all edges of a graph as `(src, dst)` pairs.
fn collect_edges(g: &Graph) -> BTreeSet<EdgeRef> {
    let mut out = BTreeSet::new();
    for (dst, node) in &g.nodes {
        for src in &node.inputs {
            out.insert(EdgeRef { src: *src, dst: *dst });
        }
    }
    out
}

/// Compute the delta between two graphs.
///
/// The diff is computed by node id. A node whose `effect_id`, `label`, or
/// `inputs` differ between sides is reported as removal-plus-addition; a
/// node that differs only in parameter values appears in
/// [`GraphDiff::node_changed`].
pub fn diff_graphs(a: &Graph, b: &Graph) -> GraphDiff {
    let mut diff = GraphDiff::default();

    for (id, bn) in &b.nodes {
        match a.nodes.get(id) {
            None => {
                diff.node_added.insert(*id, bn.clone());
            }
            Some(an) => {
                if an.effect_id != bn.effect_id
                    || an.label != bn.label
                    || an.inputs != bn.inputs
                {
                    diff.node_removed.insert(*id);
                    diff.node_added.insert(*id, bn.clone());
                } else {
                    let pd = diff_params(&an.params, &bn.params);
                    if !pd.is_empty() {
                        diff.node_changed.insert(*id, pd);
                    }
                }
            }
        }
    }
    for id in a.nodes.keys() {
        if !b.nodes.contains_key(id) {
            diff.node_removed.insert(*id);
        }
    }

    let ae = collect_edges(a);
    let be = collect_edges(b);
    for e in &be {
        if !ae.contains(e) {
            diff.edge_added.insert(*e);
        }
    }
    for e in &ae {
        if !be.contains(e) {
            diff.edge_removed.insert(*e);
        }
    }

    diff
}

/// Apply `diff` to `target` in place.
///
/// Returns [`Error::Graph`] if a removal references a node id that isn't
/// in the target, or if an edge_added/edge_removed names a missing node.
/// `node_changed` entries for nodes that aren't in the target are also
/// reported as graph errors.
pub fn apply_graph_diff(target: &mut Graph, diff: &GraphDiff) -> Result<()> {
    // 1. Validate up-front so a partial application can't leave a
    //    half-mutated graph behind.
    for id in &diff.node_removed {
        if !target.nodes.contains_key(id) {
            return Err(Error::Graph(format!(
                "cannot remove missing node {id}"
            )));
        }
    }
    for id in diff.node_changed.keys() {
        if !target.nodes.contains_key(id) {
            return Err(Error::Graph(format!(
                "cannot change params on missing node {id}"
            )));
        }
    }
    for e in diff.edge_removed.iter().chain(diff.edge_added.iter()) {
        // After applying node_added/node_removed both endpoints must
        // exist. Tolerate references that are satisfied by node_added.
        let src_present = target.nodes.contains_key(&e.src)
            || diff.node_added.contains_key(&e.src);
        let dst_present = target.nodes.contains_key(&e.dst)
            || diff.node_added.contains_key(&e.dst);
        if !src_present {
            return Err(Error::Graph(format!(
                "edge references missing node {}",
                e.src
            )));
        }
        if !dst_present {
            return Err(Error::Graph(format!(
                "edge references missing node {}",
                e.dst
            )));
        }
    }

    // 2. Mutations.
    for id in &diff.node_removed {
        target.nodes.remove(id);
        target.sinks.retain(|s| s != id);
        // Strip dangling input refs from the remaining nodes.
        for n in target.nodes.values_mut() {
            n.inputs.retain(|i| i != id);
        }
    }
    for (id, node) in &diff.node_added {
        target.nodes.insert(*id, node.clone());
    }
    for (id, pd) in &diff.node_changed {
        if let Some(n) = target.nodes.get_mut(id) {
            apply_param_diff(&mut n.params, pd);
        }
    }
    for e in &diff.edge_removed {
        if let Some(dst) = target.nodes.get_mut(&e.dst) {
            dst.inputs.retain(|i| i != &e.src);
        }
    }
    for e in &diff.edge_added {
        if let Some(dst) = target.nodes.get_mut(&e.dst) {
            if !dst.inputs.contains(&e.src) {
                dst.inputs.push(e.src);
            }
        }
    }

    Ok(())
}

// -----------------------------------------------------------------------
// Preset store
// -----------------------------------------------------------------------

/// Stable identifier for a stored preset.
pub type PresetId = Uuid;

/// In-memory, JSON-serializable bag of named effect presets.
///
/// Backed by [`lumen_core::project::Preset`] so a preset can be lifted
/// directly into [`lumen_core::Project::presets`] without conversion.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PresetStore {
    /// Ordered map of presets keyed by id.
    presets: BTreeMap<PresetId, Preset>,
}

impl PresetStore {
    /// Construct an empty store.
    pub fn new() -> Self { Self::default() }

    /// Save a preset under a freshly minted [`PresetId`] and return that id.
    ///
    /// `params` is taken as a [`ParamValues`] and serialized to JSON to
    /// match the [`Preset::params`] field shape.
    pub fn save_preset(
        &mut self,
        name: impl Into<String>,
        effect_id: impl Into<String>,
        params: &ParamValues,
    ) -> PresetId {
        let id = Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext));
        let preset = Preset {
            id,
            name: name.into(),
            effect_id: effect_id.into(),
            params: serde_json::to_value(params)
                .expect("ParamValues serializes infallibly"),
        };
        self.presets.insert(id, preset);
        id
    }

    /// Insert a preset constructed externally. Returns the previous value
    /// at this id, if any.
    pub fn insert_preset(&mut self, preset: Preset) -> Option<Preset> {
        self.presets.insert(preset.id, preset)
    }

    /// Look up a preset by id.
    pub fn load_preset(&self, id: PresetId) -> Option<&Preset> {
        self.presets.get(&id)
    }

    /// Iterate all stored presets in deterministic id order.
    pub fn list_presets(&self) -> Vec<&Preset> {
        self.presets.values().collect()
    }

    /// Number of stored presets.
    pub fn len(&self) -> usize { self.presets.len() }

    /// `true` when the store has no presets.
    pub fn is_empty(&self) -> bool { self.presets.is_empty() }

    /// Remove a preset by id, returning it if found.
    pub fn remove_preset(&mut self, id: PresetId) -> Option<Preset> {
        self.presets.remove(&id)
    }

    /// Serialize the store as pretty JSON to `path`. Writes via a
    /// temporary file and renames into place to avoid leaving a
    /// half-written file on crash.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let tmp = path.with_extension("presets.tmp");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load a store from a JSON file written by [`save_to_file`].
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&s)?)
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::graph::{Node, NodeId};

    fn pv(entries: &[(&str, ParamValue)]) -> ParamValues {
        let mut p = ParamValues::new();
        for (k, v) in entries {
            p.insert(*k, v.clone());
        }
        p
    }

    #[test]
    fn empty_diff_between_identical_params() {
        let a = pv(&[
            ("strength", ParamValue::Float(0.5)),
            ("preset", ParamValue::String("auto".into())),
        ]);
        let b = a.clone();
        let d = diff_params(&a, &b);
        assert!(d.is_empty(), "expected empty diff, got {d:?}");
    }

    #[test]
    fn apply_param_diff_round_trip() {
        let a = pv(&[
            ("strength", ParamValue::Float(0.5)),
            ("legacy", ParamValue::Bool(true)),
        ]);
        let b = pv(&[
            ("strength", ParamValue::Float(0.9)), // changed
            ("radius", ParamValue::Int(3)),       // added
                                                  // legacy removed
        ]);
        let d = diff_params(&a, &b);
        assert_eq!(d.changes.get("strength"), Some(&ParamValue::Float(0.9)));
        assert_eq!(d.additions.get("radius"), Some(&ParamValue::Int(3)));
        assert!(d.removals.contains("legacy"));

        let mut t = a.clone();
        apply_param_diff(&mut t, &d);
        assert_eq!(t.get_float("strength"), Some(0.9));
        assert_eq!(t.get_int("radius"), Some(3));
        assert!(t.get("legacy").is_none());
    }

    #[test]
    fn graph_node_add_and_remove_round_trip() {
        let mut a = Graph::new();
        let n1 = a.insert(Node::new("io.source", "src"));
        let _n2 = a.insert(Node::new("io.export", "out").with_input(n1));

        let mut b = a.clone();
        // Drop the export sink, add a brightness node downstream of n1.
        let to_remove: Vec<NodeId> = b
            .nodes
            .values()
            .filter(|n| n.effect_id == "io.export")
            .map(|n| n.id)
            .collect();
        for id in to_remove {
            b.nodes.remove(&id);
        }
        let bright = b.insert(Node::new("fx.brightness", "bright").with_input(n1));
        b.add_sink(bright);

        let d = diff_graphs(&a, &b);
        assert!(!d.node_added.is_empty(), "expected at least one added node");
        assert!(!d.node_removed.is_empty(), "expected at least one removed node");

        let mut t = a.clone();
        apply_graph_diff(&mut t, &d).unwrap();

        // Compare nodes by id and effect_id (sinks aren't part of GraphDiff).
        let t_ids: BTreeSet<NodeId> = t.nodes.keys().copied().collect();
        let b_ids: BTreeSet<NodeId> = b.nodes.keys().copied().collect();
        assert_eq!(t_ids, b_ids);
        for (id, n) in &b.nodes {
            assert_eq!(t.nodes.get(id).unwrap().effect_id, n.effect_id);
            assert_eq!(t.nodes.get(id).unwrap().inputs, n.inputs);
        }
    }

    #[test]
    fn apply_graph_diff_missing_node_errors() {
        let mut g = Graph::new();
        let _n = g.insert(Node::new("io.source", "src"));

        let mut diff = GraphDiff::default();
        let bogus = NodeId::new();
        diff.node_removed.insert(bogus);

        let r = apply_graph_diff(&mut g, &diff);
        match r {
            Err(Error::Graph(msg)) => assert!(msg.contains("missing node")),
            other => panic!("expected Error::Graph, got {other:?}"),
        }
    }

    #[test]
    fn preset_store_round_trips_through_disk() {
        let mut store = PresetStore::new();
        let p = pv(&[
            ("strength", ParamValue::Float(0.7)),
            ("mode", ParamValue::String("smooth".into())),
        ]);
        let id = store.save_preset("Soft Glow", "fx.brightness", &p);
        assert_eq!(store.len(), 1);
        assert!(store.load_preset(id).is_some());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("presets.json");
        store.save_to_file(&path).unwrap();

        let loaded = PresetStore::load_from_file(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        let lp = loaded.load_preset(id).expect("preset survives round-trip");
        assert_eq!(lp.name, "Soft Glow");
        assert_eq!(lp.effect_id, "fx.brightness");

        // params field is the JSON form of ParamValues — sanity check it
        // contains the expected key.
        let params_json = lp.params.to_string();
        assert!(params_json.contains("strength"));
        assert!(params_json.contains("mode"));
    }

    #[test]
    fn graph_diff_param_changes_only() {
        let mut a = Graph::new();
        let mut params_a = ParamValues::new();
        params_a.insert("strength", ParamValue::Float(0.5));
        let id = a.insert(
            Node::new("fx.brightness", "bright").with_params(params_a),
        );

        let mut b = a.clone();
        let mut params_b = ParamValues::new();
        params_b.insert("strength", ParamValue::Float(0.9));
        b.nodes.get_mut(&id).unwrap().params = params_b;

        let d = diff_graphs(&a, &b);
        assert!(d.node_added.is_empty());
        assert!(d.node_removed.is_empty());
        assert_eq!(d.node_changed.len(), 1);

        let mut t = a.clone();
        apply_graph_diff(&mut t, &d).unwrap();
        assert_eq!(t.nodes.get(&id).unwrap().params.get_float("strength"), Some(0.9));
    }
}
