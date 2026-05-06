//! Pipeline DAG.
//!
//! A [`Graph`] is a set of named [`Node`]s connected by edges. Each
//! node references an effect by id and carries its parameter values.
//! The [`Scheduler`] computes a topological order and calls effects in
//! sequence (or in parallel for independent branches — Phase 1 is
//! sequential; parallelism arrives with `lumen-perf` in Phase 4).

use std::collections::{BTreeMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::params::ParamValues;

/// Stable, content-agnostic id for a node within a graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(pub Uuid);

impl NodeId {
    pub fn new() -> Self { Self(Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext))) }
}

impl Default for NodeId {
    fn default() -> Self { Self::new() }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.0.fmt(f) }
}

/// One node in the DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    /// Effect id (e.g. `"lumen-fx-exposure.brightness_contrast"`).
    /// Source nodes use `"lumen-io.source"`; sinks use `"lumen-io.export"`.
    pub effect_id: String,
    /// Display label shown in node-graph UI.
    pub label: String,
    /// Upstream node ids in input order. Source nodes have no inputs.
    pub inputs: Vec<NodeId>,
    /// Parameter values bound to this node.
    pub params: ParamValues,
}

impl Node {
    pub fn new(effect_id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: NodeId::new(),
            effect_id: effect_id.into(),
            label: label.into(),
            inputs: Vec::new(),
            params: ParamValues::new(),
        }
    }

    pub fn with_input(mut self, id: NodeId) -> Self {
        self.inputs.push(id);
        self
    }

    pub fn with_params(mut self, params: ParamValues) -> Self {
        self.params = params;
        self
    }
}

/// A DAG of nodes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Graph {
    /// All nodes keyed by id. Stored in an ordered map for deterministic
    /// JSON output.
    pub nodes: BTreeMap<NodeId, Node>,
    /// Designated output node ids — terminal points the scheduler will
    /// drive toward.
    pub sinks: Vec<NodeId>,
}

impl Graph {
    pub fn new() -> Self { Self::default() }

    pub fn insert(&mut self, node: Node) -> NodeId {
        let id = node.id;
        self.nodes.insert(id, node);
        id
    }

    pub fn add_sink(&mut self, id: NodeId) -> &mut Self {
        if !self.sinks.contains(&id) {
            self.sinks.push(id);
        }
        self
    }

    pub fn get(&self, id: NodeId) -> Option<&Node> { self.nodes.get(&id) }

    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    /// Validate the graph: every input id must exist; no cycles.
    pub fn validate(&self) -> Result<()> {
        // Check all inputs reference existing nodes.
        for n in self.nodes.values() {
            for in_id in &n.inputs {
                if !self.nodes.contains_key(in_id) {
                    return Err(Error::Graph(format!(
                        "node {} references missing input {in_id}",
                        n.id
                    )));
                }
            }
        }
        // Check sinks exist.
        for sink in &self.sinks {
            if !self.nodes.contains_key(sink) {
                return Err(Error::Graph(format!("sink {sink} not in graph")));
            }
        }
        // Cycle detection via Kahn's algorithm.
        self.topological_order()?;
        Ok(())
    }

    /// Return a topological ordering of nodes. Errors on cycle.
    pub fn topological_order(&self) -> Result<Vec<NodeId>> {
        let mut indeg: BTreeMap<NodeId, usize> =
            self.nodes.keys().map(|k| (*k, 0)).collect();
        for n in self.nodes.values() {
            for in_id in &n.inputs {
                if let Some(d) = indeg.get_mut(&n.id) {
                    *d += 1;
                }
                let _ = in_id;
            }
        }
        // Inputs flow source -> dependent, so in-degree of a node equals
        // its input count. Recompute correctly:
        let mut indeg: BTreeMap<NodeId, usize> = self
            .nodes
            .iter()
            .map(|(id, n)| (*id, n.inputs.len()))
            .collect();

        let mut queue: VecDeque<NodeId> = indeg
            .iter()
            .filter_map(|(id, d)| (*d == 0).then_some(*id))
            .collect();

        // Build reverse edges: node -> dependents.
        let mut dependents: BTreeMap<NodeId, Vec<NodeId>> =
            self.nodes.keys().map(|k| (*k, Vec::new())).collect();
        for n in self.nodes.values() {
            for in_id in &n.inputs {
                dependents.entry(*in_id).or_default().push(n.id);
            }
        }

        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(id) = queue.pop_front() {
            order.push(id);
            if let Some(deps) = dependents.get(&id) {
                for &d in deps {
                    let entry = indeg.entry(d).or_insert(0);
                    if *entry > 0 {
                        *entry -= 1;
                    }
                    if *entry == 0 {
                        queue.push_back(d);
                    }
                }
            }
        }
        if order.len() != self.nodes.len() {
            return Err(Error::Graph("cycle detected in pipeline graph".into()));
        }
        Ok(order)
    }

    /// All node ids reachable backward from `sink`, including itself.
    pub fn ancestors(&self, sink: NodeId) -> HashSet<NodeId> {
        let mut visited = HashSet::new();
        let mut stack = vec![sink];
        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            if let Some(n) = self.nodes.get(&id) {
                for &i in &n.inputs {
                    stack.push(i);
                }
            }
        }
        visited
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_chain_topo() {
        let mut g = Graph::new();
        let a = g.insert(Node::new("io.source", "src"));
        let b = g.insert(Node::new("fx.brightness", "bright").with_input(a));
        let c = g.insert(Node::new("io.export", "out").with_input(b));
        g.add_sink(c);

        let order = g.topological_order().unwrap();
        assert_eq!(order, vec![a, b, c]);
        g.validate().unwrap();
    }

    #[test]
    fn cycle_detected() {
        let mut g = Graph::new();
        let a_id = NodeId::new();
        let b_id = NodeId::new();

        let mut a = Node::new("a", "a");
        a.id = a_id;
        a.inputs.push(b_id);
        let mut b = Node::new("b", "b");
        b.id = b_id;
        b.inputs.push(a_id);

        g.nodes.insert(a_id, a);
        g.nodes.insert(b_id, b);

        assert!(g.topological_order().is_err());
    }

    #[test]
    fn ancestors_collects_upstream() {
        let mut g = Graph::new();
        let a = g.insert(Node::new("a", "a"));
        let b = g.insert(Node::new("b", "b").with_input(a));
        let c = g.insert(Node::new("c", "c").with_input(b));
        let d = g.insert(Node::new("d", "d")); // not connected

        let anc = g.ancestors(c);
        assert!(anc.contains(&a));
        assert!(anc.contains(&b));
        assert!(anc.contains(&c));
        assert!(!anc.contains(&d));
    }
}
