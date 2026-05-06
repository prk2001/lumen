//! Pipeline scheduler.
//!
//! Walks a [`Graph`] in topological order and runs each node's effect.
//! Phase 1 is sequential; Phase 4 (`lumen-perf`) parallelizes
//! independent branches.

use std::collections::BTreeMap;

use crate::context::Context;
use crate::error::{Error, Result};
use crate::frame::Frame;
use crate::graph::{Graph, NodeId};
use crate::registry::EffectRegistry;

/// Trait for resolving source-node frames. The scheduler is generic
/// over how source nodes load their pixels — for the CLI this is
/// "open the asset whose id is in `params.asset_id`"; for unit tests
/// it can be a closure that returns a synthetic frame.
pub trait SourceLoader {
    fn load(&mut self, node_id: NodeId, node_params: &crate::ParamValues) -> Result<Frame>;
}

/// Trait for handling sink-node outputs (export). Same generic
/// shape as [`SourceLoader`].
pub trait SinkWriter {
    fn write(
        &mut self,
        node_id: NodeId,
        node_params: &crate::ParamValues,
        frame: Frame,
    ) -> Result<()>;
}

/// Source / sink callbacks built from closures.
pub struct ClosureIo<L, W> {
    pub load: L,
    pub write: W,
}

impl<L, W> SourceLoader for ClosureIo<L, W>
where
    L: FnMut(NodeId, &crate::ParamValues) -> Result<Frame>,
{
    fn load(&mut self, node_id: NodeId, params: &crate::ParamValues) -> Result<Frame> {
        (self.load)(node_id, params)
    }
}

impl<L, W> SinkWriter for ClosureIo<L, W>
where
    W: FnMut(NodeId, &crate::ParamValues, Frame) -> Result<()>,
{
    fn write(
        &mut self,
        node_id: NodeId,
        params: &crate::ParamValues,
        frame: Frame,
    ) -> Result<()> {
        (self.write)(node_id, params, frame)
    }
}

/// Effect ids that the scheduler treats specially.
pub mod special_effect_ids {
    /// A node with this id is a source — its output frame is produced
    /// by the [`SourceLoader`].
    pub const SOURCE: &str = "lumen-io.source";
    /// A node with this id is a sink — its input frame is handed to the
    /// [`SinkWriter`].
    pub const SINK: &str = "lumen-io.sink";
}

/// The scheduler.
pub struct Scheduler<'a, S, W> {
    pub registry: &'a EffectRegistry,
    pub ctx: &'a mut Context,
    pub source_loader: S,
    pub sink_writer: W,
}

impl<'a, S: SourceLoader, W: SinkWriter> Scheduler<'a, S, W> {
    /// Run the graph. Returns the frame produced at each sink.
    pub fn run(&mut self, graph: &Graph) -> Result<BTreeMap<NodeId, Frame>> {
        graph.validate()?;
        let order = graph.topological_order()?;
        let mut frames: BTreeMap<NodeId, Frame> = BTreeMap::new();
        let mut sink_outputs: BTreeMap<NodeId, Frame> = BTreeMap::new();

        for id in order {
            let node = graph
                .get(id)
                .ok_or_else(|| Error::Graph(format!("missing node {id} during run")))?;

            let frame = match node.effect_id.as_str() {
                special_effect_ids::SOURCE => self.source_loader.load(id, &node.params)?,
                special_effect_ids::SINK => {
                    // Sink consumes one input frame; produces nothing for
                    // downstream — but we still give it to the writer.
                    let in_id = node
                        .inputs
                        .first()
                        .copied()
                        .ok_or_else(|| Error::Graph(format!("sink {id} has no input")))?;
                    let f = frames
                        .remove(&in_id)
                        .ok_or_else(|| Error::Graph(format!("sink {id} input {in_id} not produced")))?;
                    self.sink_writer.write(id, &node.params, f.clone())?;
                    if graph.sinks.contains(&id) {
                        sink_outputs.insert(id, f);
                    }
                    continue;
                }
                effect_id => {
                    let effect = self
                        .registry
                        .get(effect_id)
                        .ok_or_else(|| Error::EffectNotFound(effect_id.to_string()))?;

                    // Phase 1: every fx node is single-input. Multi-input
                    // composites (blend, merge) land with `lumen-fx-mask`
                    // in Phase 3.
                    let in_id = node
                        .inputs
                        .first()
                        .copied()
                        .ok_or_else(|| Error::Graph(format!("node {id} ({effect_id}) has no input")))?;
                    let f = frames
                        .remove(&in_id)
                        .ok_or_else(|| Error::Graph(format!("node {id} input {in_id} not produced")))?;

                    // Validate + fill defaults on a clone of the node's params.
                    let mut params = node.params.clone();
                    params.validate_and_fill(effect.parameters())?;
                    effect.apply(self.ctx, f, &params)?
                }
            };

            // Stash the produced frame for any downstream consumers.
            frames.insert(id, frame.clone());

            // If this node is itself a designated sink with no SINK effect,
            // surface its frame as an output.
            if graph.sinks.contains(&id) && node.effect_id != special_effect_ids::SINK {
                sink_outputs.insert(id, frame);
            }
        }

        Ok(sink_outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Capabilities, Category, ColorSpace, Effect, EffectMetadata, EffectRegistry, Frame, Node,
        ParamSpec, ParamValues, PixelData,
    };
    use std::sync::Arc;

    /// An effect that adds a constant to every R channel.
    #[derive(Debug, Default)]
    struct AddRed;

    static M: EffectMetadata = EffectMetadata {
        id: "test.add_red",
        display_name: "Add Red",
        description: "",
        category: Category::Qa,
        version: 1,
    };

    static P: &[ParamSpec] = &[];

    impl Effect for AddRed {
        fn metadata(&self) -> &EffectMetadata { &M }
        fn parameters(&self) -> &[ParamSpec] { P }
        fn capabilities(&self) -> Capabilities {
            Capabilities::cpu_only_deterministic()
        }
        fn apply(&self, _ctx: &mut Context, mut input: Frame, _p: &ParamValues) -> Result<Frame> {
            if let PixelData::Rgba8(v) = &mut input.data {
                for px in v.chunks_exact_mut(4) {
                    px[0] = px[0].saturating_add(50);
                }
            }
            Ok(input)
        }
    }

    #[test]
    fn runs_simple_chain() {
        let r = EffectRegistry::new();
        r.register(Arc::new(AddRed)).unwrap();

        let mut g = Graph::new();
        let src = g.insert(Node::new(special_effect_ids::SOURCE, "src"));
        let mid = g.insert(Node::new(M.id, "add").with_input(src));
        let snk = g.insert(Node::new(special_effect_ids::SINK, "out").with_input(mid));
        g.add_sink(snk);

        // Source = solid gray (R=100,G=100,B=100,A=255).
        let synth = Frame::new(
            2,
            2,
            PixelData::Rgba8(vec![100, 100, 100, 255, 100, 100, 100, 255, 100, 100, 100, 255, 100, 100, 100, 255]),
            ColorSpace::SRgb,
            None,
        )
        .unwrap();

        let out_capture = std::cell::RefCell::new(None::<Frame>);

        let mut ctx = Context::for_still_srgb();
        let mut sched = Scheduler {
            registry: &r,
            ctx: &mut ctx,
            source_loader: |_, _: &ParamValues| Ok(synth.clone()),
            sink_writer: |_, _: &ParamValues, f: Frame| {
                *out_capture.borrow_mut() = Some(f);
                Ok(())
            },
        };
        let _ = sched.run(&g).unwrap();

        let out = out_capture.into_inner().unwrap();
        let PixelData::Rgba8(v) = out.data else { panic!() };
        // Each R should be 100+50 = 150.
        for px in v.chunks_exact(4) {
            assert_eq!(px[0], 150);
            assert_eq!(px[1], 100);
            assert_eq!(px[2], 100);
        }
    }

    impl<F> SourceLoader for F
    where
        F: FnMut(NodeId, &ParamValues) -> Result<Frame>,
    {
        fn load(&mut self, id: NodeId, p: &ParamValues) -> Result<Frame> {
            self(id, p)
        }
    }

    impl<F> SinkWriter for F
    where
        F: FnMut(NodeId, &ParamValues, Frame) -> Result<()>,
    {
        fn write(&mut self, id: NodeId, p: &ParamValues, f: Frame) -> Result<()> {
            self(id, p, f)
        }
    }
}
