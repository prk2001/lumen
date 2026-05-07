//! `lumen project` — operate on `.lumenproj` files.
//!
//! Today this surfaces the project file Lumen-core has had since
//! Phase 1. Provides:
//!
//! - `lumen project show <X.lumenproj>` — pretty-print metadata.
//! - `lumen project run  <X.lumenproj>` — execute the project's graph
//!   via the same Scheduler used by `lumen pipeline`. The first
//!   asset is the source; the first sink node's effect_id `lumen-io.sink`
//!   is the target output.

use anyhow::{anyhow, Context as _, Result};
use lumen_core::{
    scheduler::special_effect_ids, AssetKind, Context, Frame, Graph, NodeId,
    ParamValues, Project, Scheduler,
};
use lumen_io::{decode_image, encode_image, ImageEncodeOptions};

pub fn cmd_project_show(path: &std::path::Path) -> Result<()> {
    let proj = Project::load(path).with_context(|| format!("load {}", path.display()))?;
    println!(
        "id        : {}\nschema    : {}\ncreated   : {}\nmodified  : {}\nassets    : {}\nnodes     : {}\nsinks     : {}\npresets   : {}\nmodels    : {}\nhistory   : {} entries",
        proj.id, proj.schema, proj.created, proj.modified,
        proj.assets.len(), proj.graph.nodes.len(), proj.graph.sinks.len(),
        proj.presets.len(), proj.models.len(), proj.history.len(),
    );
    if !proj.assets.is_empty() {
        println!("\nassets:");
        for a in &proj.assets {
            println!("  {} {} {}  {:?}",
                a.id, a.uri, a.display_name, a.kind);
        }
    }
    if !proj.graph.nodes.is_empty() {
        println!("\nnodes (in storage order):");
        for (id, n) in proj.graph.nodes.iter() {
            println!("  {} ({})  {} inputs", id, n.effect_id, n.inputs.len());
        }
    }
    Ok(())
}

pub fn cmd_project_run(
    path: &std::path::Path,
    output: &std::path::Path,
    jpeg_quality: u8,
) -> Result<()> {
    let proj = Project::load(path).with_context(|| format!("load {}", path.display()))?;
    let registry = crate::build_registry()?;

    // Resolve the source asset: the first asset whose kind is StillImage
    // or Video. For Phase 1 we only know how to render still-image
    // graphs through this CLI helper; video is `video-pipeline`'s job.
    let still_asset = proj.assets.iter().find(|a| a.kind == AssetKind::StillImage)
        .ok_or_else(|| anyhow!("project has no still-image asset"))?;
    let input_path = uri_to_path(&still_asset.uri)?;

    // Validate the graph and topo-sort it. We piggy-back on the existing
    // Scheduler — but lumen-core's Project graph uses real source/sink
    // nodes with a specific effect_id convention. If the project doesn't
    // specify a SOURCE/SINK node, we wrap the existing graph with one.
    let graph: Graph = proj.graph.clone();
    let needs_source = !graph.nodes.values().any(|n| n.effect_id == special_effect_ids::SOURCE);
    let needs_sink   = !graph.nodes.values().any(|n| n.effect_id == special_effect_ids::SINK);
    if needs_source || needs_sink {
        return Err(anyhow!(
            "project graph must contain at least one '{}' node and one '{}' node",
            special_effect_ids::SOURCE, special_effect_ids::SINK,
        ));
    }
    graph.validate()?;

    let mut ctx = Context::for_still_srgb();
    let written = std::cell::RefCell::new(None::<std::path::PathBuf>);
    let in_path = input_path.clone();
    let out_path = output.to_path_buf();
    let source = crate::CliSource(move |_id: NodeId, _params: &ParamValues| -> lumen_core::Result<Frame> {
        decode_image(&in_path)
    });
    let sink = crate::CliSink(move |_id: NodeId, _params: &ParamValues, frame: Frame| -> lumen_core::Result<()> {
        let p = encode_image(
            frame,
            &out_path,
            ImageEncodeOptions { jpeg_quality, format: None },
        )?;
        *written.borrow_mut() = Some(p);
        Ok(())
    });
    let mut sched = Scheduler {
        registry: &registry,
        ctx: &mut ctx,
        source_loader: source,
        sink_writer: sink,
    };
    sched.run(&graph).map_err(|e| anyhow!("scheduler: {e}"))?;
    println!("wrote {}", output.display());
    Ok(())
}

fn uri_to_path(uri: &str) -> Result<std::path::PathBuf> {
    if let Some(rest) = uri.strip_prefix("file://") {
        Ok(std::path::PathBuf::from(rest))
    } else {
        Err(anyhow!("only file:// URIs supported in this CLI; got {uri}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uri_to_path_strips_file_scheme() {
        assert_eq!(
            uri_to_path("file:///tmp/x.png").unwrap(),
            std::path::PathBuf::from("/tmp/x.png")
        );
    }

    #[test]
    fn uri_to_path_rejects_non_file() {
        assert!(uri_to_path("https://example.com/x.png").is_err());
    }
}
