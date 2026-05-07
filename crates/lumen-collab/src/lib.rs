//! # lumen-collab
//!
//! Collaboration & project management: portable bundles, share links,
//! and three-way-style project diff/merge.
//!
//! ## Cat 24 scope
//!
//! - Pack a [`Project`] into a self-contained `.lumenbundle` ZIP that
//!   carries the project JSON plus (optionally) every referenced asset
//!   file, keyed by BLAKE3 hash so two projects sharing an asset don't
//!   bloat the archive.
//! - Mint and verify Ed25519-signed [`ShareLink`]s that bind a random
//!   token to the BLAKE3 hash of the project's canonical JSON.
//! - Compute a [`ProjectDiff`] for review UIs (graph delta plus
//!   top-level asset / preset / model deltas).
//! - Merge two projects with a last-writer-wins graph policy and an
//!   append-only policy for `assets` and `presets`, surfacing per-node
//!   parameter conflicts in [`MergeOutcome::conflicts`].
//!
//! Phase 1 is intentionally offline / file-based; cloud sync arrives
//! later.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, rust_2018_idioms)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use rand::RngCore;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use lumen_auth::{SigningKey, VerifyingKey};
use lumen_core::asset::{Asset, AssetId};
use lumen_core::error::{Error, Result};
use lumen_core::graph::NodeId;
use lumen_core::project::Project;
use lumen_workflow::{diff_graphs, GraphDiff, ParamDiff};

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// Schema identifier embedded in [`BundleManifest`] for forward compat.
pub const BUNDLE_SCHEMA: &str = "lumen.bundle.manifest/v1";

/// File name used for the project document inside a bundle archive.
pub const BUNDLE_PROJECT_NAME: &str = "project.json";

/// File name used for the bundle manifest inside a bundle archive.
pub const BUNDLE_MANIFEST_NAME: &str = "bundle.json";

/// Directory inside a bundle archive that stores embedded asset files,
/// keyed by hash.
pub const BUNDLE_ASSETS_DIR: &str = "assets";

// ---------------------------------------------------------------------------
// Bundles
// ---------------------------------------------------------------------------

/// Manifest for a `.lumenbundle` archive — describes its structure so a
/// reader can validate the contents without reaching outside the ZIP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleManifest {
    /// Stable schema identifier. Always [`BUNDLE_SCHEMA`].
    pub schema: String,
    /// Whether asset bytes are embedded under [`BUNDLE_ASSETS_DIR`].
    pub embeds_assets: bool,
    /// Hashes (BLAKE3, hex without prefix) of the embedded asset files
    /// keyed under `assets/<hash>`. Empty when `embeds_assets == false`
    /// or when no asset had a known hash.
    #[serde(default)]
    pub embedded_hashes: BTreeSet<String>,
    /// Name of the project file inside the archive — usually
    /// [`BUNDLE_PROJECT_NAME`].
    pub project_filename: String,
    /// Wall-clock time the bundle was packed.
    #[serde(with = "time::serde::rfc3339")]
    pub packed_at: OffsetDateTime,
    /// Software name that packed the bundle.
    pub packer_software: String,
    /// Packer software version.
    pub packer_version: String,
}

impl BundleManifest {
    fn fresh(embeds_assets: bool, embedded_hashes: BTreeSet<String>) -> Self {
        Self {
            schema: BUNDLE_SCHEMA.to_string(),
            embeds_assets,
            embedded_hashes,
            project_filename: BUNDLE_PROJECT_NAME.to_string(),
            packed_at: OffsetDateTime::now_utc(),
            packer_software: "lumen".to_string(),
            packer_version: CRATE_VERSION.to_string(),
        }
    }
}

/// One asset embedded inside a [`ProjectBundle`] alongside the
/// originating [`AssetId`] and the bytes-on-disk it was loaded from.
#[derive(Debug, Clone)]
pub struct EmbeddedAsset {
    /// Project-side asset id.
    pub asset_id: AssetId,
    /// BLAKE3 hash of the asset bytes, lowercase hex (no prefix).
    pub hash_hex: String,
    /// Filename inside the bundle archive (relative to its root).
    pub archive_path: String,
    /// On-disk path the asset was extracted to (only populated by
    /// [`unpack_bundle`]).
    pub extracted_path: Option<PathBuf>,
}

/// A loaded `.lumenbundle` archive: manifest, project, and any embedded
/// asset files referenced by hash.
#[derive(Debug, Clone)]
pub struct ProjectBundle {
    /// Bundle-level manifest.
    pub manifest: BundleManifest,
    /// The packed [`Project`] document.
    pub project: Project,
    /// Embedded asset files; empty when the bundle was packed without
    /// embedded assets.
    pub embedded_assets: Vec<EmbeddedAsset>,
}

/// Pack a [`Project`] into a `.lumenbundle` archive at `out`.
///
/// When `embed_assets` is `true` every asset whose URI is a `file://`
/// path on the local filesystem is hashed (BLAKE3) and copied into
/// `assets/<hex>` inside the archive. Assets with non-file URIs or whose
/// files are missing are skipped silently — the bundle still travels,
/// it just isn't fully self-contained for those assets.
pub fn pack_bundle<P: AsRef<Path>>(project: &Project, out: P, embed_assets: bool) -> Result<()> {
    let out = out.as_ref();
    let file = fs::File::create(out)?;
    let mut writer = zip::ZipWriter::new(file);
    let options: zip::write::FileOptions<'_, ()> =
        zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    // Project JSON.
    let project_json = serde_json::to_vec_pretty(project)?;
    writer
        .start_file(BUNDLE_PROJECT_NAME, options)
        .map_err(zip_to_error)?;
    writer.write_all(&project_json)?;

    // Optional asset embedding.
    let mut embedded_hashes: BTreeSet<String> = BTreeSet::new();
    if embed_assets {
        for asset in &project.assets {
            let Some(local_path) = file_uri_to_path(&asset.uri) else {
                continue;
            };
            let bytes = match fs::read(&local_path) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let hash = blake3::hash(&bytes).to_hex().to_string();
            // Skip duplicates within the bundle.
            if !embedded_hashes.insert(hash.clone()) {
                continue;
            }
            let archive_path = format!("{BUNDLE_ASSETS_DIR}/{hash}");
            writer
                .start_file(&archive_path, options)
                .map_err(zip_to_error)?;
            writer.write_all(&bytes)?;
        }
    }

    // Bundle manifest last so it sees the final embedded hash list.
    let manifest = BundleManifest::fresh(embed_assets, embedded_hashes);
    let manifest_json = serde_json::to_vec_pretty(&manifest)?;
    writer
        .start_file(BUNDLE_MANIFEST_NAME, options)
        .map_err(zip_to_error)?;
    writer.write_all(&manifest_json)?;

    writer.finish().map_err(zip_to_error)?;
    Ok(())
}

/// Extract a `.lumenbundle` archive from `path` into `dest_dir`.
///
/// Returns the loaded [`ProjectBundle`]. Embedded assets are written
/// under `dest_dir/assets/<hash>` and their on-disk paths recorded in
/// each [`EmbeddedAsset::extracted_path`].
pub fn unpack_bundle<P: AsRef<Path>>(path: P, dest_dir: P) -> Result<ProjectBundle> {
    let path = path.as_ref();
    let dest_dir = dest_dir.as_ref();
    fs::create_dir_all(dest_dir)?;

    let file = fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(file).map_err(zip_to_error)?;

    let mut project: Option<Project> = None;
    let mut manifest: Option<BundleManifest> = None;
    let mut embedded: Vec<EmbeddedAsset> = Vec::new();

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(zip_to_error)?;
        let name = entry.name().to_string();
        // Defend against zip-slip — only allow simple, relative paths.
        if name.contains("..") || name.starts_with('/') {
            return Err(Error::Other(format!(
                "rejected suspicious bundle entry: {name}"
            )));
        }

        if name == BUNDLE_PROJECT_NAME {
            let mut buf = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut buf)?;
            let p: Project = serde_json::from_slice(&buf)?;
            project = Some(p);
        } else if name == BUNDLE_MANIFEST_NAME {
            let mut buf = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut buf)?;
            let m: BundleManifest = serde_json::from_slice(&buf)?;
            manifest = Some(m);
        } else if let Some(rest) = name.strip_prefix(&format!("{BUNDLE_ASSETS_DIR}/")) {
            // Asset file embedded by hash.
            if rest.is_empty() {
                continue;
            }
            let mut buf = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut buf)?;
            let asset_dir = dest_dir.join(BUNDLE_ASSETS_DIR);
            fs::create_dir_all(&asset_dir)?;
            let extracted = asset_dir.join(rest);
            fs::write(&extracted, &buf)?;
            embedded.push(EmbeddedAsset {
                asset_id: AssetId::default(),
                hash_hex: rest.to_string(),
                archive_path: name,
                extracted_path: Some(extracted),
            });
        }
    }

    let project = project.ok_or_else(|| {
        Error::Other(format!(
            "bundle is missing required entry '{BUNDLE_PROJECT_NAME}'"
        ))
    })?;
    let manifest = manifest.unwrap_or_else(|| {
        // Backwards-compatible default: pretend it's a project-only bundle.
        BundleManifest::fresh(!embedded.is_empty(), BTreeSet::new())
    });

    // Best-effort: associate each embedded blob with the asset whose
    // recorded hash matches. Multiple assets may share the same hash —
    // that's fine, all of them point at the same extracted file.
    let mut resolved: Vec<EmbeddedAsset> = Vec::new();
    for blob in &embedded {
        let mut linked = false;
        for asset in &project.assets {
            if asset_hash_hex(asset).as_deref() == Some(blob.hash_hex.as_str()) {
                resolved.push(EmbeddedAsset {
                    asset_id: asset.id,
                    hash_hex: blob.hash_hex.clone(),
                    archive_path: blob.archive_path.clone(),
                    extracted_path: blob.extracted_path.clone(),
                });
                linked = true;
            }
        }
        if !linked {
            resolved.push(blob.clone());
        }
    }

    Ok(ProjectBundle {
        manifest,
        project,
        embedded_assets: resolved,
    })
}

// ---------------------------------------------------------------------------
// Share links
// ---------------------------------------------------------------------------

/// A signed share link: random token plus the BLAKE3 hash of the
/// project's canonical JSON, bound together by an Ed25519 signature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShareLink {
    /// 32-byte random token, lowercase hex (64 chars).
    pub token_hex: String,
    /// BLAKE3 hash of the project's canonical JSON, prefixed `"blake3:"`.
    pub project_hash: String,
    /// When the link was minted.
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    /// Hex-encoded Ed25519 signature over `token || project_hash`.
    pub signature_hex: String,
}

/// Mint a [`ShareLink`] for `project`, signed with `signing_key`.
///
/// The signature covers the token bytes followed by the canonical
/// project hash bytes — verifiers must hold both to validate.
pub fn make_share_link(project: &Project, signing_key: &SigningKey) -> Result<ShareLink> {
    let mut token = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut token);
    let token_hex = hex_encode(token);

    let project_hash = project_canonical_hash(project)?;
    let to_sign = signed_payload(&token_hex, &project_hash);
    let sig = ed25519_dalek::Signer::sign(signing_key, &to_sign);
    let signature_hex = hex_encode(sig.to_bytes());

    Ok(ShareLink {
        token_hex,
        project_hash,
        created_at: OffsetDateTime::now_utc(),
        signature_hex,
    })
}

/// Verify that `link` was minted for `project` by the holder of the
/// private key corresponding to `public_key`.
pub fn verify_share_link(link: &ShareLink, project: &Project, public_key: &VerifyingKey) -> bool {
    // The hash must match the project on hand.
    let Ok(expected_hash) = project_canonical_hash(project) else {
        return false;
    };
    if expected_hash != link.project_hash {
        return false;
    }

    // Reconstruct the signature.
    let Some(sig_bytes) = hex_decode(&link.signature_hex) else {
        return false;
    };
    let Ok(sig_arr): std::result::Result<[u8; 64], _> = sig_bytes.as_slice().try_into() else {
        return false;
    };
    let sig = ed25519_dalek::Signature::from_bytes(&sig_arr);

    let to_verify = signed_payload(&link.token_hex, &link.project_hash);
    public_key.verify_strict(&to_verify, &sig).is_ok()
}

fn signed_payload(token_hex: &str, project_hash: &str) -> Vec<u8> {
    let mut buf = Vec::with_capacity(token_hex.len() + 1 + project_hash.len());
    buf.extend_from_slice(token_hex.as_bytes());
    buf.push(b'|');
    buf.extend_from_slice(project_hash.as_bytes());
    buf
}

// ---------------------------------------------------------------------------
// Diff & merge
// ---------------------------------------------------------------------------

/// Top-level project diff used by review UIs.
///
/// Combines the [`GraphDiff`] from `lumen-workflow` with simple
/// add/remove deltas across `assets`, `presets`, and `models`.
///
/// `AssetId` and the `Preset` UUIDs do not implement `Ord`, so the
/// asset / preset deltas are kept as `Vec`s in insertion-stable order
/// (the iteration order of the underlying project).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProjectDiff {
    /// Pipeline graph delta.
    pub graph: GraphDiff,
    /// Asset ids present in `b` but not `a`.
    pub assets_added: Vec<AssetId>,
    /// Asset ids present in `a` but not `b`.
    pub assets_removed: Vec<AssetId>,
    /// Preset ids present in `b` but not `a`.
    pub presets_added: Vec<uuid::Uuid>,
    /// Preset ids present in `a` but not `b`.
    pub presets_removed: Vec<uuid::Uuid>,
    /// Pinned model entries added on the `b` side: `model_id -> hash`.
    pub models_added: BTreeMap<String, String>,
    /// Pinned model entries removed on the `b` side.
    pub models_removed: BTreeSet<String>,
    /// Pinned model entries whose hash changed: `model_id -> b_hash`.
    pub models_changed: BTreeMap<String, String>,
}

impl ProjectDiff {
    /// `true` when the diff would produce no changes if applied.
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
            && self.assets_added.is_empty()
            && self.assets_removed.is_empty()
            && self.presets_added.is_empty()
            && self.presets_removed.is_empty()
            && self.models_added.is_empty()
            && self.models_removed.is_empty()
            && self.models_changed.is_empty()
    }
}

/// Compute a [`ProjectDiff`] between two projects.
pub fn diff_projects(a: &Project, b: &Project) -> ProjectDiff {
    let graph = diff_graphs(&a.graph, &b.graph);

    let a_assets: BTreeSet<uuid::Uuid> = a.assets.iter().map(|x| x.id.0).collect();
    let b_assets: BTreeSet<uuid::Uuid> = b.assets.iter().map(|x| x.id.0).collect();
    let assets_added: Vec<AssetId> = b
        .assets
        .iter()
        .filter(|x| !a_assets.contains(&x.id.0))
        .map(|x| x.id)
        .collect();
    let assets_removed: Vec<AssetId> = a
        .assets
        .iter()
        .filter(|x| !b_assets.contains(&x.id.0))
        .map(|x| x.id)
        .collect();

    let a_presets: BTreeSet<uuid::Uuid> = a.presets.iter().map(|p| p.id).collect();
    let b_presets: BTreeSet<uuid::Uuid> = b.presets.iter().map(|p| p.id).collect();
    let presets_added: Vec<uuid::Uuid> = b
        .presets
        .iter()
        .map(|p| p.id)
        .filter(|id| !a_presets.contains(id))
        .collect();
    let presets_removed: Vec<uuid::Uuid> = a
        .presets
        .iter()
        .map(|p| p.id)
        .filter(|id| !b_presets.contains(id))
        .collect();

    let mut models_added = BTreeMap::new();
    let mut models_changed = BTreeMap::new();
    for (k, bv) in &b.models {
        match a.models.get(k) {
            None => {
                models_added.insert(k.clone(), bv.clone());
            }
            Some(av) if av != bv => {
                models_changed.insert(k.clone(), bv.clone());
            }
            _ => {}
        }
    }
    let mut models_removed = BTreeSet::new();
    for k in a.models.keys() {
        if !b.models.contains_key(k) {
            models_removed.insert(k.clone());
        }
    }

    ProjectDiff {
        graph,
        assets_added,
        assets_removed,
        presets_added,
        presets_removed,
        models_added,
        models_removed,
        models_changed,
    }
}

/// One per-node parameter conflict surfaced by [`merge_projects`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    /// Node id where both sides edited a parameter from the same base.
    pub node_id: NodeId,
    /// Parameter keys that disagree between `base` and `theirs`.
    pub param_keys: BTreeSet<String>,
}

/// Result of [`merge_projects`].
#[derive(Debug, Clone)]
pub struct MergeOutcome {
    /// The merged project document.
    pub merged: Project,
    /// Per-node parameter conflicts. Empty when the merge was clean.
    pub conflicts: Vec<Conflict>,
}

/// Three-way-style merge of `theirs` into `base`.
///
/// Strategy:
///
/// - **Graph nodes:** `theirs.graph.nodes` overlay `base.graph.nodes`
///   keyed by [`NodeId`] (last-writer-wins). Sinks are union-merged.
/// - **Assets:** assets from `theirs` whose ids are not present in
///   `base` are appended.
/// - **Presets:** presets from `theirs` whose ids are not present in
///   `base` are appended.
/// - **Models:** entries from `theirs` overlay `base`; if both sides
///   pin a different hash for the same model id, the conflict is
///   surfaced via [`Conflict::param_keys`] under the synthetic node id
///   [`MODELS_CONFLICT_NODE`].
///
/// The merge function does not currently consult an explicit common
/// ancestor — `base` plays that role. For nodes both sides edited
/// (i.e. params differ between sides), the diverging parameter keys
/// are surfaced in [`MergeOutcome::conflicts`] and `theirs` wins for
/// those keys.
pub fn merge_projects(base: &Project, theirs: &Project) -> Result<MergeOutcome> {
    let mut merged = base.clone();
    let mut conflicts: Vec<Conflict> = Vec::new();

    // Detect node conflicts first (before mutating). A "conflict" here
    // is: a node that exists on both sides whose parameter values
    // disagree. We surface every diverging key under a single Conflict
    // entry per node id.
    for (id, their_node) in &theirs.graph.nodes {
        if let Some(base_node) = base.graph.nodes.get(id) {
            let pd: ParamDiff = lumen_workflow::diff_params(&base_node.params, &their_node.params);
            if !pd.is_empty() {
                let mut keys: BTreeSet<String> = BTreeSet::new();
                keys.extend(pd.additions.keys().cloned());
                keys.extend(pd.removals.iter().cloned());
                keys.extend(pd.changes.keys().cloned());
                conflicts.push(Conflict {
                    node_id: *id,
                    param_keys: keys,
                });
            }
        }
    }

    // Last-writer-wins overlay of nodes.
    for (id, their_node) in &theirs.graph.nodes {
        merged.graph.nodes.insert(*id, their_node.clone());
    }
    // Sinks: union-preserve order.
    for sink in &theirs.graph.sinks {
        if !merged.graph.sinks.contains(sink) {
            merged.graph.sinks.push(*sink);
        }
    }

    // Assets: append ones that aren't already there (by id).
    let known_assets: BTreeSet<uuid::Uuid> = merged.assets.iter().map(|a| a.id.0).collect();
    for asset in &theirs.assets {
        if !known_assets.contains(&asset.id.0) {
            merged.assets.push(asset.clone());
        }
    }

    // Presets: append ones that aren't already there (by id).
    let known_presets: BTreeSet<uuid::Uuid> = merged.presets.iter().map(|p| p.id).collect();
    for preset in &theirs.presets {
        if !known_presets.contains(&preset.id) {
            merged.presets.push(preset.clone());
        }
    }

    // Models: overlay, surfacing per-model-id conflicts.
    let mut model_conflicts: BTreeSet<String> = BTreeSet::new();
    for (model_id, their_hash) in &theirs.models {
        match base.models.get(model_id) {
            Some(base_hash) if base_hash != their_hash => {
                model_conflicts.insert(model_id.clone());
            }
            _ => {}
        }
        merged.models.insert(model_id.clone(), their_hash.clone());
    }
    if !model_conflicts.is_empty() {
        conflicts.push(Conflict {
            node_id: MODELS_CONFLICT_NODE,
            param_keys: model_conflicts,
        });
    }

    merged.modified = OffsetDateTime::now_utc();
    Ok(MergeOutcome { merged, conflicts })
}

/// Synthetic [`NodeId`] used to attach model-pin conflicts to a
/// [`Conflict`] entry. Uses the all-zero UUID so it can never collide
/// with a real node id minted via UUIDv7.
pub const MODELS_CONFLICT_NODE: NodeId = NodeId(uuid::Uuid::nil());

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Hash a [`Project`] in canonical (sorted-key) JSON form.
///
/// Returns the BLAKE3 hash with the workspace's `"blake3:<hex>"`
/// self-describing prefix.
fn project_canonical_hash(project: &Project) -> Result<String> {
    let value: serde_json::Value = serde_json::to_value(project)?;
    let canonical = canonicalise_value(value);
    let bytes = serde_json::to_vec(&canonical)?;
    let h = blake3::hash(&bytes);
    Ok(format!("blake3:{}", h.to_hex()))
}

fn canonicalise_value(v: serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match v {
        Value::Object(map) => {
            let mut entries: Vec<(String, Value)> = map.into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let mut out = serde_json::Map::with_capacity(entries.len());
            for (k, v) in entries {
                out.insert(k, canonicalise_value(v));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(canonicalise_value).collect()),
        other => other,
    }
}

fn asset_hash_hex(asset: &Asset) -> Option<String> {
    let h = asset.hash.as_deref()?;
    let bare = h.strip_prefix("blake3:").unwrap_or(h);
    Some(bare.to_string())
}

fn file_uri_to_path(uri: &str) -> Option<PathBuf> {
    let rest = uri.strip_prefix("file://")?;
    // Accept both `file:///abs/path` and `file://localhost/abs/path`.
    let rest = rest.strip_prefix("localhost").unwrap_or(rest);
    if rest.is_empty() {
        return None;
    }
    Some(PathBuf::from(rest))
}

fn zip_to_error(e: zip::result::ZipError) -> Error {
    Error::Other(format!("zip error: {e}"))
}

fn hex_encode<T: AsRef<[u8]>>(bytes: T) -> String {
    let mut s = String::with_capacity(bytes.as_ref().len() * 2);
    for b in bytes.as_ref() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    for i in (0..s.len()).step_by(2) {
        let byte = u8::from_str_radix(&s[i..i + 2], 16).ok()?;
        out.push(byte);
    }
    Some(out)
}

// Force `Seek` to be in scope for `ZipWriter::finish` even though our
// minimal use doesn't surface it directly — keeps the import live for
// future expansions.
#[allow(dead_code)]
fn _seek_in_scope<W: Seek>(_w: &W) {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_auth::keypair_generate;
    use lumen_core::asset::AssetKind;
    use lumen_core::graph::Node;
    use lumen_core::params::{ParamValue, ParamValues};
    use lumen_core::project::Preset;
    use tempfile::tempdir;
    use uuid::Uuid;

    fn sample_project() -> Project {
        let mut p = Project::empty();
        let src = p.graph.insert(Node::new("io.source", "src"));
        let mut params = ParamValues::new();
        params.insert("strength", ParamValue::Float(0.5));
        let bright = p.graph.insert(
            Node::new("fx.brightness", "bright")
                .with_input(src)
                .with_params(params),
        );
        let sink = p
            .graph
            .insert(Node::new("io.export", "out").with_input(bright));
        p.graph.add_sink(sink);

        p.assets.push(Asset::new(
            "file:///tmp/some-photo.jpg",
            "Photo",
            AssetKind::StillImage,
        ));
        p
    }

    #[test]
    fn pack_unpack_round_trips_project() {
        let project = sample_project();
        let dir = tempdir().unwrap();
        let bundle_path = dir.path().join("share.lumenbundle");
        let dest = dir.path().join("extracted");

        pack_bundle(&project, &bundle_path, false).unwrap();
        assert!(bundle_path.exists());

        let loaded = unpack_bundle(bundle_path.as_path(), dest.as_path()).unwrap();
        assert_eq!(loaded.project.id, project.id);
        assert_eq!(
            loaded.project.graph.nodes.keys().collect::<Vec<_>>(),
            project.graph.nodes.keys().collect::<Vec<_>>(),
        );
        assert_eq!(loaded.manifest.schema, BUNDLE_SCHEMA);
        assert!(!loaded.manifest.embeds_assets);
        assert!(loaded.embedded_assets.is_empty());
    }

    #[test]
    fn pack_with_embedded_asset_round_trips_bytes() {
        let dir = tempdir().unwrap();
        let asset_path = dir.path().join("photo.bin");
        let payload = b"some-bytes-here";
        std::fs::write(&asset_path, payload).unwrap();
        let payload_hash = blake3::hash(payload).to_hex().to_string();

        let mut project = sample_project();
        // Replace the placeholder asset with a real local file we can hash.
        let uri = format!("file://{}", asset_path.display());
        let mut asset = Asset::new(&uri, "Photo", AssetKind::StillImage);
        asset.hash = Some(format!("blake3:{}", payload_hash));
        project.assets.clear();
        project.assets.push(asset);

        let bundle_path = dir.path().join("share.lumenbundle");
        pack_bundle(&project, &bundle_path, true).unwrap();

        let dest = dir.path().join("extracted");
        let loaded = unpack_bundle(bundle_path.as_path(), dest.as_path()).unwrap();
        assert!(loaded.manifest.embeds_assets);
        assert_eq!(loaded.embedded_assets.len(), 1);
        let extracted = loaded.embedded_assets[0]
            .extracted_path
            .as_ref()
            .expect("path");
        assert_eq!(std::fs::read(extracted).unwrap(), payload);
    }

    #[test]
    fn share_link_verifies_with_right_key_only() {
        let project = sample_project();
        let (sk, vk) = keypair_generate();
        let (_other_sk, other_vk) = keypair_generate();

        let link = make_share_link(&project, &sk).unwrap();
        assert_eq!(link.token_hex.len(), 64);
        assert!(link.project_hash.starts_with("blake3:"));
        assert!(verify_share_link(&link, &project, &vk));
        assert!(!verify_share_link(&link, &project, &other_vk));
    }

    #[test]
    fn share_link_rejects_tampered_project() {
        let project = sample_project();
        let (sk, vk) = keypair_generate();
        let link = make_share_link(&project, &sk).unwrap();

        // Mutate the project after minting the link.
        let mut altered = project.clone();
        altered.graph.insert(Node::new("fx.contrast", "c"));
        assert!(!verify_share_link(&link, &altered, &vk));
    }

    #[test]
    fn diff_identical_projects_is_empty() {
        let a = sample_project();
        let b = a.clone();
        let d = diff_projects(&a, &b);
        assert!(d.is_empty(), "expected empty diff, got {d:?}");
    }

    #[test]
    fn diff_detects_graph_and_preset_changes() {
        let a = sample_project();
        let mut b = a.clone();

        // Tweak a node's params -> graph delta.
        let some_node_id = *a
            .graph
            .nodes
            .iter()
            .find(|(_, n)| n.effect_id == "fx.brightness")
            .map(|(id, _)| id)
            .unwrap();
        let mut new_params = ParamValues::new();
        new_params.insert("strength", ParamValue::Float(0.99));
        b.graph.nodes.get_mut(&some_node_id).unwrap().params = new_params;

        // Add a preset on the b side.
        b.presets.push(Preset {
            id: Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)),
            name: "b-only".to_string(),
            effect_id: "fx.brightness".to_string(),
            params: serde_json::json!({"strength": 0.7}),
        });

        let d = diff_projects(&a, &b);
        assert!(!d.is_empty());
        assert_eq!(d.graph.node_changed.len(), 1);
        assert_eq!(d.presets_added.len(), 1);
        assert!(d.presets_removed.is_empty());
    }

    #[test]
    fn merge_appends_new_presets_without_duplication() {
        let mut base = sample_project();
        let shared = Preset {
            id: Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)),
            name: "shared".to_string(),
            effect_id: "fx.brightness".to_string(),
            params: serde_json::json!({"strength": 0.5}),
        };
        base.presets.push(shared.clone());

        let mut theirs = base.clone();
        // Theirs adds a new preset.
        theirs.presets.push(Preset {
            id: Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)),
            name: "theirs-only".to_string(),
            effect_id: "fx.brightness".to_string(),
            params: serde_json::json!({"strength": 0.9}),
        });
        // And re-adds the shared one — should not duplicate.
        theirs.presets.push(shared.clone());

        let outcome = merge_projects(&base, &theirs).unwrap();
        // Expect: original `shared` (kept once) + `theirs-only`.
        assert_eq!(outcome.merged.presets.len(), 2);
        let names: Vec<&str> = outcome
            .merged
            .presets
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        assert!(names.contains(&"shared"));
        assert!(names.contains(&"theirs-only"));
    }

    #[test]
    fn merge_flags_param_conflicts() {
        let base = sample_project();
        let bright_id = *base
            .graph
            .nodes
            .iter()
            .find(|(_, n)| n.effect_id == "fx.brightness")
            .map(|(id, _)| id)
            .unwrap();

        let mut theirs = base.clone();
        let mut their_params = ParamValues::new();
        their_params.insert("strength", ParamValue::Float(0.123));
        their_params.insert("mode", ParamValue::String("aggressive".into()));
        theirs.graph.nodes.get_mut(&bright_id).unwrap().params = their_params;

        let outcome = merge_projects(&base, &theirs).unwrap();
        // The merged node uses theirs (last-writer-wins).
        assert_eq!(
            outcome
                .merged
                .graph
                .nodes
                .get(&bright_id)
                .unwrap()
                .params
                .get_float("strength"),
            Some(0.123),
        );
        // And the conflict is surfaced.
        assert_eq!(outcome.conflicts.len(), 1);
        assert_eq!(outcome.conflicts[0].node_id, bright_id);
        assert!(outcome.conflicts[0].param_keys.contains("strength"));
        assert!(outcome.conflicts[0].param_keys.contains("mode"));
    }
}
