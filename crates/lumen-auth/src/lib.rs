//! # lumen-auth
//!
//! Authentication & integrity: chain-of-custody, hashing, C2PA.
//!
//! This crate currently implements **Cat 22 — chain-of-custody manifest
//! signing**: a small, dependency-light building block for forensic and
//! professional workflows.
//!
//! A [`Manifest`] captures *what was rendered, from what input, with what
//! recipe, by what software, when*. It is canonicalised to deterministic
//! sorted-key JSON and signed with Ed25519, then bundled into a
//! [`SignedManifest`] you can drop alongside the rendered output as a
//! `<output>.lumen-cco.json` sidecar.
//!
//! The verification side is symmetric: parse the sidecar, re-canonicalise
//! the embedded manifest, and check the signature with the bundled public
//! key (or a known-trusted one of your choosing).
//!
//! ## Quick example
//!
//! ```no_run
//! use lumen_auth::{
//!     build_manifest, keypair_generate, save_signed, sign, verify, SignedManifest,
//! };
//!
//! # fn run() -> lumen_core::Result<()> {
//! let manifest = build_manifest("input.jpg", "output.jpg", "recipe.json")?;
//! let (sk, vk) = keypair_generate();
//! let sig = sign(&manifest, &sk);
//! assert!(verify(&manifest, &sig, &vk));
//!
//! let signed = SignedManifest {
//!     manifest,
//!     signature_hex: hex_encode(sig.to_bytes()),
//!     public_key_hex: hex_encode(vk.to_bytes()),
//! };
//! save_signed(&signed, "output.jpg.lumen-cco.json")?;
//! # Ok(())
//! # }
//! # fn hex_encode<T: AsRef<[u8]>>(b: T) -> String {
//! #     b.as_ref().iter().map(|x| format!("{x:02x}")).collect()
//! # }
//! ```

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, rust_2018_idioms)]

use std::path::Path;

use ed25519_dalek::Signer;
pub use ed25519_dalek::{Signature, SigningKey, VerifyingKey};
use lumen_core::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;
use uuid::Uuid;

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// Schema identifier embedded in every manifest.
pub const MANIFEST_SCHEMA: &str = "lumen.cco.manifest";

/// Schema version. Bump when the manifest field set changes in a
/// non-additive way.
pub const MANIFEST_VERSION: u32 = 1;

/// Software name auto-filled into manifests built with [`build_manifest`].
pub const MANIFEST_SOFTWARE: &str = "lumen";

/// A chain-of-custody manifest: what was rendered, from what, by what,
/// when.
///
/// All hash fields use the workspace's `"blake3:<hex>"` self-describing
/// format so the algorithm is unambiguous in JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// Stable schema identifier. Always [`MANIFEST_SCHEMA`].
    pub schema: String,
    /// Manifest schema version. See [`MANIFEST_VERSION`].
    pub version: u32,
    /// Unique render identifier (UUIDv7).
    pub render_id: Uuid,
    /// BLAKE3 hash of the input file, prefixed `"blake3:"`.
    pub input_hash: String,
    /// BLAKE3 hash of the rendered output file, prefixed `"blake3:"`.
    pub output_hash: String,
    /// BLAKE3 hash of the recipe (parameters / pipeline) file, prefixed `"blake3:"`.
    pub recipe_hash: String,
    /// Signing software name, e.g. `"lumen"`.
    pub software: String,
    /// Signing software version (Cargo package version).
    pub software_version: String,
    /// RFC 3339 timestamp at the moment the manifest was assembled.
    pub timestamp: String,
}

/// A fully-detached signed manifest, suitable for shipping alongside the
/// rendered output as `<output>.lumen-cco.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedManifest {
    /// The signed manifest body.
    pub manifest: Manifest,
    /// Lowercase hex-encoded Ed25519 signature (64 bytes -> 128 hex chars).
    pub signature_hex: String,
    /// Lowercase hex-encoded Ed25519 public key (32 bytes -> 64 hex chars).
    pub public_key_hex: String,
}

// ---------------------------------------------------------------------------
// Manifest building
// ---------------------------------------------------------------------------

/// Build a [`Manifest`] from the three on-disk artefacts of a render.
///
/// Hashes input/output/recipe with BLAKE3, auto-fills `software`,
/// `software_version`, `timestamp`, and assigns a fresh UUIDv7
/// `render_id`.
pub fn build_manifest<P, Q, R>(input_path: P, output_path: Q, recipe_path: R) -> Result<Manifest>
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
    R: AsRef<Path>,
{
    let input_hash = hash_file(input_path.as_ref())?;
    let output_hash = hash_file(output_path.as_ref())?;
    let recipe_hash = hash_file(recipe_path.as_ref())?;

    let timestamp = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .map_err(|e| Error::Other(format!("rfc3339 format: {e}")))?;

    Ok(Manifest {
        schema: MANIFEST_SCHEMA.to_string(),
        version: MANIFEST_VERSION,
        render_id: Uuid::now_v7(),
        input_hash,
        output_hash,
        recipe_hash,
        software: MANIFEST_SOFTWARE.to_string(),
        software_version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp,
    })
}

fn hash_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).map_err(|e| {
        Error::Io(std::io::Error::new(
            e.kind(),
            format!("hash_file({}): {e}", path.display()),
        ))
    })?;
    let h = blake3::hash(&bytes);
    Ok(format!("blake3:{}", h.to_hex()))
}

// ---------------------------------------------------------------------------
// Canonicalisation
// ---------------------------------------------------------------------------

/// Produce the canonical byte sequence to sign: sorted-key JSON of the
/// manifest, with no insignificant whitespace.
///
/// This is deterministic across runs and platforms — that's the whole
/// point. Any change to the manifest contents (or to its serialised
/// shape) produces a different byte string and therefore a different
/// signature.
pub fn manifest_canonical_bytes(m: &Manifest) -> Vec<u8> {
    // Round-trip through `serde_json::Value`, then re-serialise with
    // sorted keys. `serde_json` orders BTreeMap keys lexicographically,
    // so `Value::Object(BTreeMap-backed)` gives us the canonical form.
    //
    // We unwrap here only on internal serde failures, which would mean
    // `Manifest` itself (a fixed struct of strings/numbers/Uuid) failed
    // to serialise — that would be a programmer error in this crate, not
    // a runtime input problem.
    let value: Value = serde_json::to_value(m).expect("Manifest is always serialisable");
    let canonical = canonicalise_value(value);
    serde_json::to_vec(&canonical).expect("canonical Value is always serialisable")
}

/// Recursively re-key any `Object` so its entries are emitted in sorted
/// order. `serde_json::Value::Object` is already a `Map<String, Value>`
/// which preserves insertion order; we rebuild it here so the output is
/// canonical regardless of struct field order.
fn canonicalise_value(v: Value) -> Value {
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
        Value::Array(items) => {
            Value::Array(items.into_iter().map(canonicalise_value).collect())
        }
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Signing / verification
// ---------------------------------------------------------------------------

/// Sign a manifest's canonical bytes with the given Ed25519 signing key.
pub fn sign(m: &Manifest, signing_key: &SigningKey) -> Signature {
    let bytes = manifest_canonical_bytes(m);
    signing_key.sign(&bytes)
}

/// Verify a manifest's signature against the given Ed25519 public key.
///
/// Returns `false` for any failure (mismatched signature, tampered
/// manifest, wrong key) — callers don't need to distinguish *why*
/// verification failed for the security boundary, only *whether*.
pub fn verify(m: &Manifest, sig: &Signature, verifying_key: &VerifyingKey) -> bool {
    let bytes = manifest_canonical_bytes(m);
    verifying_key.verify_strict(&bytes, sig).is_ok()
}

/// Generate a fresh Ed25519 keypair using the OS RNG.
///
/// Convenience for tests and one-off CLI flows. Production deployments
/// should load keys from a managed store rather than calling this.
pub fn keypair_generate() -> (SigningKey, VerifyingKey) {
    let mut rng = rand_core::OsRng;
    let signing_key = SigningKey::generate(&mut rng);
    let verifying_key = signing_key.verifying_key();
    (signing_key, verifying_key)
}

// ---------------------------------------------------------------------------
// Sidecar I/O
// ---------------------------------------------------------------------------

/// Atomically write a [`SignedManifest`] to disk as pretty-printed JSON.
///
/// Writes to a sibling `<path>.tmp` file then renames into place, so a
/// crash mid-write can never leave a half-written sidecar in the
/// canonical location.
pub fn save_signed<P: AsRef<Path>>(s: &SignedManifest, path: P) -> Result<()> {
    let path = path.as_ref();
    let json = serde_json::to_vec_pretty(s)?;

    let tmp = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => {
            let mut p = parent.to_path_buf();
            let stem = path
                .file_name()
                .ok_or_else(|| Error::Other(format!("invalid path: {}", path.display())))?;
            let mut name = stem.to_owned();
            name.push(".tmp");
            p.push(name);
            p
        }
        _ => {
            let mut name = path
                .file_name()
                .ok_or_else(|| Error::Other(format!("invalid path: {}", path.display())))?
                .to_owned();
            name.push(".tmp");
            std::path::PathBuf::from(name)
        }
    };

    std::fs::write(&tmp, &json)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Load a [`SignedManifest`] from a JSON file on disk.
pub fn load_signed<P: AsRef<Path>>(path: P) -> Result<SignedManifest> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    let s: SignedManifest = serde_json::from_slice(&bytes)?;
    Ok(s)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    /// Build three small artefacts in a temp dir and return their paths
    /// plus the dir guard (drop = cleanup).
    fn make_artifacts() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
        let dir = tempdir().expect("tempdir");
        let input = dir.path().join("input.bin");
        let output = dir.path().join("output.bin");
        let recipe = dir.path().join("recipe.json");
        fs::write(&input, b"input-bytes").unwrap();
        fs::write(&output, b"output-bytes-rendered").unwrap();
        fs::write(&recipe, br#"{"effect":"denoise","strength":0.5}"#).unwrap();
        (dir, input, output, recipe)
    }

    #[test]
    fn round_trip_build_sign_save_load_verify() {
        let (_dir, input, output, recipe) = make_artifacts();
        let manifest = build_manifest(&input, &output, &recipe).expect("build");

        // Sanity: hash format and auto-fields.
        assert!(manifest.input_hash.starts_with("blake3:"));
        assert!(manifest.output_hash.starts_with("blake3:"));
        assert!(manifest.recipe_hash.starts_with("blake3:"));
        assert_eq!(manifest.schema, MANIFEST_SCHEMA);
        assert_eq!(manifest.version, MANIFEST_VERSION);
        assert_eq!(manifest.software, "lumen");
        assert_eq!(manifest.software_version, env!("CARGO_PKG_VERSION"));
        assert!(!manifest.timestamp.is_empty());

        let (sk, vk) = keypair_generate();
        let sig = sign(&manifest, &sk);
        assert!(verify(&manifest, &sig, &vk));

        let signed = SignedManifest {
            manifest: manifest.clone(),
            signature_hex: hex_encode(sig.to_bytes()),
            public_key_hex: hex_encode(vk.to_bytes()),
        };

        let sidecar = _dir.path().join("output.bin.lumen-cco.json");
        save_signed(&signed, &sidecar).expect("save");
        assert!(sidecar.exists());

        let loaded = load_signed(&sidecar).expect("load");
        assert_eq!(loaded, signed);

        // Reconstruct signature + key from hex and re-verify.
        let sig_bytes = hex_decode(&loaded.signature_hex).expect("sig hex");
        let sig_arr: [u8; Signature::BYTE_SIZE] = sig_bytes
            .as_slice()
            .try_into()
            .expect("sig length");
        let sig2 = Signature::from_bytes(&sig_arr);

        let vk_bytes = hex_decode(&loaded.public_key_hex).expect("vk hex");
        let vk_arr: [u8; 32] = vk_bytes.as_slice().try_into().expect("vk length");
        let vk2 = VerifyingKey::from_bytes(&vk_arr).expect("vk from bytes");

        assert!(verify(&loaded.manifest, &sig2, &vk2));
    }

    #[test]
    fn tampered_manifest_fails_verification() {
        let (_dir, input, output, recipe) = make_artifacts();
        let manifest = build_manifest(&input, &output, &recipe).expect("build");
        let (sk, vk) = keypair_generate();
        let sig = sign(&manifest, &sk);
        assert!(verify(&manifest, &sig, &vk));

        // Mutate any field and re-verify with the same signature.
        let mut tampered = manifest.clone();
        tampered.output_hash = "blake3:0000000000000000000000000000000000000000000000000000000000000000".to_string();
        assert!(!verify(&tampered, &sig, &vk));

        let mut tampered2 = manifest.clone();
        tampered2.software = "not-lumen".to_string();
        assert!(!verify(&tampered2, &sig, &vk));
    }

    #[test]
    fn wrong_public_key_fails_verification() {
        let (_dir, input, output, recipe) = make_artifacts();
        let manifest = build_manifest(&input, &output, &recipe).expect("build");
        let (sk, _vk) = keypair_generate();
        let (_sk_other, vk_other) = keypair_generate();
        let sig = sign(&manifest, &sk);
        assert!(!verify(&manifest, &sig, &vk_other));
    }

    #[test]
    fn canonical_bytes_are_deterministic() {
        let (_dir, input, output, recipe) = make_artifacts();
        let manifest = build_manifest(&input, &output, &recipe).expect("build");

        // Multiple canonicalisations of the same manifest produce identical bytes.
        let a = manifest_canonical_bytes(&manifest);
        let b = manifest_canonical_bytes(&manifest);
        let c = manifest_canonical_bytes(&manifest);
        assert_eq!(a, b);
        assert_eq!(b, c);

        // Round-trip through serde_json::Value (which is HashMap-ordered
        // on insertion but our canonicaliser sorts keys) and back must
        // match the canonical form.
        let v: Value = serde_json::from_slice(&a).expect("canonical parses");
        let again = serde_json::to_vec(&canonicalise_value(v)).expect("re-canon");
        assert_eq!(a, again);

        // And keys really are sorted: the first few field names in the
        // canonical bytes appear in lexicographic order.
        let s = std::str::from_utf8(&a).expect("utf8");
        let i_input = s.find("\"input_hash\"").expect("input_hash present");
        let i_output = s.find("\"output_hash\"").expect("output_hash present");
        let i_recipe = s.find("\"recipe_hash\"").expect("recipe_hash present");
        let i_render = s.find("\"render_id\"").expect("render_id present");
        let i_schema = s.find("\"schema\"").expect("schema present");
        let i_software = s.find("\"software\"").expect("software present");
        let i_timestamp = s.find("\"timestamp\"").expect("timestamp present");
        let i_version = s.find("\"version\"").expect("version present");
        // input_hash < output_hash < recipe_hash < render_id < schema <
        // software < software_version < timestamp < version
        assert!(i_input < i_output);
        assert!(i_output < i_recipe);
        assert!(i_recipe < i_render);
        assert!(i_render < i_schema);
        assert!(i_schema < i_software);
        assert!(i_software < i_timestamp);
        assert!(i_timestamp < i_version);
    }

    // -- tiny hex helpers, kept private to tests ---------------------------

    fn hex_encode<T: AsRef<[u8]>>(b: T) -> String {
        let mut s = String::with_capacity(b.as_ref().len() * 2);
        for byte in b.as_ref() {
            s.push_str(&format!("{byte:02x}"));
        }
        s
    }

    fn hex_decode(s: &str) -> std::result::Result<Vec<u8>, &'static str> {
        if !s.len().is_multiple_of(2) {
            return Err("odd hex length");
        }
        let mut out = Vec::with_capacity(s.len() / 2);
        for i in (0..s.len()).step_by(2) {
            let byte = u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| "bad hex")?;
            out.push(byte);
        }
        Ok(out)
    }
}
