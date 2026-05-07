//! # lumen-platform
//!
//! Platform & distribution: installers, updates, licensing.
//!
//! ## Cat 29 — Phase 1
//!
//! This crate provides three building blocks for shipping Lumen:
//!
//! 1. **Platform detection** — best-effort machine introspection (OS, arch,
//!    CPU count, memory) for diagnostics and feature gating. See
//!    [`detect_platform`].
//! 2. **License issuance & verification** — Ed25519-signed [`License`]
//!    descriptors with edition tier, expiry, and feature flags. Signing
//!    re-uses the sorted-key JSON canonicalisation pattern from
//!    `lumen-auth`. See [`issue_license`], [`verify_license`],
//!    [`license_has_feature`].
//! 3. **Offline update check** — read a static manifest JSON from disk
//!    and compare against the compiled-in `CARGO_PKG_VERSION`. See
//!    [`check_updates_offline`].
//!
//! Phase 1 is intentionally network-free. Real online updates and key
//! distribution land in Phase 5 alongside the Tauri auto-update plugin.
//!
//! ## Quick example
//!
//! ```no_run
//! use std::collections::BTreeSet;
//! use lumen_auth::keypair_generate;
//! use lumen_platform::{
//!     detect_platform, issue_license, verify_license, license_has_feature,
//!     License, LicenseEdition,
//! };
//!
//! # fn run() -> lumen_core::Result<()> {
//! let _info = detect_platform();
//!
//! let (sk, vk) = keypair_generate();
//! let mut features = BTreeSet::new();
//! features.insert("ai-upscale".to_string());
//! let license = License {
//!     customer_id: "cust-001".to_string(),
//!     product: "lumen".to_string(),
//!     edition: LicenseEdition::Pro,
//!     expires: None,
//!     features,
//! };
//! let signed = issue_license(&license, &sk)?;
//! verify_license(&signed, &vk)?;
//! assert!(license_has_feature(&signed, "ai-upscale"));
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, rust_2018_idioms)]

use std::collections::BTreeSet;
use std::path::Path;

use ed25519_dalek::Signer;
use lumen_auth::{Signature, SigningKey, VerifyingKey};
use lumen_core::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use time::OffsetDateTime;

/// Crate-level version string surfaced for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifier used in logs and telemetry.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

// ---------------------------------------------------------------------------
// Platform detection
// ---------------------------------------------------------------------------

/// Best-effort snapshot of the host machine.
///
/// Fields are filled from `std::env::consts`, `std::thread`, and (for
/// memory) the `sysinfo` crate. The GPU adapter name is reserved for a
/// future `wgpu` probe; in Phase 1 it is always [`None`] to keep this
/// crate free of GPU dependencies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlatformInfo {
    /// Target OS, e.g. `"macos"`, `"linux"`, `"windows"`. From
    /// [`std::env::consts::OS`].
    pub os: &'static str,
    /// Target architecture, e.g. `"x86_64"`, `"aarch64"`. From
    /// [`std::env::consts::ARCH`].
    pub arch: &'static str,
    /// Available parallelism / logical CPU count. Always `>= 1`.
    pub cpu_count: usize,
    /// Total system memory in bytes, if `sysinfo` could probe it.
    pub total_memory_bytes: Option<u64>,
    /// Name of the active GPU adapter as reported by `wgpu`, when
    /// probed. Phase 1 returns [`None`] unconditionally.
    pub gpu_adapter_name: Option<String>,
}

/// Probe the host and return a [`PlatformInfo`].
///
/// Never fails: every field falls back to a sensible default (e.g.
/// `cpu_count = 1` if `available_parallelism` errors). The intent is
/// diagnostic, not authoritative — callers should not gate critical
/// behaviour on this output.
pub fn detect_platform() -> PlatformInfo {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    let cpu_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let total_memory_bytes = {
        use sysinfo::System;
        let mut sys = System::new();
        sys.refresh_memory();
        let mem = sys.total_memory();
        if mem == 0 {
            None
        } else {
            Some(mem)
        }
    };

    PlatformInfo {
        os,
        arch,
        cpu_count,
        total_memory_bytes,
        gpu_adapter_name: None,
    }
}

// ---------------------------------------------------------------------------
// Licensing
// ---------------------------------------------------------------------------

/// Edition tier of a [`License`].
///
/// Ordered roughly by capability: `Community < Pro < Studio < Enterprise`.
/// The string representation in JSON is lower-case (`"community"`, `"pro"`,
/// `"studio"`, `"enterprise"`) so wire-format stays stable across
/// case-sensitive consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LicenseEdition {
    /// Free tier.
    Community,
    /// Paid tier for individual professionals.
    Pro,
    /// Paid tier for studios and small teams.
    Studio,
    /// Paid tier for enterprises and large deployments.
    Enterprise,
}

/// A license body to be signed. Field set is intentionally minimal —
/// add fields cautiously since they become part of the canonical
/// signing payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct License {
    /// Stable customer identifier (issued by Lumen's licensing server).
    pub customer_id: String,
    /// Product the license applies to, e.g. `"lumen"`, `"lumen-server"`.
    pub product: String,
    /// Edition tier.
    pub edition: LicenseEdition,
    /// Optional expiry. `None` means a perpetual license.
    #[serde(default, with = "time::serde::rfc3339::option")]
    pub expires: Option<OffsetDateTime>,
    /// Feature flags this license unlocks. Sorted (`BTreeSet`) for
    /// deterministic canonicalisation.
    pub features: BTreeSet<String>,
}

/// A signed [`License`], suitable for distribution as a small JSON file
/// alongside the installer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedLicense {
    /// The signed license body.
    pub license: License,
    /// Lowercase hex-encoded Ed25519 signature.
    pub signature_hex: String,
    /// Lowercase hex-encoded Ed25519 public key the signature verifies
    /// against. Stored for diagnostics; production verification should
    /// always pass an *expected* public key explicitly.
    pub public_key_hex: String,
}

/// Produce the canonical byte sequence to sign for a [`License`].
///
/// Sorted-key JSON, no insignificant whitespace — same approach as
/// `lumen_auth::manifest_canonical_bytes`. Determinism is the goal;
/// any change to the license body (or its serialised shape) yields
/// different bytes and therefore a different signature.
fn license_canonical_bytes(l: &License) -> Vec<u8> {
    let value: Value = serde_json::to_value(l).expect("License is always serialisable");
    let canonical = canonicalise_value(value);
    serde_json::to_vec(&canonical).expect("canonical Value is always serialisable")
}

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
        Value::Array(items) => Value::Array(items.into_iter().map(canonicalise_value).collect()),
        other => other,
    }
}

fn hex_encode<T: AsRef<[u8]>>(b: T) -> String {
    let mut s = String::with_capacity(b.as_ref().len() * 2);
    for byte in b.as_ref() {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

fn hex_decode(s: &str) -> std::result::Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) {
        return Err("odd hex length".to_string());
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    for i in (0..s.len()).step_by(2) {
        let byte = u8::from_str_radix(&s[i..i + 2], 16)
            .map_err(|e| format!("bad hex byte at {i}: {e}"))?;
        out.push(byte);
    }
    Ok(out)
}

/// Sign a [`License`] and bundle the signature with the body in a
/// [`SignedLicense`].
///
/// The canonical-bytes step matches `lumen_auth::sign`: serialise to
/// JSON, sort keys recursively, sign that byte sequence with Ed25519.
pub fn issue_license(license: &License, signing_key: &SigningKey) -> Result<SignedLicense> {
    let bytes = license_canonical_bytes(license);
    let signature: Signature = signing_key.sign(&bytes);
    let verifying_key = signing_key.verifying_key();

    Ok(SignedLicense {
        license: license.clone(),
        signature_hex: hex_encode(signature.to_bytes()),
        public_key_hex: hex_encode(verifying_key.to_bytes()),
    })
}

/// Verify a [`SignedLicense`] against an *expected* public key.
///
/// Returns `Err(Error::Other(...))` with a clear message for any
/// failure mode (tampered body, wrong key, malformed hex, expired). The
/// expected public key is supplied by the caller — never trusted from
/// the signed payload itself.
pub fn verify_license(signed: &SignedLicense, expected_pubkey: &VerifyingKey) -> Result<()> {
    // Decode signature.
    let sig_bytes = hex_decode(&signed.signature_hex)
        .map_err(|e| Error::Other(format!("license signature hex invalid: {e}")))?;
    let sig_arr: [u8; Signature::BYTE_SIZE] = sig_bytes
        .as_slice()
        .try_into()
        .map_err(|_| Error::Other("license signature wrong length".to_string()))?;
    let signature = Signature::from_bytes(&sig_arr);

    // Verify against the expected public key (NOT the bundled one).
    let bytes = license_canonical_bytes(&signed.license);
    expected_pubkey
        .verify_strict(&bytes, &signature)
        .map_err(|e| Error::Other(format!("license signature verification failed: {e}")))?;

    // Check expiry, if present.
    if let Some(exp) = signed.license.expires {
        let now = OffsetDateTime::now_utc();
        if now >= exp {
            return Err(Error::Other(format!(
                "license expired at {exp} (now is {now})"
            )));
        }
    }

    Ok(())
}

/// Convenience: does this signed license advertise `feature`?
///
/// This does **not** verify the signature — call [`verify_license`]
/// first if the source of `signed` is untrusted.
pub fn license_has_feature(signed: &SignedLicense, feature: &str) -> bool {
    signed.license.features.contains(feature)
}

// ---------------------------------------------------------------------------
// Offline update check
// ---------------------------------------------------------------------------

/// Result of an update check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpdateInfo {
    /// Currently-running version (compile-time `CARGO_PKG_VERSION`).
    pub current_version: &'static str,
    /// Latest version reported by the manifest, if any.
    pub latest_version: Option<String>,
    /// Download URL for the latest version, if any.
    pub download_url: Option<String>,
}

impl UpdateInfo {
    /// Convenience: does the manifest advertise a version that differs
    /// from the running one?
    pub fn update_available(&self) -> bool {
        match &self.latest_version {
            Some(v) => v != self.current_version,
            None => false,
        }
    }
}

/// Read a static update manifest from disk and compare against the
/// compiled-in version.
///
/// The manifest is JSON of the shape:
///
/// ```json
/// { "latest_version": "0.2.0", "download_url": "https://..." }
/// ```
///
/// Both fields are optional. Phase 1 makes no network calls; real
/// online update checks land in Phase 5 alongside the Tauri auto-update
/// plugin.
pub fn check_updates_offline(static_manifest_path: &Path) -> Result<UpdateInfo> {
    let bytes = std::fs::read(static_manifest_path).map_err(|e| {
        Error::Io(std::io::Error::new(
            e.kind(),
            format!(
                "check_updates_offline({}): {e}",
                static_manifest_path.display()
            ),
        ))
    })?;

    #[derive(Deserialize)]
    struct Manifest {
        #[serde(default)]
        latest_version: Option<String>,
        #[serde(default)]
        download_url: Option<String>,
    }

    let manifest: Manifest = serde_json::from_slice(&bytes)?;

    Ok(UpdateInfo {
        current_version: env!("CARGO_PKG_VERSION"),
        latest_version: manifest.latest_version,
        download_url: manifest.download_url,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_auth::keypair_generate;
    use std::fs;
    use tempfile::tempdir;
    use time::Duration;

    fn make_license(features: &[&str], expires: Option<OffsetDateTime>) -> License {
        let mut feats = BTreeSet::new();
        for f in features {
            feats.insert((*f).to_string());
        }
        License {
            customer_id: "cust-001".to_string(),
            product: "lumen".to_string(),
            edition: LicenseEdition::Pro,
            expires,
            features: feats,
        }
    }

    #[test]
    fn detect_platform_returns_sensible_values() {
        let info = detect_platform();
        // OS string is one of the well-known targets.
        assert!(
            matches!(info.os, "macos" | "linux" | "windows" | "ios" | "android"),
            "unexpected os: {}",
            info.os
        );
        assert!(!info.arch.is_empty(), "arch should not be empty");
        assert!(info.cpu_count >= 1, "cpu_count must be >= 1");
        // total_memory_bytes is best-effort; if Some, it should be plausible.
        if let Some(mem) = info.total_memory_bytes {
            // Any modern host has >= 1 MiB; this guards against a 0
            // sentinel sneaking through.
            assert!(mem >= 1024 * 1024, "memory implausibly small: {mem}");
        }
        // GPU adapter name is None in Phase 1.
        assert!(info.gpu_adapter_name.is_none());
    }

    #[test]
    fn license_round_trip_issue_then_verify() {
        let (sk, vk) = keypair_generate();
        let license = make_license(&["ai-upscale", "denoise"], None);

        let signed = issue_license(&license, &sk).expect("issue");
        verify_license(&signed, &vk).expect("verify");

        assert!(license_has_feature(&signed, "ai-upscale"));
        assert!(license_has_feature(&signed, "denoise"));
        assert!(!license_has_feature(&signed, "missing-feature"));
    }

    #[test]
    fn tampered_license_fails_verification() {
        let (sk, vk) = keypair_generate();
        let license = make_license(&["ai-upscale"], None);
        let mut signed = issue_license(&license, &sk).expect("issue");

        // Mutate the license body after signing.
        signed.license.edition = LicenseEdition::Enterprise;
        let err = verify_license(&signed, &vk).expect_err("must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("verification failed"),
            "unexpected error message: {msg}"
        );

        // Mutate features and re-check with a fresh issuance.
        let signed2 = issue_license(&license, &sk).expect("issue2");
        let mut tampered = signed2.clone();
        tampered.license.features.insert("studio-only".to_string());
        let err2 = verify_license(&tampered, &vk).expect_err("must fail");
        assert!(format!("{err2}").contains("verification failed"));
    }

    #[test]
    fn expired_license_fails_with_clear_message() {
        let (sk, vk) = keypair_generate();
        let past = OffsetDateTime::now_utc() - Duration::days(1);
        let license = make_license(&["ai-upscale"], Some(past));
        let signed = issue_license(&license, &sk).expect("issue");

        let err = verify_license(&signed, &vk).expect_err("must fail (expired)");
        let msg = format!("{err}");
        assert!(
            msg.contains("expired"),
            "expected 'expired' in error, got: {msg}"
        );
    }

    #[test]
    fn future_expiry_passes_verification() {
        let (sk, vk) = keypair_generate();
        let future = OffsetDateTime::now_utc() + Duration::days(30);
        let license = make_license(&["ai-upscale"], Some(future));
        let signed = issue_license(&license, &sk).expect("issue");
        verify_license(&signed, &vk).expect("future-expiry license should verify");
    }

    #[test]
    fn wrong_public_key_fails_verification() {
        let (sk, _vk) = keypair_generate();
        let (_sk_other, vk_other) = keypair_generate();
        let license = make_license(&["ai-upscale"], None);
        let signed = issue_license(&license, &sk).expect("issue");

        let err = verify_license(&signed, &vk_other).expect_err("must fail");
        assert!(format!("{err}").contains("verification failed"));
    }

    #[test]
    fn check_updates_offline_reports_mismatch() {
        let dir = tempdir().expect("tempdir");
        let manifest_path = dir.path().join("update.json");
        fs::write(
            &manifest_path,
            br#"{"latest_version":"99.99.99","download_url":"https://example.com/lumen.dmg"}"#,
        )
        .unwrap();

        let info = check_updates_offline(&manifest_path).expect("read manifest");
        assert_eq!(info.current_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(info.latest_version.as_deref(), Some("99.99.99"));
        assert_eq!(
            info.download_url.as_deref(),
            Some("https://example.com/lumen.dmg")
        );
        assert!(
            info.update_available(),
            "version mismatch should report update"
        );
    }

    #[test]
    fn check_updates_offline_no_update_when_versions_match() {
        let dir = tempdir().expect("tempdir");
        let manifest_path = dir.path().join("update.json");
        let body = format!(
            r#"{{"latest_version":"{}","download_url":null}}"#,
            env!("CARGO_PKG_VERSION")
        );
        fs::write(&manifest_path, body.as_bytes()).unwrap();

        let info = check_updates_offline(&manifest_path).expect("read manifest");
        assert_eq!(
            info.latest_version.as_deref(),
            Some(env!("CARGO_PKG_VERSION"))
        );
        assert!(!info.update_available());
    }

    #[test]
    fn check_updates_offline_handles_partial_manifest() {
        let dir = tempdir().expect("tempdir");
        let manifest_path = dir.path().join("update.json");
        // Empty object — both fields omitted.
        fs::write(&manifest_path, b"{}").unwrap();

        let info = check_updates_offline(&manifest_path).expect("read manifest");
        assert_eq!(info.latest_version, None);
        assert_eq!(info.download_url, None);
        assert!(!info.update_available());
    }
}
