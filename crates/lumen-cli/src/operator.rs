//! Operator identity — `~/.lumen/operator.json`.
//!
//! Forensic and law-enforcement workflows require every render to be
//! attributable to a specific person. Lumen stores a single operator
//! identity per user account: an Ed25519 signing keypair plus
//! identifying metadata (display name, agency, badge / employee ID).
//!
//! The keypair is generated locally (offline; never transmitted) and
//! lives at `~/.lumen/operator.json` with mode 0600. Every signed
//! manifest, every audit-log entry, and every forensic report is
//! signed with this key, so a reviewer with the operator's published
//! public key can verify the operator's involvement.
//!
//! Multi-operator setups (where a department issues separate keys)
//! work by setting `LUMEN_OPERATOR` env var to a different file path.

use std::path::PathBuf;

use anyhow::{anyhow, Context as _, Result};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

/// Public-facing identity. Distributed alongside outputs / reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorIdentity {
    pub display_name: String,
    pub agency: String,
    /// Badge number / employee ID / personnel identifier. Free-form.
    pub identifier: String,
    pub public_key_hex: String,
    #[serde(with = "time::serde::rfc3339")]
    pub created: OffsetDateTime,
}

/// On-disk operator file. Holds the secret key — keep it local.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorFile {
    pub schema: String,
    pub identity: OperatorIdentity,
    pub secret_key_hex: String,
}

const SCHEMA: &str = "lumen.operator/v1";

pub fn operator_path() -> PathBuf {
    if let Ok(p) = std::env::var("LUMEN_OPERATOR") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home).join(".lumen").join("operator.json")
}

/// Generate a new keypair and write it to disk. Errors if a file
/// already exists at the target path (use `--force` to overwrite).
pub fn init(
    display_name: &str,
    agency: &str,
    identifier: &str,
    force: bool,
) -> Result<OperatorIdentity> {
    let path = operator_path();
    if path.exists() && !force {
        return Err(anyhow!(
            "operator file already exists at {}. Use --force to overwrite (this destroys the existing key — back it up first).",
            path.display()
        ));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("creating {}", parent.display())
        })?;
    }
    let (sk, vk) = lumen_auth::keypair_generate();
    let identity = OperatorIdentity {
        display_name: display_name.to_string(),
        agency: agency.to_string(),
        identifier: identifier.to_string(),
        public_key_hex: hex(&vk.to_bytes()),
        created: OffsetDateTime::now_utc(),
    };
    let file = OperatorFile {
        schema: SCHEMA.to_string(),
        identity: identity.clone(),
        secret_key_hex: hex(&sk.to_bytes()),
    };
    let s = serde_json::to_string_pretty(&file)?;
    std::fs::write(&path, s)?;
    set_secure_perms(&path)?;
    Ok(identity)
}

pub fn load() -> Result<OperatorFile> {
    let path = operator_path();
    let s = std::fs::read_to_string(&path)
        .with_context(|| format!("reading operator file at {}", path.display()))?;
    let f: OperatorFile = serde_json::from_str(&s)?;
    if f.schema != SCHEMA {
        return Err(anyhow!(
            "operator file schema mismatch (expected {}, got {})",
            SCHEMA,
            f.schema
        ));
    }
    Ok(f)
}

pub fn current_identity() -> Result<OperatorIdentity> {
    Ok(load()?.identity)
}

/// Decode hex secret key, build the SigningKey.
pub fn signing_key_from_file(file: &OperatorFile) -> Result<lumen_auth::SigningKey> {
    let bytes = unhex(&file.secret_key_hex)
        .ok_or_else(|| anyhow!("operator file has malformed secret_key_hex"))?;
    if bytes.len() != 32 {
        return Err(anyhow!("operator secret key must be 32 bytes"));
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Ok(lumen_auth::SigningKey::from_bytes(&arr))
}

/// Decode hex public key into a VerifyingKey.
pub fn verifying_key_from_hex(hex_str: &str) -> Result<lumen_auth::VerifyingKey> {
    let bytes = unhex(hex_str)
        .ok_or_else(|| anyhow!("malformed public_key_hex"))?;
    if bytes.len() != 32 {
        return Err(anyhow!("public key must be 32 bytes"));
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    lumen_auth::VerifyingKey::from_bytes(&arr)
        .map_err(|e| anyhow!("invalid public key: {e}"))
}

#[cfg(unix)]
fn set_secure_perms(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)?.permissions();
    perms.set_mode(0o600);
    std::fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_secure_perms(_path: &std::path::Path) -> Result<()> { Ok(()) }

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) { return None; }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}
