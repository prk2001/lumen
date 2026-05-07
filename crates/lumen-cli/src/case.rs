//! Forensic case management — case folder + hash-chained audit log.
//!
//! A "case" is a directory containing:
//!
//! - `case.json` — case metadata: ID, evidence ID, name, agency,
//!   created-by operator identity, original input hash.
//! - `audit.jsonl` — append-only newline-delimited JSON. Each entry
//!   is signed with the operator's Ed25519 key AND chains to the
//!   previous entry's signature. Tampering with any past entry
//!   invalidates the chain from that point forward.
//! - `inputs/`, `outputs/`, `stages/`, `recipes/`, `manifests/` — the
//!   actual artifacts. Never modified after creation.
//! - `reports/` — generated forensic-grade HTML reports.
//!
//! The whole case folder zips into a single tamper-evident package
//! via `lumen case export`.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context as _, Result};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::operator::{self, OperatorIdentity};

const CASE_SCHEMA: &str = "lumen.case/v1";
const ENTRY_SCHEMA: &str = "lumen.audit-entry/v1";
/// Genesis "previous-signature" — every audit log starts here.
const GENESIS: &str = "blake3:0000000000000000000000000000000000000000000000000000000000000000";

/// Top-level case metadata stored in `case.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseMetadata {
    pub schema: String,
    /// Stable case identifier — UUID v7.
    pub case_uuid: Uuid,
    /// User-supplied case ID (e.g. "2026-CCTV-LOTB").
    pub case_id: String,
    /// User-supplied evidence ID (e.g. "EVD-2026-7842").
    pub evidence_id: String,
    pub case_name: String,
    pub agency: String,
    /// Original input file's BLAKE3 hash; recorded at intake.
    pub original_input_hash: Option<String>,
    pub created_by: OperatorIdentity,
    #[serde(with = "time::serde::rfc3339")]
    pub created: OffsetDateTime,
}

/// One append-only audit log entry. Entries form a hash chain via
/// `prev_entry_signature_hex`, mirroring how Git commits chain SHAs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub schema: String,
    pub seq: u64,
    #[serde(with = "time::serde::rfc3339")]
    pub at: OffsetDateTime,
    /// Operator pubkey hex — answers "who".
    pub operator_public_key_hex: String,
    /// Action verb: "intake" / "render" / "export" / "verify" / "note".
    pub action: String,
    /// Free-form description for humans.
    pub note: String,
    /// Optional hashes of relevant artifacts.
    pub input_hash: Option<String>,
    pub output_hash: Option<String>,
    pub recipe_hash: Option<String>,
    /// Hex Ed25519 signature of `(prev_entry_signature || canonical entry bytes)`.
    pub prev_entry_signature_hex: String,
    pub entry_signature_hex: String,
}

#[derive(Debug, Default, Clone)]
pub struct AuditEntryDraft {
    pub action: String,
    pub note: String,
    pub input_hash: Option<String>,
    pub output_hash: Option<String>,
    pub recipe_hash: Option<String>,
}

pub fn case_metadata_path(case_dir: &Path) -> PathBuf {
    case_dir.join("case.json")
}
pub fn audit_log_path(case_dir: &Path) -> PathBuf {
    case_dir.join("audit.jsonl")
}

pub fn init(
    case_dir: &Path,
    case_id: &str,
    evidence_id: &str,
    case_name: &str,
    agency: &str,
    original_input: Option<&Path>,
) -> Result<CaseMetadata> {
    let op = operator::load().context(
        "no operator identity. Run `lumen operator init` first.",
    )?;

    if case_metadata_path(case_dir).exists() {
        return Err(anyhow!(
            "case already initialized at {}",
            case_dir.display()
        ));
    }

    for sub in ["inputs", "outputs", "stages", "recipes", "manifests", "reports"] {
        std::fs::create_dir_all(case_dir.join(sub))?;
    }

    // Copy original input into inputs/ if supplied.
    let original_input_hash = match original_input {
        Some(p) => {
            let dst = case_dir.join("inputs").join(
                p.file_name().ok_or_else(|| anyhow!("input has no filename"))?,
            );
            std::fs::copy(p, &dst)?;
            Some(blake3_hex_of_file(&dst)?)
        }
        None => None,
    };

    let metadata = CaseMetadata {
        schema: CASE_SCHEMA.to_string(),
        case_uuid: Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)),
        case_id: case_id.to_string(),
        evidence_id: evidence_id.to_string(),
        case_name: case_name.to_string(),
        agency: agency.to_string(),
        original_input_hash: original_input_hash.clone(),
        created_by: op.identity.clone(),
        created: OffsetDateTime::now_utc(),
    };

    // Write case.json
    let s = serde_json::to_string_pretty(&metadata)?;
    std::fs::write(case_metadata_path(case_dir), s)?;

    // Touch audit.jsonl (empty file).
    std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(audit_log_path(case_dir))?;

    // Append the first entry: case-init.
    append_entry(
        case_dir,
        AuditEntryDraft {
            action: "case-init".to_string(),
            note: format!(
                "Case '{}' (Evidence {}) opened by {} ({})",
                case_id, evidence_id, op.identity.display_name, op.identity.agency
            ),
            input_hash: original_input_hash,
            output_hash: None,
            recipe_hash: None,
        },
    )?;

    Ok(metadata)
}

pub fn load_metadata(case_dir: &Path) -> Result<CaseMetadata> {
    let s = std::fs::read_to_string(case_metadata_path(case_dir))
        .with_context(|| format!("reading {}", case_metadata_path(case_dir).display()))?;
    let m: CaseMetadata = serde_json::from_str(&s)?;
    if m.schema != CASE_SCHEMA {
        return Err(anyhow!(
            "case schema mismatch (expected {CASE_SCHEMA}, got {})",
            m.schema
        ));
    }
    Ok(m)
}

/// Read every audit-log entry in order.
pub fn read_audit_log(case_dir: &Path) -> Result<Vec<AuditEntry>> {
    let s = std::fs::read_to_string(audit_log_path(case_dir)).unwrap_or_default();
    let mut out = Vec::new();
    for line in s.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        out.push(serde_json::from_str::<AuditEntry>(trimmed)
            .with_context(|| format!("parsing audit entry: {trimmed}"))?);
    }
    Ok(out)
}

/// Append a new entry, signed with the local operator's key, chained
/// to the previous entry's signature.
pub fn append_entry(case_dir: &Path, draft: AuditEntryDraft) -> Result<AuditEntry> {
    let op = operator::load()?;
    let signing_key = operator::signing_key_from_file(&op)?;

    let mut entries = read_audit_log(case_dir)?;
    let prev = entries
        .last()
        .map(|e| e.entry_signature_hex.clone())
        .unwrap_or_else(|| GENESIS.to_string());
    let seq = entries.len() as u64;

    // Build a partially-populated entry with placeholder signature, then
    // canonicalize it for signing, then fill in the signature.
    let mut entry = AuditEntry {
        schema: ENTRY_SCHEMA.to_string(),
        seq,
        at: OffsetDateTime::now_utc(),
        operator_public_key_hex: op.identity.public_key_hex.clone(),
        action: draft.action,
        note: draft.note,
        input_hash: draft.input_hash,
        output_hash: draft.output_hash,
        recipe_hash: draft.recipe_hash,
        prev_entry_signature_hex: prev.clone(),
        entry_signature_hex: String::new(), // filled below
    };

    let signing_payload = canonical_signing_payload(&entry);
    use ed25519_dalek::Signer as _;
    let sig = signing_key.sign(&signing_payload);
    entry.entry_signature_hex = hex(&sig.to_bytes());

    // Append as a single JSON line.
    let line = serde_json::to_string(&entry)? + "\n";
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(audit_log_path(case_dir))?;
    f.write_all(line.as_bytes())?;

    entries.push(entry.clone());
    Ok(entry)
}

/// Verify the entire audit log: every entry's signature is valid AND
/// the chain links each entry back to the previous via
/// `prev_entry_signature_hex`. Returns the verified entries on success.
pub fn verify_audit_log(case_dir: &Path) -> Result<Vec<AuditEntry>> {
    let entries = read_audit_log(case_dir)?;
    let mut prev = GENESIS.to_string();
    for (i, entry) in entries.iter().enumerate() {
        if entry.prev_entry_signature_hex != prev {
            return Err(anyhow!(
                "audit log broken at seq {}: expected prev sig {} but got {}",
                entry.seq, prev, entry.prev_entry_signature_hex
            ));
        }
        if entry.seq != i as u64 {
            return Err(anyhow!(
                "audit log seq mismatch at index {i}: entry seq {}",
                entry.seq
            ));
        }
        let payload = canonical_signing_payload(entry);
        let pk = operator::verifying_key_from_hex(&entry.operator_public_key_hex)?;
        let sig_bytes = unhex(&entry.entry_signature_hex)
            .ok_or_else(|| anyhow!("seq {} sig is malformed hex", entry.seq))?;
        if sig_bytes.len() != 64 {
            return Err(anyhow!("seq {} sig wrong length", entry.seq));
        }
        let mut arr = [0u8; 64];
        arr.copy_from_slice(&sig_bytes);
        let sig = lumen_auth::Signature::from_bytes(&arr);
        use ed25519_dalek::Verifier as _;
        if pk.verify(&payload, &sig).is_err() {
            return Err(anyhow!(
                "seq {} signature does not verify against operator pubkey",
                entry.seq
            ));
        }
        prev = entry.entry_signature_hex.clone();
    }
    Ok(entries)
}

/// Sign-off summary: who has reviewed the case and what they decided.
/// `has_independent_approval` is true iff at least one `sign-off`
/// entry with decision == approve was signed by a pubkey *different*
/// from the operator who opened the case (the analyst). This is the
/// analyst-vs-reviewer separation that real forensic labs require —
/// you cannot self-approve.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SignoffStatus {
    pub signoff_count: usize,
    pub approvals: usize,
    pub rejections: usize,
    pub has_independent_approval: bool,
    pub independent_reviewers: Vec<String>, // pubkey hex
}

pub fn signoff_status(case_dir: &Path, metadata: &CaseMetadata) -> Result<SignoffStatus> {
    let entries = read_audit_log(case_dir)?;
    let analyst_pk = &metadata.created_by.public_key_hex;
    let mut count = 0;
    let mut approvals = 0;
    let mut rejections = 0;
    let mut independent: std::collections::BTreeSet<String> = Default::default();
    for e in entries {
        if e.action != "sign-off" {
            continue;
        }
        count += 1;
        let lower = e.note.to_lowercase();
        let approved = lower.contains("decision: approve");
        let rejected = lower.contains("decision: reject");
        if approved {
            approvals += 1;
        }
        if rejected {
            rejections += 1;
        }
        if approved && &e.operator_public_key_hex != analyst_pk {
            independent.insert(e.operator_public_key_hex.clone());
        }
    }
    Ok(SignoffStatus {
        signoff_count: count,
        approvals,
        rejections,
        has_independent_approval: !independent.is_empty(),
        independent_reviewers: independent.into_iter().collect(),
    })
}

/// Per-artifact strict-audit result. Every entry that referenced a
/// hash gets one row per hash; the row records the location searched
/// and whether a matching file was found.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StrictArtifactCheck {
    pub seq: u64,
    pub kind: &'static str,    // "input" | "output" | "recipe"
    pub claimed_hash: String,
    pub matched_path: Option<String>,
    pub ok: bool,
}

/// Strict audit result. The audit log signatures still must verify
/// (delegated to `verify_audit_log`); on top of that, every artifact
/// hash in the log must be backed by a real file in the case folder
/// whose BLAKE3 still matches.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StrictAuditReport {
    pub entries: Vec<AuditEntry>,
    pub artifacts: Vec<StrictArtifactCheck>,
    pub all_artifacts_match: bool,
}

/// `lumen case audit --strict`. Runs the regular signature-chain audit,
/// then re-hashes every file referenced by any entry's
/// `input_hash` / `output_hash` / `recipe_hash` and confirms the
/// recorded BLAKE3 still matches a file in the case folder. Returns
/// a per-artifact report so a reviewer can see exactly which file
/// failed (if any).
///
/// An artifact is considered matched if:
///   1. We can find a file in `inputs/`, `outputs/`, `outputs/stages/`,
///      `stages/`, or `recipes/` whose BLAKE3 == the recorded hash, OR
///   2. The recorded hash equals the hash of `inputs/<filename>` /
///      `outputs/<filename>` / `recipes/<filename>` for any file in
///      those subtrees (recursive).
///
/// Genesis-style "blake3:000…000" sentinels are ignored.
pub fn verify_audit_log_strict(case_dir: &Path) -> Result<StrictAuditReport> {
    let entries = verify_audit_log(case_dir)?; // chain still must hold
    let mut all_files: Vec<(String, std::path::PathBuf)> = Vec::new();
    for sub in ["inputs", "outputs", "stages", "recipes"] {
        let dir = case_dir.join(sub);
        if dir.is_dir() {
            collect_files_recursive(&dir, &mut all_files)?;
        }
    }

    let mut artifacts: Vec<StrictArtifactCheck> = Vec::new();
    let mut all_ok = true;
    for entry in &entries {
        for (kind, claimed) in [
            ("input", entry.input_hash.as_deref()),
            ("output", entry.output_hash.as_deref()),
            ("recipe", entry.recipe_hash.as_deref()),
        ] {
            let Some(claimed_hash) = claimed else { continue };
            // Skip genesis sentinel.
            if claimed_hash.contains("0000000000000000") {
                continue;
            }
            let matched = all_files
                .iter()
                .find(|(h, _)| h == claimed_hash)
                .map(|(_, p)| {
                    p.strip_prefix(case_dir)
                        .unwrap_or(p)
                        .display()
                        .to_string()
                });
            let ok = matched.is_some();
            if !ok {
                all_ok = false;
            }
            artifacts.push(StrictArtifactCheck {
                seq: entry.seq,
                kind,
                claimed_hash: claimed_hash.to_string(),
                matched_path: matched,
                ok,
            });
        }
    }

    Ok(StrictAuditReport {
        entries,
        artifacts,
        all_artifacts_match: all_ok,
    })
}

fn collect_files_recursive(
    dir: &Path,
    acc: &mut Vec<(String, std::path::PathBuf)>,
) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        let kind = entry.file_type()?;
        if kind.is_dir() {
            collect_files_recursive(&p, acc)?;
        } else if kind.is_file() {
            // Cheap path: file is small enough that hashing is fine.
            let h = blake3_hex_of_file(&p)?;
            acc.push((h, p));
        }
    }
    Ok(())
}

/// Canonical bytes used for signing. Stable across re-serializations
/// because all object keys are sorted.
fn canonical_signing_payload(entry: &AuditEntry) -> Vec<u8> {
    let mut copy = entry.clone();
    copy.entry_signature_hex.clear();
    let v = serde_json::to_value(&copy).unwrap();
    canonicalize_value(&v).into_bytes()
}

fn canonicalize_value(v: &serde_json::Value) -> String {
    use serde_json::Value;
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => serde_json::to_string(s).unwrap(),
        Value::Array(a) => {
            let parts: Vec<String> = a.iter().map(canonicalize_value).collect();
            format!("[{}]", parts.join(","))
        }
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let parts: Vec<String> = keys
                .into_iter()
                .map(|k| format!("{}:{}", serde_json::to_string(k).unwrap(), canonicalize_value(&m[k])))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
    }
}

/// Export a case as a tamper-evident zip file. The audit log proves
/// authorship + ordering; the zip is just a transport.
pub fn export_zip(case_dir: &Path, output_zip: &Path) -> Result<()> {
    use std::io::Read;
    use zip::write::FileOptions;
    let f = std::fs::File::create(output_zip)?;
    let mut zip = zip::ZipWriter::new(f);
    let opts: FileOptions<'_, ()> = FileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    fn add_dir(
        zip: &mut zip::ZipWriter<std::fs::File>,
        opts: &zip::write::FileOptions<'_, ()>,
        base: &Path,
        rel: &Path,
    ) -> Result<()> {
        let abs = base.join(rel);
        for entry in std::fs::read_dir(&abs)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            let entry_rel = rel.join(entry.file_name());
            if kind.is_dir() {
                let dir_name = format!("{}/", entry_rel.to_string_lossy());
                zip.add_directory(&dir_name, *opts)?;
                add_dir(zip, opts, base, &entry_rel)?;
            } else if kind.is_file() {
                let mut f = std::fs::File::open(entry.path())?;
                let mut buf = Vec::new();
                f.read_to_end(&mut buf)?;
                zip.start_file(entry_rel.to_string_lossy(), *opts)?;
                zip.write_all(&buf)?;
            }
        }
        Ok(())
    }
    add_dir(&mut zip, &opts, case_dir, Path::new(""))?;
    zip.finish()?;
    Ok(())
}

fn blake3_hex_of_file(p: &Path) -> Result<String> {
    let bytes = std::fs::read(p)?;
    Ok(format!("blake3:{}", blake3::hash(&bytes).to_hex()))
}

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

/// Test-only mutex serializing every test in the crate that touches
/// the `LUMEN_OPERATOR` env var. Lives outside the `tests` module so
/// other modules' tests (e.g. `report::tests`) can share it.
#[cfg(test)]
pub(crate) static TEST_OPERATOR_SERIALIZER: std::sync::Mutex<()> =
    std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use TEST_OPERATOR_SERIALIZER as SERIALIZER;

    fn fresh_dir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "lumen-case-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn with_test_operator(f: impl FnOnce()) {
        // Serialize across the whole test binary — we mutate a global
        // env var. `lock()` may return Err if a previous test paniced
        // while holding the guard; we recover and proceed.
        let _guard = SERIALIZER.lock().unwrap_or_else(|e| e.into_inner());
        let op_path = fresh_dir().join("operator.json");
        std::env::set_var("LUMEN_OPERATOR", &op_path);
        operator::init("Test Op", "Test PD", "TST-001", false).unwrap();
        f();
        std::env::remove_var("LUMEN_OPERATOR");
        let _ = std::fs::remove_file(&op_path);
    }

    #[test]
    fn case_init_creates_metadata_and_first_entry() {
        with_test_operator(|| {
            let dir = fresh_dir();
            let _meta = init(&dir, "C-001", "EVD-001", "Test Case", "Test PD", None).unwrap();
            assert!(case_metadata_path(&dir).exists());
            let entries = read_audit_log(&dir).unwrap();
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].seq, 0);
            assert_eq!(entries[0].action, "case-init");
        });
    }

    #[test]
    fn audit_log_chains_correctly() {
        with_test_operator(|| {
            let dir = fresh_dir();
            init(&dir, "C-002", "EVD-002", "Chain Test", "Test PD", None).unwrap();
            append_entry(&dir, AuditEntryDraft {
                action: "render".into(),
                note: "ran clarify aggressive".into(),
                ..Default::default()
            }).unwrap();
            append_entry(&dir, AuditEntryDraft {
                action: "verify".into(),
                note: "verified by reviewer".into(),
                ..Default::default()
            }).unwrap();
            let verified = verify_audit_log(&dir).unwrap();
            assert_eq!(verified.len(), 3); // case-init + 2 appended
            // Each entry's prev points to the previous entry's signature.
            assert_eq!(verified[0].prev_entry_signature_hex, GENESIS);
            assert_eq!(verified[1].prev_entry_signature_hex, verified[0].entry_signature_hex);
            assert_eq!(verified[2].prev_entry_signature_hex, verified[1].entry_signature_hex);
        });
    }

    #[test]
    fn strict_audit_passes_when_artifacts_present() {
        with_test_operator(|| {
            let dir = fresh_dir();
            // Stage a real input file so init can hash it.
            let staged = dir.join("staged-input.png");
            std::fs::write(&staged, b"fake png bytes for test").unwrap();
            init(
                &dir, "C-strict-1", "EVD-S1", "Strict OK Test", "Test PD",
                Some(&staged),
            )
            .unwrap();
            let report = verify_audit_log_strict(&dir).unwrap();
            assert!(report.all_artifacts_match, "strict should pass; report: {:?}", report);
            // Exactly one artifact ref (the case-init input hash).
            assert_eq!(report.artifacts.len(), 1, "expected 1 artifact ref, got {}", report.artifacts.len());
            assert!(report.artifacts[0].ok);
            assert_eq!(report.artifacts[0].kind, "input");
        });
    }

    #[test]
    fn strict_audit_fails_when_artifact_modified() {
        with_test_operator(|| {
            let dir = fresh_dir();
            let staged = dir.join("staged-input.png");
            std::fs::write(&staged, b"original bytes").unwrap();
            init(
                &dir, "C-strict-2", "EVD-S2", "Strict Tamper Test",
                "Test PD", Some(&staged),
            )
            .unwrap();
            // Regular audit still passes (log signatures are intact).
            assert!(verify_audit_log(&dir).is_ok());
            // Modify the copy under inputs/ — no entry in audit.jsonl
            // catches this without `--strict`.
            let copied = dir.join("inputs").join("staged-input.png");
            assert!(copied.exists());
            std::fs::write(&copied, b"TAMPERED bytes").unwrap();
            let report = verify_audit_log_strict(&dir).unwrap();
            assert!(
                !report.all_artifacts_match,
                "strict audit should detect post-hoc artifact tampering"
            );
            assert!(!report.artifacts[0].ok);
            assert_eq!(report.artifacts[0].matched_path, None);
        });
    }

    #[test]
    fn signoff_separates_analyst_from_reviewer() {
        // Acquire the LUMEN_OPERATOR mutex inline — we swap operator
        // identities mid-test, so we can't use the with_test_operator
        // helper.
        let _guard = SERIALIZER.lock().unwrap_or_else(|e| e.into_inner());
        let dir = fresh_dir();

        // Analyst opens the case.
        let analyst_op_path = dir.join("analyst.json");
        std::env::set_var("LUMEN_OPERATOR", &analyst_op_path);
        operator::init("Det. Analyst", "Lab", "ANL-1", false).unwrap();
        let analyst_pk = operator::current_identity()
            .unwrap()
            .public_key_hex;
        let metadata = init(
            &dir, "C-SO-1", "EVD-SO-1", "Sign-off Test", "Lab", None,
        )
        .unwrap();

        // Self-signoff (analyst signs their own case).
        append_entry(
            &dir,
            AuditEntryDraft {
                action: "sign-off".into(),
                note: "decision: approve - looks fine to me".into(),
                ..Default::default()
            },
        )
        .unwrap();
        let s = signoff_status(&dir, &metadata).unwrap();
        assert_eq!(s.signoff_count, 1);
        assert_eq!(s.approvals, 1);
        assert!(
            !s.has_independent_approval,
            "self-signoff must NOT count as independent approval"
        );

        // Switch to reviewer identity, sign-off again.
        let reviewer_op_path = dir.join("reviewer.json");
        std::env::set_var("LUMEN_OPERATOR", &reviewer_op_path);
        operator::init("Det. Reviewer", "Lab", "REV-1", false).unwrap();
        let reviewer_pk = operator::current_identity()
            .unwrap()
            .public_key_hex;
        assert_ne!(analyst_pk, reviewer_pk);
        append_entry(
            &dir,
            AuditEntryDraft {
                action: "sign-off".into(),
                note: "decision: approve - chain of custody verified".into(),
                ..Default::default()
            },
        )
        .unwrap();
        let s = signoff_status(&dir, &metadata).unwrap();
        assert_eq!(s.signoff_count, 2);
        assert_eq!(s.approvals, 2);
        assert!(
            s.has_independent_approval,
            "reviewer's signoff must count as independent approval"
        );
        assert_eq!(s.independent_reviewers, vec![reviewer_pk]);

        // Chain still verifies (sign-off entries are normal entries).
        let verified = verify_audit_log(&dir).unwrap();
        assert_eq!(verified.len(), 3); // case-init + 2 sign-offs

        std::env::remove_var("LUMEN_OPERATOR");
    }

    #[test]
    fn tampered_entry_breaks_verification() {
        with_test_operator(|| {
            let dir = fresh_dir();
            init(&dir, "C-003", "EVD-003", "Tamper Test", "Test PD", None).unwrap();
            append_entry(&dir, AuditEntryDraft {
                action: "render".into(),
                note: "applied clarify aggressive".into(),
                ..Default::default()
            }).unwrap();
            // Tamper: rewrite the appended entry's note. The note is
            // covered by the signature, so changing it must invalidate.
            let p = audit_log_path(&dir);
            let s = std::fs::read_to_string(&p).unwrap();
            assert!(
                s.contains("applied clarify aggressive"),
                "test setup broken — note should be present"
            );
            let tampered = s.replace("applied clarify aggressive", "applied innocent crop");
            std::fs::write(&p, tampered).unwrap();
            // Now verification must fail because the signature no longer
            // covers the modified note bytes.
            let r = verify_audit_log(&dir);
            assert!(r.is_err(), "tampered log should fail verification");
        });
    }
}
