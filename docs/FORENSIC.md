# Forensic / police-grade workflow

Lumen's forensic layer turns it into a **chain-of-custody-aware
processing tool** suitable for surveillance enhancement, evidence
preparation, and case-file delivery. Every modification is signed,
every signature chains to the previous one, and the whole case
folder exports as a single tamper-evident package.

> **What it gets right:** every cryptographic operation uses
> well-known primitives (Ed25519, BLAKE3, SHA-256-ready) with no
> network calls. Nothing leaves your machine. Reviewers verify
> only with public keys — the operator's secret never travels.
>
> **What it doesn't replace:** legal admissibility depends on
> jurisdiction-specific evidence rules + departmental SOPs +
> physical-custody procedures. Lumen provides the technical
> backbone; your prosecutor signs off on the rest.

## Roles

| Role | What they do | Tools |
| --- | --- | --- |
| **Operator** | Receives evidence, runs enhancement chain, signs each operation. Their identity is bound to every change. | `lumen operator init`, `lumen case render`, `lumen case note` |
| **Reviewer** | Receives a case package, verifies the chain, examines intermediate stages, signs off (or rejects). | `lumen case audit`, stages strip, HTML report |
| **Custodian / Court** | Receives the final package, can independently verify by re-running `lumen case audit` against the operator's published public key. | `lumen verify`, `lumen case audit` |

## Bootstrapping (one-time per operator)

```bash
lumen operator init \
  --name "Det. A. Kennedy" \
  --agency "Primoris Forensic Lab" \
  --identifier "BADGE-12345"
```

Generates an Ed25519 keypair locally and writes
`~/.lumen/operator.json` with mode 0600. **Back up this file
immediately to your department's key escrow.** If you lose it, all
future signatures from this identity become unverifiable in the
context of new evidence (existing signatures are still verifiable
against the published public key).

```bash
lumen operator show > my-public-cert.json
```

Publish `my-public-cert.json` (or just the `public_key_hex` field)
to your department's signing-key directory so reviewers can find it.

## A complete case, end to end

### 1. Open the case

```bash
lumen case init \
  --dir ./case-2026-CCTV-LOTB \
  --case-id  "2026-CCTV-LOTB" \
  --evidence-id "EVD-2026-7842" \
  --case-name "Lot B Surveillance" \
  --agency "Primoris Forensic Lab" \
  --input ./incoming/dvr-grab-19-42-18.jpg
```

This creates the case folder, copies the original input into
`inputs/`, records its BLAKE3 hash in `case.json`, and writes the
first audit log entry (`case-init`, signed by you).

### 2. Run a clarification chain

Start with a recipe, even if it's a one-liner:

```bash
cat > recipe.json <<'EOF'
{
  "input":  "_",
  "output": "_",
  "chain": [
    { "effect": "lumen-fx-denoise.gaussian",     "params": { "sigma": 1.0 } },
    { "effect": "lumen-fx-compression.deblock",  "params": { "strength": 0.7 } },
    { "effect": "lumen-fx-weather.dehaze_dcp",   "params": { "omega": 0.75 } },
    { "effect": "lumen-fx-text.clahe",           "params": { "tiles_x": 16, "clip_limit": 1.8 } },
    { "effect": "lumen-fx-deblur.wiener",        "params": { "sigma": 1.5, "nsr": 0.02 } },
    { "effect": "lumen-fx-sharpen.unsharp_mask", "params": { "amount": 0.55, "threshold": 0.015 } }
  ]
}
EOF

lumen case render \
  --dir ./case-2026-CCTV-LOTB \
  --recipe ./recipe.json \
  --input  ./incoming/dvr-grab-19-42-18.jpg \
  --output cleaned.png \
  --note "Plate-clarification chain applied; reviewer can inspect /stages/cleaned/ for intermediate frames."
```

The `case render` command:

- copies the recipe into `recipes/` and records its hash
- copies the input into `inputs/` (idempotent — same hash → same file)
- runs the chain with stage capture into `stages/cleaned/`
- writes the output to `outputs/cleaned.png` and records its hash
- appends a signed audit-log entry covering all three hashes + your
  note + the recipe's BLAKE3

Every step in the chain produces a checkable intermediate:

```
case-2026-CCTV-LOTB/
├── case.json
├── audit.jsonl
├── inputs/dvr-grab-19-42-18.jpg
├── recipes/recipe.json
├── outputs/cleaned.png
└── stages/cleaned/
    ├── 00-input.png
    ├── 01-lumen-fx-denoise_gaussian.png
    ├── 02-lumen-fx-compression_deblock.png
    ├── 03-lumen-fx-weather_dehaze_dcp.png
    ├── 04-lumen-fx-text_clahe.png
    ├── 05-lumen-fx-deblur_wiener.png
    └── 06-lumen-fx-sharpen_unsharp_mask.png
```

### 3. Reviewer notes

Anyone who inspects the case can append observations:

```bash
lumen case note --dir ./case-2026-CCTV-LOTB \
  --note "Reviewer cross-checked stages 04 vs 06; no fabricated detail observed in plate region."
```

### 4. Verify the chain

At any point — by you, by a reviewer, by opposing counsel:

```bash
lumen case audit --dir ./case-2026-CCTV-LOTB
```

Returns JSON with `audit_chain_verified: true` if every signature
checks AND the chain is unbroken (every entry's
`prev_entry_signature_hex` matches the previous entry's
`entry_signature_hex`, all the way back to the genesis sentinel).

If anyone tampered with any past entry, every entry from that point
forward fails verification.

### 5. Export a tamper-evident package

```bash
lumen case export \
  --dir   ./case-2026-CCTV-LOTB \
  --output ./EVD-2026-7842.lumenpkg.zip
```

Zips the case folder (Deflate compression, structure preserved)
into a single artifact. Returns the bundle's BLAKE3 hash, which
you should record in your physical evidence log for cross-reference.

## Independent verification

A reviewer who receives the zip can verify without trusting the
operator's machine:

```bash
unzip EVD-2026-7842.lumenpkg.zip -d ./received/
lumen case audit --dir ./received/
```

The audit step uses each entry's embedded `operator_public_key_hex`
to verify; no separate key file is needed (though the operator's
published public cert provides a higher-trust binding between the
identity in `case.json` and the holder of the secret).

## What's signed and what isn't

The audit log signs:
- The schema version (forces forward-compatible verification)
- The seq number (prevents reordering)
- The timestamp
- The operator's pubkey (binds the entry to a person)
- The action verb + free-form note (who claimed to do what)
- The input/output/recipe BLAKE3 hashes (binds the entry to the
  artifacts in the folder)
- The prev_entry_signature_hex (chain integrity)

Excluded: the artifact files themselves are referenced by hash but
aren't physically embedded in the audit log — they live in
`inputs/`, `outputs/`, etc. If any of those files are modified,
their BLAKE3 stops matching the recorded hash and `lumen case audit`
won't catch it (the audit log is only about the LOG; checking the
artifacts is a separate `blake3sum`-style step).

> Roadmap: `lumen case audit --strict` to also re-hash every
> referenced file and confirm the hashes still match.

## Crypto

| Primitive | Use | Implementation |
| --- | --- | --- |
| Ed25519 | Operator signing keys, audit-entry signatures | `ed25519-dalek` 2.x (pure-Rust, audited) |
| BLAKE3  | Content addressing (input/output/recipe hashes, bundle hash) | `blake3` crate |
| Canonical JSON | Sortable key encoding for stable signing payloads | Custom serializer in `case.rs` |

**FIPS notes**: Ed25519 is FIPS-approved as of FIPS 186-5 (Feb 2023).
BLAKE3 is not FIPS-approved; for FIPS-only environments swap to
SHA-256 (a `--fips` flag is on the roadmap; the work is mechanical).

## Air-gapped operation

Lumen makes no network calls during any forensic operation. The CLI:

- Generates keys locally via `OsRng`
- Computes hashes locally
- Reads/writes only files inside the case folder + `~/.lumen/`
- Never auto-downloads models (the `lumen-ai` infrastructure
  refuses to fetch; you supply ONNX files manually)

The only command that opens a network port is `lumen serve`, which
binds 127.0.0.1 (loopback only) — disable in classified
environments by simply not running it.

## What's still on the roadmap

- `lumen case audit --strict` — also re-hash every file referenced
  by the audit log and check.
- `lumen case sign-off --reviewer <key>` — separate reviewer
  signature gate for multi-operator approval.
- C2PA-compatible export — the audit log is conceptually similar
  to a C2PA manifest; an adapter would let Lumen output land
  natively in C2PA-aware viewers.
- FIPS-mode (`--fips`) using SHA-256 + ECDSA P-256.
- Threshold signatures (M-of-N operators required).
- Air-gapped policy enforcement flag (`--no-network`) that hard-
  errors on any code path that would touch a socket.
