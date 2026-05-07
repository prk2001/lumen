# Federal-grade evidence handling

Lumen's forensic mode targets the standards used by federal forensic
labs: FBI's FAVIAU (Forensic Audio, Video, and Image Analysis Unit),
DEA, ATF, and the regional crime labs that share their procedures.
This document maps Lumen's features to the relevant standards.

> Lumen provides the technical backbone. Legal admissibility depends
> on jurisdiction-specific evidence rules, departmental SOPs, and
> physical-custody procedures. Your prosecutor signs off on the rest.

## Standards alignment

| Standard | Requirement | Lumen feature |
| --- | --- | --- |
| **FIPS 180-4** | SHA-256 / SHA-384 / SHA-512 are the only approved hash families for federal data integrity | `--fips` adds SHA-256 alongside BLAKE3 on every artifact; `case audit --require-fips` enforces it |
| **FIPS 186-5** (Feb 2023) | Ed25519 is FIPS-approved for digital signatures | Lumen signs every audit entry with Ed25519 |
| **FIPS 140-3** | Cryptographic modules must use only approved primitives | FIPS-mode hashing routes through `sha2` (RustCrypto, FIPS-target); BLAKE3 is informational, never the only hash |
| **NIST SP 800-86** (Guide to Integrating Forensic Techniques into Incident Response) | Hash + timestamp every operation; preserve original; document chain | Per-step hashes (signed), per-step timestamps, original input copied verbatim into `inputs/`, signed log binds it all together |
| **NIST SP 800-88r1** | Media handling for forensic preservation | Copy-on-write into the case folder — original on the source media is never written |
| **SWGDE Best Practices for Digital and Multimedia Evidence Imaging** | Document every algorithm used; preserve master copy; record working copies' hashes | Recipe JSON + signed audit log records every algorithm; original input hash is the genesis sentinel |
| **SWGDE Best Practices for Image Authentication** | Algorithms must be peer-reviewed and reproducible | All clarify-chain algorithms cite published methods (see "Algorithm provenance" below) |
| **CJIS Security Policy v5.9** §5.5 (Access Control) | Two-person integrity for sensitive operations | `case sign-off --decision approve` + `audit --require-signoff` enforces analyst-vs-reviewer separation cryptographically |
| **CJIS Security Policy** §5.10 (System and Information Integrity) | Audit logs must be tamper-evident | Hash-chained Ed25519 audit log; any past entry tampering invalidates every entry from that point forward |
| **Daubert v. Merrell Dow** (1993) | Scientific evidence must be testable, peer-reviewed, with known error rates, and generally accepted | All algorithms are published with citations; the audit log records exact parameters so any operation is bit-exact reproducible |
| **Federal Rules of Evidence 901, 902** | Authentication of digital evidence | Operator's Ed25519 pubkey + signed chain provides cryptographic authentication; pubkey can be cross-referenced against the lab's published key directory |
| **C2PA / Content Credentials** | Industry-standard provenance manifest | Lumen's audit log is conceptually equivalent; native C2PA export is on the roadmap |

## Air-gapped operation

Federal labs handling classified or grand-jury material run in
SCIF-style air-gapped environments. Lumen makes no network calls
during any forensic operation:

- Keys are generated locally via `OsRng`
- All hashing is local
- Reads/writes are confined to the case folder + `~/.lumen/`
- AI models are never auto-downloaded — `lumen-ai` requires you to
  supply ONNX files manually
- The only network-binding command is `lumen serve` (loopback only)

To enforce this at the policy layer:

```bash
lumen --no-network <any subcommand>
# or:
LUMEN_NO_NETWORK=1 lumen <subcommand>
```

`--no-network` causes `lumen serve` to refuse to start. All other
commands work unchanged because they don't open sockets.

## FIPS mode

```bash
# Operator init under FIPS:
lumen --fips operator init --name "Det. X" \
    --agency "FBI Lab" --identifier "BADGE-1"

# Open a FIPS case — every audit entry will include both BLAKE3 and SHA-256:
lumen --fips case init --dir ./case --case-id 2026-X \
    --evidence-id EVD-1 --case-name "..." --agency "FBI Lab" \
    --input ./incoming/dvr-grab.jpg

# Every render under FIPS dual-records:
lumen --fips case render --dir ./case --recipe ./r.json \
    --input ./incoming/dvr-grab.jpg --output cleaned.png \
    --note "Forensic clarify chain applied"

# Verify a received bundle requires SHA-256 on every entry:
lumen case audit --dir ./received --strict --require-fips --require-signoff
```

`case audit --require-fips` exits non-zero if any entry recorded a
BLAKE3 hash without a SHA-256 sibling. This is the gate to put on
the receive side of an inter-lab transfer.

## Receiving evidence (verify-export)

Cases arrive as `.lumenpkg.zip` bundles. To verify one without
unpacking it manually:

```bash
lumen case verify-export --input ./EVD-2026-7842.lumenpkg.zip \
    --require-signoff
```

Returns JSON with:
- `bundle_hash`: BLAKE3 + SHA-256 + size of the zip itself (record
  this in your physical evidence log so the bundle is doubly bound
  to its container)
- `case`: full metadata block
- `audit_chain_verified`: every signature checks, chain unbroken
- `strict.all_artifacts_match`: every artifact hash in the log
  matches a real file inside the bundle (no post-hoc swap)
- `signoff`: independent reviewer count, approval/rejection counts

Exit codes:
- 0: bundle is intact and (if `--require-signoff`) independently
  approved
- 1: chain broken, artifact missing/modified, or signoff missing
- 2: zip invalid or unreadable

## Algorithm provenance

Every operation in the forensic clarify chain comes from peer-reviewed
literature. This is the table to put in front of a defense expert:

| Algorithm | Citation | Lumen module |
| --- | --- | --- |
| Bilateral filter | Tomasi & Manduchi, ICCV 1998 | `lumen-fx-denoise.bilateral` |
| CLAHE | Pizer et al., CVGIP 1987; Zuiderveld 1994 | `lumen-fx-text.clahe` |
| Dark Channel Prior dehaze | He, Sun & Tang, CVPR 2009 | `lumen-fx-weather.dehaze_dcp` |
| Wiener deconvolution | Wiener 1949; FFT formulation Brigham 1988 | `lumen-fx-deblur.wiener` |
| Richardson-Lucy deconvolution | Richardson, J. Opt. Soc. Am. 1972; Lucy, Astronom. J. 1974 | `lumen-fx-deblur.richardson_lucy` |
| Biggs-Andrews damping | Biggs & Andrews, Appl. Optics 1997 | RL `damping` parameter |
| Lanczos resampling | Duchon 1979 | `lumen-fx-upscale.lanczos` |
| Ed25519 signatures | Bernstein et al., 2011; FIPS 186-5 (2023) | `lumen-auth` (via `ed25519-dalek`) |
| BLAKE3 hashing | O'Connor et al., 2020 | `blake3` crate |
| SHA-256 hashing | NIST FIPS 180-4 (2015) | `sha2` crate (FIPS-mode only) |

Every parameter is in the recipe JSON inside the case bundle. A defense
expert can take that recipe, run `lumen pipeline --recipe r.json` on
the original input (also in the bundle), and produce a bit-exact match
of the analyst's output.

## End-to-end FAVIAU-style pipeline

The standard federal-lab workflow, from a DVR-grabbed `.mp4` to a
court-deliverable evidence package:

```bash
# 0. One-time: bootstrap analyst + reviewer identities.
LUMEN_OPERATOR=~/.lumen/analyst.json   lumen --fips operator init \
    --name "Det. A. Analyst"  --agency "FBI Lab" --identifier "ANL-1"
LUMEN_OPERATOR=~/.lumen/reviewer.json  lumen --fips operator init \
    --name "Det. B. Reviewer" --agency "FBI Lab" --identifier "REV-1"

# 1. Extract a contiguous range of CCTV frames from the source video.
LUMEN_OPERATOR=~/.lumen/analyst.json \
lumen video-extract --input ./incoming/dvr-grab.mp4 \
    --output-dir ./frames --start 100 --end 130 --stride 1 --format png

# 2. Open a FIPS-mode case bound to the source video as the original.
lumen --fips case init --dir ./case --case-id 2026-X --evidence-id EVD-1 \
    --case-name "Lot B Surveillance" --agency "FBI Lab" \
    --input ./incoming/dvr-grab.mp4

# 3. Multi-frame super-resolve all 30 frames into a 4x SR composite,
#    with the per-frame shift report recorded in the signed audit log.
lumen --fips case super-resolve --dir ./case \
    --input ./frames/frame_000100.png \
    --input ./frames/frame_000101.png \
    [...] \
    --output sr-composite.png --scale 4 --fuse median \
    --note "FAVIAU SR pipeline: 30 frames, 4x scale"

# 4. Run the forensic clarify chain on the SR composite.
lumen --fips case render --dir ./case \
    --recipe ./forensic-recipe.json \
    --input ./case/outputs/sr-composite.png \
    --output cleaned.png --note "Forensic clarify chain"

# 5. Reviewer (different identity) cross-checks and signs off.
LUMEN_OPERATOR=~/.lumen/reviewer.json \
lumen case sign-off --dir ./case --decision approve \
    --note "Cross-checked SR registration vs original frames; \
            no fabricated detail observed in plate region."

# 6. Verify everything before export.
lumen case audit --dir ./case --strict --require-fips --require-signoff

# 7. Render reviewer-friendly HTML + tamper-evident bundle.
lumen case report --dir ./case
lumen case export --dir ./case --output EVD-1.lumenpkg.zip

# 8. Receiving lab: single-command verification.
lumen case verify-export --input ./EVD-1.lumenpkg.zip \
    --require-signoff
```

Every step in this pipeline produces a signed audit entry. The shift
report from step 3 is embedded in the signed note — a defense expert
can read every per-frame (dx, dy, peak_score) without having to re-run
the algorithm.

## What's still on the federal roadmap

- **RFC 3161 timestamping** — third-party Time-Stamping Authority
  signatures on every audit entry. Currently timestamps are operator-
  asserted (signed by the operator's key but using the operator's
  clock). RFC 3161 would prove a lower bound on when each step
  happened without trusting the operator.
- **HSM-backed operator keys** — PIV/CAC smartcard storage for
  `~/.lumen/operator.json`'s private key. Currently stored at mode
  0600 in the filesystem.
- **C2PA manifest export** — adapter that emits a C2PA-compatible
  provenance manifest derived from the audit log, so Lumen output
  works in C2PA-aware viewers (Adobe, Microsoft, BBC, etc.).
- **Iterative back-projection refinement** — Keren-style IBP on top
  of median fusion to recover even more detail. Median is robust;
  IBP is sharper.
- **Blind PSF estimation** — fit the actual PSF from observable
  motion vectors / edge profile in the source rather than assuming
  Gaussian sigma. Plug into Wiener / Richardson-Lucy.
- **16-bit / 32-bit float export** — for evidence preservation
  (current pipeline runs 32-bit float internally; export is 8-bit).
- **Threshold signatures (M-of-N)** — multiple operators must
  collectively sign before a case can be exported.
