# Source Feature Spec — Summary

Lumen is built from a 30-category, ~1,140-feature engineering spec
generated in prior Claude sessions. The full spec lives at:

- `/Users/patrickkennedy/Downloads/features_5levels.md`
  (Categories 1–21, 106,593 lines, 2.5 MB)
- `/Users/patrickkennedy/Downloads/features_5levels_part2.md`
  (Categories 20–30, 32,832 lines, 768 KB)

> **Note.** The two files have overlap on Cat 20 (Measurement & Analysis)
> and the first file contains a few duplicated `## N. …` headers from
> multi-session generation runs. We treat **part2** as authoritative for
> Cat 20+ and **part1** as authoritative for Cat 1–19.
> Reconciliation tracked in `docs/architecture/adr-0010-spec-merge.md`
> (planned).

## Category map

Each category maps 1:1 to a Rust crate (or to the Tauri UI for Cat 3).

| #  | Category                                  | Owning crate / surface       |
| -- | ----------------------------------------- | ---------------------------- |
| 1  | Input, Formats & Codecs                   | `lumen-io`                   |
| 2  | Playback & Navigation                     | `lumen-playback`             |
| 3  | UI, Workspace & Viewing                   | `apps/desktop`, `ui/`        |
| 4  | Exposure, Tone & Dynamic Range            | `lumen-fx-exposure`          |
| 5  | Color Science & Grading                   | `lumen-fx-color`             |
| 6  | Sharpening & Detail Recovery              | `lumen-fx-sharpen`           |
| 7  | Noise Reduction & Cleanup                 | `lumen-fx-denoise`           |
| 8  | Compression Artifact Removal              | `lumen-fx-compression`       |
| 9  | Geometric & Lens Correction               | `lumen-fx-geometric`         |
| 10 | Stabilization & Motion Correction         | `lumen-fx-stabilize`         |
| 11 | Deblurring & Deconvolution                | `lumen-fx-deblur`            |
| 12 | Super-Resolution & Upscaling              | `lumen-fx-upscale`           |
| 13 | Frame Rate & Temporal                     | `lumen-fx-temporal`          |
| 14 | AI-Powered Enhancement                    | `lumen-fx-ai`                |
| 15 | Face / Skin / Portrait                    | `lumen-fx-face`              |
| 16 | Text / Plate / Object Clarification       | `lumen-fx-text`              |
| 17 | Masking / Selection / ROI                 | `lumen-fx-mask`              |
| 18 | Weather / Atmospheric / Environmental     | `lumen-fx-weather`           |
| 19 | Advanced Imaging Modalities               | `lumen-fx-modalities`        |
| 20 | Measurement & Analysis                    | `lumen-measure`              |
| 21 | Audio Enhancement                         | `lumen-audio`                |
| 22 | Authentication & Integrity                | `lumen-auth`                 |
| 23 | Workflow & Non-Destructive                | `lumen-workflow`             |
| 24 | Collaboration & Project Management        | `lumen-collab`               |
| 25 | Reporting / Visualization / Presentation  | `lumen-report`               |
| 26 | Export / Delivery / Encoding              | `lumen-export`               |
| 27 | Performance & Hardware                    | `lumen-perf`                 |
| 28 | Extensibility / Automation / API          | `lumen-api`                  |
| 29 | Platform & Distribution                   | `lumen-platform`             |
| 30 | Quality Assurance & Monitoring            | `lumen-qa`                   |

## Spec structure

The source files use a 5-level outline:

```text
##  Category
###   Feature   (L1)
-     Sub-feature (L2)
  -   Implementation detail (L3)
    - Sub-detail (L4)
      - Concrete spec / parameter / default (L5)
```

Each L5 leaf is a defensible engineering deliverable — a parameter, a
default value, a code path, or a concrete library binding. **Lumen does
not promise to ship every leaf**; the spec is a coverage map. The
roadmap in [`PLAN.md`](PLAN.md) selects which leaves get implemented in
which phase.

## Working with the spec

Practical advice when implementing a category:

1. **Don't read the whole file.** It's too large to hold in working
   memory. Use `grep`/`rg` to extract one category at a time.
2. **Treat L1 as features and L2 as the test plan.** Each L2 item maps
   roughly to a unit test or property test.
3. **L3+ becomes documentation.** Once an effect is implemented, the
   L3–L5 detail goes into the effect's `mod` doc-comment so future
   readers understand why the defaults were chosen.
4. **Flag conflicts.** If the spec says X but a load-bearing library
   forces Y, file an ADR rather than silently diverging.

## Re-generating per-category extracts

```bash
# Extract category N from the source file:
awk '/^## [0-9]+\. / {p=0} /^## 7\. /{p=1} p' \
  /Users/patrickkennedy/Downloads/features_5levels.md \
  > docs/specs/cat-07-noise-reduction.md
```

Per-category extracts will be checked into `docs/specs/` as we work
through them. Doing this *lazily* (only when starting work on a
category) avoids 3 MB of duplicated content in the repo.

## Spec drift policy

- The Downloads files are the **historical record**.
- Once Lumen ships a category, **the code becomes authoritative** and
  the spec extract in `docs/specs/cat-NN.md` is updated to match
  reality (with the original tagged in git history).
- New ideas go in `docs/specs/proposals/` and graduate via PR.
