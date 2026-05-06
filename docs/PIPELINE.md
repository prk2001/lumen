# Pipeline recipe format

`lumen pipeline --recipe recipe.json` runs a multi-effect chain
described by a JSON recipe. This document specifies the format.

> **Phase 1 status:** linear chains only. Branched DAGs land in Phase 3
> alongside multi-input effects (composite, blend, mask-merge).

## Schema (informal)

```jsonc
{
  // Path to the source media. Relative paths are resolved against the
  // recipe file's directory.
  "input":  "samples/in.png",

  // Path to the output. Format inferred from extension
  // (PNG / JPEG / TIFF / WebP / BMP in Phase 1).
  "output": "out/result.png",

  // Ordered list of effects to apply.
  "chain": [
    {
      // Globally unique effect id. Run `lumen list-effects` to enumerate.
      "effect": "lumen-fx-denoise.gaussian",

      // Optional human label, surfaced in logs.
      "label": "Soften noise",

      // Parameter values keyed by parameter id.
      // Unknown keys cause validation failure.
      "params": {
        "sigma": 0.6
      }
    },
    { "effect": "lumen-fx-sharpen.unsharp_mask",
      "params": { "amount": 1.4, "radius": 1.2, "threshold": 0.0 } },
    { "effect": "lumen-fx-color.saturation",
      "params": { "amount": 1.25 } },
    { "effect": "lumen-fx-exposure.brightness_contrast",
      "params": { "brightness": 0.05, "contrast": 1.1 } }
  ]
}
```

## Parameter coercion

Recipe params accept JSON `bool` / `number` / `string` directly. Numbers
that fit `i64` are passed as `Int`; otherwise as `Float`. The host
performs type-checking and range validation against each effect's
[`ParamSpec`](../crates/lumen-core/src/params.rs); promotion from `Int`
to `Float` happens automatically when an effect declares a
`ParamKind::Float`.

## Defaulting

Any parameter not listed in `params` is filled from the effect's
default. So this works for a quick run:

```json
{
  "input":  "in.png",
  "output": "out.png",
  "chain": [{ "effect": "lumen-fx-denoise.gaussian" }]
}
```

## Error reporting

Errors carry:

- the recipe path,
- the index of the offending step,
- the parameter name (if applicable),
- a stable error code (e.g. `INVALID_PARAMETER`, `EFFECT_NOT_FOUND`).

Failed pipelines do not write a partial output file.

## Beyond Phase 1

Planned extensions (in roughly the order they'll appear):

- **Multiple inputs** — `"sources": [...]` with named ids and an
  optional `"node_inputs"` map per step (Phase 3).
- **Branches** — replace `chain` with a `graph` object (full DAG) when
  the work justifies it.
- **Per-effect color-space override** — `"working_space": "aces_cg"`.
- **Render scope** — `"frames": [start, end]` for video clips
  (Phase 1.1.b once FFmpeg lands).
- **Includes** — `"include": "preset.json"` for shared chains.
- **Inline LUTs** — base64 / file refs once `fx-color` ships LUT
  loading.

The schema is versioned. Phase 2 introduces `"schema": "lumen-recipe/v2"`
when these extensions ship; v1 recipes will continue to work via a
migration shim.
