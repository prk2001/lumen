# Desktop app (Tauri 2 + React)

The Lumen desktop application — a Tauri 2 shell hosting a React +
TypeScript UI built with Vite. The Rust side links the `lumen-*` crates
directly and exposes them to the frontend through Tauri IPC commands;
no separate `lumen serve` HTTP process is required.

## What this currently is

A native window that talks to the in-process pipeline. The React UI:

1. Loads the effect registry at startup via `invoke('list_effects')`.
2. Lets you pick an input file path, output file path, and an effect.
3. Renders sliders / toggles / dropdowns from each effect's parameter
   spec.
4. Calls `invoke('apply_effect', …)` on demand and renders the output
   image inline (via the `convertFileSrc` asset protocol).

To run it locally:

```bash
cd ~/Lumen/apps/desktop
pnpm install
pnpm tauri dev
```

## Tauri commands exposed

Defined in `src-tauri/src/commands.rs` and registered in
`src-tauri/src/lib.rs`:

| Command         | Signature                                                                                                | Notes                                       |
| --------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `list_effects`  | `() -> Vec<EffectInfo>`                                                                                  | id + display_name + parameter specs         |
| `probe`         | `(path: String) -> Result<AssetMetadataDto, String>`                                                     | wraps `lumen_io::probe`                     |
| `apply_effect`  | `(input_path, output_path, effect_id, params: serde_json::Value) -> Result<(), String>`                  | single-effect render                        |
| `run_pipeline`  | `(recipe_json, input_path, output_path) -> Result<RenderStats, String>`                                  | multi-stage chain; `RenderStats { duration_ms, output_bytes }` |

The starter set of fx crates linked into the binary is:
`lumen-fx-exposure`, `lumen-fx-color`, `lumen-fx-sharpen`,
`lumen-fx-denoise`, `lumen-fx-geometric`. Add more by extending
`registry()` in `commands.rs` and `Cargo.toml`.

## Workspace boundary

The `src-tauri/` crate is intentionally **standalone** — it is *not* a
member of the root Lumen Cargo workspace. The empty `[workspace]` table
in `src-tauri/Cargo.toml` makes that explicit. This avoids dragging the
Tauri build graph into every `cargo build` at the workspace root. The
`lumen-*` crates are pulled in through `path = "../../../crates/…"`
references rather than by joining the workspace.

## Layout

```text
apps/desktop/
├── package.json              # pnpm + Vite + React + @tauri-apps/api
├── index.html                # Vite entry
├── vite.config.ts
├── src/                      # React UI
│   ├── App.tsx               # IPC client, effects UI
│   ├── App.css               # dark theme
│   ├── main.tsx
│   └── assets/
├── src-tauri/
│   ├── Cargo.toml            # standalone; depends on lumen-* by path
│   ├── tauri.conf.json       # title "Lumen", 1280x800, asset protocol on
│   ├── build.rs
│   ├── capabilities/
│   ├── icons/
│   └── src/
│       ├── main.rs
│       ├── lib.rs            # registers IPC commands
│       └── commands.rs       # IPC command implementations
└── public/
```

## Useful commands

```bash
pnpm install                              # install JS deps
pnpm dev                                  # Vite-only (no Tauri window)
pnpm tauri dev                            # full desktop app, hot-reload
pnpm tauri build --debug --no-bundle      # smoke-test compile, no installer
pnpm tauri build                          # release build with installer
```

## Identifier

The bundle identifier is `com.primorispartners.lumen` and the product
name is `Lumen`, set in `src-tauri/tauri.conf.json`.

## Recommended IDE setup

[VS Code](https://code.visualstudio.com/) plus the
[Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode)
and
[rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
extensions.
