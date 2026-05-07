# Desktop app (Tauri 2 + React)

The Lumen desktop application — a Tauri 2 shell hosting a React + TypeScript
UI built with Vite. Scaffolded in Phase 1 / Milestone 1.5.

## What this currently is (Phase 1)

A thin desktop shell that embeds an `<iframe>` pointed at the `lumen-cli`
`serve` subcommand. To use it locally:

```bash
# Terminal 1 — start the live preview server (separate process)
cargo run -p lumen-cli -- serve --recipe path/to/recipe.toml
#  -> serves the live preview UI at http://127.0.0.1:8723/

# Terminal 2 — launch the desktop shell
cd ~/Lumen/apps/desktop
pnpm tauri dev
```

The window opens at 1280x800 with the title "Lumen" and loads the iframe
from `http://127.0.0.1:8723/`. A small banner at the top reminds you that
`lumen serve` must be running.

## What this will become (Phase 2+)

Once the IPC layer is in place, the iframe goes away and the React UI
talks to the pipeline directly through Tauri commands defined in
`src-tauri/src/commands/`. At that point the desktop app no longer
depends on the HTTP server — it links the `lumen-*` crates directly into
`src-tauri/Cargo.toml`.

## Workspace boundary

The `src-tauri/` crate is intentionally **standalone** — it is *not* a
member of the root Lumen Cargo workspace. The empty `[workspace]` table
in `src-tauri/Cargo.toml` makes that explicit. This avoids dragging the
heavy Tauri build graph (webview-sys, wry, gtk, etc.) into every
`cargo build` at the workspace root. Once we wire the desktop app to
`lumen-core` and friends, those crates will be added as
`{ path = "../../../crates/lumen-core" }` dependencies — still without
joining the workspace.

## Layout

```text
apps/desktop/
├── package.json              # pnpm + Vite + React + @tauri-apps/api
├── index.html                # Vite entry
├── vite.config.ts
├── src/                      # React UI
│   ├── App.tsx               # iframe shell -> http://127.0.0.1:8723/
│   ├── main.tsx
│   ├── App.css
│   └── assets/
├── src-tauri/
│   ├── Cargo.toml            # standalone; will gain lumen-* deps later
│   ├── tauri.conf.json       # title "Lumen", 1280x800
│   ├── build.rs
│   ├── capabilities/
│   ├── icons/
│   └── src/
│       ├── main.rs
│       └── lib.rs            # commands/ will live here in Phase 2
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
