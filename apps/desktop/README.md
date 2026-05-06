# Desktop app (Tauri 2 + React)

This directory will host the Lumen desktop application. Initial scaffold
is **deferred to Phase 1 / Milestone 1.5** to avoid pulling Node and
Tauri tooling in before the core engine has anything to display.

## When ready, scaffold with:

```bash
cd ~/Lumen/apps
npm create tauri-app@latest desktop -- \
  --template react-ts \
  --identifier com.primorispartners.lumen \
  --manager pnpm
```

Then add `crates/lumen-core` and friends to `src-tauri/Cargo.toml` so
the desktop app shares the same engine as the CLI and server.

## Why deferred

Spinning up a Tauri app installs several hundred MB of Node modules and
adds a second build system (Vite). Until the Rust core does something
worth previewing, the engineering value is zero. Scaffolding it
prematurely also locks in framework versions that may shift before
we're ready to commit.

## Once scaffolded, expected layout:

```text
apps/desktop/
├── package.json              # pnpm + Vite + React
├── src/                      # React UI
│   ├── App.tsx
│   ├── main.tsx
│   ├── pages/
│   ├── panels/               # Inspector, Timeline, Viewer, Layers
│   └── lib/
├── src-tauri/
│   ├── Cargo.toml            # Tauri runtime + lumen-* dependencies
│   ├── tauri.conf.json
│   └── src/
│       ├── main.rs
│       └── commands/         # IPC bridge to lumen-core
└── public/
```
