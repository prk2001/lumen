//! Tauri entry point for the Lumen desktop app.
//!
//! The Rust side of this binary links the `lumen-*` crates directly
//! (see `Cargo.toml`) and exposes them to the React UI through Tauri
//! IPC commands defined in [`commands`]. The frontend never speaks to
//! a separate `lumen serve` HTTP process — every render happens
//! in-process.

mod commands;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            commands::list_effects,
            commands::probe,
            commands::apply_effect,
            commands::run_pipeline,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
