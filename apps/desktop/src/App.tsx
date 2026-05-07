import { useEffect, useMemo, useState } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import "./App.css";

type ParamSpec = {
  id: string;
  display_name: string;
  description: string;
  kind: "bool" | "int" | "float" | "choice" | "string";
  default: boolean | number | string;
  min: number | null;
  max: number | null;
  options: string[] | null;
};

type EffectInfo = {
  id: string;
  display_name: string;
  description: string;
  category: string;
  parameters: ParamSpec[];
};

type ParamValue = boolean | number | string;

function defaultsFor(effect: EffectInfo): Record<string, ParamValue> {
  const out: Record<string, ParamValue> = {};
  for (const p of effect.parameters) out[p.id] = p.default as ParamValue;
  return out;
}

function App() {
  const [effects, setEffects] = useState<EffectInfo[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [paramValues, setParamValues] = useState<Record<string, ParamValue>>({});
  const [inputPath, setInputPath] = useState<string>("");
  const [outputPath, setOutputPath] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);
  const [renderError, setRenderError] = useState<string | null>(null);
  const [renderedAt, setRenderedAt] = useState<number | null>(null);

  // Load effects from Rust on mount.
  useEffect(() => {
    invoke<EffectInfo[]>("list_effects")
      .then((list) => {
        setEffects(list);
        if (list.length > 0) {
          setSelectedId(list[0].id);
          setParamValues(defaultsFor(list[0]));
        }
      })
      .catch((e) => setLoadError(String(e)));
  }, []);

  const selected = useMemo(
    () => effects.find((e) => e.id === selectedId) ?? null,
    [effects, selectedId],
  );

  function pickEffect(id: string) {
    setSelectedId(id);
    const eff = effects.find((e) => e.id === id);
    if (eff) setParamValues(defaultsFor(eff));
    setRenderError(null);
  }

  function setParam(id: string, v: ParamValue) {
    setParamValues((prev) => ({ ...prev, [id]: v }));
  }

  async function render() {
    if (!selected) return;
    if (!inputPath || !outputPath) {
      setRenderError("input and output paths are required");
      return;
    }
    setBusy(true);
    setRenderError(null);
    try {
      await invoke<void>("apply_effect", {
        inputPath,
        outputPath,
        effectId: selected.id,
        params: paramValues,
      });
      setRenderedAt(Date.now());
    } catch (e) {
      setRenderError(String(e));
    } finally {
      setBusy(false);
    }
  }

  // Cache-bust the preview <img> after every successful render.
  const previewSrc = useMemo(() => {
    if (!outputPath || renderedAt === null) return null;
    try {
      return convertFileSrc(outputPath) + `?t=${renderedAt}`;
    } catch {
      return null;
    }
  }, [outputPath, renderedAt]);

  return (
    <div className="lumen-app">
      <header className="lumen-header">
        <span className="lumen-brand">LUMEN</span>
        <span className="lumen-meta">
          desktop · in-process pipeline ·{" "}
          <b>{effects.length}</b> effects loaded
        </span>
      </header>

      {loadError && (
        <div className="lumen-banner lumen-banner-error">
          failed to load effects: {loadError}
        </div>
      )}

      <main className="lumen-main">
        <section className="lumen-panel">
          <h2>Files</h2>
          <div className="lumen-row">
            <label>
              <span>Input path</span>
              <input
                type="text"
                value={inputPath}
                placeholder="/path/to/input.png"
                onChange={(e) => setInputPath(e.target.value)}
              />
            </label>
          </div>
          <div className="lumen-row">
            <label>
              <span>Output path</span>
              <input
                type="text"
                value={outputPath}
                placeholder="/path/to/output.png"
                onChange={(e) => setOutputPath(e.target.value)}
              />
            </label>
          </div>
        </section>

        <section className="lumen-panel">
          <h2>Effect</h2>
          <div className="lumen-row">
            <select
              value={selectedId ?? ""}
              onChange={(e) => pickEffect(e.target.value)}
            >
              {effects.map((eff) => (
                <option key={eff.id} value={eff.id}>
                  {eff.display_name} — {eff.id}
                </option>
              ))}
            </select>
          </div>

          {selected && (
            <>
              <p className="lumen-desc">{selected.description}</p>
              <div className="lumen-params">
                {selected.parameters.length === 0 && (
                  <span className="lumen-dim">no parameters</span>
                )}
                {selected.parameters.map((p) => (
                  <ParamRow
                    key={p.id}
                    spec={p}
                    value={paramValues[p.id]}
                    onChange={(v) => setParam(p.id, v)}
                  />
                ))}
              </div>
            </>
          )}
        </section>

        <section className="lumen-panel">
          <h2>Render</h2>
          <div className="lumen-row">
            <button
              className="lumen-cta"
              onClick={render}
              disabled={busy || !selected}
            >
              {busy ? "Rendering…" : "Render"}
            </button>
            {renderError && (
              <span className="lumen-err">{renderError}</span>
            )}
          </div>
          <div className="lumen-preview">
            {previewSrc ? (
              <img alt="output preview" src={previewSrc} />
            ) : (
              <span className="lumen-dim">
                output preview will appear here after a successful render
              </span>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

function ParamRow({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec;
  value: ParamValue | undefined;
  onChange: (v: ParamValue) => void;
}) {
  const v = value ?? (spec.default as ParamValue);

  switch (spec.kind) {
    case "bool":
      return (
        <label className="lumen-param">
          <span>{spec.display_name}</span>
          <input
            type="checkbox"
            checked={Boolean(v)}
            onChange={(e) => onChange(e.target.checked)}
          />
          <small>{spec.description}</small>
        </label>
      );
    case "int":
      return (
        <label className="lumen-param">
          <span>{spec.display_name}</span>
          <input
            type="number"
            step={1}
            min={spec.min ?? undefined}
            max={spec.max ?? undefined}
            value={Number(v)}
            onChange={(e) => onChange(parseInt(e.target.value, 10) || 0)}
          />
          <small>{spec.description}</small>
        </label>
      );
    case "float":
      return (
        <label className="lumen-param">
          <span>{spec.display_name}</span>
          <input
            type="number"
            step={0.01}
            min={spec.min ?? undefined}
            max={spec.max ?? undefined}
            value={Number(v)}
            onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
          />
          <small>{spec.description}</small>
        </label>
      );
    case "choice":
      return (
        <label className="lumen-param">
          <span>{spec.display_name}</span>
          <select
            value={String(v)}
            onChange={(e) => onChange(e.target.value)}
          >
            {(spec.options ?? []).map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
          </select>
          <small>{spec.description}</small>
        </label>
      );
    case "string":
    default:
      return (
        <label className="lumen-param">
          <span>{spec.display_name}</span>
          <input
            type="text"
            value={String(v)}
            onChange={(e) => onChange(e.target.value)}
          />
          <small>{spec.description}</small>
        </label>
      );
  }
}

export default App;
