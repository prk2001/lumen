import "./App.css";

const SERVE_URL = "http://127.0.0.1:8723/";

function App() {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        width: "100vw",
        height: "100vh",
        margin: 0,
        padding: 0,
        background: "#0e0e10",
        color: "#e6e6e6",
        fontFamily:
          "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
      }}
    >
      <header
        style={{
          flex: "0 0 auto",
          padding: "8px 14px",
          borderBottom: "1px solid #2a2a2e",
          background: "#1a1a1d",
          fontSize: 12,
          letterSpacing: 0.2,
          display: "flex",
          alignItems: "center",
          gap: 10,
        }}
      >
        <strong style={{ color: "#f0c040" }}>Lumen</strong>
        <span style={{ color: "#9a9aa0" }}>
          live preview &mdash; <code>lumen serve</code> must be running on
          <code style={{ marginLeft: 4 }}>:8723</code>
        </span>
      </header>
      <iframe
        src={SERVE_URL}
        title="Lumen live preview"
        style={{
          flex: "1 1 auto",
          width: "100%",
          border: 0,
          background: "#0e0e10",
        }}
      />
    </div>
  );
}

export default App;
