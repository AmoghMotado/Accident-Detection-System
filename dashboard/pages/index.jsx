import { useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API || "http://localhost:8000";
// These types from older engines are still considered "accident" if they appear
const ACCIDENT_KEYS = ["collision_confirmed","collision_overlap","collision","crash","impact"];

export default function Home() {
  const [token, setToken] = useState("");
  const [health, setHealth] = useState(null);
  const [banner, setBanner] = useState(null);
  const [soundEnabled, setSoundEnabled] = useState(false);

  const wsRef = useRef(null);
  const imgRef = useRef(null);
  const alarmRef = useRef(null);

  // login once (only if API needs JWT)
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API}/login`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ email: "admin@example.com", password: "Admin@12345" })
        });
        const data = await res.json();
        if (res.ok && data.access_token) setToken(data.access_token);
      } catch {}
    })();
  }, []);

  // poll /health
  useEffect(() => {
    const tick = async () => {
      try {
        const r = await fetch(`${API}/health`);
        setHealth(await r.json());
      } catch {}
    };
    tick();
    const id = setInterval(tick, 3000);
    return () => clearInterval(id);
  }, []);

  // WebSocket: popup + alarm on accident
  useEffect(() => {
    let ws;
    let retry = 0;

    const buildWsUrl = (path) =>
      (process.env.NEXT_PUBLIC_API || "http://localhost:8000").replace("http", "ws") + path;

    // Try the newer endpoint first, then fall back (some older builds used /ws)
    const tryEndpoints = ["/events/ws", "/ws"];

    const connect = (idx = 0) => {
      const url = buildWsUrl(tryEndpoints[idx]);
      ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => { retry = 0; console.log("[WS] connected:", url); };

      ws.onmessage = (msg) => {
        try {
          const e = JSON.parse(msg.data);
          if (e?.type !== "heartbeat") console.log("[WS msg]", e);

          // New server always sends { type: "accident", ... }.
          // Keep a fallback for engines that emit other names.
          const t = String(e?.type || "").toLowerCase();
          const sev = String(e?.severity || "").toLowerCase();
          const isAccident = t === "accident" || ACCIDENT_KEYS.some(k => t.includes(k)) || sev === "critical";

          if (isAccident) {
            setBanner("🚨 ACCIDENT DETECTED");
            if (soundEnabled && alarmRef.current) {
              alarmRef.current.currentTime = 0;
              alarmRef.current.play().catch(()=>{});
            }
            setTimeout(() => setBanner(null), 5000);
          }
        } catch {}
      };

      ws.onclose = () => {
        if (idx === 0) {
          console.warn("[WS] primary closed, trying fallback…");
          connect(1);
          return;
        }
        const delay = Math.min(30000, 1000 * 2 ** retry++);
        console.warn("[WS] closed, retrying in", delay, "ms");
        setTimeout(() => connect(idx), delay);
      };
    };

    connect();
    return () => ws && ws.close();
  }, [soundEnabled]);

  // refresh still frame as a “stream”
  useEffect(() => {
    const id = setInterval(() => {
      if (imgRef.current) imgRef.current.src = `${API}/frame?ts=${Date.now()}`;
    }, 200);
    return () => clearInterval(id);
  }, []);

  // user must click once to allow audio
  const enableAlerts = () => {
    setSoundEnabled(true);
    const a = alarmRef.current;
    if (a) { a.play().then(() => a.pause()); }
  };

  // ====== Styles ======
  const page = {
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
    background: "#f4f6f8",
    color: "#111827",
    minHeight: "100vh",
    margin: 0,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: 16,
    boxSizing: "border-box"
  };
  const h1 = { margin: 0, marginBottom: 8, textAlign: "center", fontWeight: 800 };
  const sub = { marginBottom: 12, fontSize: 14, opacity: .8, textAlign: "center" };
  const shell = {
    position: "relative",
    width: "min(96vw, 1650px)",
    height: "min(82vh, 950px)",
    borderRadius: 12,
    overflow: "hidden",
    boxShadow: "0 10px 30px rgba(0,0,0,.12)",
    background: "transparent"
  };
  const img = {
    position: "absolute",
    inset: 0,
    width: "100%",
    height: "100%",
    objectFit: "contain",
    background: "transparent"
  };
  const bannerBox = {
    position: "absolute",
    left: "50%",
    bottom: 24,
    transform: "translateX(-50%)",
    padding: "12px 18px",
    borderRadius: 999,
    background: "rgba(220, 38, 38, 0.95)",
    color: "#fff",
    fontWeight: 800,
    letterSpacing: 0.6,
    boxShadow: "0 8px 20px rgba(220,38,38,.35)",
    border: "1px solid rgba(255,255,255,.25)"
  };
  const btn = (active) => ({
    marginTop: 14,
    padding: "10px 14px",
    borderRadius: 10,
    border: "1px solid " + (active ? "#22c55e" : "#9ca3af"),
    background: active ? "#86efac" : "#e5e7eb",
    color: "#111827",
    cursor: "pointer",
    fontWeight: 700
  });

  return (
    <div style={page}>
      <h1 style={h1}>🚦 Accident Detection – Live Monitor</h1>
      <div style={sub}>
        {health && (
          <>
            <b>Status:</b> {health.ok ? "✅ OK" : "❌ OFF"} &nbsp;•&nbsp;
            <b>FPS:</b> {health.fps?.toFixed?.(1)} &nbsp;•&nbsp;
            <b>ROI pts:</b> {health.roi_pts} &nbsp;•&nbsp;
            <b>Resize:</b> {health.resize ? health.resize.join("×") : "—"}
          </>
        )}
      </div>

      <div style={shell}>
        <img ref={imgRef} src={`${API}/frame`} alt="live" style={img} />
        {banner && <div style={bannerBox}>{banner}</div>}
      </div>

      <button onClick={enableAlerts} style={btn(soundEnabled)}>
        {soundEnabled ? "🔔 Alerts Enabled" : "🔕 Enable Alerts"}
      </button>

      {/* Place siren at dashboard/public/alarm.mp3 */}
      <audio ref={alarmRef} src="/alarm.mp3" preload="auto" />
    </div>
  );
}
