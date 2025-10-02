# app/fastapi_app.py
#!/usr/bin/env python3
from __future__ import annotations

import os, io, cv2, sys, yaml, json, time, queue, threading
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Depends, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi  # for custom OpenAPI (Authorize button)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from traffic_core.detector_yolov8 import Yolov8Detector
from traffic_core.tracker_deepsort import DeepSortTracker
from traffic_core.accident_logic import AccidentEngine
from traffic_core.roi import inside_roi
from traffic_core.vis import draw_roi, draw_track, draw_fps
from traffic_core.homography import parse_h_from_cfg, image_to_world_m, speed_kmh_from_tracks

from .db import SessionLocal, engine, Base
from .models import User, Event
from .auth import hash_password, verify_password, make_token, decode_token

load_dotenv()
Base.metadata.create_all(bind=engine)

def load_cfg(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

app = FastAPI(title="Accident Detection API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ---------- State ----------
class State:
    def __init__(self):
        self.cfg: Dict[str, Any] = {}
        self.detector: Optional[Yolov8Detector] = None
        self.tracker: Optional[DeepSortTracker] = None
        self.engine: Optional[AccidentEngine] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.resize: Optional[Tuple[int, int]] = None
        self.stride: int = 1
        self.fps: float = 30.0
        self.allowed: List[str] = []
        self.polygon: List[List[int]] = []
        self.H: Optional[np.ndarray] = None
        self.ppm: float = 10.0
        self._frame_lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.q: "queue.Queue[dict]" = queue.Queue(512)
        self.stop = threading.Event()
        self.worker: Optional[threading.Thread] = None

STATE = State()

# ---------- Startup ----------
@app.on_event("startup")
def startup():
    cfg_path = os.getenv("ACCIDENT_CFG", "configs/infer.yaml")
    STATE.cfg = load_cfg(cfg_path)

    # Ensure admin user exists
    db = SessionLocal()
    try:
        admin = STATE.cfg.get("auth", {})
        email = admin.get("admin_user", "admin@example.com")
        password = admin.get("admin_pass", "Admin@12345")
        if not db.query(User).filter_by(email=email).first():
            db.add(User(email=email, password_hash=hash_password(password), role="admin"))
            db.commit()
    finally:
        db.close()

    # Detector
    det = STATE.cfg["model"]
    STATE.detector = Yolov8Detector(
        weights=det["weights"],
        device=det.get("device", 0),
        conf_thresh=det.get("conf_thresh", 0.45),
        iou_thresh=det.get("iou_thresh", 0.45),
        imgsz=det.get("imgsz", 640),
        target_classes=det.get("target_classes", None),
    )
    STATE.allowed = det.get("target_classes", ["car", "bus", "truck", "motorbike"])

    # Input
    inp = STATE.cfg["input"]
    src = inp["source"]
    STATE.resize = tuple(inp["resize"]) if inp.get("resize") else None
    STATE.stride = int(inp.get("stride", 1))
    fps_fb = float(inp.get("fallback_fps", 30))
    STATE.cap = cv2.VideoCapture(src)
    if not STATE.cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {src}")
    cap_fps = STATE.cap.get(cv2.CAP_PROP_FPS) or 0.0
    STATE.fps = cap_fps if cap_fps > 0 else fps_fb

    # ROI + calibration
    STATE.polygon = STATE.cfg["roi"]["polygon"]
    STATE.H = parse_h_from_cfg(STATE.cfg["calibration"])
    STATE.ppm = float(STATE.cfg["calibration"].get("pixels_per_meter", 10.0))

    # Tracker & engine
    STATE.tracker = DeepSortTracker()
    acc = STATE.cfg["accident"]
    STATE.engine = AccidentEngine(
        polygon=STATE.polygon,
        allowed_labels=STATE.allowed,
        fps=STATE.fps,
        min_speed_kmh=float(acc.get("min_speed_kmh", 3)),
        stopped_frames=int(acc.get("stopped_frames", 45)),
        sudden_decel_kmh=float(acc.get("sudden_decel_kmh", 25)),
        collision_iou=float(acc.get("collision_iou", 0.25)),
        near_miss_dist_px=float(acc.get("near_miss_dist_px", 40)),
    )

    # Start worker
    STATE.stop.clear()
    STATE.worker = threading.Thread(target=loop, daemon=True)
    STATE.worker.start()

@app.on_event("shutdown")
def shutdown():
    STATE.stop.set()
    if STATE.worker:
        STATE.worker.join(timeout=2)
    try:
        if STATE.cap:
            STATE.cap.release()
    except Exception:
        pass

# ---------- Worker ----------
def loop():
    frame_id = 0
    t_last = time.time()
    sm_fps = STATE.fps
    while not STATE.stop.is_set():
        ok, f = STATE.cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        if STATE.resize:
            f = cv2.resize(f, STATE.resize)

        if STATE.stride > 1 and (frame_id % STATE.stride != 0):
            frame_id += 1
            continue

        dets = STATE.detector.infer(f)
        dets = [d for d in dets if d["label"] in STATE.allowed and inside_roi(d["bbox"], STATE.polygon)]
        tracks = STATE.tracker.update(f, dets, fps=STATE.fps)

        # homography-based km/h
        for t in tracks:
            x, y, w, h = t["bbox"]
            cx = x + w / 2
            cy = y + h / 2
            if STATE.H is not None:
                prev = t.get("_prev_world")
                wx, wy = image_to_world_m(STATE.H, cx, cy)
                if prev:
                    t["speed_kmh"] = speed_kmh_from_tracks(prev, (wx, wy), STATE.fps)
                else:
                    t["speed_kmh"] = 0.0
                t["_prev_world"] = (wx, wy)
            else:
                pxps = t.get("speed_px_per_sec", 0.0)
                t["speed_kmh"] = (pxps / max(STATE.ppm, 1e-6)) * 3.6

        events = STATE.engine.step(tracks, pixels_per_meter=STATE.ppm)

        # persist + broadcast events
        if events:
            db = SessionLocal()
            try:
                for e in events:
                    db.add(
                        Event(
                            ts=datetime_from_epoch(e.get("time", time.time())),
                            etype=e.get("type", ""),
                            severity=e.get("severity", ""),
                            track_id=e.get("track_id", None),
                            a=e.get("a", None),
                            b=e.get("b", None),
                            extra={k: v for k, v in e.items() if k not in ("time", "type", "severity", "track_id", "a", "b")},
                        )
                    )
                db.commit()
            finally:
                db.close()

            # ---- Normalize and always broadcast { type: "accident", ... } ----
            for e in events:
                original_type = (e.get("type") or "").lower()
                looks_like_accident = any(k in original_type for k in [
                    "collision_confirmed","collision","crash","impact","accident"
                ])
                unified = {
                    "type": "accident" if looks_like_accident else (original_type or "event"),
                    "original_type": original_type,
                    "severity": e.get("severity") or ("critical" if looks_like_accident else "info"),
                    "event_id": str(e.get("id") or e.get("uid") or e.get("frame_index") or int(time.time()*1000)),
                    "track_id": e.get("track_id"),
                    "a": e.get("a"),
                    "b": e.get("b"),
                    "time": e.get("time", time.time()),
                    "extra": {k: v for k, v in e.items() if k not in {"type","severity","track_id","a","b","time"}},
                }
                try:
                    STATE.q.put_nowait(unified)
                    print("WS broadcast:", unified)  # debug; remove if noisy
                except queue.Full:
                    pass

        # draw preview frame
        draw_roi(f, STATE.polygon)
        for t in tracks:
            draw_track(f, t, show_speed=True)
        now = time.time()
        inst = 1.0 / max(1e-6, now - t_last)
        sm_fps = 0.9 * sm_fps + 0.1 * inst
        draw_fps(f, sm_fps)
        t_last = now
        with STATE._frame_lock:
            STATE.frame = f.copy()
        frame_id += 1

def datetime_from_epoch(ts: float):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "fps": STATE.fps, "resize": STATE.resize, "roi_pts": len(STATE.polygon)}

@app.post("/login")
def login(payload: Dict[str, str]):
    email = payload.get("email")
    password = payload.get("password")
    if not email or not password:
        raise HTTPException(status_code=422, detail="email and password are required")

    db = SessionLocal()
    try:
        u = db.query(User).filter_by(email=email).first()
        if not u or not verify_password(password, u.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        claims = {"sub": str(u.id), "email": u.email, "role": u.role}
        token = make_token(claims)
        return {"access_token": token, "token_type": "bearer"}
    finally:
        db.close()

def get_user(authorization: str = Header(default="")) -> Dict[str, Any]:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        return decode_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

@app.get("/events")
def list_events(limit: int = 100, user=Depends(get_user)):
    db = SessionLocal()
    try:
        q = db.query(Event).order_by(Event.id.desc()).limit(limit).all()
        return [
            {
                "id": e.id,
                "ts": e.ts.isoformat(),
                "type": e.etype,
                "severity": e.severity,
                "track_id": e.track_id,
                "a": e.a,
                "b": e.b,
                "extra": e.extra,
            }
            for e in q
        ]
    finally:
        db.close()

# Quick check of latest persisted event
@app.get("/events/latest")
def latest_event():
    db = SessionLocal()
    try:
        row = db.query(Event).order_by(Event.id.desc()).first()
        if not row:
            return {"ok": True, "latest": None}
        return {
            "ok": True,
            "latest": {
                "id": row.id,
                "ts": row.ts.isoformat(),
                "type": row.etype,
                "severity": row.severity,
                "extra": row.extra,
            }
        }
    finally:
        db.close()

# Manual test endpoint to force a popup on the UI
@app.post("/debug/accident")
def debug_accident():
    fake = {
        "type": "accident",
        "original_type": "debug_injected",
        "severity": "critical",
        "event_id": f"debug-{int(time.time())}",
        "time": time.time(),
        "extra": {"note": "manual trigger"}
    }
    try:
        STATE.q.put_nowait(fake)
    except queue.Full:
        pass
    return {"ok": True}

@app.get("/frame")
def frame():
    with STATE._frame_lock:
        f = None if STATE.frame is None else STATE.frame.copy()
    if f is None:
        return PlainTextResponse("No frame", status_code=404)
    ok, buf = cv2.imencode(".jpg", f)
    if not ok:
        return PlainTextResponse("Encode error", status_code=500)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")

@app.websocket("/events/ws")
async def ws(ws):
    await ws.accept()
    try:
        while True:
            try:
                e = STATE.q.get(timeout=1.0)
                await ws.send_text(json.dumps(e))
            except queue.Empty:
                await ws.send_text(json.dumps({"type": "heartbeat", "ts": time.time()}))
    except WebSocketDisconnect:
        return

# ---------- Swagger "Authorize" (Bearer JWT) ----------
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
        description="API for Accident Detection (YOLOv8 + DeepSORT).",
    )

    components = schema.setdefault("components", {}).setdefault("securitySchemes", {})
    components["BearerAuth"] = {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}

    paths = schema.get("paths", {})
    if "/events" in paths and "get" in paths["/events"]:
        paths["/events"]["get"]["security"] = [{"BearerAuth": []}]

    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi
