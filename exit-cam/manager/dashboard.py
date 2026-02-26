"""
manager/dashboard.py  —  Exit-Cam Dashboard
=============================================
WebSocket  ws://localhost:8001/ws
    {"event": "heartbeat",  "active_count": N, "unique_count": N, "ts": epoch}
    {"event": "enter",      "track_id": N, "conf": 0.xx, "zone": "...", "ts": epoch}
    {"event": "exit",       "track_id": N, "dwell": N.N, "ts": epoch, "refresh_recent": true}
    {"event": "new_entry",  "unique_count": N, "ts": epoch}

REST
    GET /api/stream              → MJPEG live frame stream
    GET /api/stats/session       → unique_total, today_count, active_now
    GET /api/stats/hourly        → [{hour, count}] for today
    GET /api/stats/emotions      → [{emotion, count}]
    GET /api/recent              → last 20 archived persons (with image paths)
    GET /api/gallery/flat        → all images in output/best/ (live buffer)
    GET /api/gallery/days        → tree: [{day, hours: [{hour, images:[]}]}]
    GET /api/gallery/session     → session gallery ordered by last_seen DESC
    GET /output/...              → static file serving for archived images
    GET /                        → dashboard UI
"""
from __future__ import annotations

import asyncio
import base64
import datetime
import json
import os
import sqlite3
import threading
import time
from typing import Optional

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

import config

# ── Backend WS config — read from config.py ───────────────────────────────────
BACKEND_WS_URL    = getattr(config, "BACKEND_WS_URL",    "")
BACKEND_WS_TOKEN  = getattr(config, "BACKEND_WS_TOKEN",  "changeme")
BACKEND_CAM_ID    = getattr(config, "BACKEND_CAM_ID",    "exit-cam")
BACKEND_FRAME_FPS = float(getattr(config, "BACKEND_FRAME_FPS", 1))

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUTPUT_DIR = os.path.join(_BASE_DIR, "output")
_STATIC_DIR = os.path.join(_BASE_DIR, "static")
_BEST_DIR   = os.path.join(_OUTPUT_DIR, "best")
_ARCHIVE_DB = os.path.join(_OUTPUT_DIR, "metadata.db")
_SESSION_DB = os.path.join(_OUTPUT_DIR, "exit_session.db")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Exit-Cam Dashboard")

for _d in (_STATIC_DIR, _OUTPUT_DIR, _BEST_DIR):
    os.makedirs(_d, exist_ok=True)

app.mount("/output", StaticFiles(directory=_OUTPUT_DIR), name="output")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")





# ── Image path normalisation ───────────────────────────────────────────────────
def _web_path(raw: str) -> str:
    """Convert an absolute filesystem path to a web-servable relative path."""
    if not raw:
        return ""
    p   = raw.replace("\\", "/")
    idx = p.find("output/")
    if idx >= 0:
        return p[idx:]
    return ""


# ── MJPEG stream ───────────────────────────────────────────────────────────────
_latest_frame: Optional[bytes] = None
_frame_lock = threading.Lock()
_push_frame_last: float = 0.0
_push_frame_interval: float = 1.0 / max(BACKEND_FRAME_FPS, 1.0) if BACKEND_FRAME_FPS > 0 else 0.1


def push_frame(frame) -> None:
    """Encode a BGR frame at most PUSH_FRAME_FPS times per second, cache for MJPEG,
    and forward to backend WS.  Skips encoding on frames that arrive too fast."""
    global _latest_frame, _push_frame_last
    _interval = 1.0 / max(float(getattr(config, "PUSH_FRAME_FPS", 10)), 1.0)
    now = time.time()
    if now - _push_frame_last < _interval:
        return
    _push_frame_last = now
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
    if ok:
        frame_bytes = buf.tobytes()
        with _frame_lock:
            _latest_frame = frame_bytes
        _backend_enqueue_frame(frame_bytes)


def _mjpeg_gen():
    while True:
        with _frame_lock:
            f = _latest_frame
        if f:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
        time.sleep(0.033)


@app.get("/api/stream")
async def stream():
    return StreamingResponse(
        _mjpeg_gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── WebSocket ──────────────────────────────────────────────────────────────────
_ws_clients: list[WebSocket] = []
_ws_lock    = threading.Lock()
_event_loop: Optional[asyncio.AbstractEventLoop] = None


# ── Outbound backend WebSocket client ─────────────────────────────────────────
# Queues — asyncio.Queue is safe to put() from threads via run_coroutine_threadsafe
_backend_event_q: Optional[asyncio.Queue] = None   # JSON-serialisable dicts
_backend_frame_q: Optional[asyncio.Queue] = None   # raw JPEG bytes


async def _backend_ws_loop() -> None:
    """
    Persistent outbound WebSocket client.
    - Connects to BACKEND_WS_URL with Bearer token auth.
    - Drains _backend_event_q and _backend_frame_q and sends them.
    - Reconnects automatically with exponential backoff on any error.
    - Runs entirely inside the uvicorn event loop — never blocks the pipeline.
    """
    import websockets  # optional dep; only imported when URL is configured
    import ssl as _ssl

    backoff  = 1.0
    attempt  = 0
    headers  = {"Authorization": f"Bearer {BACKEND_WS_TOKEN}"}
    # Allow self-signed / ngrok certs; backend token is the auth layer
    _ssl_ctx = _ssl.create_default_context()
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode    = _ssl.CERT_NONE

    while True:
        attempt += 1
        try:
            async with websockets.connect(
                BACKEND_WS_URL,
                additional_headers=headers,
                ssl=_ssl_ctx if BACKEND_WS_URL.startswith("wss") else None,
                ping_interval=20,
                ping_timeout=10,
            ) as ws:
                if attempt > 1:
                    print(f"[Backend WS] reconnected (attempt {attempt}) → {BACKEND_WS_URL}")
                else:
                    print(f"[Backend WS] connected → {BACKEND_WS_URL}")
                backoff = 1.0  # reset on successful connect
                attempt = 0

                while True:
                    # Drain both queues, prioritising events over frames
                    try:
                        payload = _backend_event_q.get_nowait()
                        await ws.send(json.dumps(payload))
                        _backend_event_q.task_done()
                        continue
                    except asyncio.QueueEmpty:
                        pass

                    try:
                        frame_b64 = _backend_frame_q.get_nowait()
                        await ws.send(json.dumps({
                            "type":   "frame",
                            "cam":    BACKEND_CAM_ID,
                            "image":  frame_b64,
                            "ts":     time.time(),
                        }))
                        _backend_frame_q.task_done()
                        continue
                    except asyncio.QueueEmpty:
                        pass

                    # Nothing queued — yield briefly
                    await asyncio.sleep(0.05)

        except Exception as exc:
            print(f"[Backend WS] disconnected — {exc} | retry in {backoff:.0f}s (attempt {attempt})")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


def _backend_enqueue_event(payload: dict) -> None:
    """Thread-safe: put an event dict onto the backend event queue."""
    if not _event_loop or not _backend_event_q or not BACKEND_WS_URL:
        return
    asyncio.run_coroutine_threadsafe(
        _backend_event_q.put(payload), _event_loop
    )


# Frame throttle state
_backend_last_frame_t: float = 0.0


def _backend_enqueue_frame(frame_bytes: bytes) -> None:
    """Thread-safe: enqueue a JPEG frame for backend forwarding (throttled)."""
    global _backend_last_frame_t
    if not _event_loop or not _backend_frame_q or not BACKEND_WS_URL:
        return
    if BACKEND_FRAME_FPS <= 0:
        return
    now = time.time()
    if now - _backend_last_frame_t < 1.0 / BACKEND_FRAME_FPS:
        return
    _backend_last_frame_t = now
    b64 = base64.b64encode(frame_bytes).decode()
    # maxsize=1 queue: drop old frame if backend is slow (always send latest)
    try:
        _backend_frame_q.get_nowait()  # discard stale frame
        _backend_frame_q.task_done()
    except asyncio.QueueEmpty:
        pass
    asyncio.run_coroutine_threadsafe(
        _backend_frame_q.put(b64), _event_loop
    )


@app.on_event("startup")
async def _grab_loop():
    global _event_loop, _backend_event_q, _backend_frame_q
    _event_loop       = asyncio.get_running_loop()
    _backend_event_q  = asyncio.Queue(maxsize=256)
    _backend_frame_q  = asyncio.Queue(maxsize=1)
    if BACKEND_WS_URL:
        asyncio.create_task(_backend_ws_loop())
        print(f"[Backend WS] client configured → {BACKEND_WS_URL} (cam={BACKEND_CAM_ID})")
    else:
        print("[Backend WS] disabled (BACKEND_WS_URL not set)")


def _broadcast(payload: dict) -> None:
    if not _event_loop:
        return
    with _ws_lock:
        clients = list(_ws_clients)
    dead = []
    for ws in clients:
        if ws.client_state != WebSocketState.CONNECTED:
            dead.append(ws)
            continue
        try:
            asyncio.run_coroutine_threadsafe(ws.send_json(payload), _event_loop)
        except Exception:
            dead.append(ws)
    if dead:
        with _ws_lock:
            for ws in dead:
                if ws in _ws_clients:
                    _ws_clients.remove(ws)


def _emit(payload: dict) -> None:
    """Broadcast to local WebSocket clients AND forward to backend WS."""
    _broadcast(payload)
    _backend_enqueue_event({"type": "event", "cam": BACKEND_CAM_ID, "data": payload})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    with _ws_lock:
        _ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, Exception):
        with _ws_lock:
            if ws in _ws_clients:
                _ws_clients.remove(ws)


# ── DB helpers ─────────────────────────────────────────────────────────────────
def _q_archive(sql: str, params: tuple = ()) -> list[dict]:
    """Run a read-only query against the archive metadata DB."""
    if not os.path.exists(_ARCHIVE_DB):
        return []
    try:
        con = sqlite3.connect(_ARCHIVE_DB)
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, params).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _q_session(sql: str, params: tuple = ()) -> list[dict]:
    """Run a read-only query against the session gallery DB."""
    if not os.path.exists(_SESSION_DB):
        return []
    try:
        con = sqlite3.connect(_SESSION_DB)
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, params).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _today_epoch() -> float:
    """Return Unix timestamp for the start of today (midnight local time)."""
    d = datetime.date.today()
    return float(int(datetime.datetime(d.year, d.month, d.day).timestamp()))


# ── REST endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    p = os.path.join(_STATIC_DIR, "index.html")
    return (
        FileResponse(p)
        if os.path.exists(p)
        else JSONResponse({"error": "index.html missing"}, status_code=404)
    )


@app.get("/api/stats/session")
async def stats_session():
    """Total unique exits (all-time), today's count, and currently active count."""
    total = _q_session("SELECT COUNT(*) AS n FROM session_gallery")
    today = _q_session(
        "SELECT COUNT(*) AS n FROM session_gallery WHERE first_exit >= ?",
        (_today_epoch(),),
    )
    return {
        "unique_total": total[0]["n"] if total else 0,
        "today_count":  today[0]["n"] if today  else 0,
        "active_now":   _active_ref[0],
    }


@app.get("/api/stats/hourly")
async def stats_hourly():
    """Exit counts per hour for today (from session gallery)."""
    return _q_session(
        """
        SELECT CAST(
                   strftime('%H', datetime(first_exit, 'unixepoch', 'localtime'))
               AS INTEGER) AS hour,
               COUNT(*) AS count
        FROM   session_gallery
        WHERE  first_exit >= ?
        GROUP  BY hour
        ORDER  BY hour
        """,
        (_today_epoch(),),
    )


@app.get("/api/stats/emotions")
async def stats_emotions():
    """Emotion distribution from all archived persons."""
    return _q_archive(
        """
        SELECT COALESCE(emotion, 'Undetected') AS emotion, COUNT(*) AS count
        FROM   person
        GROUP  BY emotion
        ORDER  BY count DESC
        """
    )


@app.get("/api/recent")
async def recent():
    """Last 20 archived persons with web-accessible image paths."""
    rows = _q_archive(
        """
        SELECT track_id, best_conf, first_seen, last_seen,
               ROUND(last_seen - first_seen, 1) AS dwell,
               emotion, emotion_score, image_path
        FROM   person
        ORDER  BY id DESC
        LIMIT  20
        """
    )
    for r in rows:
        r["image_path"] = _web_path(r.get("image_path", ""))
    return rows


@app.get("/api/gallery/session")
async def gallery_session():
    """Session gallery: all unique exits, most-recent first."""
    rows = _q_session(
        """
        SELECT cid, exit_count, first_exit, last_seen, image_path
        FROM   session_gallery
        ORDER  BY last_seen DESC
        """
    )
    for r in rows:
        r["image_path"] = _web_path(r.get("image_path", ""))
    return rows


@app.get("/api/gallery/flat")
async def gallery_flat():
    """All JPEG files directly inside output/best/ (live buffer)."""
    if not os.path.exists(_BEST_DIR):
        return []
    files = sorted(
        [f for f in os.listdir(_BEST_DIR) if f.endswith(".jpg")],
        key=lambda f: os.path.getmtime(os.path.join(_BEST_DIR, f)),
        reverse=True,
    )
    return [{"path": f"output/best/{f}", "name": f} for f in files]


@app.get("/api/gallery/days")
async def gallery_days():
    """Archived images organised by day → hour → files."""
    if not os.path.exists(_BEST_DIR):
        return []
    result   = []
    day_dirs = sorted(
        [
            d for d in os.listdir(_BEST_DIR)
            if os.path.isdir(os.path.join(_BEST_DIR, d)) and d.startswith("day_")
        ],
        reverse=True,
    )
    for day in day_dirs:
        dp    = os.path.join(_BEST_DIR, day)
        hours = []
        for hour in sorted(
            [
                h for h in os.listdir(dp)
                if os.path.isdir(os.path.join(dp, h)) and h.startswith("hour_")
            ],
            reverse=True,
        ):
            hp     = os.path.join(dp, hour)
            images = sorted(
                [f for f in os.listdir(hp) if f.endswith(".jpg")],
                key=lambda f: os.path.getmtime(os.path.join(hp, f)),
                reverse=True,
            )
            hours.append(
                {
                    "hour":   hour,
                    "images": [
                        {"path": f"output/best/{day}/{hour}/{f}", "name": f}
                        for f in images
                    ],
                }
            )
        result.append({"day": day, "hours": hours})
    return result


# ── Pipeline state ─────────────────────────────────────────────────────────────
_prev_ids:   set       = set()
_seen_cache: dict      = {}   # track_id → first_seen unix timestamp
_active_ref: list[int] = [0]  # mutable single-element list so REST can read it
_last_hb:    float     = 0.0
_state:      dict      = {}
_TWO_HOURS             = 7200


# ── Pipeline hook ──────────────────────────────────────────────────────────────
def notify(tracks: list[dict], identity_manager) -> None:
    """
    Called every pipeline tick with the current list of active track dicts.

    Emits events to both local WebSocket clients and the VPS:
      • enter      — a new track_id appeared in the frame
      • exit       — a track_id left the frame (includes dwell time)
      • new_entry  — the global unique-exit counter incremented
      • heartbeat  — sent at most once per second with live counts
    """
    global _prev_ids, _last_hb, _seen_cache

    now         = time.time()
    current_ids = {t["track_id"] for t in tracks}
    _active_ref[0] = len(current_ids)

    # Purge stale seen-cache entries (older than 2 h)
    stale = [k for k, v in _seen_cache.items() if now - v > _TWO_HOURS]
    for tid in stale:
        del _seen_cache[tid]

    # Populate seen-cache from identity_manager state
    for tid, state in identity_manager.get_state().items():
        if tid not in _seen_cache:
            _seen_cache[tid] = state.first_seen

    # Enter events
    for tid in current_ids - _prev_ids:
        t = next((x for x in tracks if x["track_id"] == tid), None)
        if t:
            _emit({
                "event":    "enter",
                "track_id": tid,
                "conf":     round(t.get("conf", 0), 3),
                "zone":     t.get("zone") or "outside",
                "ts":       now,
            })

    # Exit events
    for tid in _prev_ids - current_ids:
        first = _seen_cache.pop(tid, None)
        _emit({
            "event":          "exit",
            "track_id":       tid,
            "dwell":          round(now - first, 1) if first else 0.0,
            "ts":             now,
            "refresh_recent": True,
        })

    # New unique-exit event
    unique = identity_manager.unique_exit_count
    if unique != _state.get("last_unique", unique):
        _emit({"event": "new_entry", "unique_count": unique, "ts": now})
    _state["last_unique"] = unique

    _prev_ids = current_ids

    # Heartbeat (max once per second)
    if now - _last_hb >= 1.0:
        _emit({
            "event":        "heartbeat",
            "active_count": len(current_ids),
            "unique_count": unique,
            "ts":           now,
        })
        _last_hb = now


# ── Server entrypoint ──────────────────────────────────────────────────────────
def start(host: str = "0.0.0.0", port: int = 8001) -> None:
    # Wire up emotion-archived callback so background archive threads
    # (which run outside asyncio) can forward the result to the backend WS.
    from . import identity_manager as _im_mod
    _im_mod._on_archived = lambda ev: _backend_enqueue_event(
        {"type": "event", "cam": BACKEND_CAM_ID, "data": ev}
    )

    cfg = uvicorn.Config(app, host=host, port=port, log_level="warning")
    srv = uvicorn.Server(cfg)
    threading.Thread(target=srv.run, daemon=True).start()
    print(f"[Dashboard] http://localhost:{port}")
