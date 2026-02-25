"""
manager/dashboard.py
====================
FastAPI server — runs in a background daemon thread inside main.py.

WebSocket  ws://localhost:8000/ws
    {"event": "heartbeat",  "active_count": N, "ts": epoch}
    {"event": "enter",      "track_id": N, "conf": 0.xx, "ts": epoch}
    {"event": "exit",       "track_id": N, "dwell": N.N, "ts": epoch, "refresh_recent": true}

REST
    GET /api/stream              → MJPEG live frame stream
    GET /api/stats/today         → total, avg_dwell, avg_conf
    GET /api/stats/hourly        → [{hour, count}] for today
    GET /api/stats/emotions      → [{emotion, count}]
    GET /api/recent              → last 20 archived persons (with image paths)
    GET /api/gallery/flat        → all images in output/best/ (live buffer)
    GET /api/gallery/days        → tree: [{day, hours: [{hour, images:[]}]}]
    GET /output/...              → static file serving for archived images
    GET /                        → dashboard UI
"""
from __future__ import annotations

import asyncio
import glob
import os
import re
import sqlite3
import threading
import time
from typing import Optional

import config

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

# ── paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUTPUT_DIR = os.path.join(_BASE_DIR, "output")
_STATIC_DIR = os.path.join(_BASE_DIR, "static")
_BEST_DIR   = os.path.join(_BASE_DIR, "output", "best")


def _get_db_path() -> Optional[str]:
    for p in [
        os.path.join(_BASE_DIR, "metadata.db"),
        os.path.join(_BEST_DIR, "metadata.db"),
    ]:
        if os.path.exists(p):
            return p
    return None


# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Exit-Cam Dashboard")

os.makedirs(_STATIC_DIR, exist_ok=True)
os.makedirs(_OUTPUT_DIR, exist_ok=True)

app.mount("/output", StaticFiles(directory=_OUTPUT_DIR), name="output")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ── MJPEG live stream ──────────────────────────────────────────────────────────
# latest_frame is set by push_frame() called from main.py pipeline
_latest_frame: Optional[bytes] = None
_frame_lock = threading.Lock()


def push_frame(frame) -> None:
    """Call from pipeline() with the annotated BGR frame."""
    global _latest_frame
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if ok:
        with _frame_lock:
            _latest_frame = buf.tobytes()


def _mjpeg_generator():
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(0.04)   # ~25 fps cap


@app.get("/api/stream")
async def stream():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── WebSocket ──────────────────────────────────────────────────────────────────
_ws_clients: list[WebSocket] = []
_ws_lock = threading.Lock()
_event_loop: Optional[asyncio.AbstractEventLoop] = None


@app.on_event("startup")
async def _grab_loop():
    global _event_loop
    _event_loop = asyncio.get_running_loop()


def _broadcast(payload: dict):
    if _event_loop is None:
        return
    with _ws_lock:
        clients = list(_ws_clients)
    dead = []
    for ws in clients:
        # Skip sockets already known to be disconnected
        if ws.client_state != WebSocketState.CONNECTED:
            dead.append(ws)
            continue
        try:
            asyncio.run_coroutine_threadsafe(ws.send_json(payload), _event_loop)
        except Exception:
            dead.append(ws)
    # Clean up dead connections so the list doesn't grow over days
    if dead:
        with _ws_lock:
            for ws in dead:
                if ws in _ws_clients:
                    _ws_clients.remove(ws)


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


# ── DB ─────────────────────────────────────────────────────────────────────────
def _query(sql: str, params: tuple = ()) -> list[dict]:
    db = _get_db_path()
    if not db:
        return []
    try:
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


# ── REST ───────────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    p = os.path.join(_STATIC_DIR, "index.html")
    return FileResponse(p) if os.path.exists(p) else JSONResponse(
        {"error": "index.html missing from static/"}, status_code=404)


@app.get("/api/config")
async def api_config():
    """Expose pipeline tuning values the frontend needs (read-only)."""
    return {
        "reid_buffer_seconds": getattr(config, "REID_BUFFER_SECONDS",
                                        getattr(config, "REID_BUFFER_FRAMES", 90) / 30),
    }


@app.get("/api/stats/today")
async def stats_today():
    rows = _query("""
        SELECT COUNT(*) AS total,
               ROUND(AVG(last_seen - first_seen), 2) AS avg_dwell,
               ROUND(AVG(best_conf), 3) AS avg_conf
        FROM person
        WHERE first_seen >= strftime('%s','now','start of day')
    """)
    return rows[0] if rows else {"total": 0, "avg_dwell": 0, "avg_conf": 0}


@app.get("/api/stats/hourly")
async def stats_hourly():
    return _query("""
        SELECT CAST(strftime('%H', datetime(first_seen,'unixepoch','localtime'))
                    AS INTEGER) AS hour,
               COUNT(*) AS count
        FROM person
        WHERE first_seen >= strftime('%s','now','start of day')
        GROUP BY hour ORDER BY hour
    """)


@app.get("/api/stats/emotions")
async def stats_emotions():
    return _query("""
        SELECT COALESCE(emotion,'Undetected') AS emotion, COUNT(*) AS count
        FROM person GROUP BY emotion ORDER BY count DESC
    """)


@app.get("/api/recent")
async def recent():
    return _query("""
        SELECT track_id, best_conf, first_seen, last_seen,
               ROUND(last_seen - first_seen, 1) AS dwell,
               emotion, emotion_score, image_path
        FROM person ORDER BY id DESC LIMIT 20
    """)


@app.get("/api/gallery/flat")
async def gallery_flat():
    """All .jpg files currently in output/best/ (live best-frame buffer)."""
    best = os.path.join(_BEST_DIR)
    if not os.path.exists(best):
        return []
    files = sorted(
        [f for f in os.listdir(best) if f.endswith(".jpg")],
        key=lambda f: os.path.getmtime(os.path.join(best, f)),
        reverse=True,
    )
    return [{"path": f"output/best/{f}", "name": f} for f in files]


@app.get("/api/gallery/days")
async def gallery_days():
    """
    Returns tree structure:
    [{ day: "day_20260223", hours: [{ hour: "hour_09", images: [...] }] }]
    """
    if not os.path.exists(_BEST_DIR):
        return []
    result = []
    day_dirs = sorted(
        [d for d in os.listdir(_BEST_DIR)
         if os.path.isdir(os.path.join(_BEST_DIR, d)) and d.startswith("day_")],
        reverse=True,
    )
    for day in day_dirs:
        day_path = os.path.join(_BEST_DIR, day)
        hours = []
        hour_dirs = sorted(
            [h for h in os.listdir(day_path)
             if os.path.isdir(os.path.join(day_path, h)) and h.startswith("hour_")],
            reverse=True,
        )
        for hour in hour_dirs:
            hour_path = os.path.join(day_path, hour)
            images = sorted(
                [f for f in os.listdir(hour_path) if f.endswith(".jpg")],
                key=lambda f: os.path.getmtime(os.path.join(hour_path, f)),
                reverse=True,
            )
            hours.append({
                "hour": hour,
                "images": [
                    {"path": f"output/best/{day}/{hour}/{f}", "name": f}
                    for f in images
                ],
            })
        result.append({"day": day, "hours": hours})
    return result


# ── Pipeline hooks ─────────────────────────────────────────────────────────────
_prev_active_ids: set[int] = set()
_last_heartbeat: float = 0.0
# Store first_seen per track so we have it after person leaves active state
_first_seen_cache: dict[int, float] = {}


def notify(tracks: list[dict], identity_manager) -> None:
    global _prev_active_ids, _last_heartbeat, _first_seen_cache

    now = time.time()
    current_ids = {t["track_id"] for t in tracks}

    # Safety net: purge cache entries older than 2 hours.
    # Normally every entry is removed on exit, but if an exit event is
    # ever missed (e.g. frame drop at the exact moment of disappearance)
    # this prevents the dict growing unboundedly over a 3-day run.
    _TWO_HOURS = 7200
    stale_ids = [tid for tid, ts in _first_seen_cache.items()
                 if now - ts > _TWO_HOURS]
    for tid in stale_ids:
        del _first_seen_cache[tid]

    # Cache first_seen while person is active (so we have it at exit time)
    active_states = identity_manager.get_state()
    for tid, state in active_states.items():
        if tid not in _first_seen_cache:
            _first_seen_cache[tid] = state.first_seen

    # ── Enter events ──────────────────────────────────────────────────────────
    for tid in current_ids - _prev_active_ids:
        track = next((t for t in tracks if t["track_id"] == tid), None)
        if track:
            _broadcast({
                "event"   : "enter",
                "track_id": tid,
                "conf"    : round(track["conf"], 3),
                "ts"      : now,
            })

    # ── Exit events ───────────────────────────────────────────────────────────
    for tid in _prev_active_ids - current_ids:
        first = _first_seen_cache.pop(tid, None)
        dwell = round(now - first, 1) if first else 0.0
        _broadcast({
            "event"         : "exit",
            "track_id"      : tid,
            "dwell"         : dwell,
            "ts"            : now,
            "refresh_recent": True,   # tells frontend to re-fetch gallery
        })

    _prev_active_ids = current_ids

    # ── Heartbeat every 1 second ──────────────────────────────────────────────
    if now - _last_heartbeat >= 1.0:
        _broadcast({
            "event"       : "heartbeat",
            "active_count": len(current_ids),
            "ts"          : now,
        })
        _last_heartbeat = now


# ── Zone entry hooks ───────────────────────────────────────────────────────────
# Track which zones have already been broadcast per track so we fire the
# zone_count WebSocket event exactly once per track per zone.
_zone_broadcast_seen: dict[int, set] = {}


def notify_zone_entry(track_id: int, zone: str) -> None:
    """
    Call from main.py pipeline() each frame after identity.process() returns
    a new zone entry for a track.  Broadcasts a zone_count WS event once per
    track per zone — subsequent calls for the same (track_id, zone) are no-ops.
    """
    seen = _zone_broadcast_seen.setdefault(track_id, set())
    if zone not in seen:
        seen.add(zone)
        _broadcast({
            "event"   : "zone_count",
            "zone"    : zone,
            "track_id": track_id,
            "ts"      : time.time(),
        })


def cleanup_zone_broadcast(track_id: int) -> None:
    """Call when a track is permanently gone to free memory."""
    _zone_broadcast_seen.pop(track_id, None)


def start(host: str = "0.0.0.0", port: int = 8001) -> None:
    cfg    = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    print(f"[Dashboard] http://localhost:{port}")