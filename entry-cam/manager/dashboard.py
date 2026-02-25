"""
manager/dashboard.py  —  Entry-Cam Dashboard Server
=====================================================

Bug fixed: image_path in session gallery was stored as a relative path
("output/best/id_N.jpg") but gallery_session() was calling
os.path.relpath(relative, absolute_base) which produces broken paths like
"../../../output/best/id_N.jpg".

Fix: normalise image_path to always be a web-accessible URL fragment
starting with "output/" regardless of how it was stored (absolute or relative).

FIX-E (identity_manager.py sync): notify() signature restored to
    notify(tracks: list[dict], identity_manager)
  The previous version changed it to notify(snapshot: dict) which broke
  the dashboard because this file was never updated to match.

WebSocket events
----------------
  heartbeat   { active_count, unique_count, ts }
  enter       { track_id, conf, zone, ts }
  exit        { track_id, dwell, ts, refresh_recent }
  new_entry   { unique_count, ts }

REST
----
  GET /api/stream           MJPEG live feed
  GET /api/stats/session    { unique_total, today_count, active_now }
  GET /api/stats/hourly     [{ hour, count }] today from session gallery
  GET /api/recent           last 20 archived persons from metadata.db
  GET /api/gallery/session  all session-gallery persons + fixed image URLs
  GET /api/gallery/flat     all output/best/*.jpg (live buffer)
  GET /api/gallery/days     archive tree by day/hour
  GET /output/...           static file serving
  GET /                     dashboard UI
"""
from __future__ import annotations

import asyncio
import datetime
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

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUTPUT_DIR = os.path.join(_BASE_DIR, "output")
_STATIC_DIR = os.path.join(_BASE_DIR, "static")
_BEST_DIR   = os.path.join(_OUTPUT_DIR, "best")
_ARCHIVE_DB = os.path.join(_OUTPUT_DIR, "metadata.db")
_SESSION_DB = os.path.join(_OUTPUT_DIR, "entry_session.db")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Entry-Cam Dashboard")

for d in (_STATIC_DIR, _OUTPUT_DIR, _BEST_DIR):
    os.makedirs(d, exist_ok=True)

app.mount("/output", StaticFiles(directory=_OUTPUT_DIR), name="output")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ── Image path normalisation ───────────────────────────────────────────────────
def _web_path(raw: str) -> str:
    """
    Convert any stored image_path to a web-accessible URL fragment
    starting with "output/".

    Handles:
      ""                              -> ""
      "output/best/id_5.jpg"          -> "output/best/id_5.jpg"
      "/abs/path/output/best/id_5.jpg"-> "output/best/id_5.jpg"
      "../../../output/best/id_5.jpg" -> ""
    """
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


def push_frame(frame) -> None:
    global _latest_frame
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
    if ok:
        with _frame_lock:
            _latest_frame = buf.tobytes()


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
_ws_lock = threading.Lock()
_event_loop: Optional[asyncio.AbstractEventLoop] = None


@app.on_event("startup")
async def _grab_loop():
    global _event_loop
    _event_loop = asyncio.get_running_loop()


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
    if not os.path.exists(_ARCHIVE_DB):
        return []
    try:
        c = sqlite3.connect(_ARCHIVE_DB)
        c.row_factory = sqlite3.Row
        rows = c.execute(sql, params).fetchall()
        c.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _q_session(sql: str, params: tuple = ()) -> list[dict]:
    if not os.path.exists(_SESSION_DB):
        return []
    try:
        c = sqlite3.connect(_SESSION_DB)
        c.row_factory = sqlite3.Row
        rows = c.execute(sql, params).fetchall()
        c.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _today_epoch() -> float:
    d = datetime.date.today()
    return float(int(datetime.datetime(d.year, d.month, d.day).timestamp()))


# ── REST endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    p = os.path.join(_STATIC_DIR, "index.html")
    return FileResponse(p) if os.path.exists(p) else JSONResponse(
        {"error": "index.html missing"}, status_code=404
    )


@app.get("/api/stats/session")
async def stats_session():
    total = _q_session("SELECT COUNT(*) AS n FROM session_gallery")
    today = _q_session(
        "SELECT COUNT(*) AS n FROM session_gallery WHERE first_entry >= ?",
        (_today_epoch(),),
    )
    return {
        "unique_total": total[0]["n"] if total else 0,
        "today_count":  today[0]["n"] if today  else 0,
        "active_now":   _active_ref[0],
    }


@app.get("/api/stats/hourly")
async def stats_hourly():
    return _q_session("""
        SELECT CAST(strftime('%H', datetime(first_entry,'unixepoch','localtime'))
                    AS INTEGER) AS hour,
               COUNT(*) AS count
        FROM session_gallery
        WHERE first_entry >= ?
        GROUP BY hour ORDER BY hour
    """, (_today_epoch(),))


@app.get("/api/recent")
async def recent():
    rows = _q_archive("""
        SELECT track_id, best_conf, first_seen, last_seen,
               ROUND(last_seen - first_seen, 1) AS dwell, image_path
        FROM person ORDER BY id DESC LIMIT 20
    """)
    for r in rows:
        r["image_path"] = _web_path(r.get("image_path", ""))
    return rows


@app.get("/api/gallery/session")
async def gallery_session():
    rows = _q_session("""
        SELECT cid, entry_count, first_entry, last_seen, image_path
        FROM session_gallery ORDER BY last_seen DESC
    """)
    for r in rows:
        r["image_path"] = _web_path(r.get("image_path", ""))
    return rows


@app.get("/api/gallery/flat")
async def gallery_flat():
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
    if not os.path.exists(_BEST_DIR):
        return []
    result = []
    day_dirs = sorted(
        [d for d in os.listdir(_BEST_DIR)
         if os.path.isdir(os.path.join(_BEST_DIR, d)) and d.startswith("day_")],
        reverse=True,
    )
    for day in day_dirs:
        dp = os.path.join(_BEST_DIR, day)
        hours = []
        for hour in sorted(
            [h for h in os.listdir(dp)
             if os.path.isdir(os.path.join(dp, h)) and h.startswith("hour_")],
            reverse=True,
        ):
            hp = os.path.join(dp, hour)
            images = sorted(
                [f for f in os.listdir(hp) if f.endswith(".jpg")],
                key=lambda f: os.path.getmtime(os.path.join(hp, f)),
                reverse=True,
            )
            hours.append({
                "hour":   hour,
                "images": [
                    {"path": f"output/best/{day}/{hour}/{f}", "name": f}
                    for f in images
                ],
            })
        result.append({"day": day, "hours": hours})
    return result


# ── Pipeline hooks ─────────────────────────────────────────────────────────────
_prev_ids:   set        = set()
_seen_cache: dict       = {}
_active_ref: list[int]  = [0]
_last_hb:    float      = 0.0
_state:      dict       = {}
_TWO_HOURS              = 7200


def notify(tracks: list[dict], identity_manager) -> None:
    """
    FIX-E: signature is (tracks, identity_manager) — unchanged from original.
    The broken intermediate version passed a snapshot dict which caused
    AttributeError on identity_manager.get_state() and .unique_entry_count.
    """
    global _prev_ids, _last_hb, _seen_cache
    now         = time.time()
    current_ids = {t["track_id"] for t in tracks}
    _active_ref[0] = len(current_ids)

    # Purge stale seen cache
    for tid in [k for k, v in _seen_cache.items() if now - v > _TWO_HOURS]:
        del _seen_cache[tid]

    # Cache first_seen from identity_manager state
    for tid, state in identity_manager.get_state().items():
        if tid not in _seen_cache:
            _seen_cache[tid] = state.first_seen

    # Enter events
    for tid in current_ids - _prev_ids:
        t = next((x for x in tracks if x["track_id"] == tid), None)
        if t:
            _broadcast({
                "event":    "enter",
                "track_id": tid,
                "conf":     round(t.get("conf", 0), 3),
                "zone":     t.get("zone") or "outside",
                "ts":       now,
            })

    # Exit events
    for tid in _prev_ids - current_ids:
        first = _seen_cache.pop(tid, None)
        _broadcast({
            "event":          "exit",
            "track_id":       tid,
            "dwell":          round(now - first, 1) if first else 0.0,
            "ts":             now,
            "refresh_recent": True,
        })

    # New unique entry event
    unique = identity_manager.unique_entry_count
    if unique != _state.get("last_unique", unique):
        _broadcast({"event": "new_entry", "unique_count": unique, "ts": now})
    _state["last_unique"] = unique

    _prev_ids = current_ids

    # Heartbeat every 1 s
    if now - _last_hb >= 1.0:
        _broadcast({
            "event":        "heartbeat",
            "active_count": len(current_ids),
            "unique_count": unique,
            "ts":           now,
        })
        _last_hb = now


def start(host: str = "0.0.0.0", port: int = 8000) -> None:
    cfg = uvicorn.Config(app, host=host, port=port, log_level="warning")
    s   = uvicorn.Server(cfg)
    threading.Thread(target=s.run, daemon=True).start()
    print(f"[Dashboard] http://localhost:{port}")