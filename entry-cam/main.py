"""
main.py  —  Entry-cam pipeline

FIX SUMMARY:
  - _fps_buf replaced with thread-safe collections.deque(maxlen=30)
  - identity._zones.draw() replaced with public identity.draw_zones()
  - dashboard.notify() called with (tracks, identity) — correct signature
"""
from __future__ import annotations

import argparse
import collections
import time

import cv2

import config
from manager.cv2_manager      import CV2Manager
from manager.yolo_detect      import YoloDetector
from manager.identity_manager import IdentityManager
from manager import dashboard

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--source", default="webcam",
                    choices=["webcam", "video", "rtsp"])
parser.add_argument("--path",   default=None)
parser.add_argument("--no-dashboard", action="store_true",
                    help="Disable the web dashboard")
args = parser.parse_args()

# ── Source ────────────────────────────────────────────────────────────────────
if args.source == "video" and args.path:
    from sources.video import VideoFileSource
    source = VideoFileSource(args.path)
elif args.source == "rtsp" and args.path:
    from sources.rtsp import RTSPSource
    source = RTSPSource(args.path)
else:
    from sources.webcam import WebcamSource
    source = WebcamSource(0)

# ── Modules ───────────────────────────────────────────────────────────────────
detector = YoloDetector()
identity = IdentityManager()

if not args.no_dashboard:
    dashboard.start(
        host=getattr(config, "DASHBOARD_HOST", "0.0.0.0"),
        port=getattr(config, "DASHBOARD_PORT", 8002),
    )

# ── FPS (thread-safe deque) ───────────────────────────────────────────────────
# FIX: replaced plain list with deque — safe for multi-threaded dashboard reads
_fps_buf: collections.deque = collections.deque(maxlen=30)

def _fps() -> float:
    now = time.perf_counter()
    _fps_buf.append(now)
    if len(_fps_buf) < 2:
        return 0.0
    return (len(_fps_buf) - 1) / (_fps_buf[-1] - _fps_buf[0])


# ── Pipeline ──────────────────────────────────────────────────────────────────
def pipeline(frame):
    # 1. Detect + track
    raw_tracks = detector.process_with_tracking(frame)

    # 2. ReID + zone crossing + dedup + count
    tracks = identity.process(frame, raw_tracks)

    # 3. Zone overlay
    # FIX: use public method instead of accessing private _zones attribute
    identity.draw_zones(frame)

    # 4. Bounding boxes + labels
    for t in tracks:
        x1, y1, x2, y2 = t["x1"], t["y1"], t["x2"], t["y2"]
        zone  = t.get("zone") or "outside"
        conf  = t.get("conf", 0.0)
        label = f"ID {t['track_id']}  [{zone}]  {conf:.0%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA,
        )

    # 5. HUD
    hud_lines = [
        f"Unique entries : {identity.unique_entry_count}",
        f"In frame       : {identity.active_count()}",
        f"FPS            : {_fps():.1f}",
    ]
    for i, line in enumerate(hud_lines):
        y = 28 + i * 28
        cv2.putText(frame, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    # 6. Dashboard hooks
    if not args.no_dashboard:
        dashboard.push_frame(frame)
        dashboard.notify(tracks, identity)

    return frame


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    manager = CV2Manager(show=config.SHOW_WINDOW)
    manager.add_processor(pipeline)
    try:
        manager.run(source)
    finally:
        identity.shutdown()
        print("Done.")