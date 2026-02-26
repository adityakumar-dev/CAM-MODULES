"""
main.py  —  Exit-cam pipeline
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
        port=getattr(config, "DASHBOARD_PORT", 8001),
    )

# ── FPS (thread-safe deque) ───────────────────────────────────────────────────
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
    identity.draw_zones(frame)

    # 4. Bounding boxes + labels
    _exit_zone_name = getattr(config, "EXIT_ZONE_NAME", "exit")
    in_zone_count   = sum(1 for t in tracks if t.get("zone") == _exit_zone_name)
    for t in tracks:
        x1, y1, x2, y2 = t["x1"], t["y1"], t["x2"], t["y2"]
        in_zone = t.get("zone") == _exit_zone_name
        zone    = t.get("zone") or "outside"
        conf    = t.get("conf", 0.0)
        # Green = in exit zone, cyan = in some other zone, gray = outside
        colour  = (0, 255, 0) if in_zone else ((0, 220, 220) if t.get("zone") else (160, 160, 160))
        label   = f"ID {t['track_id']}  [{zone}]  {conf:.0%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            frame, label, (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA,
        )

    # 5. HUD
    hud_lines = [
        f"Unique exits : {identity.unique_exit_count}",
        f"In frame     : {identity.active_count()}",
        f"In zone      : {in_zone_count}",
        f"FPS          : {_fps():.1f}",
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