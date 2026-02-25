"""
main.py  —  Emotion Zone Counter pipeline
"""
from __future__ import annotations
import time
import argparse
import cv2
import numpy as np
import config
from manager.cv2_manager import CV2Manager
from manager.yolo_detect import YoloDetector
from manager.identity_manager import IdentityManager
from manager import dashboard
# ── Source ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--source", default="webcam",
                    choices=["webcam", "video", "rtsp"],
                    help="Input source type")
parser.add_argument("--path", default=None,
                    help="File path or RTSP URL (required for video/rtsp)")
args = parser.parse_args()

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

# ── Dashboard ─────────────────────────────────────────────────────────────────
dashboard.start()

# ── Zone polygon overlay helper ───────────────────────────────────────────────
_ZONE_COLOURS = [
    (0, 200, 255), (0, 255, 100), (255, 100, 0),
    (200, 0, 255), (0, 150, 255), (255, 200, 0),
]

def _draw_zones(frame: np.ndarray) -> None:
    zones = getattr(config, "ZONES", {})
    for idx, (name, poly) in enumerate(zones.items()):
        colour = _ZONE_COLOURS[idx % len(_ZONE_COLOURS)]
        pts    = np.array(poly, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], colour)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [pts], True, colour, 2)
        # Label at centroid
        cx = int(np.mean([p[0] for p in poly]))
        cy = int(np.mean([p[1] for p in poly]))
        cv2.putText(frame, name, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


# ── Track zone broadcast state ────────────────────────────────────────────────
# Keep the last counted_zones set per track so we only broadcast new entries.
_prev_counted: dict[int, set] = {}

# ── Pipeline ──────────────────────────────────────────────────────────────────
def pipeline(frame: np.ndarray) -> np.ndarray:
    global _prev_counted

    # 1. YOLO + ByteTrack
    raw_tracks = detector.process_with_tracking(frame)

    # 2. Zone tracking + best-frame capture + (within-session) re-ID
    tracks = identity.process(frame, raw_tracks)

    # 3. Broadcast new zone entries via WebSocket
    current_ids = {t["track_id"] for t in tracks}
    for t in tracks:
        tid = t["track_id"]
        new_zones = set(t.get("counted_zones", [])) - _prev_counted.get(tid, set())
        for zone in new_zones:
            dashboard.notify_zone_entry(tid, zone)
        _prev_counted[tid] = set(t.get("counted_zones", []))

    # Clean up state for gone tracks
    gone = set(_prev_counted) - current_ids
    for tid in gone:
        del _prev_counted[tid]
        dashboard.cleanup_zone_broadcast(tid)

    # 4. Notify dashboard (enter / exit / heartbeat)
    dashboard.notify(tracks, identity)

    # 5. Draw zones
    _draw_zones(frame)

    # 6. Draw bounding boxes, IDs, and current zone label
    for t in tracks:
        x1, y1, x2, y2 = t["x1"], t["y1"], t["x2"], t["y2"]
        zone  = t.get("zone") or ""
        label = f"ID {t['track_id']}"
        if zone:
            label += f" [{zone}]"

        # Colour box green if in a zone, white otherwise
        colour = (0, 255, 0) if zone else (200, 200, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    # 7. HUD — active count + per-zone counts
    hud_lines = [f"Active: {len(tracks)}"]
    zone_totals: dict[str, int] = {}
    for t in tracks:
        z = t.get("zone")
        if z:
            zone_totals[z] = zone_totals.get(z, 0) + 1
    for z, cnt in zone_totals.items():
        hud_lines.append(f"  {z}: {cnt}")

    for i, line in enumerate(hud_lines):
        cv2.putText(frame, line, (10, 28 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # 8. Push to MJPEG stream
    dashboard.push_frame(frame)

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