"""
tools/draw_zones.py
===================
Run once to set ENTRY_ZONES in config.py.

Usage
-----
    python tools/draw_zones.py --source 0          # webcam
    python tools/draw_zones.py --source rtsp://... # RTSP
    python tools/draw_zones.py --source video.mp4  # video file
    python tools/draw_zones.py --image frame.jpg   # static image

Controls
--------
    Left click   Add point
    Right click  Undo last point
    ENTER        Finish zone, start next
    R            Reset all
    S            Print config output
    Q / ESC      Quit (prints config if zones exist)
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

ZONE_NAMES = ["detector", "inside"]
COLOURS    = [(0, 200, 255), (0, 255, 100)]


def grab_frame(source: str) -> np.ndarray:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open: {source}")
        sys.exit(1)
    for _ in range(5):
        ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("ERROR: Could not read frame.")
        sys.exit(1)
    return frame


def render(base, zones, current_pts, zone_idx):
    overlay = base.copy()
    canvas  = base.copy()
    for i, pts in enumerate(zones):
        if len(pts) >= 3:
            poly = np.array(pts, dtype=np.int32)
            cv2.fillPoly(overlay, [poly], COLOURS[i % len(COLOURS)])
            cv2.polylines(canvas, [poly], True, COLOURS[i % len(COLOURS)], 2)
        for pt in pts:
            cv2.circle(canvas, pt, 4, COLOURS[i % len(COLOURS)], -1)
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    col = COLOURS[zone_idx % len(COLOURS)] if zone_idx < len(COLOURS) else (200, 200, 200)
    for pt in current_pts:
        cv2.circle(canvas, pt, 5, col, -1)
    if len(current_pts) >= 2:
        for i in range(len(current_pts) - 1):
            cv2.line(canvas, current_pts[i], current_pts[i+1], col, 1)

    zone_name = ZONE_NAMES[zone_idx] if zone_idx < len(ZONE_NAMES) else "done"
    lines = [
        f"Zone: [{zone_name}]  ({len(current_pts)} pts)",
        "LClick=add  RClick=undo  ENTER=finish  R=reset  S=save  Q=quit",
    ]
    for i, ln in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(canvas, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(canvas, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    return canvas


def print_config(zones):
    print("\n" + "="*60)
    print("Paste into config.py:")
    print("="*60)
    print("ENTRY_ZONES = {")
    for i, pts in enumerate(zones):
        name = ZONE_NAMES[i] if i < len(ZONE_NAMES) else f"zone_{i}"
        print(f'    "{name}": {pts},')
    print("}")
    if len(zones) >= 2:
        n1 = ZONE_NAMES[0]
        n2 = ZONE_NAMES[1]
        print(f'ENTRY_COUNT_TRIGGER = "{n1}\u2192{n2}"')
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    
    parser.add_argument("--image",  default=None)
    args = parser.parse_args()

    if args.image:
        base = cv2.imread(args.image)
        if base is None:
            print(f"ERROR: Cannot read image: {args.image}")
            sys.exit(1)
    
    elif args.source == "video":
        base = grab_frame("./../../test2.mp4")
    else:
        print(f"Grabbing frame from {args.source} ...")
        base = grab_frame(args.source)

    print(f"Frame: {base.shape[1]}x{base.shape[0]}")

    completed: list = []
    current:   list = []
    zone_idx        = 0

    def on_mouse(event, x, y, flags, param):
        nonlocal current
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and current:
            current.pop()

    cv2.namedWindow("draw_zones", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("draw_zones", on_mouse)

    while True:
        cv2.imshow("draw_zones", render(base.copy(), completed, current, zone_idx))
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 10):  # ENTER
            if zone_idx < len(ZONE_NAMES):
                if len(current) < 3:
                    print("Need at least 3 points.")
                else:
                    completed.append(list(current))
                    print(f"Zone '{ZONE_NAMES[zone_idx]}' saved ({len(current)} pts).")
                    current  = []
                    zone_idx += 1
                    if zone_idx >= len(ZONE_NAMES):
                        print("All zones done. Press S to save, Q to quit.")

        elif key in (ord('r'), ord('R')):
            completed.clear(); current.clear(); zone_idx = 0
            print("Reset.")

        elif key in (ord('s'), ord('S')):
            if completed:
                print_config(completed)
            else:
                print("No completed zones yet.")

        elif key in (ord('q'), ord('Q'), 27):
            if completed:
                print_config(completed)
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()