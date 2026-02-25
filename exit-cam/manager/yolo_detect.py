"""
YoloDetector  — DEBUG BUILD
============================
All original logic preserved. DEBUG lines marked with  # <<< DEBUG
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
from ultralytics import YOLO
import config


def _to_float(val) -> float:
    if hasattr(val, "item"):
        return float(val.item())
    if isinstance(val, np.ndarray):
        return float(val.flat[0])
    return float(val)


class YoloDetector:

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path     = model_path or config.YOLO_MODEL_PATH
        self.conf_threshold = config.YOLO_CONFIDENCE_THRESHOLD
        self.iou_threshold  = config.YOLO_IOU_THRESHOLD
        self.device         = config.YOLO_DEVICE
        self.model          = YOLO(self.model_path)
        self._unconfirmed_counter: int = -1

        # <<< DEBUG
        self._frame_count: int = 0
        print(f"[YOLO] Detector init | conf_thresh={self.conf_threshold} | "
              f"iou_thresh={self.iou_threshold} | device={self.device} | "
              f"model={self.model_path}")
        # <<< DEBUG END

    # ── Plain detection ───────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            classes=[0],
            verbose=False,
        )[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = _to_float(box.conf)
            detections.append((x1, y1, x2, y2, conf))
        return detections

    # ── Detection + ByteTrack ─────────────────────────────────────────────────

    def process_with_tracking(self, frame: np.ndarray) -> List[dict]:
        # <<< DEBUG
        self._frame_count += 1
        frame_no = self._frame_count
        # <<< DEBUG END

        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            classes=[0],
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        tracks = []
        if results.boxes is None:
            # <<< DEBUG
            print(f"[YOLO] frame={frame_no} | results.boxes is None — no detections at all")
            # <<< DEBUG END
            return tracks

        # <<< DEBUG
        total_boxes = len(results.boxes)
        confirmed_count   = 0
        unconfirmed_count = 0
        # <<< DEBUG END

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = _to_float(box.conf)
            box_w = x2 - x1
            box_h = y2 - y1

            if box.id is not None:
                track_id = int(_to_float(box.id))
                # <<< DEBUG
                confirmed_count += 1
                print(f"[YOLO] frame={frame_no} | CONFIRMED track_id={track_id} | "
                      f"conf={conf:.3f} | box=({x1},{y1},{x2},{y2}) | "
                      f"size={box_w}x{box_h}px")
                # <<< DEBUG END
            else:
                track_id = self._unconfirmed_counter
                self._unconfirmed_counter -= 1
                # <<< DEBUG
                unconfirmed_count += 1
                print(f"[YOLO] frame={frame_no} | UNCONFIRMED box.id=None → "
                      f"synthetic_id={track_id} | conf={conf:.3f} | "
                      f"box=({x1},{y1},{x2},{y2}) | size={box_w}x{box_h}px  "
                      f"⚠️  This will be DROPPED by IdentityManager (bt_id <= 0 filter)")
                # <<< DEBUG END

            tracks.append(dict(
                track_id=track_id,
                x1=x1, y1=y1, x2=x2, y2=y2,
                conf=conf,
            ))

        # <<< DEBUG
        if total_boxes > 0:
            print(f"[YOLO] frame={frame_no} | SUMMARY: {total_boxes} box(es) — "
                  f"{confirmed_count} confirmed, {unconfirmed_count} unconfirmed/synthetic")
        # <<< DEBUG END

        return tracks