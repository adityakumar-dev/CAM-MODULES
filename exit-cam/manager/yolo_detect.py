"""
YoloDetector
============
Two modes:
  process()               → plain detection, returns [(x1,y1,x2,y2,conf)]
  process_with_tracking() → ByteTrack via model.track(), returns list of dicts
                            with keys: track_id, x1, y1, x2, y2, conf, is_temp_id

Temporary ID Strategy:
  When ByteTrack hasn't yet assigned a confirmed integer ID to a detection
  (box.id is None — common on first 2-3 frames or after occlusion), we
  assign a unique temporary string ID like "tmp_<uuid4_short>" instead of
  skipping the detection or using -1.

  WHY: Skipping means the person is invisible to IdentityManager for those
  frames. If they cross the counting zone during that window they are MISSED.
  Using -1 is worse — all untracked detections collide under the same fake ID.

  IdentityManager is responsible for:
    1. Tracking tmp IDs through zone transitions normally.
    2. Merging a tmp ID -> real ID when ByteTrack later confirms it,
       by spatial proximity (centroid distance matching).
    3. Ensuring the merged entry is counted only once.

  The 'is_temp_id' flag in the returned dict lets IdentityManager know
  which entries need merge-watching.
"""
from __future__ import annotations

import uuid
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

import config


def _to_float(val) -> float:
    """Safely convert any tensor / ndarray / scalar to a Python float."""
    if hasattr(val, "item"):
        return float(val.item())
    if isinstance(val, np.ndarray):
        return float(val.flat[0])
    return float(val)


def _safe_int(val) -> int | None:
    """
    Extract integer from a tensor/ndarray/scalar.
    Returns None if val is None or extraction fails.
    """
    if val is None:
        return None
    try:
        return int(_to_float(val))
    except Exception:
        return None


def _new_temp_id() -> str:
    """Generate a short unique temporary track ID string."""
    return f"tmp_{uuid.uuid4().hex[:8]}"


class YoloDetector:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path     = model_path or config.YOLO_MODEL_PATH
        self.conf_threshold = config.YOLO_CONFIDENCE_THRESHOLD
        self.iou_threshold  = config.YOLO_IOU_THRESHOLD
        self.device         = config.YOLO_DEVICE
        self.model          = YOLO(self.model_path)

    # ------------------------------------------------------------------
    # Plain detection (no tracking)
    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=config.YOLO_IMGSZ,
            classes=[0],   # person only
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = _to_float(box.conf)
            detections.append((x1, y1, x2, y2, conf))
        return detections

    # ------------------------------------------------------------------
    # Detection + ByteTrack via model.track()
    # ------------------------------------------------------------------
    def process_with_tracking(self, frame: np.ndarray) -> List[dict]:
        """
        Returns list of dicts with keys:
            track_id  : int (confirmed) or str like "tmp_a3f9b2c1" (unconfirmed)
            x1, y1, x2, y2 : int
            conf      : float
            is_temp_id: bool  True when ByteTrack has not confirmed the ID yet

        NO detections are dropped. Every visible person is returned.
        IdentityManager handles merging tmp -> real IDs via spatial matching.
        """
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=config.YOLO_IMGSZ,
            classes=[0],          # person only
            persist=True,         # keep tracker state between calls
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        tracks: List[dict] = []

        if results.boxes is None:
            return tracks

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf    = _to_float(box.conf)
            real_id = _safe_int(box.id)

            if real_id is not None:
                # Confirmed ByteTrack ID
                tracks.append(dict(
                    track_id=real_id,
                    x1=x1, y1=y1,
                    x2=x2, y2=y2,
                    conf=conf,
                    is_temp_id=False,
                ))
            else:
                # Unconfirmed detection: assign unique temp ID.
                # Each unconfirmed box gets its OWN temp ID (not shared),
                # so two unconfirmed people do not collide under one ID.
                tracks.append(dict(
                    track_id=_new_temp_id(),
                    x1=x1, y1=y1,
                    x2=x2, y2=y2,
                    conf=conf,
                    is_temp_id=True,
                ))

        return tracks