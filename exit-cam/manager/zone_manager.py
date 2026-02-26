"""
ZoneManager
===========
Named polygonal zones on the camera frame + crossing detection.

Usage
-----
    zm = ZoneManager(config.EXIT_ZONES)

    # Each frame per active track:
    event = zm.update(cid, (cx, cy))
    # Returns "exit->outside", "outside->exit", or None

    # Draw zones for debug:
    zm.draw(frame)

Zone setup
----------
Run manager/draw_zone.py once against your camera to determine coordinates,
then paste the output into config.py as EXIT_ZONES.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_ZONE_COLOURS = [
    (0,   200, 255),
    (0,   255, 100),
    (255, 100,   0),
    (180,   0, 255),
]


class ZoneManager:

    def __init__(self, zones: Dict[str, List[List[int]]]):
        """
        zones: { zone_name: [[x,y], [x,y], ...] }
        At least 3 points per polygon.
        """
        self._polys: Dict[str, np.ndarray] = {
            name: np.array(pts, dtype=np.int32)
            for name, pts in zones.items()
        }
        self._zone_names = list(self._polys.keys())
        # cid -> last known zone (None = outside all zones / first frame)
        self._track_zone: Dict[int, Optional[str]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, cid: int, centre: Tuple[int, int]) -> Optional[str]:
        """
        Call once per frame per active track.
        Returns crossing string e.g. "exit->outside" or None.
        First frame for a new cid: records zone, no event fired.
        """
        current = self._point_zone(centre)
        prev    = self._track_zone.get(cid, "UNSET")

        self._track_zone[cid] = current

        if prev == "UNSET":
            return None

        if prev != current and prev is not None and current is not None:
            return f"{prev}\u2192{current}"

        return None

    def current_zone(self, cid: int) -> Optional[str]:
        return self._track_zone.get(cid)

    def remove(self, cid: int) -> None:
        """Free memory when a track is permanently gone."""
        self._track_zone.pop(cid, None)

    def transfer(self, old_cid, new_cid) -> None:
        """Transfer zone state from a temp ID to a real ID after merge."""
        if old_cid in self._track_zone:
            self._track_zone[new_cid] = self._track_zone.pop(old_cid)

    def draw(self, frame: np.ndarray, alpha: float = 0.20) -> np.ndarray:
        """Semi-transparent zone overlay + outlines + labels. Modifies in-place."""
        overlay = frame.copy()
        for idx, (name, poly) in enumerate(self._polys.items()):
            colour = _ZONE_COLOURS[idx % len(_ZONE_COLOURS)]
            cv2.fillPoly(overlay, [poly], colour)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        for idx, (name, poly) in enumerate(self._polys.items()):
            colour = _ZONE_COLOURS[idx % len(_ZONE_COLOURS)]
            cv2.polylines(frame, [poly], isClosed=True, color=colour, thickness=2)
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            cv2.putText(
                frame, name.upper(), (cx - 30, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA,
            )
        return frame

    # ── Internal ──────────────────────────────────────────────────────────────

    def _point_zone(self, point: Tuple[int, int]) -> Optional[str]:
        pt = (float(point[0]), float(point[1]))
        for name, poly in self._polys.items():
            if cv2.pointPolygonTest(poly, pt, False) >= 0:
                return name
        return None
