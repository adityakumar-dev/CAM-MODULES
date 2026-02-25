"""Webcam video source with auto-reconnect and frame statistics."""

import cv2
import time
import logging
from typing import Optional, Tuple, Dict, Any
from .base import VideoSource

logger = logging.getLogger(__name__)


class WebcamSource(VideoSource):
    """
    Webcam capture with:
    - Auto-detected or user-set FPS throttling
    - Resolution querying
    - Frame counting & drop statistics
    - Auto-reconnect on failure
    """

    def __init__(
        self,
        camera_index: int = 0,
        target_fps: Optional[int] = None,
        resolution: Optional[Tuple[int, int]] = None,
        max_reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
    ) -> None:
        """
        Args:
            camera_index:   Webcam device index (0 = default).
            target_fps:     Limit frame rate; None = use camera native FPS.
            resolution:     (width, height) to request from camera, or None.
            max_reconnect_attempts: Retries on read failure before giving up.
            reconnect_delay:        Seconds between reconnect attempts.
        """
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.resolution = resolution
        self._max_reconnects = max_reconnect_attempts
        self._reconnect_delay = reconnect_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame_time = 0.0
        self.actual_fps: float = 0.0

        # Statistics
        self._frame_count = 0
        self._drop_count = 0
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open webcam at index {self.camera_index}")

        if self.resolution is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        self.actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self.actual_fps <= 0:
            self.actual_fps = 30.0

        if self.target_fps is None:
            self.target_fps = self.actual_fps

        self._start_time = time.time()
        logger.info(
            "Webcam %d opened — native %.1f FPS, target %.1f FPS, res %dx%d",
            self.camera_index,
            self.actual_fps,
            self.target_fps,
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Webcam %d closed — %s", self.camera_index, self.get_stats())

    def _reconnect(self) -> bool:
        """Try to re-open the camera."""
        for attempt in range(1, self._max_reconnects + 1):
            logger.warning(
                "Webcam %d reconnect attempt %d/%d",
                self.camera_index, attempt, self._max_reconnects,
            )
            self._close()
            time.sleep(self._reconnect_delay)
            try:
                self._open()
                return True
            except RuntimeError:
                continue
        return False

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def is_frame_ready(self) -> bool:
        """Check if enough time has passed for next frame based on target FPS."""
        if self.target_fps is None:
            return True
        now = time.time()
        return (now - self._last_frame_time) >= (1.0 / self.target_fps)
    def read(self) -> Optional[Tuple[bool, "cv2.Mat"]]:
        """Read one frame.  Returns (True, frame) or None."""
        if self._cap is None:
            raise RuntimeError("Webcam not opened. Use as context manager.")

        # FPS throttling
        if self.target_fps is not None:
            now = time.time()
            if now - self._last_frame_time < 1.0 / self.target_fps:
                return None
            self._last_frame_time = now

        ret, frame = self._cap.read()
        if not ret:
            self._drop_count += 1
            if not self._reconnect():
                return None
            ret, frame = self._cap.read()
            if not ret:
                return None

        self._frame_count += 1
        return True, frame

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_resolution(self) -> Tuple[int, int]:
        """Return current (width, height)."""
        if self._cap is None:
            return (0, 0)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def get_stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "frames_read": self._frame_count,
            "frames_dropped": self._drop_count,
            "elapsed_s": round(elapsed, 1),
            "avg_fps": round(self._frame_count / max(elapsed, 0.001), 1),
        }

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._close()
