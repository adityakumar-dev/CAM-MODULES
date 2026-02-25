"""RTSP stream source with buffering and reconnection handling."""

import cv2
import time
import logging
from typing import Optional, Tuple, Dict, Any
from .base import VideoSource

logger = logging.getLogger(__name__)


class RTSPSource(VideoSource):
    """
    RTSP stream capture with:
    - Auto-reconnect on connection loss
    - Configurable buffer size
    - Frame statistics and latency tracking
    - Transport protocol selection (TCP/UDP)
    """

    def __init__(
        self,
        rtsp_url: str,
        target_fps: Optional[int] = None,
        buffer_size: int = 1,
        use_tcp: bool = True,
        max_reconnect_attempts: int = 5,
        reconnect_delay: float = 2.0,
    ) -> None:
        """
        Args:
            rtsp_url:       RTSP stream URL (e.g., rtsp://192.168.1.100:554/stream)
            target_fps:     Limit frame rate; None = no throttling
            buffer_size:    OpenCV buffer size (1 = minimal latency)
            use_tcp:        Use TCP transport (more reliable) vs UDP (lower latency)
            max_reconnect_attempts: Retries on connection failure
            reconnect_delay:        Seconds between reconnect attempts
        """
        self.rtsp_url = rtsp_url
        self.target_fps = target_fps
        self._buffer_size = buffer_size
        self._use_tcp = use_tcp
        self._max_reconnects = max_reconnect_attempts
        self._reconnect_delay = reconnect_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame_time = 0.0
        self.actual_fps: float = 0.0

        # Statistics
        self._frame_count = 0
        self._drop_count = 0
        self._reconnect_count = 0
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(self.rtsp_url)
        
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open RTSP stream: {self.rtsp_url}")

        # Configure for RTSP streaming
        if self._use_tcp:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)

        self.actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self.actual_fps <= 0:
            self.actual_fps = 25.0  # Common default for streams

        self._start_time = time.time()
        logger.info(
            "RTSP stream opened: %s — %.1f FPS, res=%dx%d, transport=%s",
            self.rtsp_url,
            self.actual_fps,
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "TCP" if self._use_tcp else "UDP",
        )

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("RTSP stream closed: %s — %s", self.rtsp_url, self.get_stats())

    def _reconnect(self) -> bool:
        """Try to re-open the RTSP stream."""
        for attempt in range(1, self._max_reconnects + 1):
            logger.warning(
                "RTSP reconnect attempt %d/%d for %s",
                attempt, self._max_reconnects, self.rtsp_url,
            )
            self._close()
            time.sleep(self._reconnect_delay)
            try:
                self._open()
                self._reconnect_count += 1
                return True
            except RuntimeError:
                continue
        logger.error("RTSP reconnect failed after %d attempts", self._max_reconnects)
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
        """Read one frame. Returns (True, frame) or None."""
        if self._cap is None:
            raise RuntimeError("RTSP stream not opened. Use as context manager.")

        # FPS throttling
        if self.target_fps is not None:
            now = time.time()
            if now - self._last_frame_time < 1.0 / self.target_fps:
                return None
            self._last_frame_time = now

        ret, frame = self._cap.read()
        if not ret:
            self._drop_count += 1
            logger.warning("Frame read failed from RTSP stream, attempting reconnect...")
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
            "reconnects": self._reconnect_count,
            "elapsed_s": round(elapsed, 1),
            "avg_fps": round(self._frame_count / max(elapsed, 0.001), 1),
        }
