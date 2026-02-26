"""RTSP stream source with background reader thread and reconnection handling.

The reader thread runs continuously, storing the latest decoded frame in
_latest_bgr and signalling _new_frame_event.  The pipeline thread calls
read() which is non-blocking: it waits on _new_frame_event (up to 200 ms)
and returns the latest frame immediately without blocking on network I/O.

Benefit: network latency (normally 1/fps ≈ 50-67 ms for a 15-20 fps RTSP
camera) is completely removed from the processing pipeline, so the pipeline
can run at the full rate allowed by YOLO inference speed.
"""

import cv2
import time
import logging
import threading
from typing import Optional, Tuple, Dict, Any
from .base import VideoSource

logger = logging.getLogger(__name__)


class RTSPSource(VideoSource):
    """
    RTSP stream capture with:
    - Background reader thread (pipeline never blocks on network I/O)
    - Auto-reconnect on connection loss
    - Configurable FPS throttle
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
        self.rtsp_url          = rtsp_url
        self.target_fps        = target_fps
        self._buffer_size      = buffer_size
        self._use_tcp          = use_tcp
        self._max_reconnects   = max_reconnect_attempts
        self._reconnect_delay  = reconnect_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame_time  = 0.0
        self.actual_fps: float = 0.0

        # Statistics
        self._frame_count    = 0
        self._drop_count     = 0
        self._reconnect_count = 0
        self._start_time: Optional[float] = None

        # Background reader state
        self._latest_bgr: Optional["cv2.Mat"] = None
        self._bgr_lock     = threading.Lock()
        self._new_frame    = threading.Event()   # set by reader, cleared by read()
        self._stop_event   = threading.Event()   # set by _close() to stop reader
        self._reader_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _open(self) -> None:
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open RTSP stream: {self.rtsp_url}")
        if self._use_tcp:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)
        self._cap = cap

        self.actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self.actual_fps <= 0:
            self.actual_fps = 25.0

        self._start_time = time.time()
        logger.info(
            "RTSP stream opened: %s — %.1f FPS, res=%dx%d, transport=%s",
            self.rtsp_url,
            self.actual_fps,
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "TCP" if self._use_tcp else "UDP",
        )

        # Start background reader
        self._stop_event.clear()
        self._new_frame.clear()
        self._latest_bgr = None
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="rtsp-reader",
            daemon=True,
        )
        self._reader_thread.start()
        logger.debug("RTSP background reader thread started")

    def _close(self) -> None:
        # Signal the reader thread to stop
        self._stop_event.set()
        self._new_frame.set()  # unblock any waiting read()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=3.0)
            self._reader_thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("RTSP stream closed: %s — %s", self.rtsp_url, self.get_stats())

    # ------------------------------------------------------------------
    # Background reader — runs in daemon thread
    # ------------------------------------------------------------------

    def _reconnect_cap(self) -> bool:
        """Re-open only the VideoCapture object (called from reader thread)."""
        for attempt in range(1, self._max_reconnects + 1):
            if self._stop_event.is_set():
                return False
            logger.warning(
                "RTSP reconnect attempt %d/%d for %s",
                attempt, self._max_reconnects, self.rtsp_url,
            )
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            time.sleep(self._reconnect_delay)
            try:
                cap = cv2.VideoCapture(self.rtsp_url)
                if not cap.isOpened():
                    continue
                if self._use_tcp:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)
                self._cap = cap
                self._reconnect_count += 1
                logger.info("RTSP reconnected: %s", self.rtsp_url)
                return True
            except Exception as exc:
                logger.warning("Reconnect attempt %d failed: %s", attempt, exc)
        logger.error("RTSP reconnect failed after %d attempts", self._max_reconnects)
        return False

    def _reader_loop(self) -> None:
        """Continuously read frames and store the latest in _latest_bgr."""
        while not self._stop_event.is_set():
            if self._cap is None:
                time.sleep(0.01)
                continue

            ret, frame = self._cap.read()
            if not ret:
                self._drop_count += 1
                logger.warning("RTSP frame read failed, attempting reconnect…")
                if not self._reconnect_cap():
                    logger.error("RTSP reader: giving up after failed reconnect")
                    break
                continue

            with self._bgr_lock:
                self._latest_bgr = frame
                self._frame_count += 1
            self._new_frame.set()

    # ------------------------------------------------------------------
    # Reading — called from pipeline thread (non-blocking)
    # ------------------------------------------------------------------

    def is_frame_ready(self) -> bool:
        """Check if enough time has passed for next frame based on target FPS."""
        if self.target_fps is None:
            return True
        return (time.time() - self._last_frame_time) >= (1.0 / self.target_fps)

    def read(self) -> Optional[Tuple[bool, "cv2.Mat"]]:
        """
        Return the latest decoded frame.

        Blocks up to 200 ms waiting for the background reader to deliver a
        new frame (matching the behaviour of a blocking cap.read() call but
        without holding the pipeline thread across network stalls).

        Returns (True, frame) on success, None on timeout / not-opened.
        """
        if self._cap is None and self._reader_thread is None:
            raise RuntimeError("RTSP stream not opened. Use as context manager.")

        # Wait for the reader thread to signal a new frame (up to 200 ms)
        if not self._new_frame.wait(timeout=0.2):
            return None  # timeout — camera stalled or reconnecting
        if self._stop_event.is_set():
            return None
        self._new_frame.clear()

        # Optional FPS throttle (useful when target_fps < camera fps)
        if self.target_fps is not None:
            now = time.time()
            if now - self._last_frame_time < 1.0 / self.target_fps:
                return None
            self._last_frame_time = now

        with self._bgr_lock:
            frame = self._latest_bgr
        if frame is None:
            return None
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
            "frames_read":    self._frame_count,
            "frames_dropped": self._drop_count,
            "reconnects":     self._reconnect_count,
            "elapsed_s":      round(elapsed, 1),
            "avg_fps":        round(self._frame_count / max(elapsed, 0.001), 1),
        }
