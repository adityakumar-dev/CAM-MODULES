"""Video file source with progress tracking, looping, and frame statistics."""

import cv2
import time
import logging
from typing import Optional, Tuple, Dict, Any
from .base import VideoSource

logger = logging.getLogger(__name__)


class VideoFileSource(VideoSource):
    """
    Video file capture with:
    - Auto-detected or user-set FPS throttling
    - Total frame count and progress %
    - Optional looping
    - Frame skip for fast seeking
    - Resolution / codec info
    """

    def __init__(
        self,
        path: str,
        target_fps: Optional[int] = None,
        loop: bool = False,
        start_frame: int = 0,
    ) -> None:
        """
        Args:
            path:        Path to video file.
            target_fps:  Limit playback rate; None = use file's native FPS.
            loop:        Restart from beginning when video ends.
            start_frame: Seek to this frame on open.
        """
        self.path = path
        self.target_fps = target_fps
        self.loop = loop
        self._start_frame = start_frame

        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame_time = 0.0
        self.actual_fps: float = 0.0
        self.total_frames: int = 0
        self.codec: str = ""

        # Stats
        self._frame_count = 0
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.path}")

        self.actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self.actual_fps <= 0:
            self.actual_fps = 30.0

        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        self.codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))

        if self.target_fps is None:
            self.target_fps = self.actual_fps

        if self._start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)

        self._start_time = time.time()
        logger.info(
            "Video opened: %s — %.1f FPS, %d frames, codec=%s, res=%dx%d",
            self.path,
            self.actual_fps,
            self.total_frames,
            self.codec,
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Video closed: %s — %s", self.path, self.get_stats())

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
            raise RuntimeError("Video not opened. Use as context manager.")

        # FPS throttling
        if self.target_fps is not None:
            now = time.time()
            if now - self._last_frame_time < 1.0 / self.target_fps:
                return None
            self._last_frame_time = now

        ret, frame = self._cap.read()
        if not ret:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    return None
            else:
                return None

        self._frame_count += 1
        return True, frame

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_resolution(self) -> Tuple[int, int]:
        if self._cap is None:
            return (0, 0)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def progress(self) -> float:
        """Playback progress 0.0–1.0."""
        if self._cap is None or self.total_frames <= 0:
            return 0.0
        pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        return min(pos / self.total_frames, 1.0)

    @property
    def current_frame_index(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    def seek(self, frame_index: int) -> None:
        """Seek to a specific frame."""
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def get_stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "frames_read": self._frame_count,
            "total_frames": self.total_frames,
            "progress_pct": round(self.progress * 100, 1),
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
