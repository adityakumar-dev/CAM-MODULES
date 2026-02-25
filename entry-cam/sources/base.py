"""Abstract base class for video sources."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import cv2


class VideoSource(ABC):
    """
    Abstract base class for all video sources (webcam, video file, RTSP stream).
    
    Provides a common interface for:
    - Opening/closing video capture
    - Reading frames with FPS throttling
    - Querying resolution and statistics
    - Context manager support
    """

    @abstractmethod
    def _open(self) -> None:
        """Initialize and open the video source."""
        pass

    @abstractmethod
    def _close(self) -> None:
        """Release resources and close the video source."""
        pass

    @abstractmethod
    def is_frame_ready(self) -> bool:
        """
        Check if enough time has passed to process the next frame based on FPS throttling.
        
        Returns:
            bool: True if a frame should be read now, False if throttling
        """
        pass

    @abstractmethod
    def read(self) -> Optional[Tuple[bool, "cv2.Mat"]]:
        """
        Read one frame from the source.
        
        Returns:
            Optional[Tuple[bool, cv2.Mat]]: (True, frame) if successful, None if throttled or failed
        """
        pass

    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """
        Get the current resolution of the video source.
        
        Returns:
            Tuple[int, int]: (width, height)
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the video source.
        
        Returns:
            Dict[str, Any]: Statistics like frames read, FPS, etc.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        self._open()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Context manager exit."""
        self._close()
