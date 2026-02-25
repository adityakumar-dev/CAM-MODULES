from __future__ import annotations

import traceback
from typing import Callable, List

import cv2
from sources.base import VideoSource

FrameProcessor = Callable[["cv2.Mat"], "cv2.Mat"]


class CV2Manager:
    def __init__(
        self,
        window_name: str = "source",
        show: bool = True,
    ) -> None:
        self.window_name = window_name
        self.show = show
        self._processors: List[FrameProcessor] = []

    def add_processor(self, processor: FrameProcessor) -> None:
        self._processors.append(processor)

    def run(self, source: VideoSource) -> None:
        with source:
            while True:
                result = source.read()
                if result is None:
                    continue
                ok, frame = result
                if not ok:
                    break

                for proc in self._processors:
                    try:
                        new = proc(frame)
                        if new is not None:
                            frame = new
                    except Exception:
                        # Print FULL traceback so we can see the exact line
                        traceback.print_exc()

                if self.show:
                    cv2.imshow(self.window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        if self.show:
            cv2.destroyAllWindows()