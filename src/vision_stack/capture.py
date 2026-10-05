"""Webcam, video-file, and still-image frame sources."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from vision_stack.exhaust import assert_never

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SourceKind(str, Enum):
    VIDEO = "video"
    IMAGE = "image"


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_ms: float
    image: np.ndarray


def parse_source(source: str) -> int | str:
    text = source.strip()
    if text.isdigit():
        return int(text)
    return text


class FrameSource:
    """Yields BGR frames from a webcam index, a video path, or a still image."""

    def __init__(self, source: str, max_frames: int | None = None) -> None:
        self.source = source
        self.max_frames = max_frames
        self.kind = SourceKind.VIDEO
        self.fps = 30.0
        self._cap: cv2.VideoCapture | None = None
        self._still: np.ndarray | None = None
        self._opened_at = 0.0

    def __enter__(self) -> FrameSource:
        parsed = parse_source(self.source)
        if isinstance(parsed, int):
            self._open_capture(parsed)
        else:
            self._open_path(Path(parsed))
        self._opened_at = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def frames(self) -> Iterator[Frame]:
        match self.kind:
            case SourceKind.IMAGE:
                if self._still is None:
                    raise RuntimeError("Image source is not open")
                yield Frame(0, 0.0, self._still)
                return
            case SourceKind.VIDEO:
                yield from self._video_frames()
                return
            case _:
                assert_never(self.kind)

    def _open_path(self, path: Path) -> None:
        if path.suffix.lower() in IMAGE_SUFFIXES:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {path}")
            self.kind = SourceKind.IMAGE
            self._still = image
            self.fps = 30.0
            return
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")
        self._open_capture(str(path))

    def _open_capture(self, target: int | str) -> None:
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            cap.release()
            raise FileNotFoundError(
                f"Could not open source {self.source!r}. "
                "Pass a webcam index such as 0, or a path to a video or image."
            )
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps < 1.0 or fps > 240.0:
            fps = 30.0
        self._cap = cap
        self.fps = fps
        self.kind = SourceKind.VIDEO

    def _video_frames(self) -> Iterator[Frame]:
        if self._cap is None:
            raise RuntimeError("Video source is not open")
        index = 0
        while self.max_frames is None or index < self.max_frames:
            ok, image = self._cap.read()
            if not ok:
                break
            yield Frame(index, self._timestamp_ms(index), _as_bgr(image).copy())
            index += 1

    def _timestamp_ms(self, index: int) -> float:
        if self._cap is None:
            return 0.0
        reported = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if reported > 0.0:
            return reported
        if index == 0:
            return 0.0
        return index * 1000.0 / self.fps


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image
