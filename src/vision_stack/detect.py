"""Detector factory.

`yolo_detector` imports Ultralytics. When that import fails, `build_detector`
still constructs the MobileNet-SSD backend.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from vision_stack.config import Backend, Settings
from vision_stack.detections import Detection
from vision_stack.exhaust import assert_never
from vision_stack.ssd_detector import MobileNetSsdDetector

try:
    from vision_stack import yolo_detector as yolo_backend
except ImportError:
    yolo_backend = None


class Detector(Protocol):
    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Return boxes for one BGR frame."""


def build_detector(settings: Settings) -> Detector:
    match settings.backend:
        case Backend.YOLO:
            if yolo_backend is None:
                raise RuntimeError(
                    "Ultralytics is not installed. "
                    "pip install -r requirements.txt, or rerun with --backend mobilenet_ssd."
                )
            return yolo_backend.YoloDetector(
                model=settings.model,
                device=settings.device,
                confidence=settings.confidence,
                imgsz=settings.imgsz,
            )
        case Backend.MOBILENET_SSD:
            return MobileNetSsdDetector(
                prototxt=settings.mobilenet_prototxt,
                weights=settings.mobilenet_weights,
                confidence=settings.confidence,
                device=settings.device,
            )
        case _:
            assert_never(settings.backend)
