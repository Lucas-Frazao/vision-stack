"""Detection records and frame annotation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """One box in pixel coordinates on the source frame."""

    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    infer_ms: float,
) -> np.ndarray:
    """Return a copy of `frame` with boxes and a one-line timing HUD."""
    canvas = frame.copy()
    for detection in detections:
        cv2.rectangle(
            canvas,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            (0, 220, 0),
            2,
        )
        cv2.putText(
            canvas,
            f"{detection.label} {detection.confidence:.2f}",
            (detection.x1, max(16, detection.y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"{len(detections)} det  {infer_ms:.1f} ms",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas
