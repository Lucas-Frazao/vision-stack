"""OpenCV DNN MobileNet-SSD fallback (VOC labels, CPU inference)."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from vision_stack.config import DeviceName
from vision_stack.detections import Detection

logger = logging.getLogger(__name__)

VOC_LABELS = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


class MobileNetSsdDetector:
    """Caffe MobileNet-SSD. OpenCV DNN runs on CPU in the stock wheel."""

    def __init__(
        self,
        prototxt: Path,
        weights: Path,
        confidence: float,
        device: DeviceName,
    ) -> None:
        if not prototxt.is_file() or not weights.is_file():
            raise FileNotFoundError(
                "MobileNet-SSD files are missing at "
                f"{prototxt} and {weights}. "
                "Run: python scripts/download_mobilenet_ssd.py"
            )
        if device is not DeviceName.CPU:
            logger.warning(
                "backend mobilenet_ssd uses the OpenCV CPU DNN runtime; "
                "device %s is recorded and applied on the YOLO backend",
                device.value,
            )
        self.device = device
        self.confidence = confidence
        self._net = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))

    def predict(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self._net.setInput(blob)
        raw = self._net.forward()
        detections: list[Detection] = []
        for index in range(int(raw.shape[2])):
            confidence = float(raw[0, 0, index, 2])
            if confidence < self.confidence:
                continue
            class_id = int(raw[0, 0, index, 1])
            if class_id <= 0 or class_id >= len(VOC_LABELS):
                continue
            x1 = int(raw[0, 0, index, 3] * width)
            y1 = int(raw[0, 0, index, 4] * height)
            x2 = int(raw[0, 0, index, 5] * width)
            y2 = int(raw[0, 0, index, 6] * height)
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(VOC_LABELS[class_id], confidence, x1, y1, x2, y2)
            )
        return detections
