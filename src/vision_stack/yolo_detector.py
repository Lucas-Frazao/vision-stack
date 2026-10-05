"""Ultralytics YOLO backend. `device` is forwarded; nothing here selects CUDA."""

from __future__ import annotations

import numpy as np
import torch
from ultralytics import YOLO

from vision_stack.config import DeviceName
from vision_stack.detections import Detection
from vision_stack.exhaust import assert_never


class YoloDetector:
    def __init__(
        self,
        model: str,
        device: DeviceName,
        confidence: float,
        imgsz: int,
    ) -> None:
        _require_device(device)
        self.device = device
        self.confidence = confidence
        self.imgsz = imgsz
        self._model = YOLO(model)

    def predict(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            source=frame,
            device=self.device.value,
            conf=self.confidence,
            imgsz=self.imgsz,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        names = result.names
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        height, width = frame.shape[:2]
        detections: list[Detection] = []
        for coords, confidence, class_id in zip(xyxy, confidences, class_ids):
            label = _label(names, int(class_id))
            box = _clip_box(coords, width, height)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            detections.append(Detection(label, float(confidence), x1, y1, x2, y2))
        return detections


def _require_device(device: DeviceName) -> None:
    match device:
        case DeviceName.CPU:
            return
        case DeviceName.CUDA:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "device 'cuda' was requested but this PyTorch build cannot see a CUDA device. "
                    "Run with --device cpu."
                )
            return
        case DeviceName.MPS:
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "device 'mps' was requested but MPS is not available. "
                    "Run with --device cpu."
                )
            return
        case _:
            assert_never(device)


def _label(names: dict[int, str] | list[str], class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _clip_box(
    coords: np.ndarray,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = (int(round(float(value))) for value in coords)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2
