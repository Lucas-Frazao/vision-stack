"""Connect capture, detection, annotation, and the latency log."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from vision_stack.capture import FrameSource, SourceKind
from vision_stack.config import Settings
from vision_stack.detect import Detector
from vision_stack.detections import draw_detections
from vision_stack.exhaust import assert_never
from vision_stack.latency import LatencyLog


@dataclass(frozen=True)
class RunSummary:
    frames: int
    detections: int
    output_media: Path
    latency_csv: Path
    device: str
    backend: str
    median_infer_ms: float
    mean_total_ms: float


def run_pipeline(settings: Settings, detector: Detector) -> RunSummary:
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "latency.csv"
    infer_ms_values: list[float] = []
    total_ms_values: list[float] = []
    detection_total = 0
    frames_written = 0
    output_media: Path | None = None
    interrupted = False

    with FrameSource(settings.source, settings.max_frames) as source, LatencyLog(csv_path) as latency:
        writer: cv2.VideoWriter | None = None
        writer_size: tuple[int, int] | None = None
        try:
            for frame in source.frames():
                started = time.perf_counter()
                infer_started = time.perf_counter()
                detections = detector.predict(frame.image)
                infer_ms = (time.perf_counter() - infer_started) * 1000.0
                annotated = draw_detections(frame.image, detections, infer_ms)
                match source.kind:
                    case SourceKind.IMAGE:
                        output_media = output_dir / "annotated.jpg"
                        if not cv2.imwrite(str(output_media), annotated):
                            raise RuntimeError(f"Could not write {output_media}")
                    case SourceKind.VIDEO:
                        if writer is None or writer_size is None:
                            writer_size = _even_size(annotated.shape[1], annotated.shape[0])
                            output_media = output_dir / "annotated.mp4"
                            writer = _open_writer(output_media, source.fps, writer_size)
                        frame_to_write = annotated
                        if (annotated.shape[1], annotated.shape[0]) != writer_size:
                            frame_to_write = cv2.resize(annotated, writer_size)
                        writer.write(frame_to_write)
                    case _:
                        assert_never(source.kind)
                total_ms = (time.perf_counter() - started) * 1000.0
                latency.write(
                    frame_idx=frame.index,
                    timestamp_ms=frame.timestamp_ms,
                    infer_ms=infer_ms,
                    total_ms=total_ms,
                    num_detections=len(detections),
                )
                frames_written += 1
                detection_total += len(detections)
                infer_ms_values.append(infer_ms)
                total_ms_values.append(total_ms)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            if writer is not None:
                writer.release()

    if frames_written == 0 or output_media is None:
        if interrupted:
            raise RuntimeError("Interrupted before any frame was processed.")
        raise RuntimeError(f"No frames read from source {settings.source!r}.")

    return RunSummary(
        frames=frames_written,
        detections=detection_total,
        output_media=output_media,
        latency_csv=csv_path,
        device=settings.device.value,
        backend=settings.backend.value,
        median_infer_ms=statistics.median(infer_ms_values),
        mean_total_ms=statistics.fmean(total_ms_values),
    )


def _even_size(width: int, height: int) -> tuple[int, int]:
    even_width = width if width % 2 == 0 else width - 1
    even_height = height if height % 2 == 0 else height - 1
    if even_width < 2 or even_height < 2:
        raise RuntimeError(f"Frame is too small to write: {width}x{height}")
    return even_width, even_height


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer
