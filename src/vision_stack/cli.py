"""Command-line entry for the Day-1 perception pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from argparse import Namespace

from vision_stack.config import load_settings
from vision_stack.detect import build_detector
from vision_stack.pipeline import RunSummary, run_pipeline


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser(
        description="Read a webcam or video, detect objects, write annotated frames and a latency CSV."
    )
    parser.add_argument(
        "--source",
        help="Webcam index (0) or a path to a video or image. Overrides the config file.",
    )
    parser.add_argument(
        "--out",
        help="Directory for annotated.mp4 (or annotated.jpg) and latency.csv.",
    )
    parser.add_argument("--device", help="cpu, cuda, or mps. Forwarded to the YOLO backend.")
    parser.add_argument("--model", help="Ultralytics model name or .pt path. Default: yolov8n.pt.")
    parser.add_argument("--confidence", type=float, help="Score threshold in [0, 1].")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Stop after this many frames. 0 means the whole file, or until Ctrl+C on a webcam.",
    )
    parser.add_argument("--imgsz", type=int, help="YOLO inference image size. Default: 640.")
    parser.add_argument(
        "--backend",
        choices=["yolo", "mobilenet_ssd"],
        help="yolo (default) or mobilenet_ssd (OpenCV DNN fallback).",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML defaults. CLI flags and VISION_STACK_* env vars override it.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        settings = load_settings(parse_args(argv))
        logging.info(
            "source=%s device=%s backend=%s model=%s",
            settings.source,
            settings.device.value,
            settings.backend.value,
            settings.model,
        )
        summary = run_pipeline(settings, build_detector(settings))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_summary(summary))
    return 0


def format_summary(summary: RunSummary) -> str:
    return "\n".join(
        [
            f"device: {summary.device}",
            f"backend: {summary.backend}",
            f"frames: {summary.frames}",
            f"detections: {summary.detections}",
            f"median_infer_ms: {summary.median_infer_ms:.2f}",
            f"mean_total_ms: {summary.mean_total_ms:.2f}",
            f"annotated: {summary.output_media}",
            f"latency_csv: {summary.latency_csv}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
