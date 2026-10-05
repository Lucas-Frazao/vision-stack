#!/usr/bin/env python3
"""Write a short synthetic clip used by the CPU quickstart."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def render_frame(index: int, width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (48, 36, 24)
    horizon = int(height * 0.72)
    cv2.rectangle(frame, (0, horizon), (width, height), (40, 110, 48), thickness=-1)
    origin_x = 48 + index * 16
    cv2.rectangle(frame, (origin_x, 210), (origin_x + 70, 360), (190, 150, 90), thickness=-1)
    cv2.circle(frame, (origin_x + 35, 180), 28, (210, 190, 170), thickness=-1)
    cv2.rectangle(frame, (420, 80), (560, 200), (70, 70, 180), thickness=-1)
    return frame


def write_clip(path: Path, frames: int, width: int, height: int, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for index in range(frames):
            writer.write(render_frame(index, width, height))
    finally:
        writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate data/sample/synthetic_demo.mp4")
    parser.add_argument(
        "--out",
        default="data/sample/synthetic_demo.mp4",
        help="Output mp4 path.",
    )
    parser.add_argument("--frames", type=int, default=24)
    args = parser.parse_args()
    destination = Path(args.out)
    write_clip(destination, frames=args.frames, width=640, height=480, fps=10.0)
    print(f"wrote {destination} ({args.frames} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
