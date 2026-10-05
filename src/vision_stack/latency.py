"""Per-frame latency CSV."""

from __future__ import annotations

import csv
from pathlib import Path

CSV_COLUMNS = (
    "frame_idx",
    "timestamp_ms",
    "infer_ms",
    "total_ms",
    "num_detections",
)


class LatencyLog:
    """Append one row per processed frame. Flushed so Ctrl+C keeps the file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        self._handle.flush()

    def write(
        self,
        frame_idx: int,
        timestamp_ms: float,
        infer_ms: float,
        total_ms: float,
        num_detections: int,
    ) -> None:
        self._writer.writerow(
            {
                "frame_idx": frame_idx,
                "timestamp_ms": f"{timestamp_ms:.3f}",
                "infer_ms": f"{infer_ms:.3f}",
                "total_ms": f"{total_ms:.3f}",
                "num_detections": num_detections,
            }
        )
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> LatencyLog:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
