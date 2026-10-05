"""CPU pipeline tests that do not download model weights."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_stack.capture import FrameSource
from vision_stack.config import DeviceName, load_settings
from vision_stack.detections import Detection
from vision_stack.latency import CSV_COLUMNS, LatencyLog
from vision_stack.pipeline import run_pipeline
from vision_stack.ssd_detector import MobileNetSsdDetector


def _args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "source": None,
        "out": None,
        "device": None,
        "model": None,
        "confidence": None,
        "max_frames": None,
        "imgsz": None,
        "backend": None,
        "config": str(ROOT / "configs" / "default.yaml"),
    }
    values.update(overrides)
    return Namespace(**values)


def _write_video(path: Path, frames: int, size: tuple[int, int] = (64, 48)) -> None:
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not write {path}")
    try:
        for index in range(frames):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[:, :] = (20, 20, index * 10)
            writer.write(image)
    finally:
        writer.release()


class _OneBox:
    def predict(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        return [Detection("widget", 0.91, 1, 1, width // 2, height // 2)]


class LatencyTests(unittest.TestCase):
    def test_csv_columns_and_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latency.csv"
            with LatencyLog(path) as log:
                log.write(0, 0.0, 12.5, 14.25, 2)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0].keys()), list(CSV_COLUMNS))
            self.assertEqual(rows[0]["frame_idx"], "0")
            self.assertEqual(rows[0]["timestamp_ms"], "0.000")
            self.assertEqual(rows[0]["infer_ms"], "12.500")
            self.assertEqual(rows[0]["total_ms"], "14.250")
            self.assertEqual(rows[0]["num_detections"], "2")


class SettingsTests(unittest.TestCase):
    def test_cli_overrides_yaml_and_env(self) -> None:
        settings = load_settings(
            _args(source="clip.mp4", device="cpu", max_frames=8),
            environ={"VISION_STACK_DEVICE": "cuda", "VISION_STACK_SOURCE": "env.mp4"},
        )
        self.assertEqual(settings.source, "clip.mp4")
        self.assertEqual(settings.device, DeviceName.CPU)
        self.assertEqual(settings.max_frames, 8)
        self.assertEqual(settings.model, "yolov8n.pt")

    def test_env_overrides_yaml(self) -> None:
        settings = load_settings(
            _args(),
            environ={"VISION_STACK_DEVICE": "mps", "VISION_STACK_BACKEND": "mobilenet_ssd"},
        )
        self.assertEqual(settings.device, DeviceName.MPS)
        self.assertEqual(settings.backend.value, "mobilenet_ssd")
        self.assertIsNone(settings.max_frames)

    def test_unknown_device_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_settings(_args(device="tpu"))


class PipelineTests(unittest.TestCase):
    def test_video_writes_annotation_and_latency_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "in.mp4"
            out = root / "out"
            _write_video(video, frames=4)
            settings = load_settings(
                _args(source=str(video), out=str(out), device="cpu", max_frames=4)
            )
            summary = run_pipeline(settings, _OneBox())
            self.assertEqual(summary.frames, 4)
            self.assertEqual(summary.detections, 4)
            self.assertEqual(summary.device, "cpu")
            self.assertTrue(summary.output_media.is_file())
            self.assertEqual(summary.output_media.name, "annotated.mp4")
            with summary.latency_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual([row["num_detections"] for row in rows], ["1", "1", "1", "1"])
            self.assertEqual([row["frame_idx"] for row in rows], ["0", "1", "2", "3"])
            captured = cv2.VideoCapture(str(summary.output_media))
            try:
                self.assertTrue(captured.isOpened())
                read = 0
                while True:
                    ok, _frame = captured.read()
                    if not ok:
                        break
                    read += 1
                self.assertEqual(read, 4)
            finally:
                captured.release()

    def test_max_frames_stops_early(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "in.mp4"
            _write_video(video, frames=5)
            settings = load_settings(
                _args(source=str(video), out=str(root / "out"), device="cpu", max_frames=2)
            )
            summary = run_pipeline(settings, _OneBox())
            self.assertEqual(summary.frames, 2)

    def test_image_writes_jpg_and_one_csv_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "still.png"
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[:, :] = (10, 20, 30)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            settings = load_settings(
                _args(source=str(image_path), out=str(root / "out"), device="cpu")
            )
            summary = run_pipeline(settings, _OneBox())
            self.assertEqual(summary.frames, 1)
            self.assertEqual(summary.output_media.name, "annotated.jpg")
            self.assertTrue(summary.output_media.is_file())

    def test_missing_video_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = load_settings(
                _args(source=str(root / "missing.mp4"), out=str(root / "out"), device="cpu")
            )
            with self.assertRaises(FileNotFoundError):
                run_pipeline(settings, _OneBox())

    def test_checked_in_sample_clip(self) -> None:
        sample = ROOT / "data" / "sample" / "synthetic_demo.mp4"
        with FrameSource(str(sample), max_frames=None) as source:
            frames = list(source.frames())
        self.assertEqual(len(frames), 24)
        self.assertEqual(frames[0].image.shape, (480, 640, 3))
        self.assertEqual(frames[0].timestamp_ms, 0.0)
        self.assertGreater(frames[-1].timestamp_ms, frames[0].timestamp_ms)

    def test_frame_source_reads_every_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "in.mp4"
            _write_video(video, frames=3)
            with FrameSource(str(video), max_frames=None) as source:
                frames = list(source.frames())
            self.assertEqual([frame.index for frame in frames], [0, 1, 2])
            self.assertGreaterEqual(frames[-1].timestamp_ms, 0.0)


class FallbackTests(unittest.TestCase):
    def test_missing_mobilenet_files_explain_the_download_script(self) -> None:
        with self.assertRaises(FileNotFoundError) as raised:
            MobileNetSsdDetector(
                Path("models/missing.prototxt"),
                Path("models/missing.caffemodel"),
                confidence=0.4,
                device=DeviceName.CPU,
            )
        self.assertIn("download_mobilenet_ssd.py", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
