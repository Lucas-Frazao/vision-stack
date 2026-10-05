"""Runtime settings from YAML, environment, and CLI flags."""

from __future__ import annotations

import os
from argparse import Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

import yaml

ENV_DEVICE = "VISION_STACK_DEVICE"
ENV_MODEL = "VISION_STACK_MODEL"
ENV_SOURCE = "VISION_STACK_SOURCE"
ENV_BACKEND = "VISION_STACK_BACKEND"


class DeviceName(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class Backend(str, Enum):
    YOLO = "yolo"
    MOBILENET_SSD = "mobilenet_ssd"


def normalize_device(value: str) -> DeviceName:
    key = value.strip().lower()
    try:
        return DeviceName(key)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in DeviceName)
        raise ValueError(f"Unknown device {value!r}. Expected one of: {allowed}.") from exc


def normalize_backend(value: str) -> Backend:
    key = value.strip().lower()
    try:
        return Backend(key)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in Backend)
        raise ValueError(f"Unknown backend {value!r}. Expected one of: {allowed}.") from exc


@dataclass(frozen=True)
class Settings:
    source: str
    output_dir: Path
    device: DeviceName
    model: str
    confidence: float
    max_frames: int | None
    imgsz: int
    backend: Backend
    mobilenet_prototxt: Path
    mobilenet_weights: Path


def load_settings(
    args: Namespace,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """CLI overrides env, env overrides the YAML file."""
    env = os.environ if environ is None else environ
    file_values = _read_yaml(Path(args.config))
    source = _pick(args.source, env.get(ENV_SOURCE), file_values.get("source"), "0")
    output_dir = _pick(args.out, None, file_values.get("output_dir"), "outputs")
    device = _pick(args.device, env.get(ENV_DEVICE), file_values.get("device"), "cpu")
    model = _pick(args.model, env.get(ENV_MODEL), file_values.get("model"), "yolov8n.pt")
    backend = _pick(
        args.backend,
        env.get(ENV_BACKEND),
        file_values.get("backend"),
        "yolo",
    )
    confidence = _pick(
        args.confidence,
        None,
        file_values.get("confidence"),
        0.25,
    )
    imgsz = _pick(args.imgsz, None, file_values.get("imgsz"), 640)
    max_frames = _pick(
        args.max_frames,
        None,
        file_values.get("max_frames"),
        0,
    )
    prototxt = _pick(
        None,
        None,
        file_values.get("mobilenet_prototxt"),
        "models/mobilenet_ssd.prototxt",
    )
    weights = _pick(
        None,
        None,
        file_values.get("mobilenet_weights"),
        "models/mobilenet_ssd.caffemodel",
    )
    confidence_value = float(confidence)
    imgsz_value = int(imgsz)
    if not 0.0 <= confidence_value <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if imgsz_value <= 0:
        raise ValueError("imgsz must be a positive integer")
    return Settings(
        source=str(source),
        output_dir=Path(str(output_dir)),
        device=normalize_device(str(device)),
        model=str(model),
        confidence=confidence_value,
        max_frames=_optional_limit(int(max_frames)),
        imgsz=imgsz_value,
        backend=normalize_backend(str(backend)),
        mobilenet_prototxt=Path(str(prototxt)),
        mobilenet_weights=Path(str(weights)),
    )


def _read_yaml(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Config not found: {path}. Run from the repository root, "
            "or pass --config with a YAML file."
        )
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return loaded


def _pick(cli_value: object, env_value: object, file_value: object, default: object) -> object:
    if cli_value is not None:
        return cli_value
    if env_value is not None and env_value != "":
        return env_value
    if file_value is not None:
        return file_value
    return default


def _optional_limit(value: int) -> int | None:
    if value < 0:
        raise ValueError("max_frames must be >= 0")
    if value == 0:
        return None
    return value
