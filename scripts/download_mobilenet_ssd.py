#!/usr/bin/env python3
"""Download the Caffe MobileNet-SSD files used by --backend mobilenet_ssd."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

PROTOTXT_URLS = (
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
    "https://raw.githubusercontent.com/PINTO0309/MobileNet-SSD-RealSense/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.prototxt",
)
WEIGHTS_URLS = (
    "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
    "https://github.com/PINTO0309/MobileNet-SSD-RealSense/raw/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.caffemodel",
)
MIN_WEIGHTS_BYTES = 1_000_000


def _download(urls: tuple[str, ...], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in urls:
        try:
            urllib.request.urlretrieve(url, destination)
        except OSError as exc:
            errors.append(f"{url}: {exc}")
            continue
        if destination.is_file() and destination.stat().st_size > 0:
            return
        errors.append(f"{url}: empty response")
    raise RuntimeError("Download failed:\n" + "\n".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download MobileNet-SSD prototxt and caffemodel.")
    parser.add_argument("--prototxt", default="models/mobilenet_ssd.prototxt")
    parser.add_argument("--weights", default="models/mobilenet_ssd.caffemodel")
    args = parser.parse_args()
    prototxt = Path(args.prototxt)
    weights = Path(args.weights)
    _download(PROTOTXT_URLS, prototxt)
    _download(WEIGHTS_URLS, weights)
    size = weights.stat().st_size
    if size < MIN_WEIGHTS_BYTES:
        weights.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded weights are only {size} bytes. "
            "The URL likely returned a Git LFS pointer instead of the caffemodel."
        )
    print(f"wrote {prototxt}")
    print(f"wrote {weights} ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
