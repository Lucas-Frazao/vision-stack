#!/usr/bin/env python3
"""Run the perception pipeline from a checkout without an editable install."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vision_stack.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
