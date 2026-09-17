"""Puts this directory on `sys.path` so `pytest tests/` works without PYTHONPATH."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
