"""Helper to run scripts directly from repository root without editable install."""

from __future__ import annotations

from pathlib import Path
import sys


def ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
