"""Inspect xLAM (HF repo id or local fixture file). See docs/KAGGLE_GUIDE.md and README."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tooldistill.cli import COMMANDS  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(COMMANDS["inspect"](sys.argv[1:]))
