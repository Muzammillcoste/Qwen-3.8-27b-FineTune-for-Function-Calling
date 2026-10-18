"""Phase 1: download the gated xLAM raw file at the pinned revision."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tooldistill.cli import COMMANDS  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(COMMANDS["download"](sys.argv[1:]))
