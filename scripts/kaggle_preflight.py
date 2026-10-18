"""Phase 3 stub: Kaggle GPU/environment preflight."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tooldistill.cli import COMMANDS  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(COMMANDS["kaggle-preflight"](sys.argv[1:]))
