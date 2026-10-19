"""Phase 3B: fixed eval sets (val_eval_400 bundled; test_eval_1000 sealed) + arm checks + bundle."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tooldistill.cli import COMMANDS  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(COMMANDS["export-phase3b"](sys.argv[1:]))
