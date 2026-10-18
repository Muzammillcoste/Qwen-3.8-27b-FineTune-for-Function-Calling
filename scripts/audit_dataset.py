"""Phase 1: factual data audit (requiredness, labels, defaults, duplicates, token lengths, sources)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tooldistill.cli import COMMANDS  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(COMMANDS["audit-data"](sys.argv[1:]))
