"""Regenerate notebooks/kaggle_train_eval.ipynb (thin Kaggle driver; logic lives in src/).

Phase 0 skeleton: cells call CLI commands that are stubs until Phase 3 is approved.
Run:  python notebooks/build_notebook.py
"""

import json
from pathlib import Path

MD = "markdown"
CODE = "code"

CELLS = [
    (MD, """# ToolDistill-Lab — Kaggle driver (thin)

**Phase 0 skeleton.** All logic lives in `src/tooldistill`. Commands for preflight, pilot, training
and evaluation are stubs until Phase 3 is approved; running them prints *not implemented* and does
nothing. Never store tokens in this notebook. One training run per session; never loop over arms."""),
    (CODE, """# Cell 1 — human-set parameters (the ONLY cell you edit)
MODE = "preflight"   # one of: preflight | pilot | train | eval
RUN_ID = None        # e.g. "random_1k" — required for train/eval; set manually, one per session
PROJECT_DIR = "/kaggle/input/tooldistill-project"   # attached project snapshot (read-only)
DATA_DIR = "/kaggle/input/tooldistill-data"         # attached PRIVATE prepared data (read-only)
OUT_DIR = "/kaggle/working/outputs"
assert MODE in {"preflight", "pilot", "train", "eval"}, MODE
if MODE in {"train", "eval"}:
    assert RUN_ID, "Set RUN_ID explicitly for train/eval"
"""),
    (CODE, """# Cell 2 — environment: verify pinned versions (no upgrades mid-study)
import subprocess, sys
print(sys.version)
subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"], check=False)
"""),
    (CODE, """# Cell 3 — GPU / memory / disk (read-only checks; no allocation)
import shutil, subprocess
subprocess.run(["nvidia-smi"], check=False)
print("disk free GB:", shutil.disk_usage("/kaggle/working").free / 1e9)
"""),
    (CODE, """# Cell 4 — paths + checksums of attached data (fails loudly if missing)
import os, sys
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
os.makedirs(OUT_DIR, exist_ok=True)
print(os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else f"DATA_DIR missing: {DATA_DIR}")
"""),
    (CODE, """# Cell 5 — preflight (Phase 3 implements tooldistill.hardware.preflight)
from tooldistill.cli import main as td
rc = td(["kaggle-preflight"])
print("preflight exit code:", rc)
"""),
    (CODE, """# Cell 6 — pilot / train / eval dispatch: runs ONLY the selected MODE, ONLY for RUN_ID
if MODE == "pilot":
    rc = td(["train-adapter", "--config", "configs/train_qlora.yaml", "--pilot"])
elif MODE == "train":
    rc = td(["train-adapter", "--config", "configs/train_qlora.yaml", "--run-id", RUN_ID])
elif MODE == "eval":
    rc = td(["eval-adapter", "--config", "configs/eval.yaml", "--run-id", RUN_ID])
else:
    rc = 0
    print("MODE=preflight: no training or evaluation started.")
print("exit code:", rc)
"""),
    (CODE, """# Cell 7 — package outputs (manifest, logs, adapters only; never merged full weights)
import shutil
if os.listdir(OUT_DIR):
    shutil.make_archive("/kaggle/working/outputs_bundle", "zip", OUT_DIR)
    print("bundle written; save a notebook version to persist it")
else:
    print("no outputs to bundle")
"""),
]


def build() -> dict:
    cells = []
    for kind, src in CELLS:
        lines = src.strip("\n").splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == CODE:
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).with_name("kaggle_train_eval.ipynb")
    out.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print("wrote", out)
