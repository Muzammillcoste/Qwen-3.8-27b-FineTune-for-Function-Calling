"""Regenerate notebooks/kaggle_train_eval.ipynb (thin Kaggle driver; all logic lives in src/).

Phase 3A: technical feasibility on ONE T4. Run:  python notebooks/build_notebook.py
"""

import json
from pathlib import Path

MD = "markdown"
CODE = "code"

CELLS = [
    (MD, """# ToolDistill-Lab — Phase 3A Kaggle feasibility (one T4)

Thin driver: all logic is in `src/tooldistill/phase3a.py`. **Feasibility only — the 100-example
pilot is not an experimental result.** No 1k/3k arm is trained here.

Settings → Accelerator **GPU T4 x2** (only GPU 0 is used) · Internet **On** (model download) ·
Inputs: your PRIVATE datasets `tooldistill-data` (pilot/val jsonl + manifest) and
`tooldistill-project` (project_source.zip). Never paste tokens here (Qwen is not gated).

Edit **only Cell 1**. Recommended order: `preflight` → `format_check` → `phase3a`."""),
    (CODE, """# Cell 1 — the ONLY cell you edit
MODE = "preflight"    # preflight | format_check | baseline | phase3a
RUN_ID = None         # Phase 3A runs no experimental arm; full arms need separate approval
OUT_DIR = "/kaggle/working/outputs"
assert MODE in {"preflight", "format_check", "baseline", "phase3a"}, MODE
assert RUN_ID is None, "Experimental arms are not approved in Phase 3A"
"""),
    (CODE, """# Cell 2 — single GPU + allocator settings (must run before torch is imported)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"            # T4 x2 is NOT one 32 GB device
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
"""),
    (CODE, """# Cell 3 — locate attached inputs (fails loudly if missing)
import glob
src_hits = glob.glob("/kaggle/input/**/src/tooldistill/__init__.py", recursive=True)
data_hits = glob.glob("/kaggle/input/**/pilot_train_100.jsonl", recursive=True)
assert src_hits, "Attach the tooldistill-project dataset (project_source.zip)"
assert data_hits, "Attach the PRIVATE tooldistill-data dataset"
PROJECT_DIR = src_hits[0].rsplit("/src/", 1)[0]
DATA_DIR = os.path.dirname(data_hits[0])
print("PROJECT_DIR:", PROJECT_DIR, "\\nDATA_DIR:", DATA_DIR)
"""),
    (CODE, """# Cell 4 — deterministic install of exact pins (torch: Kaggle's preinstalled CUDA build)
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                f"{PROJECT_DIR}/requirements-kaggle.txt"], check=True)
os.makedirs(OUT_DIR, exist_ok=True)
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
open(f"{OUT_DIR}/pip_freeze.txt", "w").write(freeze.stdout)
# Resolver warnings about packages this project never uses (gradio, diffusers, ...) are
# harmless. What matters: the stack below must IMPORT cleanly (Trainer also imports `datasets`).
check = subprocess.run([sys.executable, "-c",
    "import torch, transformers, transformers.trainer, peft, accelerate, bitsandbytes, "
    "huggingface_hub, datasets; print('IMPORT OK', 'torch', torch.__version__, "
    "'transformers', transformers.__version__, 'peft', peft.__version__, "
    "'bnb', bitsandbytes.__version__, 'hub', huggingface_hub.__version__, "
    "'datasets', datasets.__version__)"], capture_output=True, text=True)
print(check.stdout or check.stderr[-3000:])
assert check.returncode == 0, "Pinned stack does not import cleanly; send this output"
"""),
    (CODE, """# Cell 5 — run Phase 3A in the selected MODE
sys.path.insert(0, f"{PROJECT_DIR}/src")
import json, yaml
from tooldistill import phase3a
train_cfg = yaml.safe_load(open(f"{PROJECT_DIR}/configs/train_qlora.yaml"))
arm_tokens = {a: json.load(open(f"{PROJECT_DIR}/reports/phase2_manifests/{a}.json"))["tokens_total"]
              for a in ("random_1k", "diversity_1k", "random_3k")} \\
    if os.path.isdir(f"{PROJECT_DIR}/reports/phase2_manifests") else None
report = phase3a.run(MODE, DATA_DIR, OUT_DIR, train_cfg, arm_tokens=arm_tokens,
                     interrupt_at=train_cfg["pilot"]["interrupt_at_step"])
print(json.dumps({k: report.get(k) for k in ("status", "data_checks", "format_check",
                  "baseline_val100", "pilot_val100", "paired_base_vs_pilot", "projections")},
                 indent=1, default=str)[:6000])
"""),
    (CODE, """# Cell 6 — package outputs (report, predictions, logs, adapter; never merged weights)
import shutil
shutil.rmtree(f"{OUT_DIR}/pilot_checkpoints", ignore_errors=True)  # keep the final adapter only
shutil.make_archive("/kaggle/working/phase3a_outputs", "zip", OUT_DIR)
print("Download /kaggle/working/phase3a_outputs.zip (or Save Version to persist it).")
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
                     "language_info": {"name": "python"},
                     "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).with_name("kaggle_train_eval.ipynb")
    out.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print("wrote", out)
