"""Regenerate notebooks/kaggle_train_eval.ipynb (thin Kaggle driver; all logic lives in src/).

Phase 3B/4: base eval + the two primary arms on ONE T4 (phase3a.py remains for the pilot). Run:  python notebooks/build_notebook.py
"""

import json
from pathlib import Path

MD = "markdown"
CODE = "code"

CELLS = [
    (MD, """# ToolDistill-Lab — Phase 3B/4 Kaggle runner (one T4)

Thin driver: logic in `src/tooldistill/experiment.py`. Frozen config (`configs/train_qlora.yaml`,
`frozen: true`) and frozen metrics (eval-v2). **One job per session.** The sealed test set is NOT in
this bundle and `eval_test` is refused.

Settings → Accelerator **GPU T4 x2** (only GPU 0 used) · Internet **On** · Inputs: PRIVATE
`tooldistill-data` (+ `val_eval_400.jsonl`, `train_random_1k.jsonl`, `train_diversity_1k.jsonl`)
and `tooldistill-project` (project_source.zip). Never paste tokens.

Edit **only Cell 1**. Sessions: (1) `eval_base_val` · (2) `train_eval` + `random_1k` ·
(3) `train_eval` + `diversity_1k`. Each needs explicit researcher approval."""),
    (CODE, """# Cell 1 — the ONLY cell you edit
MODE = "preflight"     # preflight | eval_base_val | train_eval
RUN_ID = "random_1k"   # random_1k | diversity_1k  (None for eval_base_val)
OUT_DIR = "/kaggle/working/outputs"
assert MODE in {"preflight", "eval_base_val", "train_eval"}, MODE
assert (RUN_ID is None) == (MODE == "eval_base_val"), "RUN_ID None only for eval_base_val"
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
data_hits = glob.glob("/kaggle/input/**/val_eval_400.jsonl", recursive=True)
assert src_hits, "Attach the tooldistill-project dataset (project_source.zip)"
assert data_hits, "Attach the UPDATED PRIVATE tooldistill-data dataset (with val_eval_400.jsonl)"
PROJECT_DIR = src_hits[0].rsplit("/src/", 1)[0]
DATA_DIR = os.path.dirname(data_hits[0])
print("PROJECT_DIR:", PROJECT_DIR, "\nDATA_DIR:", DATA_DIR)
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
    (CODE, """# Cell 5 — run ONE job
sys.path.insert(0, f"{PROJECT_DIR}/src")
import json, yaml
from tooldistill import experiment
train_cfg = yaml.safe_load(open(f"{PROJECT_DIR}/configs/train_qlora.yaml"))
eval_cfg = yaml.safe_load(open(f"{PROJECT_DIR}/configs/eval.yaml"))
assert train_cfg.get("frozen") is True, "training config must be frozen"
report = experiment.run(MODE, RUN_ID, DATA_DIR, OUT_DIR, train_cfg, eval_cfg)
print(open(f"{OUT_DIR}/{RUN_ID or 'base'}/report.md").read())
"""),
    (CODE, """# Cell 6 — package outputs (report, predictions, adapter; checkpoints dropped)
import shutil
for d in glob.glob(f"{OUT_DIR}/*/checkpoints"):
    shutil.rmtree(d, ignore_errors=True)
name = f"/kaggle/working/{MODE}_{RUN_ID or 'base'}_outputs"
shutil.make_archive(name, "zip", OUT_DIR)
print(f"Download {name}.zip (or Save Version to persist it).")
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
