"""Spec L #17 (Phase-0 part), #18 (notebook default), #20, and $0-API / no-training guards."""

from __future__ import annotations

import json
import re
import subprocess
import sys

import pytest

from tooldistill import PhaseNotImplemented
from tooldistill.cli import main
from tooldistill.io import REPO_ROOT

CODE_DIRS = ["src", "scripts", "tests", "notebooks"]
PAID_API_MODULES = ["openai", "anthropic", "groq", "google.generativeai", "google.genai",
                    "mistralai", "cohere", "together", "deepseek", "litellm"]
SECRET_PATTERNS = [re.compile(p) for p in (
    r"hf_[A-Za-z0-9]{30,}", r"sk-[A-Za-z0-9_-]{20,}", r"gsk_[A-Za-z0-9]{20,}",
    r"AIza[0-9A-Za-z_-]{30,}", r"(?i)(api[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"]{12,}",
)]


def _text_files():
    for d in CODE_DIRS + ["configs", "docs"]:
        for p in (REPO_ROOT / d).rglob("*"):
            if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".md", ".ipynb", ".txt",
                                            ".toml", ".json", ".jsonl"}:
                yield p


def _code_files():
    for d in CODE_DIRS:
        yield from (REPO_ROOT / d).rglob("*.py")


def test_no_paid_model_api_imports():
    pat = re.compile(r"^\s*(?:import|from)\s+(" + "|".join(map(re.escape, PAID_API_MODULES))
                     + r")\b", re.MULTILINE)
    offenders = [str(p) for p in _code_files() if pat.search(p.read_text("utf-8"))]
    assert offenders == []


def test_no_committed_secrets():
    this = __file__
    offenders = []
    for p in _text_files():
        if str(p) == this:
            continue
        text = p.read_text("utf-8", errors="ignore")
        offenders += [f"{p.name}:{pat.pattern}" for pat in SECRET_PATTERNS if pat.search(text)]
    assert offenders == []


WEIGHT_MODULES = {"train.py", "phase3a.py", "experiment.py"}  # GPU-only (Kaggle) loading/training


def test_model_weights_and_training_confined_to_phase3a_modules():
    """Weights/training code may exist ONLY in train.py/phase3a.py, which run on a CUDA GPU
    (Kaggle). Everything else may fetch tokenizer files only."""
    pat = re.compile(r"AutoModel|Trainer\(|snapshot_download|BitsAndBytesConfig|from peft")
    offenders = [p.name for p in (REPO_ROOT / "src").rglob("*.py")
                 if p.name not in WEIGHT_MODULES and pat.search(p.read_text("utf-8"))]
    assert offenders == []
    for p in (REPO_ROOT / "src").rglob("*.py"):
        if p.name in WEIGHT_MODULES:
            continue
        for line in p.read_text("utf-8").splitlines():
            if "from_pretrained(" in line:
                assert "AutoTokenizer" in line, f"{p.name}: non-tokenizer from_pretrained: {line}"
    train_src = (REPO_ROOT / "src/tooldistill/train.py").read_text("utf-8")
    assert 'device_map={"": 0}' in train_src  # quantized weights load only onto a CUDA device


def test_phase3a_pipeline_requires_gpu_and_trains_only_pilot_data():
    src = (REPO_ROOT / "src/tooldistill/phase3a.py").read_text("utf-8")
    assert "preflight(str(out), require_gpu=True)" in src
    # the only training input is `examples`, built from the pilot TRAIN file
    assert "examples, rep[\"data_checks\"] = data_and_mask_checks(tok, pilot," in src
    assert src.count("run_training(") == 2 and src.count("run_training(model, tok, examples") == 2
    assert "train_random" not in src and "train_diversity" not in src  # no arm files loaded


@pytest.mark.parametrize("module,fn", [
    ("quality", "score_quality"), ("diversity", "diversity_select"),
    ("bfcl_adapter", "run_bfcl_subset"),
])
def test_later_phase_modules_refuse_to_run(module, fn):
    import importlib

    mod = importlib.import_module(f"tooldistill.{module}")
    with pytest.raises(PhaseNotImplemented):
        getattr(mod, fn)()


@pytest.mark.parametrize("cmd", ["kaggle-preflight", "train-adapter", "eval-adapter"])
def test_later_phase_cli_commands_exit_without_doing_work(cmd, capsys):
    assert main([cmd]) == 2
    assert "not implemented" in capsys.readouterr().err


@pytest.mark.parametrize("script", ["inspect_xlam", "download_xlam", "prepare_dataset",
                                    "check_access", "audit_selection", "export_kaggle"])
def test_scripts_show_help(script):
    out = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / f"{script}.py"), "--help"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "usage" in out.stdout.lower()


def test_notebook_defaults_to_preflight_and_never_trains_by_default():
    nb = json.loads((REPO_ROOT / "notebooks" / "kaggle_train_eval.ipynb").read_text("utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert re.search(r'^MODE\s*=\s*"preflight"', src, re.MULTILINE)
    assert re.search(r'^RUN_ID\s*=\s*("random_1k"|"diversity_1k"|None)', src, re.MULTILINE)
    assert "for run_id in" not in src and "for RUN_ID in" not in src  # never loops over arms
    assert 'MODE in {"preflight", "eval_base_val", "train_eval"}' in src  # no eval_test in notebook


def test_fixture_is_small_and_labeled_synthetic():
    p = REPO_ROOT / "tests" / "fixtures" / "miniature_xlam.jsonl"
    assert p.stat().st_size < 50_000
    rows = [json.loads(line) for line in p.read_text("utf-8").splitlines()
            if line.startswith("{\"")]
    assert all(r.get("query", "[FIXTURE]").startswith("[FIXTURE]") for r in rows)


def test_data_directory_is_gitignored():
    gi = (REPO_ROOT / ".gitignore").read_text("utf-8")
    assert "data/**" in gi and "artifacts/**" in gi and ".env" in gi
