"""Phase 1 audit report on the fixture (offline: character lengths, no tokenizer download)."""

from __future__ import annotations

import pytest

from tooldistill.audit import build_audit, load_records, render_audit_md
from tooldistill.io import load_yaml
from tooldistill.pipeline import prepare
from tooldistill.token_lengths import char_lengths, load_qwen_tokenizer

from .conftest import FIXTURE


@pytest.fixture
def audit(tmp_path):
    prepare(load_yaml("configs/data.yaml"), FIXTURE, tmp_path, seed=42, fixture=True)
    rows = char_lengths(load_records(tmp_path / "normalized.jsonl"))
    return build_audit(tmp_path, None, rows, "characters (test)")


def test_audit_separates_invalid_from_ambiguous(audit):
    st = audit["label_status"]
    assert st["invalid"] >= 3 and st["ambiguous"] >= 2 and st["no_call"] == 1
    assert "missing_arg_unknown_requiredness" in audit["warning_kinds"]
    assert audit["flag_kinds"]["default_type_mismatch"]["category"] == "schema_issue"
    assert audit["flag_kinds"]["call_args_invalid"]["category"] == "label_error"


def test_audit_requiredness_and_conflicts(audit):
    rq = audit["requiredness"]
    assert set(rq["state"]) == {"REQUIRED", "OPTIONAL", "UNKNOWN"}
    assert rq["conflicts"] == {"declared_required_but_default_present": 1}


def test_audit_has_examples_for_every_emitted_kind(audit):
    kinds = set(audit["flag_kinds"]) | set(audit["warning_kinds"])
    assert kinds <= set(audit["examples"])


def test_audit_markdown_renders_all_sections(audit):
    md = render_audit_md(audit)
    for h in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7.", "## 8.", "## 9.",
              "## 10.", "## 11."):
        assert h in md


def test_tokenizer_loader_refuses_non_tokenizer_files():
    with pytest.raises(ValueError):
        load_qwen_tokenizer("Qwen/Qwen3-4B-Instruct-2507", "rev",
                            "model-00001-of-00003.safetensors")
