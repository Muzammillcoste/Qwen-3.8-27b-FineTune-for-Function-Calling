"""Spec L #14-#15 with the REAL Qwen chat template (tokenizer files only, read from local cache).

Skipped automatically if transformers or the cached tokenizer is unavailable (offline CI).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("transformers")

from tooldistill.format_qwen import render, to_messages, truncation_stats  # noqa: E402
from tooldistill.io import load_yaml  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    t = load_yaml("configs/data.yaml")["tokenizer"]
    try:
        return AutoTokenizer.from_pretrained(t["repo_id"], revision=t["revision"],
                                             local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not in local HF cache")


@pytest.mark.parametrize("sid", ["fixture:0", "fixture:40003", "fixture:5", "fixture:40016"])
def test_renders_native_tool_call_and_tool_docs(tok, by_id, sid):
    rec = by_id[sid]
    msgs, assistant, tools = to_messages(rec)
    full = tok.apply_chat_template(msgs + [assistant], tools=tools, tokenize=False)
    assert "<tools>" in full and "# Tools" in full
    for t in rec.tools:  # tool documentation (incl. dotted names) is in the prompt
        assert json.dumps(t["function"]["name"]) in full
    r = render(rec, tok)
    # the template's system text itself mentions <tool_call>; count blocks in the target only
    assert r.completion_text.count("<tool_call>") == rec.call_count  # one block per call
    assert r.calls_rendered and r.prefix_ok


def test_loss_boundary_excludes_prompt_and_includes_target(tok, by_id):
    rec = by_id["fixture:40003"]
    r = render(rec, tok)
    assert r.prompt_tokens > 0 and r.completion_tokens > 0
    assert "<tool_call>" in r.completion_text and "<|im_end|>" in r.completion_text
    assert rec.query not in r.completion_text           # user text is prompt-side only
    assert "weather.forecast.v2.get" in r.completion_text


def test_truncation_stats_counts_targets_cut(tok, by_id):
    rows = [render(by_id[s], tok) for s in ("fixture:0", "fixture:40016")]
    s = truncation_stats(rows, max_len=min(r.total_tokens for r in rows))
    assert s["truncated"] == 1 and s["completion_cut"] == 1
