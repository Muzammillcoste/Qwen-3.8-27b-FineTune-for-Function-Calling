"""Render records with the REAL Qwen chat template (tokenizer files only; no model weights).

Prompt = [optional system] + user query + tools (the template's own ``# Tools`` system block).
Target = one assistant turn with native ``<tool_call>`` blocks, produced by the template from
``assistant.tool_calls``. The prompt render (``add_generation_prompt=True``) must be an exact
token-level prefix of the full render; the remainder is the loss-bearing completion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .schemas import NormalizedRecord

RENDER_VERSION = "qwen-native-tools-v1"
SYSTEM_PROMPT: str | None = None  # no custom system text; the template's tool block is the system


def load_chat_tokenizer(repo_id: str, revision: str):
    """AutoTokenizer for the pinned revision. Downloads tokenizer/config files only (~16 MB)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(repo_id, revision=revision)


def template_sha256(tok) -> str:
    return hashlib.sha256((tok.chat_template or "").encode("utf-8")).hexdigest()


def to_messages(rec: NormalizedRecord, system_prompt: str | None = SYSTEM_PROMPT):
    msgs: list[dict[str, Any]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": rec.query})
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                       for c in rec.expected_calls],
    }
    return msgs, assistant, rec.tools


@dataclass
class Rendered:
    sample_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prefix_ok: bool        # prompt ids are an exact prefix of full ids (loss-mask boundary valid)
    calls_rendered: bool   # every expected call name + arguments JSON appears in the completion
    completion_text: str


def render(rec: NormalizedRecord, tok, system_prompt: str | None = SYSTEM_PROMPT) -> Rendered:
    msgs, assistant, tools = to_messages(rec, system_prompt)
    prompt = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs + [assistant], tools=tools, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    prefix_ok = full.startswith(prompt) and f_ids[: len(p_ids)] == p_ids
    completion = full[len(prompt):] if full.startswith(prompt) else ""
    calls_ok = all(
        json.dumps(c.name, ensure_ascii=False) in completion
        and json.dumps(c.arguments, ensure_ascii=False) in completion
        for c in rec.expected_calls
    )
    return Rendered(rec.sample_id, len(p_ids), len(f_ids) - len(p_ids), len(f_ids), prefix_ok,
                    calls_ok, completion)


def truncation_stats(rows: list[Rendered], max_len: int) -> dict[str, Any]:
    """At ``max_len``: records truncated at all, and records whose completion would be cut."""
    n = len(rows)
    over = sum(r.total_tokens > max_len for r in rows)
    target_cut = sum(r.total_tokens > max_len for r in rows)  # right-truncation cuts the target
    prompt_alone_over = sum(r.prompt_tokens >= max_len for r in rows)
    return {"max_len": max_len, "n": n, "truncated": over,
            "truncated_pct": round(100 * over / n, 3) if n else 0.0,
            "completion_cut": target_cut, "prompt_alone_exceeds": prompt_alone_over}
