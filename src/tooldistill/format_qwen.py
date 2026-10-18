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


IGNORE_INDEX = -100


class OverLengthError(ValueError):
    """Raised instead of truncating: a cut target must never be trained on."""


def prompt_ids(rec: NormalizedRecord, tok, system_prompt: str | None = SYSTEM_PROMPT) -> list[int]:
    """Inference prompt ids. Identical to the training prompt prefix (checked in tests)."""
    msgs, _, tools = to_messages(rec, system_prompt)
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


def build_training_example(rec: NormalizedRecord, tok, max_len: int,
                           system_prompt: str | None = SYSTEM_PROMPT) -> dict[str, list[int]]:
    """input_ids + labels for completion-only loss.

    labels = -100 on every prompt token (system/tools/user/assistant header); active on the
    assistant tool-call tokens up to and INCLUDING the first ``<|im_end|>`` (EOS). Tokens the
    template appends after EOS (a newline) are masked. Over-length -> OverLengthError.
    """
    msgs, assistant, tools = to_messages(rec, system_prompt)
    p_text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
    f_text = tok.apply_chat_template(msgs + [assistant], tools=tools, tokenize=False)
    p_ids = tok(p_text, add_special_tokens=False)["input_ids"]
    ids = tok(f_text, add_special_tokens=False)["input_ids"]
    if ids[: len(p_ids)] != p_ids:
        raise ValueError(f"{rec.sample_id}: prompt is not a token prefix of the full render")
    if len(ids) > max_len:
        raise OverLengthError(f"{rec.sample_id}: {len(ids)} tokens > max_len {max_len}")
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    labels = [IGNORE_INDEX] * len(p_ids) + ids[len(p_ids):]
    try:
        end = ids.index(eos, len(p_ids))
    except ValueError:
        raise ValueError(f"{rec.sample_id}: no <|im_end|> in the assistant target") from None
    for i in range(end + 1, len(labels)):
        labels[i] = IGNORE_INDEX
    if all(x == IGNORE_INDEX for x in labels):
        raise ValueError(f"{rec.sample_id}: empty supervised target")
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids),
            "n_prompt": len(p_ids), "n_target": end + 1 - len(p_ids)}


def truncation_stats(rows: list[Rendered], max_len: int) -> dict[str, Any]:
    """At ``max_len``: records truncated at all, and records whose completion would be cut."""
    n = len(rows)
    over = sum(r.total_tokens > max_len for r in rows)
    target_cut = sum(r.total_tokens > max_len for r in rows)  # right-truncation cuts the target
    prompt_alone_over = sum(r.prompt_tokens >= max_len for r in rows)
    return {"max_len": max_len, "n": n, "truncated": over,
            "truncated_pct": round(100 * over / n, 3) if n else 0.0,
            "completion_cut": target_cut, "prompt_alone_exceeds": prompt_alone_over}
