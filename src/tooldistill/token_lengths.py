"""Token-length measurement with the REAL Qwen tokenizer file (tokenizer.json only, no weights).

Counts are per raw segment (query text, tools JSON, expected-call blocks). Chat-template overhead
(system text, role tags, tool preamble) is NOT included: rendered lengths are measured in Phase 3
with the actual chat template. Treat totals here as lower bounds of the rendered sequence length.
"""

from __future__ import annotations

import json
from typing import Any

from .schemas import NormalizedRecord

TOKEN_METHOD = "qwen_tokenizer_json_raw_segments_v1"


def load_qwen_tokenizer(repo_id: str, revision: str, filename: str = "tokenizer.json"):
    """Download ONLY the tokenizer file at a pinned revision. Never downloads model weights."""
    if not filename.startswith("tokenizer"):
        raise ValueError(f"refusing to download non-tokenizer file: {filename}")
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = hf_hub_download(repo_id, filename, revision=revision)
    return Tokenizer.from_file(path)


def segments(rec: NormalizedRecord) -> dict[str, str]:
    tools = "\n".join(json.dumps(t, ensure_ascii=False) for t in rec.tools)
    calls = "\n".join(
        "<tool_call>\n" + json.dumps({"name": c.name, "arguments": c.arguments},
                                     ensure_ascii=False) + "\n</tool_call>"
        for c in rec.expected_calls
    )
    return {"query": rec.query, "tools": tools, "calls": calls}


def count_tokens(records: list[NormalizedRecord], tokenizer, batch: int = 2048) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for start in range(0, len(records), batch):
        chunk = records[start:start + batch]
        segs = [segments(r) for r in chunk]
        counts = {}
        for key in ("query", "tools", "calls"):
            enc = tokenizer.encode_batch([s[key] for s in segs], add_special_tokens=False)
            counts[key] = [len(e.ids) for e in enc]
        for i, r in enumerate(chunk):
            q, t, c = counts["query"][i], counts["tools"][i], counts["calls"][i]
            out.append({"sample_id": r.sample_id, "query": q, "tools": t, "calls": c,
                        "total": q + t + c})
    return out


def char_lengths(records: list[NormalizedRecord]) -> list[dict[str, Any]]:
    """Fallback when the tokenizer is unavailable: character counts (clearly labelled)."""
    rows = []
    for r in records:
        s = segments(r)
        q, t, c = len(s["query"]), len(s["tools"]), len(s["calls"])
        rows.append({"sample_id": r.sample_id, "query": q, "tools": t, "calls": c,
                     "total": q + t + c})
    return rows
