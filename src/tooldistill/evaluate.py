"""Offline parsing of Qwen-native ``<tool_call>`` output and structured-call scoring (spec I2/I3).

Never ``eval``s model output, never executes tools, never calls external services. JSON only.
Numeric canonicalization: 5 and 5.0 are equal (JSON numbers); "5" (string) and 5 are NOT.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from .io import canonical_json
from .schemas import NormalizedRecord
from .validate import compiled_validator

EVAL_VERSION = "eval-v1"
BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
END_TOKENS = ("<|im_end|>", "<|endoftext|>")


@dataclass
class ParsedOutput:
    calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    n_blocks: int = 0
    text_outside_blocks: bool = False

    @property
    def parse_ok(self) -> bool:
        return self.n_blocks > 0 and not self.errors


def parse_tool_calls(text: str) -> ParsedOutput:
    out = ParsedOutput()
    for end in END_TOKENS:
        if end in text:
            text = text.split(end, 1)[0]
    blocks = BLOCK.findall(text)
    out.n_blocks = len(blocks)
    if text.count("<tool_call>") != len(blocks):
        out.errors.append("unclosed_tool_call_block")
    if BLOCK.sub("", text).replace("<tool_call>", "").strip():
        out.text_outside_blocks = True
    for i, raw in enumerate(blocks):
        try:
            obj = json.loads(raw.strip())
        except json.JSONDecodeError:
            out.errors.append(f"block{i}:invalid_json")
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            out.errors.append(f"block{i}:missing_name")
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                out.errors.append(f"block{i}:arguments_not_json")
                continue
        if not isinstance(args, dict):
            out.errors.append(f"block{i}:arguments_not_object")
            continue
        out.calls.append({"name": obj["name"], "arguments": args})
    return out


def canonical_value(v: Any) -> Any:
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return repr(v)
        return int(v) if float(v).is_integer() else float(v)
    if isinstance(v, list):
        return [canonical_value(x) for x in v]
    if isinstance(v, dict):
        return {k: canonical_value(x) for k, x in v.items()}
    return repr(v)


def call_key(c: dict[str, Any]) -> str:
    return canonical_json({"name": c["name"], "arguments": canonical_value(c["arguments"])})


@dataclass
class ExampleScore:
    sample_id: str
    parse_ok: bool
    exact_match: bool
    tool_names_match: bool
    n_expected: int
    n_predicted: int
    hallucinated_tool: bool
    schema_valid_prediction: bool
    arg_keys_match: bool
    arg_values_correct: int
    arg_values_total: int
    text_outside_blocks: bool
    errors: list[str]
    predicted: list[dict[str, Any]]
    expected: list[dict[str, Any]]
    error_category: str


def score_example(rec: NormalizedRecord, raw_output: str) -> ExampleScore:
    parsed = parse_tool_calls(raw_output)
    expected = [{"name": c.name, "arguments": c.arguments} for c in rec.expected_calls]
    pred = parsed.calls
    available = {t["function"]["name"] for t in rec.tools}
    exact = parsed.parse_ok and Counter(map(call_key, pred)) == Counter(map(call_key, expected))
    names = Counter(c["name"] for c in pred) == Counter(c["name"] for c in expected)
    halluc = any(c["name"] not in available for c in pred)

    schema_ok = bool(pred)
    for c in pred:
        ok_any = False
        for t in rec.tools:
            if t["function"]["name"] == c["name"]:
                v, _ = compiled_validator(t["function"]["parameters"])
                ok_any |= v is not None and v.is_valid(c["arguments"])
        schema_ok &= ok_any

    arg_ok = arg_tot = 0
    keys_match = False
    if len(pred) == len(expected) == 1 and pred[0]["name"] == expected[0]["name"]:
        pa, ea = pred[0]["arguments"], expected[0]["arguments"]
        keys_match = set(pa) == set(ea)
        for k, v in ea.items():
            arg_tot += 1
            arg_ok += k in pa and canonical_value(pa[k]) == canonical_value(v)

    if exact:
        cat = "correct"
    elif not parsed.n_blocks:
        cat = "no_tool_call"
    elif parsed.errors:
        cat = "malformed_tool_call"
    elif halluc:
        cat = "hallucinated_tool"
    elif not names:
        cat = "wrong_tool_or_call_count"
    elif not keys_match and len(expected) == 1:
        cat = "wrong_argument_keys"
    else:
        cat = "wrong_argument_values"
    return ExampleScore(rec.sample_id, parsed.parse_ok, exact, names, len(expected), len(pred),
                        halluc, schema_ok, keys_match, arg_ok, arg_tot,
                        parsed.text_outside_blocks, parsed.errors, pred, expected, cat)


def _rate(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None}


def aggregate(scores: list[ExampleScore]) -> dict[str, Any]:
    n = len(scores)
    return {
        "eval_version": EVAL_VERSION,
        "n": n,
        "exact_structured_call_accuracy": _rate(sum(s.exact_match for s in scores), n),
        "parse_rate": _rate(sum(s.parse_ok for s in scores), n),
        "tool_name_accuracy": _rate(sum(s.tool_names_match for s in scores), n),
        "schema_valid_prediction_rate": _rate(sum(s.schema_valid_prediction for s in scores), n),
        "hallucinated_tool_rate": _rate(sum(s.hallucinated_tool for s in scores), n),
        "argument_value_accuracy": _rate(sum(s.arg_values_correct for s in scores),
                                         sum(s.arg_values_total for s in scores)),
        "text_outside_blocks_rate": _rate(sum(s.text_outside_blocks for s in scores), n),
        "error_categories": dict(Counter(s.error_category for s in scores).most_common()),
    }


def paired_comparison(a: list[ExampleScore], b: list[ExampleScore]) -> dict[str, Any]:
    """Exact-match agreement table on the SAME examples + exact two-sided McNemar p-value.
    Descriptive only for the 100-example pilot: no improvement claims."""
    by_b = {s.sample_id: s for s in b}
    both = only_a = only_b = neither = 0
    for s in a:
        t = by_b[s.sample_id]
        both += s.exact_match and t.exact_match
        only_a += s.exact_match and not t.exact_match
        only_b += t.exact_match and not s.exact_match
        neither += not s.exact_match and not t.exact_match
    d = only_a + only_b
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(d, i) for i in range(k + 1)) / 2 ** d) if d else 1.0
    return {"both_correct": both, "only_first": only_a, "only_second": only_b,
            "neither": neither, "discordant": d, "mcnemar_exact_p": round(p, 4)}


def score_to_dict(s: ExampleScore) -> dict[str, Any]:
    return asdict(s)
