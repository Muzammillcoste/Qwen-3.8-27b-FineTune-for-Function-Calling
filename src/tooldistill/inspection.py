"""Dataset inspection report (spec E2). Terminal-safe: never dumps the whole dataset.

Token lengths here are APPROXIMATE (whitespace tokens / characters). Real Qwen token lengths are
measured after rendering with the actual tokenizer chat template in a later phase.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from typing import Any

from .io import canonical_json
from .normalize_xlam import NormalizeOptions, normalize_rows
from .provenance import check_id_integrity
from .schemas import NormalizedRecord, RejectedRecord
from .validate import apply_validation, flag_kind


def _field_profile(rows: list[Any]) -> dict[str, dict[str, Any]]:
    prof: dict[str, dict[str, Any]] = {}
    rows = [r for r in rows if isinstance(r, dict) and "__malformed_line__" not in r]
    for row in rows:
        for k, v in row.items():
            p = prof.setdefault(k, {"types": Counter(), "nulls": 0, "json_encoded_strings": 0,
                                    "present": 0})
            p["present"] += 1
            p["types"][type(v).__name__] += 1
            if v is None:
                p["nulls"] += 1
            elif isinstance(v, str) and v[:1] in "[{":
                try:
                    json.loads(v)
                    p["json_encoded_strings"] += 1
                except json.JSONDecodeError:
                    pass
    n = len(rows)
    for p in prof.values():
        p["missing"] = n - p["present"]
        p["types"] = dict(p["types"])
    return prof


def _stats(xs: list[int]) -> dict[str, Any]:
    if not xs:
        return {}
    xs = sorted(xs)
    return {"min": xs[0], "p50": statistics.median(xs), "p95": xs[int(0.95 * (len(xs) - 1))],
            "max": xs[-1], "mean": round(statistics.fmean(xs), 1)}


def _trunc(s: str, n: int = 160) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def build_inspection(rows: list[Any], opts: NormalizeOptions) -> dict[str, Any]:
    records, rejected = normalize_rows(rows, opts)
    for r in records:
        apply_validation(r)
    integrity = check_id_integrity([r.original_id for r in records])
    flag_counts = Counter(flag_kind(f) for r in records for f in r.flags)
    note_counts = Counter(n.split(":", 1)[0] for r in records for n in r.notes)
    teacher = (dict(Counter(r.teacher_source for r in records))
               if integrity["id_field_trustworthy"] else None)
    return {
        "n_source_rows": len(rows),
        "n_normalized": len(records),
        "n_rejected": len(rejected),
        "n_unparseable_lines": sum(isinstance(r, dict) and "__malformed_line__" in r
                                   for r in rows),
        "field_profile": _field_profile(rows),
        "id_integrity": integrity,
        "id_field": opts.id_field,
        "teacher_source_counts": teacher,
        "answers_count_distribution": dict(sorted(Counter(r.call_count for r in records).items())),
        "tools_count_distribution": dict(sorted(Counter(len(r.tools) for r in records).items())),
        "n_with_multiple_available_tools": sum(len(r.tools) > 1 for r in records),
        "n_no_tool_call_label": sum(r.call_count == 0 for r in records),
        "query_chars": _stats([len(r.query) for r in records]),
        "query_ws_tokens": _stats([len(r.query.split()) for r in records]),
        "tools_json_chars": _stats([len(canonical_json(r.tools)) for r in records]),
        "rejections_by_reason": dict(Counter(x.reason for x in rejected)),
        "rejection_examples": [
            {"row": x.source_row_index, "id": x.original_id, "reason": x.reason,
             "detail": _trunc(x.detail, 100)} for x in rejected[:10]
        ],
        "validation_flag_counts": dict(flag_counts.most_common()),
        "conversion_note_counts": dict(note_counts.most_common()),
        "records": records,
        "rejected": rejected,
    }


def render_markdown(ins: dict[str, Any], source_label: str, revision: str,
                    is_fixture: bool, n_examples: int = 10) -> str:
    L: list[str] = []
    title = "Fixture Inspection" if is_fixture else "Dataset Inspection"
    L.append(f"# {title}: {source_label}\n")
    if is_fixture:
        L.append("> **FIXTURE — SYNTHETIC STRUCTURAL TEST DATA, NOT REAL xLAM RECORDS.** "
                 "Counts below describe the fixture only and are not research findings.\n")
    L.append(f"- Source: `{source_label}`  \n- Revision: `{revision}`")
    L.append(f"- Source rows: {ins['n_source_rows']} | normalized: {ins['n_normalized']} | "
             f"rejected: {ins['n_rejected']} | unparseable lines: "
             f"{ins['n_unparseable_lines']}\n")
    L.append("## Field profile (parseable rows)\n\n| field | present | missing | nulls | python types | "
             "JSON-encoded strings |\n|---|---:|---:|---:|---|---:|")
    for k, p in sorted(ins["field_profile"].items()):
        L.append(f"| `{k}` | {p['present']} | {p['missing']} | {p['nulls']} | {p['types']} | "
                 f"{p['json_encoded_strings']} |")
    idi = ins["id_integrity"]
    L.append(f"\n## IDs (field `{ins['id_field']}`)\n")
    L.append(f"- missing: {idi['n_missing']}, unique: {idi['n_unique']}, duplicate ids: "
             f"{idi['n_duplicate_ids']} {idi['duplicate_ids'][:10]}")
    L.append(f"- min/max: {idi['min_id']} / {idi['max_id']}, gaps in range: "
             f"{idi.get('n_gaps_in_range')}")
    L.append(f"- trustworthy for provenance: **{idi['id_field_trustworthy']}**\n")
    L.append("## Teacher source (from original IDs only)\n")
    if ins["teacher_source_counts"] is None:
        L.append("NOT COMPUTED — ID field not verified as present, integer and unique.\n")
    else:
        L.append(f"{ins['teacher_source_counts']}\n")
    L.append("## Calls and tools\n")
    L.append(f"- expected-call count distribution: {ins['answers_count_distribution']}")
    L.append(f"- available-tools count distribution: {ins['tools_count_distribution']}")
    L.append(f"- rows with >1 available tool: {ins['n_with_multiple_available_tools']}")
    L.append(f"- rows with empty answers (potential no-tool): {ins['n_no_tool_call_label']}\n")
    L.append("## Lengths (approximate; NOT Qwen tokens)\n")
    L.append(f"- query chars: {ins['query_chars']}\n- query whitespace tokens: "
             f"{ins['query_ws_tokens']}\n- tools JSON chars: {ins['tools_json_chars']}\n")
    L.append("## Malformed / rejected rows\n")
    L.append(f"- by reason: {ins['rejections_by_reason']}")
    for ex in ins["rejection_examples"]:
        L.append(f"  - row {ex['row']} id={ex['id']}: `{ex['reason']}` {ex['detail']}")
    L.append("\n## Validation flags (occurrences per flag kind)\n")
    for k, v in ins["validation_flag_counts"].items():
        L.append(f"- `{k}`: {v}")
    L.append("\n## Conversion notes\n")
    for k, v in ins["conversion_note_counts"].items():
        L.append(f"- `{k}`: {v}")
    L.append("\n## Task domain\n\nNot inferred: no reliable domain classifier is used "
             "(spec E2 forbids faking one).\n")
    L.append(f"## First {n_examples} normalized records (truncated)\n")
    for r in ins["records"][:n_examples]:
        L.append(f"### `{r.sample_id}` teacher=`{r.teacher_source}` calls={r.call_count}\n")
        L.append(f"- query: {_trunc(r.query)}")
        L.append(f"- tools: {[t['function']['name'] for t in r.tools]}")
        calls = [{"name": c.name, "arguments": c.arguments} for c in r.expected_calls]
        L.append(f"- expected_calls: `{_trunc(canonical_json(calls), 240)}`")
        L.append(f"- flags: {r.flags or '[]'}\n")
    return "\n".join(L) + "\n"


def record_preview(r: NormalizedRecord) -> dict[str, Any]:
    """Compact dict for terminal display."""
    return {
        "sample_id": r.sample_id,
        "original_id": r.original_id,
        "teacher_source": r.teacher_source,
        "query": _trunc(r.query, 100),
        "tools": {t["function"]["name"]: {
            "required": t["function"]["parameters"]["required"],
            "properties": {k: v.get("type", v.get("anyOf", v.get("enum", "UNCONSTRAINED")))
                           for k, v in t["function"]["parameters"]["properties"].items()},
        } for t in r.tools},
        "expected_calls": [{"name": c.name, "arguments": c.arguments} for c in r.expected_calls],
        "flags": r.flags,
    }


def rejected_preview(x: RejectedRecord) -> dict[str, Any]:
    return {"row": x.source_row_index, "original_id": x.original_id, "reason": x.reason,
            "detail": _trunc(x.detail, 100)}
