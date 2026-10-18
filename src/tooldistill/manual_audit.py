"""Blinded semantic-quality audit sample (spec F2). Labels are entered by HUMANS only.

Produces two files:
- ``annotation_sheet.csv``: what annotators see. Random audit ids, shuffled order, NO arm names,
  NO validation flags, NO teacher source.
- ``audit_key.csv``: hidden mapping audit_id -> sample_id, arms, stratum, label_status. Keep it
  away from annotators until labelling is finished.

Design: ``per_arm`` records per 1k arm, proportional to teacher within the arm, plus a blinded
``diagnostic`` stratum of schema-conflict / ambiguous pool records so the audit can test whether
the schema filter tracks semantic correctness at all. A record drawn for several arms is shown once.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import resolve, sha256_text
from .schemas import NormalizedRecord

AUDIT_VERSION = "semantic-audit-v1"
ANNOTATION_COLUMNS = [
    "audit_id", "query", "available_tools", "expected_call",
    "semantic_correct",     # Y / N / U  (U = cannot judge)
    "error_type",           # wrong_tool | wrong_arg_value | missing_needed_arg | extra_arg |
                            # unfaithful_to_query | query_ambiguous | other | (blank if Y)
    "severity",             # minor | major  (blank if Y)
    "notes", "annotator_id",
]


def _tool_summary(rec: NormalizedRecord) -> str:
    parts = []
    for t in rec.tools:
        fn = t["function"]
        props = fn["parameters"].get("properties", {})
        sig = ", ".join(f"{k}:{v.get('type', 'any')}" for k, v in props.items())
        parts.append(f"{fn['name']}({sig}) — {fn['description'][:160]}")
    return "\n".join(parts)


def build_audit_sample(arms: dict[str, list[str]], records: dict[str, NormalizedRecord],
                       diagnostic_pool: dict[str, list[str]], seed: int, per_arm: int = 40,
                       diagnostic: dict[str, int] | None = None) -> list[dict[str, Any]]:
    diagnostic = diagnostic or {"schema_conflict": 20, "ambiguous": 10}
    rng = random.Random(f"{AUDIT_VERSION}|{seed}")
    picked: dict[str, dict[str, Any]] = {}

    def add(sid: str, stratum: str, arm: str | None):
        e = picked.setdefault(sid, {"sample_id": sid, "strata": [], "arms": []})
        if stratum not in e["strata"]:
            e["strata"].append(stratum)
        if arm and arm not in e["arms"]:
            e["arms"].append(arm)

    for arm in sorted(arms):
        by_teacher: dict[str, list[str]] = defaultdict(list)
        for sid in sorted(arms[arm]):
            by_teacher[records[sid].teacher_source].append(sid)
        n_arm = len(arms[arm])
        for t in sorted(by_teacher):
            k = round(per_arm * len(by_teacher[t]) / n_arm)
            for sid in rng.sample(by_teacher[t], min(k, len(by_teacher[t]))):
                add(sid, f"arm:{arm}", arm)
    for kind, k in sorted(diagnostic.items()):
        cands = sorted(diagnostic_pool.get(kind, []))
        for sid in rng.sample(cands, min(k, len(cands))):
            add(sid, f"diagnostic:{kind}", None)
    # membership of every picked record in every arm (not only the arm it was drawn for)
    for e in picked.values():
        e["arms"] = sorted(a for a in arms if e["sample_id"] in set(arms[a]))
    items = sorted(picked.values(), key=lambda e: e["sample_id"])
    rng.shuffle(items)
    for e in items:
        e["audit_id"] = "A" + sha256_text(f"{AUDIT_VERSION}|{seed}|{e['sample_id']}")[:10]
    return items


def write_audit_files(items: list[dict[str, Any]], records: dict[str, NormalizedRecord],
                      out_dir: str | Path) -> dict[str, Path]:
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sheet, key = out / "annotation_sheet.csv", out / "audit_key.csv"
    with open(sheet, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS)
        w.writeheader()
        for e in items:
            r = records[e["sample_id"]]
            c = r.expected_calls[0] if len(r.expected_calls) == 1 else None
            call = ({"name": c.name, "arguments": c.arguments} if c else
                    [{"name": x.name, "arguments": x.arguments} for x in r.expected_calls])
            w.writerow({"audit_id": e["audit_id"], "query": r.query,
                        "available_tools": _tool_summary(r),
                        "expected_call": json.dumps(call, ensure_ascii=False)})
    with open(key, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["audit_id", "sample_id", "arms", "strata", "teacher",
                                          "label_status", "strict_schema_valid"])
        w.writeheader()
        for e in items:
            r = records[e["sample_id"]]
            w.writerow({"audit_id": e["audit_id"], "sample_id": e["sample_id"],
                        "arms": ";".join(e["arms"]), "strata": ";".join(e["strata"]),
                        "teacher": r.teacher_source,
                        "label_status": r.metadata.get("label_status"),
                        "strict_schema_valid": r.metadata.get("strict_schema_valid")})
    return {"sheet": sheet, "key": key}
