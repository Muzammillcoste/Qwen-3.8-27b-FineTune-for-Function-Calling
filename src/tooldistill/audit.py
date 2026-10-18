"""Phase 1 data audit: builds a factual report from prepare_dataset outputs.

Reads ``<normalized_dir>/{normalized.jsonl, rejected.jsonl, *_report.json, splits/}`` plus the raw
metadata, and writes ``reports/phase1_audit.md`` and ``data/audits/phase1_audit.json``.
No selection, no quality score: this report only describes the data.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import iter_jsonl, resolve
from .manifests import code_commit
from .schemas import NormalizedRecord
from .split import SPLITS, call_count_bucket
from .validate import LABEL_STATUSES, flag_category, flag_kind

EXAMPLES_PER_KIND = 3


def _pct(n: int, d: int) -> str:
    return f"{n} ({100 * n / d:.2f}%)" if d else f"{n}"


def _dist(xs: list[int]) -> dict[str, Any]:
    if not xs:
        return {}
    s = sorted(xs)

    def q(p: float) -> int:
        return s[min(len(s) - 1, int(p * (len(s) - 1)))]

    return {"n": len(s), "min": s[0], "p50": q(0.5), "p90": q(0.9), "p95": q(0.95),
            "p99": q(0.99), "max": s[-1], "mean": round(statistics.fmean(s), 1)}


def _trunc(s: str, n: int = 140) -> str:
    s = s.replace("\n", " ").replace("|", "\\|")
    return s if len(s) <= n else s[: n - 3] + "..."


def load_records(path: Path) -> list[NormalizedRecord]:
    return [NormalizedRecord.from_dict(d) for d in iter_jsonl(path)]


def build_audit(norm_dir: str | Path, raw_meta: dict[str, Any] | None,
                token_rows: list[dict[str, Any]] | None, token_method: str) -> dict[str, Any]:
    nd = resolve(norm_dir)
    records = load_records(nd / "normalized.jsonl")
    rejected = list(iter_jsonl(nd / "rejected.jsonl"))
    dedup = json.loads((nd / "dedup_report.json").read_text("utf-8"))
    split_summary = json.loads((nd / "split_summary.json").read_text("utf-8"))
    leakage = json.loads((nd / "leakage_audit.json").read_text("utf-8"))
    prov = json.loads((nd / "provenance_check.json").read_text("utf-8"))
    manifest = json.loads((nd / "manifest.json").read_text("utf-8"))
    split_of = {}
    for s in SPLITS:
        for d in iter_jsonl(nd / "splits" / f"{s}.jsonl"):
            split_of[d["sample_id"]] = s

    n = len(records)
    by_id = {r.sample_id: r for r in records}
    teacher = Counter(r.teacher_source for r in records)

    # --- requiredness (parameter level)
    state_counts: Counter = Counter()
    evidence: Counter = Counter()
    conflicts: Counter = Counter()
    conflict_examples: list[str] = []
    rec_with_unknown = 0
    n_params = 0
    for r in records:
        has_unknown = False
        for entry in r.metadata.get("requiredness", []):
            for p, info in entry["params"].items():
                n_params += 1
                state_counts[info["state"]] += 1
                if info["declared"] is not None:
                    evidence["declared:" + "+".join(info["declared_sources"])] += 1
                for reason in info["inferred_optional_reasons"]:
                    evidence["inferred:" + reason] += 1
                if info["declared"] is None and not info["inferred_optional_reasons"]:
                    evidence["no_evidence"] += 1
                for c in info["conflict"]:
                    conflicts[c] += 1
                    if len(conflict_examples) < EXAMPLES_PER_KIND:
                        conflict_examples.append(f"{r.sample_id} {entry['tool']}.{p}: {c}")
                has_unknown |= info["state"] == "UNKNOWN"
        rec_with_unknown += has_unknown

    # --- labels
    status = Counter(r.metadata.get("label_status") for r in records)
    status_by_teacher: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        status_by_teacher[r.teacher_source][r.metadata.get("label_status")] += 1
    flag_kinds: Counter = Counter()
    flag_records: Counter = Counter()
    warn_kinds: Counter = Counter()
    warn_records: Counter = Counter()
    args_invalid_keywords: Counter = Counter()
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    default_pairs: Counter = Counter()
    unknown_types: Counter = Counter()
    for r in records:
        seen_f, seen_w = set(), set()
        for f in r.flags:
            k = flag_kind(f)
            flag_kinds[k] += 1
            if k not in seen_f:
                flag_records[k] += 1
                seen_f.add(k)
            if k == "call_args_invalid":
                for kw in f.rsplit(":", 1)[-1].split(","):
                    args_invalid_keywords[kw] += 1
            if k == "default_type_mismatch":
                parts = f.split(":")
                default_pairs[f"expected {parts[2]} / got {parts[3]}"] += 1
            if k == "unknown_type":
                unknown_types[f.split(":", 2)[-1]] += 1
            if len(examples[k]) < EXAMPLES_PER_KIND and not any(
                    e["sample_id"] == r.sample_id for e in examples[k]):
                examples[k].append({"sample_id": r.sample_id, "detail": f,
                                    "query": _trunc(r.query),
                                    "errors": "; ".join(r.metadata.get("validation_errors",
                                                                       [])[:2])})
        for w in r.metadata.get("validation_warnings", []):
            k = flag_kind(w)
            warn_kinds[k] += 1
            if k not in seen_w:
                warn_records[k] += 1
                seen_w.add(k)
            if len(examples[k]) < EXAMPLES_PER_KIND and not any(
                    e["sample_id"] == r.sample_id for e in examples[k]):
                calls = json.dumps([{"name": c.name, "arguments": c.arguments}
                                    for c in r.expected_calls], ensure_ascii=False)
                examples[k].append({"sample_id": r.sample_id, "detail": w,
                                    "query": _trunc(r.query), "errors": _trunc(calls, 160)})

    # --- structure
    calls_by_teacher = {t: dict(sorted(Counter(r.call_count for r in records
                                               if r.teacher_source == t).items()))
                        for t in sorted(teacher)}
    tools_dist = dict(sorted(Counter(len(r.tools) for r in records).items()))

    # --- duplicates
    group_sizes = Counter(r.source_group_id for r in records)
    size_hist = Counter(min(s, 10) for s in group_sizes.values())
    largest = group_sizes.most_common(3)
    largest_examples = []
    for gid, size in largest:
        members = [r for r in records if r.source_group_id == gid][:4]
        largest_examples.append({"group": gid, "size": size,
                                 "queries": [_trunc(m.query, 100) for m in members]})
    near_examples = []
    by_group = defaultdict(list)
    for r in records:
        by_group[r.source_group_id].append(r)
    for _gid, members in sorted(by_group.items(), key=lambda kv: kv[0]):
        if len(members) == 2 and members[0].exact_dup_key != members[1].exact_dup_key and \
                members[0].query != members[1].query:
            near_examples.append([f"{m.sample_id}: {_trunc(m.query, 110)}" for m in members])
        if len(near_examples) >= EXAMPLES_PER_KIND:
            break
    exact_examples = [f"{r.sample_id} == {r.duplicate_of}: {_trunc(r.query, 100)}"
                      for r in records if r.duplicate_of][:EXAMPLES_PER_KIND]

    # --- tokens
    tok: dict[str, Any] = {"method": token_method}
    if token_rows:
        trows = {t["sample_id"]: t for t in token_rows}
        for seg in ("query", "tools", "calls", "total"):
            tok[seg] = _dist([t[seg] for t in token_rows])
        totals = [t["total"] for t in token_rows]
        tok["over"] = {str(L): _pct(sum(x > L for x in totals), len(totals))
                       for L in (512, 1024, 1536, 2048, 4096)}
        tok["total_by_teacher"] = {
            t: _dist([trows[r.sample_id]["total"] for r in records if r.teacher_source == t])
            for t in sorted(teacher)}
        tok["total_by_status"] = {
            s: _dist([trows[r.sample_id]["total"] for r in records
                      if r.metadata.get("label_status") == s]) for s in sorted(status)}

    # --- splits
    split_table = {}
    for s in SPLITS:
        members = [by_id[sid] for sid, sp in split_of.items() if sp == s]
        split_table[s] = {
            "n": len(members),
            "teacher": dict(Counter(m.teacher_source for m in members)),
            "calls": dict(Counter(call_count_bucket(m.call_count) for m in members)),
            "label_status": dict(Counter(m.metadata.get("label_status") for m in members)),
        }

    return {
        "source": raw_meta,
        "code_commit": code_commit(),
        "prepare_manifest_files": {k: v["sha256"] for k, v in manifest["files"].items()},
        "counts": manifest["counts"],
        "rejected_by_reason": dict(Counter(x["reason"] for x in rejected)),
        "rejected_examples": rejected[:EXAMPLES_PER_KIND * 3],
        "provenance": prov,
        "teacher": dict(teacher),
        "requiredness": {"n_params": n_params, "state": dict(state_counts),
                         "evidence": dict(evidence.most_common()),
                         "conflicts": dict(conflicts), "conflict_examples": conflict_examples,
                         "records_with_unknown_param": rec_with_unknown},
        "label_status": dict(status),
        "strict_valid": sum(bool(r.metadata.get("strict_schema_valid")) for r in records),
        "label_status_by_teacher": {t: dict(c) for t, c in sorted(status_by_teacher.items())},
        "flag_kinds": {k: {"occurrences": v, "records": flag_records[k],
                           "category": flag_category(k)} for k, v in flag_kinds.most_common()},
        "warning_kinds": {k: {"occurrences": v, "records": warn_records[k]}
                          for k, v in warn_kinds.most_common()},
        "args_invalid_keywords": dict(args_invalid_keywords.most_common()),
        "default_mismatch_pairs": dict(default_pairs.most_common()),
        "unknown_types": dict(unknown_types.most_common(20)),
        "examples": dict(examples),
        "calls_by_teacher": calls_by_teacher,
        "tools_count_distribution": tools_dist,
        "dedup": dedup,
        "group_size_hist_capped10": dict(sorted(size_hist.items())),
        "largest_groups": largest_examples,
        "near_duplicate_examples": near_examples,
        "exact_duplicate_examples": exact_examples,
        "tokens": tok,
        "splits": split_table,
        "split_summary": split_summary,
        "leakage": leakage,
        "n_records": n,
    }


def render_audit_md(a: dict[str, Any]) -> str:
    n = a["n_records"]
    L: list[str] = ["# Phase 1 Data Audit — Salesforce xLAM Function Calling 60K\n"]
    L.append("Factual description of the data only. No selection, no quality score, no training. "
             "Schema/label checks are syntactic; they do not establish semantic correctness.\n")
    src = a["source"] or {}
    L.append("## 1. Source and reproducibility\n")
    L.append("| item | value |\n|---|---|")
    for k in ("repo_id", "revision", "file", "row_count", "raw_sha256", "downloaded_utc",
              "license"):
        L.append(f"| {k} | `{src.get(k)}` |")
    L.append(f"| code commit | `{a['code_commit']}` |")
    L.append(f"| normalized.jsonl sha256 | `{a['prepare_manifest_files'].get('normalized')}` |\n")

    c = a["counts"]
    L.append("## 2. Totals and parse failures\n")
    L.append(f"- source rows: **{c['source_rows']}**; normalized: **{c['normalized']}**; "
             f"rejected (parse failures): **{c['rejected']}**")
    L.append(f"- rejected by reason: {a['rejected_by_reason'] or 'none'}")
    for x in a["rejected_examples"]:
        L.append(f"  - row {x['source_row_index']} id={x['original_id']}: `{x['reason']}` "
                 f"{_trunc(x['detail'], 100)}")
    L.append(f"- exact duplicates excluded from eligible pool: {c['exact_duplicates_excluded']}; "
             f"eligible: {c['eligible']}\n")

    p = a["provenance"]
    L.append("## 3. IDs and teacher source (DeepSeek vs Mixtral)\n")
    idi = p["id_integrity"]
    L.append(f"- id field trustworthy: **{idi['id_field_trustworthy']}** (missing {idi['n_missing']},"
             f" duplicates {idi['n_duplicate_ids']}, min {idi['min_id']}, max {idi['max_id']}, "
             f"gaps {idi.get('n_gaps_in_range')})")
    L.append(f"- card check computed: {p['computed']}, **match: {p['ok']}** — {p['message']}")
    L.append("\n| teacher | records | share |\n|---|---:|---:|")
    for t, v in sorted(a["teacher"].items()):
        L.append(f"| {t} | {v} | {100 * v / n:.2f}% |")
    L.append("")

    rq = a["requiredness"]
    L.append("## 4. Requiredness (parameter level, three-state)\n")
    L.append(f"- parameters: {rq['n_params']}; records with ≥1 UNKNOWN parameter: "
             f"{_pct(rq['records_with_unknown_param'], n)}")
    L.append("\n| state | params | share |\n|---|---:|---:|")
    for s in ("REQUIRED", "OPTIONAL", "UNKNOWN"):
        v = rq["state"].get(s, 0)
        L.append(f"| {s} | {v} | {100 * v / max(rq['n_params'], 1):.2f}% |")
    L.append("\nEvidence (a parameter can have several inferred reasons):\n")
    L.append("| evidence | params |\n|---|---:|")
    for k, v in rq["evidence"].items():
        L.append(f"| {k} | {v} |")
    L.append(f"\n- contradictory metadata: {rq['conflicts'] or 'none found'}")
    for e in rq["conflict_examples"]:
        L.append(f"  - {e}")
    L.append("")

    L.append("## 5. Label validity: confirmed invalid vs schema conflict vs ambiguous\n")
    L.append("`invalid` = confirmed structural label error (unknown tool, spurious argument, "
             "missing DECLARED-required argument, enum violation). `schema_conflict` = only "
             "declared-type vs supplied-value disagreements (`schema_argument_type_conflict`, "
             "e.g. a decimal in an `int` field): strictly schema-invalid, but NOT a confirmed "
             "semantic label error. `ambiguous` = warnings only. Records are kept either way.\n")
    L.append(f"- strict JSON-Schema valid: {_pct(a['strict_valid'], n)}\n")
    L.append("| label_status | records | share |\n|---|---:|---:|")
    for s in LABEL_STATUSES:
        L.append(f"| {s} | {a['label_status'].get(s, 0)} | "
                 f"{100 * a['label_status'].get(s, 0) / n:.2f}% |")
    L.append("\nBy teacher:\n\n| teacher | " + " | ".join(LABEL_STATUSES) + " |\n|---|"
             + "---:|" * len(LABEL_STATUSES))
    for t, cnt in a["label_status_by_teacher"].items():
        tot = sum(cnt.values())
        L.append(f"| {t} | " + " | ".join(_pct(cnt.get(s, 0), tot)
                                          for s in LABEL_STATUSES)
                 + " |")
    L.append(f"\n- `call_args_invalid` by JSON-Schema keyword: {a['args_invalid_keywords']}")
    L.append("\nWarnings (ambiguous channel):\n\n| warning | occurrences | records |\n|---|---:|---:|")
    for k, v in a["warning_kinds"].items():
        L.append(f"| {k} | {v['occurrences']} | {v['records']} |")
    L.append("")

    L.append("## 6. All flags by category\n")
    L.append("| flag | category | occurrences | records |\n|---|---|---:|---:|")
    for k, v in a["flag_kinds"].items():
        L.append(f"| {k} | {v['category']} | {v['occurrences']} | {v['records']} |")
    L.append("\n### Default-type mismatches (schema issue; separate from label correctness)\n")
    L.append("| expected / got | params |\n|---|---:|")
    for k, v in a["default_mismatch_pairs"].items():
        L.append(f"| {k} | {v} |")
    if a["unknown_types"]:
        L.append(f"\n- unknown type strings (top 20): {a['unknown_types']}")
    L.append("")

    L.append("## 7. Structure\n")
    L.append(f"- expected-call count by teacher: {a['calls_by_teacher']}")
    L.append(f"- available-tools count: {a['tools_count_distribution']}\n")

    d = a["dedup"]
    L.append("## 8. Duplicates and near-duplicates\n")
    L.append(f"- algorithm `{d['algorithm_version']}`, params {d['params']}")
    L.append(f"- exact duplicates: **{d['n_exact_duplicates']}**; groups: {d['n_groups']}; "
             f"largest group: {d['largest_group']}")
    L.append(f"- union edges by type: {d['edges']}")
    L.append(f"- group size histogram (10 = ≥10): {a['group_size_hist_capped10']}")
    L.append(f"- input sets with conflicting labels: {len(d['conflicting_label_sets'])}")
    L.append("- exact-duplicate examples:")
    for e in a["exact_duplicate_examples"]:
        L.append(f"  - {e}")
    L.append("- near-duplicate (2-member group) examples:")
    for pair in a["near_duplicate_examples"]:
        L.append(f"  - {pair[0]}\n    {pair[1]}")
    L.append("- largest groups:")
    for g in a["largest_groups"]:
        L.append(f"  - {g['group']} size {g['size']}: {g['queries']}")
    L.append("")

    t = a["tokens"]
    L.append("## 9. Token lengths\n")
    L.append(f"Method: `{t['method']}`. Raw segments only — chat-template overhead NOT included; "
             "totals are lower bounds of rendered length.\n")
    if "total" in t:
        L.append("| segment | min | p50 | p90 | p95 | p99 | max | mean |\n|---|---:|---:|---:|---:|"
                 "---:|---:|---:|")
        for seg in ("query", "tools", "calls", "total"):
            s = t[seg]
            L.append(f"| {seg} | {s['min']} | {s['p50']} | {s['p90']} | {s['p95']} | {s['p99']} | "
                     f"{s['max']} | {s['mean']} |")
        L.append(f"\n- records with total above N tokens: {t['over']}")
        L.append(f"- total by teacher: {t['total_by_teacher']}")
        L.append(f"- total by label_status: {t['total_by_status']}\n")

    L.append("## 10. Splits (group-aware, seed 42) and leakage audit\n")
    L.append("| split | n | teacher | calls | label_status |\n|---|---:|---|---|---|")
    for s, v in a["splits"].items():
        L.append(f"| {s} | {v['n']} | {v['teacher']} | {v['calls']} | {v['label_status']} |")
    L.append(f"\n- leakage audit ok: **{a['leakage']['ok']}** {a['leakage']['violations'] or ''}\n")

    L.append("## 11. Examples for every warning/error category\n")
    for k in sorted(a["examples"]):
        L.append(f"### `{k}` ({flag_category(k)})\n")
        for e in a["examples"][k]:
            L.append(f"- `{e['sample_id']}` — `{_trunc(e['detail'], 120)}`  \n  query: "
                     f"{e['query']}" + (f"  \n  {e['errors']}" if e.get("errors") else ""))
        L.append("")
    return "\n".join(L) + "\n"


def write_audit(a: dict[str, Any], md_path: str | Path, json_path: str | Path) -> None:
    mp, jp = resolve(md_path), resolve(json_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    jp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(render_audit_md(a), "utf-8")
    jp.write_text(json.dumps(a, indent=2, default=str, ensure_ascii=False) + "\n", "utf-8")
