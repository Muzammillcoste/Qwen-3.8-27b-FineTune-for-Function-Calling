"""Compare runs on one fixed eval set (eval-v2). Rescores RAW outputs locally, so every run is
scored by the same frozen metric code regardless of when it ran. Writes aggregates only.

Primary contrast: diversity_1k vs random_1k (paired, same examples). Also each arm vs base.
Coverage breakdown: eval items whose function is seen by both arms / one arm / neither.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluate import EVAL_VERSION, aggregate, score_example
from .io import iter_jsonl
from .schemas import NormalizedRecord
from .stats import paired_report

METRICS = ("exact_match", "schema_aware_match")
CONTRASTS = (("random_1k", "diversity_1k"), ("base", "random_1k"), ("base", "diversity_1k"))


def load_run(pred_path: Path, records: dict[str, NormalizedRecord]) -> dict[str, Any]:
    preds = {d["sample_id"]: d["raw_output"] for d in iter_jsonl(pred_path)}
    missing = set(records) - set(preds)
    if missing:
        raise ValueError(f"{pred_path.name}: {len(missing)} eval items missing")
    return {sid: score_example(records[sid], preds[sid]) for sid in sorted(records)}


def compare(runs: dict[str, Path], evalset: Path, manifest: Path) -> dict[str, Any]:
    records = {d["sample_id"]: NormalizedRecord.from_dict(d) for d in iter_jsonl(evalset)}
    man = json.loads(manifest.read_text("utf-8"))
    if sorted(records) != sorted(man["sample_ids"]):
        raise ValueError("eval set does not match its manifest")
    cov = man["coverage_by_id"]
    scored = {name: load_run(p, records) for name, p in runs.items()}
    ids = sorted(records)
    out: dict[str, Any] = {"eval_version": EVAL_VERSION, "evalset": man["name"], "n": len(ids),
                           "aggregate": {k: aggregate(list(v.values())) for k, v in scored.items()},
                           "contrasts": {}, "coverage": {}}
    for a, b in CONTRASTS:
        if a in scored and b in scored:
            out["contrasts"][f"{b} - {a}"] = {
                m: paired_report([getattr(scored[a][i], m) for i in ids],
                                 [getattr(scored[b][i], m) for i in ids]) for m in METRICS}
    for cat in sorted(set(cov.values())):
        cid = [i for i in ids if cov[i] == cat]
        row: dict[str, Any] = {"n": len(cid)}
        for name, s in scored.items():
            row[name] = round(sum(s[i].exact_match for i in cid) / len(cid), 4)
        if "random_1k" in scored and "diversity_1k" in scored and len(cid) >= 1:
            row["diversity_minus_random"] = paired_report(
                [scored["random_1k"][i].exact_match for i in cid],
                [scored["diversity_1k"][i].exact_match for i in cid])
        out["coverage"][cat] = row
    return out


def render_md(r: dict[str, Any]) -> str:
    L = [f"# Run comparison — {r['evalset']} (n={r['n']}, {r['eval_version']})\n",
         "Aggregates only. Decision test: exact McNemar; bootstrap CI descriptive "
         "(unreliable with few discordant pairs). One training seed: CIs reflect eval-sample "
         "uncertainty only.\n",
         "| run | strict exact | schema-aware | tool name | arg value | parse |",
         "|---|---|---|---|---|---|"]
    def f(a: dict[str, Any], k: str) -> str:
        return f"{a[k]['k']}/{a[k]['n']} ({a[k]['rate']})"

    for name, a in r["aggregate"].items():
        L.append(f"| {name} | {f(a, 'exact_structured_call_accuracy')} | "
                 f"{f(a, 'schema_aware_call_accuracy')} | {f(a, 'tool_name_accuracy')} | "
                 f"{f(a, 'argument_value_accuracy')} | {f(a, 'parse_rate')} |")
    L.append("\n| contrast | metric | diff | 95% bootstrap CI | discordant | McNemar p |")
    L.append("|---|---|---|---|---|---|")
    for c, ms in r["contrasts"].items():
        for m, rep in ms.items():
            b, mc = rep["bootstrap"], rep["mcnemar"]
            L.append(f"| {c} | {m} | {b['diff']:+.4f} | [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] "
                     f"| {mc['discordant']} ({mc['only_a']}/{mc['only_b']}) | {mc['p_value']} |")
    L.append("\n| coverage category | n | " + " | ".join(
        k for k in next(iter(r["coverage"].values())) if k not in ("n", "diversity_minus_random"))
        + " | diversity − random (p) |")
    L.append("|---" * (3 + len(r["aggregate"])) + "|")
    for cat, row in r["coverage"].items():
        accs = [str(v) for k, v in row.items() if k not in ("n", "diversity_minus_random")]
        dm = row.get("diversity_minus_random")
        d = f"{dm['bootstrap']['diff']:+.3f} (p={dm['mcnemar']['p_value']})" if dm else "—"
        L.append(f"| {cat} | {row['n']} | " + " | ".join(accs) + f" | {d} |")
    return "\n".join(L) + "\n"
