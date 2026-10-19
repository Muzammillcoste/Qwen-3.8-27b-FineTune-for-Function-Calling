"""Phase 3B: frozen evaluation sets, arm comparability checks, coverage categories, Kaggle bundle.

- ``val_eval_400``: fixed seeded validation subset for the initial arm comparison. Disjoint from
  the 100-example Phase 3A development sample (which was inspected to design metrics).
- ``test_eval_1000``: fixed seeded TEST subset. SEALED: written locally only, never bundled for
  Kaggle, and evaluation is refused while ``configs/eval.yaml: test.sealed`` is true.
Both: single-call, non-duplicate, rendered <= max_rendered_tokens (same exclusion as training),
ONE record per dedup group (no correlated near-duplicates inflating N), group-disjoint from
TRAIN (asserted), stratified with the shared select-v2 strata.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .format_qwen import build_training_example
from .io import REPO_ROOT, iter_jsonl, resolve, sha256_file, sha256_text, write_jsonl
from .manifests import code_commit
from .phase2 import load_split
from .schemas import NormalizedRecord
from .select import record_features, stratified_sample

EVALSET_VERSION = "evalset-v1"
COVERAGE = ("seen_by_both", "seen_by_random_1k_only", "seen_by_diversity_1k_only", "seen_by_neither")


def one_per_group(records: list[NormalizedRecord], seed: int, salt: str) -> list[NormalizedRecord]:
    groups: dict[str, list[NormalizedRecord]] = {}
    for r in sorted(records, key=lambda r: r.sample_id):
        groups.setdefault(r.source_group_id, []).append(r)
    rng = random.Random(f"{EVALSET_VERSION}|{seed}|{salt}")
    return [rng.choice(g) for _, g in sorted(groups.items())]


def eligible_heldout(nd: Path, split: str, lengths: dict[str, int], max_tok: int,
                     exclude: set[str]) -> list[NormalizedRecord]:
    return [r for r in load_split(nd, split)
            if r.call_count == 1 and r.duplicate_of is None and r.sample_id not in exclude
            and lengths[r.sample_id] <= max_tok]


def build_eval_sets(data_cfg: dict[str, Any], research_cfg: dict[str, Any], seed: int,
                    dev_ids: set[str], n_val: int = 400, n_test: int = 1000):
    nd = resolve(data_cfg["paths"]["normalized_dir"])
    lengths = {d["sample_id"]: d["total"]
               for d in iter_jsonl(resolve(data_cfg["paths"]["audits_dir"]) / "rendered_lengths.jsonl")}
    max_tok = research_cfg["primary_pool"]["max_rendered_tokens"]
    train_groups = {r.source_group_id for r in load_split(nd, "train")}
    out = {}
    for name, split, n, excl in (("val_eval_400", "val", n_val, dev_ids),
                                 ("test_eval_1000", "test", n_test, set())):
        cands = one_per_group(eligible_heldout(nd, split, lengths, max_tok, excl), seed, name)
        feats = {r.sample_id: record_features(r, lengths[r.sample_id]) for r in cands}
        ids = stratified_sample(feats, n, seed, name)
        by = {r.sample_id: r for r in cands}
        recs = [by[i] for i in ids]
        assert not {r.source_group_id for r in recs} & train_groups, f"{name}: group overlap"
        assert len({r.source_group_id for r in recs}) == len(recs)
        out[name] = recs
    assert not ({r.source_group_id for r in out["val_eval_400"]}
                & {r.source_group_id for r in out["test_eval_1000"]})
    return out, lengths


def arm_functions(path: Path) -> set[str]:
    return {d["expected_calls"][0]["name"] for d in iter_jsonl(path)}


def coverage_category(fn: str, random_fns: set[str], diversity_fns: set[str]) -> str:
    a, b = fn in random_fns, fn in diversity_fns
    return COVERAGE[0] if a and b else COVERAGE[1] if a else COVERAGE[2] if b else COVERAGE[3]


def arm_comparability(tok, arms: dict[str, list[NormalizedRecord]], max_len: int,
                      eff_batch: int) -> dict[str, Any]:
    """Measured with the real tokenizer + the exact training example builder."""
    rows = {}
    for name, recs in arms.items():
        ex = [build_training_example(r, tok, max_len) for r in recs]
        teacher = Counter(r.teacher_source for r in recs)
        rows[name] = {
            "n": len(recs), "optimizer_steps_1_epoch": -(-len(recs) // eff_batch),
            "total_tokens": sum(len(e["input_ids"]) for e in ex),
            "supervised_tokens": sum(e["n_target"] for e in ex),
            "max_tokens": max(len(e["input_ids"]) for e in ex),
            "deepseek_share": round(teacher["deepseek-v2-chat"] / len(recs), 4),
            "distinct_functions": len({r.expected_calls[0].name for r in recs}),
            "mean_tools": round(sum(len(r.tools) for r in recs) / len(recs), 3),
            "mean_args": round(sum(len(r.expected_calls[0].arguments) for r in recs) / len(recs), 3),
        }
    a, b = rows["random_1k"], rows["diversity_1k"]
    rel = lambda k: abs(a[k] - b[k]) / a[k]  # noqa: E731
    checks = {
        "same_optimizer_steps": a["optimizer_steps_1_epoch"] == b["optimizer_steps_1_epoch"],
        "total_tokens_within_5pct": rel("total_tokens") <= 0.05,
        "supervised_tokens_within_5pct": rel("supervised_tokens") <= 0.05,
        "teacher_share_within_0.02": abs(a["deepseek_share"] - b["deepseek_share"]) <= 0.02,
        "no_record_over_max_len": max(a["max_tokens"], b["max_tokens"]) <= max_len,
    }
    return {"arms": rows, "checks": checks, "all_pass": all(checks.values()),
            "relative_diff": {k: round(rel(k), 4) for k in ("total_tokens", "supervised_tokens")}}


def export_phase3b(data_cfg, research_cfg, train_cfg, seed: int = 42,
                   bundle_dir: str = "data/kaggle_upload") -> dict[str, Any]:
    from transformers import AutoTokenizer

    from .kaggle_export import zip_source

    bundle = resolve(bundle_dir) / "data"
    m3a = json.loads((bundle / "manifest.json").read_text("utf-8"))
    dev_ids = set(m3a["val_sample"]["sample_ids"])
    sets, lengths = build_eval_sets(data_cfg, research_cfg, seed, dev_ids)

    sel = resolve(data_cfg["paths"]["selected_dir"])
    arm_recs = {a: [NormalizedRecord.from_dict(d) for d in iter_jsonl(sel / f"train_{a}.jsonl")]
                for a in ("random_1k", "diversity_1k")}
    for a, recs in arm_recs.items():  # manifests from Phase 2 must still match the files
        man = json.loads((REPO_ROOT / f"reports/phase2_manifests/{a}.json").read_text("utf-8"))
        assert sorted(r.sample_id for r in recs) == man["sample_ids"], f"{a}: manifest drift"
    fn_r = {r.expected_calls[0].name for r in arm_recs["random_1k"]}
    fn_d = {r.expected_calls[0].name for r in arm_recs["diversity_1k"]}

    t = data_cfg["tokenizer"]
    tok = AutoTokenizer.from_pretrained(t["repo_id"], revision=t["revision"])
    tr = train_cfg["training"]
    comp = arm_comparability(tok, arm_recs, tr["max_seq_length"],
                             tr["per_device_train_batch_size"] * tr["gradient_accumulation_steps"])

    man_dir = REPO_ROOT / "reports" / "eval_manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir = resolve("data/eval_sealed")
    files = {}
    for name, recs in sets.items():
        target = (sealed_dir if name.startswith("test") else bundle) / f"{name}.jsonl"
        write_jsonl(target, (r.to_dict() for r in recs))
        cov = {r.sample_id: coverage_category(r.expected_calls[0].name, fn_r, fn_d) for r in recs}
        ids = [r.sample_id for r in recs]
        manifest = {
            "evalset_version": EVALSET_VERSION, "name": name, "seed": seed, "n": len(recs),
            "split": recs[0].split, "sample_ids": ids,
            "sample_ids_sha256": sha256_text("\n".join(sorted(ids))),
            "file_sha256": sha256_file(target),
            "sealed": name.startswith("test"),
            "rules": "single-call, non-duplicate, rendered<=1536, one per dedup group, "
                     "group-disjoint from TRAIN, stratified select-v2 strata"
                     + ("; excludes the 100 Phase 3A dev examples" if name.startswith("val") else ""),
            "teacher": dict(Counter(r.teacher_source for r in recs)),
            "label_status": dict(Counter(r.metadata.get("label_status") for r in recs)),
            "rendered_tokens_sum": sum(lengths[i] for i in ids),
            "coverage_counts": dict(Counter(cov.values())),
            "coverage_by_id": cov,
            "code_commit": code_commit(),
        }
        (man_dir / f"{name}.json").write_text(json.dumps(manifest, indent=1) + "\n", "utf-8")
        files[name] = {k: manifest[k] for k in ("n", "sample_ids_sha256", "file_sha256",
                                                 "coverage_counts", "teacher", "label_status")}
        files[name]["path"] = str(target)

    for a, recs in arm_recs.items():  # arm training files go into the Kaggle bundle
        write_jsonl(bundle / f"train_{a}.jsonl", (r.to_dict() for r in recs))
    m3a["phase3b"] = {
        "val_eval_400": {"file": "val_eval_400.jsonl",
                         "sha256": sha256_file(bundle / "val_eval_400.jsonl")},
        "arms": {a: {"file": f"train_{a}.jsonl", "sha256": sha256_file(bundle / f"train_{a}.jsonl"),
                     "sample_ids_sha256": json.loads(
                         (REPO_ROOT / f"reports/phase2_manifests/{a}.json").read_text("utf-8"))
                     ["sample_ids_sha256"]} for a in arm_recs},
        "test_eval_1000": "SEALED - not in this bundle",
    }
    (bundle / "manifest.json").write_text(json.dumps(m3a, indent=1) + "\n", "utf-8")
    (REPO_ROOT / "reports" / "phase3b_arm_comparability.json").write_text(
        json.dumps(comp, indent=1) + "\n", "utf-8")
    n_src = zip_source(resolve(bundle_dir) / "project" / "project_source.zip")
    return {"eval_sets": files, "arm_comparability": comp, "source_files_zipped": n_src}
