"""Phase 3A Kaggle bundle: tiny pilot data + project source (spec J2, scaled to Phase 3A).

Outputs (under ``data/kaggle_upload/``, gitignored):
  data/   -> upload as a PRIVATE Kaggle dataset (xLAM-derived, gated source)
     pilot_train_100.jsonl   stratified random 100 from the eligible TRAIN pool (not an arm)
     val_sample_100.jsonl    fixed seeded 100 from single-call VALIDATION (labels used for
                             scoring only; never for training or selection)
     manifest.json, README_DATA.md
  project/project_source.zip -> upload as a private Kaggle dataset (code only, no data)
Also writes ``data/audits/val_sample_label_check.csv`` for an independent human label check.
"""

from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from .io import REPO_ROOT, iter_jsonl, resolve, sha256_file, write_jsonl
from .manifests import LICENSE_ACKNOWLEDGEMENT, code_commit
from .phase2 import load_split
from .schemas import NormalizedRecord
from .select import build_pool, record_features, stratified_sample

EXPORT_VERSION = "kaggle-phase3a-v1"
SOURCE_INCLUDE = ["src", "configs", "scripts", "notebooks", "tests", "reports/phase2_manifests",
                  "pyproject.toml",
                  "requirements-kaggle.txt", "README.md", "LICENSE_NOTICES.md"]
SOURCE_EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", "data", ".venv",
                        ".ipynb_checkpoints", "tooldistill.egg-info"}


def _lengths(path: Path) -> dict[str, int]:
    return {d["sample_id"]: d["total"] for d in iter_jsonl(path)}


def build_samples(data_cfg: dict[str, Any], research_cfg: dict[str, Any], seed: int,
                  n_pilot: int = 100, n_val: int = 100, n_format: int = 20):
    nd = resolve(data_cfg["paths"]["normalized_dir"])
    lengths = _lengths(resolve(data_cfg["paths"]["audits_dir"]) / "rendered_lengths.jsonl")
    max_tok = research_cfg["primary_pool"]["max_rendered_tokens"]

    pool = [r for r in build_pool(load_split(nd, "train")) if lengths[r.sample_id] <= max_tok]
    pf = {r.sample_id: record_features(r, lengths[r.sample_id]) for r in pool}
    pilot_ids = stratified_sample(pf, n_pilot, seed, "pilot")

    val = [r for r in load_split(nd, "val")
           if r.call_count == 1 and r.duplicate_of is None and lengths[r.sample_id] <= max_tok]
    vf = {r.sample_id: record_features(r, lengths[r.sample_id]) for r in val}
    val_ids = stratified_sample(vf, n_val, seed, "val_sample")
    by_id = {r.sample_id: r for r in pool + val}
    return [by_id[i] for i in pilot_ids], [by_id[i] for i in val_ids], val_ids[:n_format]


def _summary(recs: list[NormalizedRecord], lengths: dict[str, int]) -> dict[str, Any]:
    toks = sorted(lengths[r.sample_id] for r in recs)
    return {"n": len(recs),
            "teacher": dict(Counter(r.teacher_source for r in recs)),
            "label_status": dict(Counter(r.metadata.get("label_status") for r in recs)),
            "rendered_tokens_sum": sum(toks), "rendered_tokens_max": toks[-1],
            "rendered_tokens_p50": toks[len(toks) // 2]}


def zip_source(out_zip: Path) -> int:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for item in SOURCE_INCLUDE:
            p = REPO_ROOT / item
            files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
            for f in files:
                rel = f.relative_to(REPO_ROOT)
                excluded_dir = bool(SOURCE_EXCLUDE_PARTS & set(rel.parts))
                data_file = f.suffix == ".jsonl" and "fixtures" not in rel.parts
                if excluded_dir or data_file or f.suffix == ".pyc":
                    continue
                z.write(f, rel.as_posix())
                n += 1
    return n


def export(data_cfg: dict[str, Any], research_cfg: dict[str, Any], train_cfg: dict[str, Any],
           seed: int, out_dir: str | Path = "data/kaggle_upload") -> dict[str, Any]:
    out = resolve(out_dir)
    pilot, val, fmt_ids = build_samples(data_cfg, research_cfg, seed)
    lengths = _lengths(resolve(data_cfg["paths"]["audits_dir"]) / "rendered_lengths.jsonl")
    assert not {r.sample_id for r in pilot} & {r.sample_id for r in val}
    assert {r.split for r in pilot} == {"train"} and {r.split for r in val} == {"val"}

    dd = out / "data"
    write_jsonl(dd / "pilot_train_100.jsonl", (r.to_dict() for r in pilot))
    write_jsonl(dd / "val_sample_100.jsonl", (r.to_dict() for r in val))
    p2 = json.loads((REPO_ROOT / "reports/phase2_manifests/random_1k.json").read_text("utf-8"))
    manifest = {
        "export_version": EXPORT_VERSION, "seed": seed, "code_commit": code_commit(),
        "source_dataset": data_cfg["source"]["repo_id"],
        "source_revision": data_cfg["source"]["revision"],
        "model_id": train_cfg["model_name"], "model_revision": train_cfg["model_revision"],
        "tokenizer": f"{data_cfg['tokenizer']['repo_id']}@{data_cfg['tokenizer']['revision']}",
        "chat_template_sha256": p2["chat_template_sha256"],
        "max_seq_len": research_cfg["primary_pool"]["max_rendered_tokens"],
        "pilot_train": {"file": "pilot_train_100.jsonl",
                        "sha256": sha256_file(dd / "pilot_train_100.jsonl"),
                        "sample_ids": [r.sample_id for r in pilot], **_summary(pilot, lengths),
                        "note": "stratified random from eligible TRAIN pool; pilot is NOT an arm"},
        "val_sample": {"file": "val_sample_100.jsonl",
                       "sha256": sha256_file(dd / "val_sample_100.jsonl"),
                       "sample_ids": [r.sample_id for r in val], **_summary(val, lengths),
                       "format_check_ids": fmt_ids,
                       "note": "fixed seeded VALIDATION sample; labels used only for scoring"},
        "license_acknowledgement": LICENSE_ACKNOWLEDGEMENT,
        "distribution": "PRIVATE Kaggle dataset only (gated source data)",
    }
    (dd / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", "utf-8")
    (dd / "README_DATA.md").write_text(
        "# ToolDistill-Lab Phase 3A pilot data (PRIVATE)\n\n"
        "Derived from Salesforce/xlam-function-calling-60k (CC BY 4.0, gated: accept the "
        "dataset's access conditions and cite APIGen, arXiv:2406.18518). Research use only.\n"
        "Keep this Kaggle dataset PRIVATE. Files: pilot_train_100.jsonl (TRAIN), "
        "val_sample_100.jsonl (VALIDATION, scoring only), manifest.json (hashes, ids).\n"
        "Schema: tooldistill.normalized.v1 records (see project docs).\n", "utf-8")

    n_src = zip_source(out / "project" / "project_source.zip")

    sheet = resolve(data_cfg["paths"]["audits_dir"]) / "val_sample_label_check.csv"
    with open(sheet, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "query", "available_tools", "expected_call",
                    "label_correct(Y/N/U)", "notes"])
        for r in val:
            c = r.expected_calls[0]
            w.writerow([r.sample_id, r.query,
                        " | ".join(t["function"]["name"] for t in r.tools),
                        json.dumps({"name": c.name, "arguments": c.arguments},
                                   ensure_ascii=False), "", ""])
    return {"out": str(out), "pilot": manifest["pilot_train"]["n"],
            "val": manifest["val_sample"]["n"], "format_check": len(fmt_ids),
            "source_files_zipped": n_src, "label_check_sheet": str(sheet),
            "val_label_status": manifest["val_sample"]["label_status"],
            "pilot_label_status": manifest["pilot_train"]["label_status"]}
