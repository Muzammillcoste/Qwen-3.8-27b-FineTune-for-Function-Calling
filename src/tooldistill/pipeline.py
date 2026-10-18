"""Phase 0/1 data pipeline: normalize -> validate -> group/dedup -> split -> leakage audit.

Order is mandatory (spec E5). Selection (Phase 2) is deliberately absent and must only ever be
fitted on the ``train`` output of this pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .deduplicate import group_records
from .io import read_records, resolve, sha256_file, write_jsonl
from .manifests import build_manifest, write_manifest
from .normalize_xlam import NormalizeOptions, normalize_rows
from .provenance import DEFAULT_RULES, check_id_integrity, verify_source_counts
from .split import SPLITS, assert_no_leakage, assign_splits
from .validate import apply_validation


def options_from_config(cfg: dict[str, Any], *, fixture: bool = False) -> NormalizeOptions:
    src, ids, norm = cfg.get("source", {}), cfg.get("ids", {}), cfg.get("normalization", {})
    rules = tuple(cfg.get("provenance", {}).get("rules") or DEFAULT_RULES)
    return NormalizeOptions(
        source_dataset="fixture:miniature_xlam" if fixture else src.get("repo_id"),
        source_revision="fixture" if fixture else str(src.get("revision")),
        id_field=ids.get("id_field", "id"),
        id_prefix="fixture" if fixture else "xlam",
        allow_source_row_index_as_id=bool(ids.get("allow_source_row_index_as_id", False)),
        additional_properties=bool(norm.get("additional_properties", False)),
        teacher_rules=rules,
    )


def check_pinned_revision(cfg: dict[str, Any], raw_metadata_path: Path) -> None:
    """Fail if the raw file's recorded revision differs from the pinned config revision."""
    meta = json.loads(raw_metadata_path.read_text("utf-8"))
    pinned = cfg["source"]["revision"]
    if meta["revision"] != pinned:
        raise RuntimeError(
            f"raw revision {meta['revision']} != pinned {pinned}; re-audit before continuing"
        )


def prepare(
    cfg: dict[str, Any], input_path: str | Path, out_dir: str | Path, seed: int,
    *, fixture: bool = False,
) -> dict[str, Any]:
    input_path = resolve(input_path)
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_meta = input_path.parent / "raw_metadata.json"
    if not fixture and raw_meta.exists():
        check_pinned_revision(cfg, raw_meta)

    rows = read_records(input_path)
    opts = options_from_config(cfg, fixture=fixture)
    records, rejected = normalize_rows(rows, opts)
    for r in records:
        apply_validation(r)

    d = cfg.get("dedup", {})
    dedup = group_records(
        records,
        near_duplicate_jaccard=float(d.get("near_duplicate_jaccard", 0.8)),
        group_same_query_text=bool(d.get("group_same_query_text", True)),
        max_block_pairwise=int(d.get("max_block_pairwise", 5000)),
    )
    # Exact duplicates leave the eligible pool (kept on disk with reason).
    excluded = [r for r in records if r.duplicate_of is not None]
    eligible = [r for r in records if r.duplicate_of is None]

    s = cfg.get("split", {})
    split_summary = assign_splits(
        eligible,
        ratios=dict(s.get("ratios", {"train": 0.8, "val": 0.1, "test": 0.1})),
        seed=seed,
        stratify_by=list(s.get("stratify_by", ["teacher_source", "call_count_bucket"])),
        min_records_per_split=int(s.get("min_records_per_split", 1)),
    )
    leakage = assert_no_leakage(eligible)

    # Spec L #7: teacher provenance vs dataset-card counts, only if the id field is trustworthy.
    integrity = check_id_integrity([r.original_id for r in records])
    expected = cfg.get("provenance", {}).get("expected_counts_full_dataset")
    if integrity["id_field_trustworthy"]:
        # Rejected rows carry no teacher label, so any rejection makes the card check fail visibly.
        chk = verify_source_counts((r.teacher_source for r in records), expected)
        provenance_check = {"computed": True, "ok": chk.ok, "observed": chk.observed,
                            "expected": chk.expected, "n_rejected_rows": len(rejected),
                            "message": chk.message}
    else:
        provenance_check = {"computed": False, "ok": False,
                            "message": "not computed: id field missing, non-integer or duplicated"}
    provenance_check["id_integrity"] = integrity

    files: dict[str, Path] = {}
    files["normalized"] = out / "normalized.jsonl"
    write_jsonl(files["normalized"], (r.to_dict() for r in records))
    files["rejected"] = out / "rejected.jsonl"
    write_jsonl(files["rejected"], (x.to_dict() for x in rejected))
    files["excluded_exact_duplicates"] = out / "excluded_exact_duplicates.jsonl"
    write_jsonl(files["excluded_exact_duplicates"],
                ({"sample_id": r.sample_id, "duplicate_of": r.duplicate_of,
                  "reason": "exact_duplicate"} for r in excluded))
    for name in SPLITS:
        files[name] = out / "splits" / f"{name}.jsonl"
        write_jsonl(files[name], (r.to_dict() for r in eligible if r.split == name))
    reports = {"dedup_report": dedup.to_dict(), "split_summary": split_summary,
               "leakage_audit": leakage.to_dict(), "provenance_check": provenance_check}
    for name, payload in reports.items():
        p = out / f"{name}.json"
        p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", "utf-8")
        files[name] = p

    manifest = build_manifest(
        files,
        source_dataset=opts.source_dataset,
        source_revision=opts.source_revision,
        input_sha256=sha256_file(input_path),
        seed=seed,
        is_fixture=fixture,
        counts={"source_rows": len(rows), "normalized": len(records), "rejected": len(rejected),
                "exact_duplicates_excluded": len(excluded), "eligible": len(eligible)},
    )
    write_manifest(out / "manifest.json", manifest)
    return {"manifest": manifest, "records": records, "eligible": eligible,
            "rejected": rejected, **reports}
