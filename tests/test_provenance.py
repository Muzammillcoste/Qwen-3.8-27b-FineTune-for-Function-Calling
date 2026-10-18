"""Spec L #6, #7, #19: ID survival, teacher provenance from IDs only, manifests."""

from __future__ import annotations

import json
import random

import pytest

from tooldistill.io import iter_jsonl, load_yaml, sha256_file, sha256_obj
from tooldistill.normalize_xlam import NormalizeOptions, normalize_row, normalize_rows
from tooldistill.pipeline import prepare
from tooldistill.provenance import (
    check_id_integrity,
    parse_original_id,
    teacher_from_id,
    verify_source_counts,
)

from .conftest import FIXTURE, FIXTURE_OPTS


@pytest.mark.parametrize("oid,teacher", [
    (0, "deepseek-v2-chat"), (33658, "deepseek-v2-chat"),
    (33659, "mixtral-8x22b-instruct"), (59999, "mixtral-8x22b-instruct"),
    (60000, "unknown"), (-1, "unknown"), (None, "unknown"),
])
def test_teacher_boundaries_from_card(oid, teacher):
    assert teacher_from_id(oid) == teacher


def test_rules_from_config_match_defaults():
    cfg = load_yaml("configs/data.yaml")
    assert [teacher_from_id(i, cfg["provenance"]["rules"]) for i in (0, 33658, 33659, 59999)] \
        == [teacher_from_id(i) for i in (0, 33658, 33659, 59999)]


@pytest.mark.parametrize("raw,value,note", [
    (7, 7, None), ("12", 12, "id_coerced_from_string"), (True, None, "id_invalid_type:bool"),
    (1.0, None, "id_invalid_type:float"), (None, None, "id_missing"),
])
def test_parse_original_id(raw, value, note):
    p = parse_original_id(raw)
    assert (p.value, p.note) == (value, note)


def test_teacher_is_derived_from_id_not_row_position(fixture_rows):
    rows = [r for r in fixture_rows if isinstance(r.get("id"), int)]
    a, _ = normalize_rows(rows, FIXTURE_OPTS)
    shuffled = rows[:]
    random.Random(1).shuffle(shuffled)
    b, _ = normalize_rows(shuffled, FIXTURE_OPTS)
    key = lambda recs: sorted((r.original_id, r.query, r.teacher_source) for r in recs)  # noqa: E731
    assert key(a) == key(b)


def test_card_counts_verified_on_synthetic_full_id_range():
    ok = verify_source_counts(teacher_from_id(i) for i in range(60000))
    assert ok.ok, ok.message
    bad = verify_source_counts(teacher_from_id(i) for i in range(59990))
    assert not bad.ok and "mixtral" in bad.message
    with_unknown = verify_source_counts([teacher_from_id(i) for i in range(60000)] + ["unknown"])
    assert not with_unknown.ok


def test_missing_id_gives_unknown_teacher_and_flag(by_id):
    rec = next(r for r in by_id.values() if r.query == "[FIXTURE] Weather in Tokyo?")
    assert rec.original_id is None and rec.id_origin == "missing"
    assert rec.teacher_source == "unknown"
    assert "missing_original_id" in rec.flags
    assert rec.sample_id == f"fixture:row{rec.source_row_index}"


def test_row_index_ids_only_when_explicitly_allowed():
    row = {"query": "q", "tools": "[]", "answers": "[]"}
    rec = normalize_row(row, 33659, NormalizeOptions(allow_source_row_index_as_id=True))
    assert rec.original_id == 33659 and rec.id_origin == "source_row_index"
    assert rec.teacher_source == "mixtral-8x22b-instruct"


def test_duplicate_ids_flagged_and_sample_ids_unique(fixture_normalized):
    records, _ = fixture_normalized
    dups = [r for r in records if r.original_id == 33659]
    assert len(dups) == 2
    assert all("duplicate_original_id" in r.flags for r in dups)
    assert len({r.sample_id for r in records}) == len(records)


def test_out_of_range_id_flagged(by_id):
    assert "teacher_source_unknown" in by_id["fixture:60005"].flags


def test_raw_and_normalized_hash_pair(fixture_rows, by_id):
    rec = by_id["fixture:0"]
    assert rec.raw_hash == sha256_obj(fixture_rows[0])
    assert len(rec.normalized_hash) == 64 and rec.normalized_hash != rec.raw_hash


def test_id_integrity_summary():
    s = check_id_integrity([0, 1, 1, None, 5])
    assert s["n_missing"] == 1 and s["duplicate_ids"] == [1] and s["gap_examples"] == [2, 3, 4]
    assert s["id_field_trustworthy"] is False
    assert check_id_integrity([0, 1, 2])["id_field_trustworthy"] is True


@pytest.fixture
def prepared(tmp_path):
    cfg = load_yaml("configs/data.yaml")
    return prepare(cfg, FIXTURE, tmp_path, seed=42, fixture=True), tmp_path


def test_original_ids_survive_normalize_dedup_split_export(prepared):
    res, out = prepared
    normalized_ids = {(r.sample_id, r.original_id) for r in res["records"]}
    exported = set()
    for name in ("train", "val", "test"):
        exported |= {(d["sample_id"], d["original_id"]) for d in iter_jsonl(out / "splits" /
                                                                             f"{name}.jsonl")}
    excluded = {d["sample_id"] for d in iter_jsonl(out / "excluded_exact_duplicates.jsonl")}
    assert {s for s, _ in exported} | excluded == {s for s, _ in normalized_ids}
    assert exported <= normalized_ids  # ids unchanged on the way out


def test_card_check_computed_only_with_trustworthy_ids(tmp_path, prepared):
    from tooldistill.io import write_jsonl

    from .conftest import synthetic_rows

    res, _ = prepared
    assert res["provenance_check"]["computed"] is False  # fixture has missing + duplicate ids
    src = tmp_path / "synth.jsonl"
    write_jsonl(src, synthetic_rows())
    res2 = prepare(load_yaml("configs/data.yaml"), src, tmp_path / "out", seed=42, fixture=True)
    chk = res2["provenance_check"]
    assert chk["computed"] is True
    assert chk["ok"] is False and "Mismatch" in chk["message"]  # synthetic != card counts


def test_manifest_has_hashes_counts_schema_and_license(prepared):
    res, out = prepared
    m = json.loads((out / "manifest.json").read_text("utf-8"))
    assert m["schema_version"] == "tooldistill.normalized.v1"
    assert "CC BY 4.0" in m["license_acknowledgement"]
    assert m["is_fixture"] is True and m["seed"] == 42
    for name, entry in m["files"].items():
        assert len(entry["sha256"]) == 64
        if name in ("train", "val", "test", "normalized", "rejected"):
            assert entry["n_records"] is not None
    assert m["files"]["train"]["sha256"] == sha256_file(out / "splits" / "train.jsonl")
    assert m["counts"]["rejected"] == 5
