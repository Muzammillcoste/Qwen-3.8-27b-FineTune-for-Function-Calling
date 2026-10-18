"""Spec E5 / L #8, #9, #10 (split part), #13: dedup grouping and group-aware split leakage."""

from __future__ import annotations

import random
from collections import Counter, defaultdict

import pytest

from tooldistill.deduplicate import group_records, normalize_query_text
from tooldistill.io import load_yaml, sha256_file
from tooldistill.normalize_xlam import NormalizeOptions, normalize_rows
from tooldistill.pipeline import prepare
from tooldistill.split import (
    EmptySplitError,
    LeakageError,
    assert_no_leakage,
    assign_splits,
    audit_leakage,
)

from .conftest import FIXTURE, synthetic_rows

RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def grouped_fixture(fixture_normalized):
    records, _ = fixture_normalized
    report = group_records(records)
    return {r.sample_id: r for r in records}, report


def synth(seed=7, n_families=300):
    records, rejected = normalize_rows(synthetic_rows(n_families, seed),
                                       NormalizeOptions(id_prefix="synth"))
    assert not rejected
    return records


def test_exact_duplicate_marked_and_grouped(fixture_normalized):
    recs, report = grouped_fixture(fixture_normalized)
    dup = recs["fixture:40008"]
    assert dup.duplicate_of == "fixture:0" and "exact_duplicate" in dup.flags
    assert dup.source_group_id == recs["fixture:0"].source_group_id
    assert report.n_exact_duplicates == 1


def test_near_duplicate_paraphrase_grouped(fixture_normalized):
    recs, _ = grouped_fixture(fixture_normalized)
    assert recs["fixture:40009"].source_group_id == recs["fixture:0"].source_group_id
    assert recs["fixture:40009"].duplicate_of is None  # near-dup grouped, NOT deleted


def test_same_query_text_with_different_label_grouped(fixture_normalized):
    recs, _ = grouped_fixture(fixture_normalized)
    a = next(r for r in recs.values() if r.original_id == 33659 and "order" in r.query)
    assert recs["fixture:40010"].source_group_id == a.source_group_id


def test_duplicate_original_ids_grouped(fixture_normalized):
    recs, _ = grouped_fixture(fixture_normalized)
    gids = {r.source_group_id for r in recs.values() if r.original_id == 33659}
    assert len(gids) == 1


def test_unrelated_records_not_grouped(fixture_normalized):
    recs, _ = grouped_fixture(fixture_normalized)
    assert recs["fixture:5"].source_group_id != recs["fixture:0"].source_group_id
    assert recs["fixture:40003"].source_group_id != recs["fixture:40016"].source_group_id


def test_group_ids_independent_of_input_order():
    a = synth()
    b = synth()
    random.Random(3).shuffle(b)
    group_records(a)
    group_records(b)
    assert {r.sample_id: r.source_group_id for r in a} == \
        {r.sample_id: r.source_group_id for r in b}


def test_no_group_spans_splits_and_ratios_reasonable():
    recs = synth()
    group_records(recs)
    summary = assign_splits(recs, RATIOS, seed=42)
    by_group = defaultdict(set)
    for r in recs:
        by_group[r.source_group_id].add(r.split)
    assert all(len(s) == 1 for s in by_group.values())
    assert audit_leakage(recs).ok
    n = len(recs)
    for split, target in RATIOS.items():
        frac = summary["splits"][split]["n_records"] / n
        assert abs(frac - target) < 0.03, (split, frac)


def test_paraphrase_families_never_straddle_splits():
    recs = synth()
    group_records(recs)
    assign_splits(recs, RATIOS, seed=42)
    fam_splits = defaultdict(set)
    for r in recs:
        fam = r.expected_calls[0].name  # one tool per synthetic family
        fam_splits[fam].add(r.split)
    assert all(len(s) == 1 for s in fam_splits.values())


def test_teacher_mix_is_stratified():
    recs = synth()
    group_records(recs)
    assign_splits(recs, RATIOS, seed=42)
    share = {}
    for s in RATIOS:
        c = Counter(r.teacher_source for r in recs if r.split == s)
        share[s] = c["deepseek-v2-chat"] / sum(c.values())
    assert max(share.values()) - min(share.values()) < 0.1, share


def test_same_seed_same_split_and_different_seed_differs():
    def run(seed):
        recs = synth()
        group_records(recs)
        assign_splits(recs, RATIOS, seed=seed)
        return {r.sample_id: r.split for r in recs}

    assert run(42) == run(42)
    assert run(42) != run(43)


def test_split_invariant_to_input_order():
    a, b = synth(), synth()
    random.Random(9).shuffle(b)
    for recs in (a, b):
        group_records(recs)
        assign_splits(recs, RATIOS, seed=42)
    assert {r.sample_id: r.split for r in a} == {r.sample_id: r.split for r in b}


def test_audit_detects_injected_leakage():
    recs = synth()
    group_records(recs)
    assign_splits(recs, RATIOS, seed=42)
    groups = defaultdict(list)
    for r in recs:
        groups[r.source_group_id].append(r)
    victim = next(g for g in groups.values() if len(g) > 1)
    victim[0].split = "test" if victim[0].split != "test" else "train"
    rep = audit_leakage(recs)
    assert not rep.ok and "source_group_id" in rep.violations
    with pytest.raises(LeakageError):
        assert_no_leakage(recs)


def test_audit_detects_query_overlap_even_if_groups_differ():
    recs = synth(n_families=20)
    group_records(recs)
    assign_splits(recs, RATIOS, seed=42)
    a = next(r for r in recs if r.split == "train")
    b = next(r for r in recs if r.split == "test")
    b.query = a.query.upper()  # same normalized text
    assert normalize_query_text(a.query) == normalize_query_text(b.query)
    assert "normalized_query" in audit_leakage(recs).violations


def test_empty_split_fails_explicitly(fixture_normalized):
    records, _ = fixture_normalized
    few = records[:2]
    group_records(few)
    with pytest.raises(EmptySplitError):
        assign_splits(few, RATIOS, seed=42)


def test_split_requires_grouping_first(fixture_normalized):
    records, _ = fixture_normalized
    with pytest.raises(ValueError, match="group before splitting"):
        assign_splits(records, RATIOS, seed=42)


def test_bad_ratios_rejected(fixture_normalized):
    records, _ = fixture_normalized
    group_records(records)
    with pytest.raises(ValueError):
        assign_splits(records, {"train": 0.9, "val": 0.2, "test": 0.1}, seed=42)


def test_fixture_pipeline_passes_leakage_audit_and_is_reproducible(tmp_path):
    cfg = load_yaml("configs/data.yaml")
    r1 = prepare(cfg, FIXTURE, tmp_path / "a", seed=42, fixture=True)
    r2 = prepare(cfg, FIXTURE, tmp_path / "b", seed=42, fixture=True)
    assert r1["leakage_audit"]["ok"]
    for name in ("train", "val", "test"):
        assert sha256_file(tmp_path / "a" / "splits" / f"{name}.jsonl") == \
            sha256_file(tmp_path / "b" / "splits" / f"{name}.jsonl")
    f1 = {k: v["sha256"] for k, v in r1["manifest"]["files"].items() if k != "manifest"}
    f2 = {k: v["sha256"] for k, v in r2["manifest"]["files"].items() if k != "manifest"}
    assert f1 == f2
    # the exact duplicate is excluded from every split
    assert all(r.sample_id != "fixture:40008" for r in r1["eligible"])
