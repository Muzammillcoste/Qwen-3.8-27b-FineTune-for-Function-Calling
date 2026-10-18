"""Spec L #9-#13 + Phase 2 decisions: deterministic, train-only, matched, distinct arms."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from tooldistill.analysis import describe, mean_pairwise_jaccard, overlap, smd, wilson
from tooldistill.deduplicate import group_records
from tooldistill.normalize_xlam import NormalizeOptions, normalize_rows
from tooldistill.select import (
    SelectionError,
    allocate,
    build_pool,
    check_teacher_balance,
    is_schema_consistent,
    record_features,
    select_all,
)
from tooldistill.split import assign_splits
from tooldistill.validate import apply_validation

from .conftest import synthetic_rows

TEACHERS = ("deepseek-v2-chat", "mixtral-8x22b-instruct")


def synth_feats(n: int = 6000, seed: int = 0, inconsistent_rate: float = 0.03) -> dict:
    """SYNTHETIC feature table shaped like the real one (no tokenizer needed)."""
    rng = random.Random(seed)
    feats = {}
    for i in range(n):
        fn = f"fn{rng.randint(0, 600)}"
        nargs = rng.choice([0, 1, 1, 2, 2, 3, 4])
        consistent = rng.random() > inconsistent_rate
        feats[f"s:{i}"] = {
            "sample_id": f"s:{i}", "teacher": TEACHERS[i % 7 < 4], "group": f"g{i // 2}",
            "function": fn, "arg_signature": f"{fn}({nargs})", "n_tools": rng.randint(1, 8),
            "n_args": nargs, "n_params_called_tool": nargs + rng.randint(0, 3),
            "n_unknown_params_called_tool": rng.randint(0, 2),
            "n_structured_args": rng.choice([0, 0, 0, 1]),
            "query_words": {f"w{rng.randint(0, 3000)}" for _ in range(rng.randint(4, 15))},
            "rendered_total": rng.randint(150, 1400), "schema_consistent": consistent,
            "label_status": "valid" if consistent else "schema_conflict",
            "strict_schema_valid": consistent, "n_warnings": 0, "has_default_mismatch": False,
            "has_type_conflict": not consistent,
        }
    return feats


@pytest.fixture(scope="module")
def selected():
    feats = synth_feats()
    subsets, params = select_all(feats, seed=42)
    return feats, subsets, params


def test_arm_sizes_and_uniqueness(selected):
    _, s, _ = selected
    assert {k: len(v.sample_ids) for k, v in s.items()} == {
        "random_1k": 1000, "schema_consistent_1k": 1000, "diversity_1k": 1000, "random_3k": 3000}
    for v in s.values():
        assert len(set(v.sample_ids)) == len(v.sample_ids)


def test_same_seed_identical_ids_and_hashes(selected):
    feats, s, _ = selected
    s2, _ = select_all(feats, seed=42)
    assert {k: v.ids_sha256 for k, v in s.items()} == {k: v.ids_sha256 for k, v in s2.items()}


def test_different_seed_changes_random_not_algorithm(selected):
    feats, s, _ = selected
    s43, _ = select_all(feats, seed=43)
    assert s43["random_1k"].ids_sha256 != s["random_1k"].ids_sha256
    assert {k: v.algorithm for k, v in s43.items()} == {k: v.algorithm for k, v in s.items()}


def test_random_1k_nested_in_random_3k(selected):
    _, s, _ = selected
    assert set(s["random_1k"].sample_ids) <= set(s["random_3k"].sample_ids)


def test_all_1k_arms_share_stratum_quotas(selected):
    feats, s, params = selected
    from tooldistill.select import stratum_key

    cuts = params["length_cuts"]
    counts = {k: Counter(stratum_key(feats[i], cuts) for i in v.sample_ids)
              for k, v in s.items() if k.endswith("_1k")}
    assert counts["random_1k"] == counts["schema_consistent_1k"] == counts["diversity_1k"]
    assert dict(counts["random_1k"]) == {k: v for k, v in params["quota_1k"].items() if v}


def test_schema_consistent_differs_only_where_random_was_inconsistent(selected):
    feats, s, _ = selected
    r1, sc = set(s["random_1k"].sample_ids), set(s["schema_consistent_1k"].sample_ids)
    assert all(feats[i]["schema_consistent"] for i in sc)
    assert all(not feats[i]["schema_consistent"] for i in r1 - sc)
    assert len(r1 - sc) == sum(not feats[i]["schema_consistent"] for i in r1)


def test_diversity_is_distinct_and_more_covering(selected):
    feats, s, _ = selected
    assert overlap(s["diversity_1k"].sample_ids, s["random_1k"].sample_ids)["jaccard"] < 0.5
    fn = lambda ids: len({feats[i]["function"] for i in ids})  # noqa: E731
    assert fn(s["diversity_1k"].sample_ids) > fn(s["random_1k"].sample_ids)
    groups = [feats[i]["group"] for i in s["diversity_1k"].sample_ids]
    assert len(groups) == len(set(groups))  # never two from one dedup group


def test_teacher_balance_enforced(selected):
    feats, s, _ = selected
    devs = check_teacher_balance(s, feats, max_dev=0.02)
    assert all(abs(v) <= 0.02 for v in devs.values())
    bad = dict(s)
    from tooldistill.select import Subset

    ds = [i for i in feats if feats[i]["teacher"] == TEACHERS[1]][:1000]
    bad["skewed"] = Subset("skewed", ds, "test")
    with pytest.raises(SelectionError):
        check_teacher_balance(bad, feats, max_dev=0.02)


def test_pool_too_small_fails_explicitly():
    with pytest.raises(SelectionError):
        select_all(synth_feats(n=2000), seed=42)


def test_insufficient_consistent_records_fails_explicitly():
    with pytest.raises(SelectionError, match="consistent"):
        select_all(synth_feats(n=4000, inconsistent_rate=0.9), seed=42)


def test_allocate_is_exact_and_capped():
    # exact shares 2.5/1.5/1.0 -> floors 2/1/1; tie on remainder broken by key -> "a"
    assert allocate({"a": 5, "b": 3, "c": 2}, 5) == {"a": 3, "b": 1, "c": 1}
    q1 = allocate({"a": 5, "b": 3, "c": 2}, 4, caps={"a": 1, "b": 3, "c": 2})
    assert q1["a"] <= 1 and sum(q1.values()) == 4
    with pytest.raises(SelectionError):
        allocate({"a": 1}, 2)


def test_pool_refuses_validation_and_test_records():
    records, _ = normalize_rows(synthetic_rows(60), NormalizeOptions(id_prefix="synth"))
    group_records(records)
    assign_splits(records, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
    with pytest.raises(SelectionError, match="TRAIN"):
        build_pool(records)
    train = [r for r in records if r.split == "train"]
    pool = build_pool(train)
    assert pool and all(r.call_count == 1 and r.duplicate_of is None for r in pool)


def test_features_use_only_the_record_itself(fixture_normalized):
    records, _ = fixture_normalized
    group_records(records)
    rec = next(r for r in records if r.sample_id == "fixture:40001")
    f = record_features(rec, rendered_total=300)
    assert f["function"] == "search_items" and f["n_args"] == 2
    assert f["arg_signature"] == "search_items(price_max,q)"
    assert f["n_unknown_params_called_tool"] == 1   # q has no requiredness evidence
    assert "50" not in f["query_words"]            # numbers excluded from novelty words


def test_schema_consistency_definition(fixture_normalized):
    records, _ = fixture_normalized
    for r in records:
        apply_validation(r)
    by = {r.sample_id: r for r in records}
    assert is_schema_consistent(by["fixture:0"])
    assert not is_schema_consistent(by["fixture:40020"])    # ambiguous (warning)
    assert not is_schema_consistent(by["fixture:40006"])    # invalid
    assert is_schema_consistent(by["fixture:40001"])        # declaration issues don't matter


def test_describe_and_stats_helpers(selected):
    feats, s, params = selected
    d = describe(s["random_1k"].sample_ids, feats, params["length_cuts"],
                 Counter({"fn1": 3, "zzz": 1}), dict.fromkeys(feats, 10))
    assert d["n"] == 1000 and d["tokens_sum_completion"] == 10000
    assert abs(smd([1, 2, 3], [1, 2, 3])) < 1e-12
    assert 0 <= mean_pairwise_jaccard([{"a"}, {"a", "b"}, {"c"}]) <= 1
    lo, hi = wilson(30, 1000)
    assert lo < 3.0 < hi
