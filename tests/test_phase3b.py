"""Phase 3B: frozen eval-v2 rules, paired statistics, eval-set construction, sealed test guard,
coverage categories and the run comparison (no GPU, no network)."""

from __future__ import annotations

import json

import pytest

from tooldistill.compare import compare, render_md
from tooldistill.evaluate import schema_aware_key, score_example, valid_defaults
from tooldistill.experiment import SealedTestError, check_request
from tooldistill.normalize_xlam import normalize_row
from tooldistill.phase3b import COVERAGE, coverage_category, one_per_group
from tooldistill.stats import mcnemar_exact, paired_bootstrap_ci, paired_report, wilson


def rec_with(params: dict, args: dict, name="f"):
    row = {"id": 1, "query": "q", "tools": json.dumps([{"name": name, "description": "d",
                                                        "parameters": params}]),
           "answers": json.dumps([{"name": name, "arguments": args}])}
    return normalize_row(row, 0)


def tc(name, **args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


# ----------------------------------------------------------------- eval-v2 secondary metric
def test_optional_arg_equal_to_valid_default_is_equivalent():
    rec = rec_with({"q": {"type": "str", "description": "x"},
                    "outputsize": {"type": "int", "description": "x", "default": 30}},
                   {"q": "AAPL"})
    s = score_example(rec, tc("f", q="AAPL", outputsize=30))
    assert not s.exact_match and s.schema_aware_match
    s = score_example(rec, tc("f", q="AAPL", outputsize=31))  # different value: not equivalent
    assert not s.schema_aware_match


@pytest.mark.parametrize("params,pred_extra", [
    ({"number": {"type": "int", "description": "x", "default": "10"}}, {"number": 10}),   # bad default
    ({"league": {"type": "int", "description": "x", "default": ""}}, {"league": None}),   # null
    ({"descending": {"type": "bool", "description": "Defaults to False."}}, {"descending": False}),
])
def test_no_equivalence_without_explicit_valid_schema_default(params, pred_extra):
    rec = rec_with({"q": {"type": "str", "description": "x"}, **params}, {"q": "a"})
    assert not score_example(rec, tc("f", q="a", **pred_extra)).schema_aware_match


@pytest.mark.parametrize("label,pred", [("BTCUSDT", "BTCUSD"), ("Ford", "ford"),
                                        ("US", "United States")])
def test_different_strings_never_equated(label, pred):
    rec = rec_with({"s": {"type": "str", "description": "x"}}, {"s": label})
    s = score_example(rec, tc("f", s=pred))
    assert not s.exact_match and not s.schema_aware_match


def test_numeric_json_equality_only():
    rec = rec_with({"lat": {"type": "float", "description": "x"}}, {"lat": 34.0522})
    assert not score_example(rec, tc("f", lat=34.05)).schema_aware_match
    rec = rec_with({"n": {"type": "int", "description": "x"}}, {"n": 5})
    assert score_example(rec, tc("f", n=5.0)).exact_match


def test_valid_defaults_require_agreement_across_same_named_tools():
    row = {"id": 1, "query": "q", "answers": "[]", "tools": json.dumps([
        {"name": "f", "description": "d", "parameters": {"p": {"type": "int", "description": "",
                                                               "default": 1}}},
        {"name": "f", "description": "d", "parameters": {"p": {"type": "int", "description": "",
                                                               "default": 2}}}])}
    assert valid_defaults(normalize_row(row, 0), "f") == {}
    assert schema_aware_key(normalize_row(row, 0), {"name": "f", "arguments": {"p": 1}}) \
        != schema_aware_key(normalize_row(row, 0), {"name": "f", "arguments": {}})


# ----------------------------------------------------------------- statistics
def test_paired_bootstrap_deterministic_and_zero_for_identical():
    a = [True, False, True, True] * 25
    r = paired_bootstrap_ci(a, a)
    assert (r["diff"], r["ci_low"], r["ci_high"]) == (0, 0, 0)
    b = [True] * 100
    assert paired_bootstrap_ci(a, b) == paired_bootstrap_ci(a, b)
    assert paired_bootstrap_ci(a, b)["diff"] == 0.25


def test_mcnemar_and_wilson():
    a = [True] * 5 + [False] * 95
    b = [False] * 100
    m = mcnemar_exact(a, b)
    assert m["discordant"] == 5 and m["p_value"] == 0.0625
    lo, hi = wilson(82, 100)
    assert lo < 0.82 < hi
    rep = paired_report(a, b)
    assert rep["acc_a"] == 0.05 and rep["mcnemar"]["only_a"] == 5


def test_bootstrap_rejects_unpaired():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([True], [True, False])


# ----------------------------------------------------------------- eval sets / coverage / guards
def test_one_per_group_is_deterministic(fixture_normalized):
    from tooldistill.deduplicate import group_records

    records, _ = fixture_normalized
    group_records(records)
    a = one_per_group(records, 42, "x")
    assert len({r.source_group_id for r in a}) == len(a) == len({r.source_group_id for r in records})
    assert [r.sample_id for r in a] == [r.sample_id for r in one_per_group(records, 42, "x")]


def test_coverage_categories():
    r, d = {"a", "b"}, {"b", "c"}
    assert [coverage_category(f, r, d) for f in ("b", "a", "c", "z")] == list(COVERAGE)


def test_sealed_test_and_run_id_guards():
    with pytest.raises(SealedTestError):
        check_request("eval_test", None, {"test": {"sealed": True}})
    with pytest.raises(SealedTestError):
        check_request("eval_test", None, {})  # missing config defaults to sealed
    with pytest.raises(ValueError):
        check_request("train_eval", "pilot_100", {})
    with pytest.raises(ValueError):
        check_request("train_eval", None, {})
    with pytest.raises(ValueError):
        check_request("eval_base_val", "random_1k", {})
    check_request("train_eval", "diversity_1k", {})
    check_request("eval_base_val", None, {})


# ----------------------------------------------------------------- comparison end-to-end
def test_compare_runs_on_fixture(tmp_path, by_id):
    ids = ["fixture:0", "fixture:33700", "fixture:40003"]
    evalset = tmp_path / "ev.jsonl"
    evalset.write_text("".join(json.dumps(by_id[i].to_dict()) + "\n" for i in ids), "utf-8")
    cov = {"fixture:0": "seen_by_both", "fixture:33700": "seen_by_neither",
           "fixture:40003": "seen_by_diversity_1k_only"}
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"name": "fx", "sample_ids": ids, "coverage_by_id": cov}), "utf-8")

    def write(name, outputs):
        p = tmp_path / f"{name}.jsonl"
        p.write_text("".join(json.dumps({"sample_id": i, "raw_output": o}) + "\n"
                             for i, o in zip(ids, outputs, strict=True)), "utf-8")
        return p

    good = [tc("get_weather", city="Paris"), tc("sum_numbers", numbers=[10, 20]),
            tc("weather.forecast.v2.get", city="Rome", days=3)]
    runs = {"base": write("base", ["x", good[1], "x"]),
            "random_1k": write("r", [good[0], good[1], "x"]),
            "diversity_1k": write("d", good)}
    res = compare(runs, evalset, man)
    assert res["aggregate"]["diversity_1k"]["exact_structured_call_accuracy"]["k"] == 3
    c = res["contrasts"]["diversity_1k - random_1k"]["exact_match"]
    assert c["mcnemar"]["only_b"] == 1 and c["bootstrap"]["diff"] == pytest.approx(1 / 3, abs=1e-3)
    assert res["coverage"]["seen_by_diversity_1k_only"]["diversity_1k"] == 1.0
    assert "diversity_1k - random_1k" in render_md(res)
    man.write_text(json.dumps({"name": "fx", "sample_ids": ids[:2], "coverage_by_id": cov}), "utf-8")
    with pytest.raises(ValueError, match="manifest"):
        compare(runs, evalset, man)
