"""Spec L #16: parser rejects malformed/hallucinated output; multiple calls scored without loss."""

from __future__ import annotations

import json

import pytest

from tooldistill.evaluate import (
    aggregate,
    canonical_value,
    paired_comparison,
    parse_tool_calls,
    score_example,
)


def tc(name, **args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def test_parse_single_and_multiple_blocks_and_stop_at_eos():
    p = parse_tool_calls(tc("a", x=1) + "\n" + tc("b", y="z") + "<|im_end|>\n" + tc("c"))
    assert p.parse_ok and [c["name"] for c in p.calls] == ["a", "b"]  # nothing after EOS


@pytest.mark.parametrize("text,err", [
    ("<tool_call>\n{not json}\n</tool_call>", "block0:invalid_json"),
    ("<tool_call>\n{\"arguments\": {}}\n</tool_call>", "block0:missing_name"),
    ("<tool_call>\n{\"name\": \"a\", \"arguments\": [1]}\n</tool_call>", "block0:arguments_not_object"),
    ("<tool_call>\n{\"name\": \"a\", \"arguments\": {}}", "unclosed_tool_call_block"),
])
def test_malformed_output_rejected(text, err):
    p = parse_tool_calls(text)
    assert err in p.errors and not p.parse_ok


def test_arguments_as_json_string_and_no_eval():
    p = parse_tool_calls('<tool_call>\n{"name": "a", "arguments": "{\\"k\\": 2}"}\n</tool_call>')
    assert p.calls == [{"name": "a", "arguments": {"k": 2}}]
    p = parse_tool_calls('<tool_call>\n{"name": "__import__(\'os\')", "arguments": {}}\n</tool_call>')
    assert p.calls[0]["name"] == "__import__('os')"  # treated as a string, never evaluated


def test_plain_text_answer_is_no_tool_call():
    p = parse_tool_calls("I cannot help with that.")
    assert p.n_blocks == 0 and not p.parse_ok and p.text_outside_blocks


def test_numeric_canonicalization():
    assert canonical_value({"a": 5.0, "b": [1.0, 2.5]}) == {"a": 5, "b": [1, 2.5]}
    assert canonical_value("5") != canonical_value(5)


def test_score_correct_and_order_insensitive(by_id):
    rec = by_id["fixture:5"]  # two calls, same tool
    out = tc("get_weather", city="Lima") + "\n" + tc("get_weather", city="Oslo")
    s = score_example(rec, out)
    assert s.exact_match and s.error_category == "correct" and s.schema_valid_prediction


def test_missing_and_extra_calls_not_lost(by_id):
    rec = by_id["fixture:5"]
    s = score_example(rec, tc("get_weather", city="Oslo"))
    assert not s.exact_match and s.n_predicted == 1 and s.n_expected == 2
    s = score_example(by_id["fixture:0"], tc("get_weather", city="Paris") * 2)
    assert not s.exact_match and s.n_predicted == 2


def test_hallucinated_tool_flagged(by_id):
    s = score_example(by_id["fixture:0"], tc("get_forecast", city="Paris"))
    assert s.hallucinated_tool and s.error_category == "hallucinated_tool"
    assert not s.schema_valid_prediction


def test_wrong_value_and_wrong_keys(by_id):
    rec = by_id["fixture:0"]
    s = score_example(rec, tc("get_weather", city="Rome"))
    assert s.error_category == "wrong_argument_values" and s.arg_values_correct == 0
    s = score_example(rec, tc("get_weather", town="Paris"))
    assert s.error_category == "wrong_argument_keys"


def test_aggregate_and_paired(by_id):
    rec = by_id["fixture:0"]
    good = score_example(rec, tc("get_weather", city="Paris"))
    bad = score_example(rec, "no call")
    agg = aggregate([good, bad])
    assert agg["exact_structured_call_accuracy"] == {"k": 1, "n": 2, "rate": 0.5}
    assert agg["error_categories"] == {"correct": 1, "no_tool_call": 1}
    pc = paired_comparison([good], [bad])
    assert pc["only_first"] == 1 and pc["discordant"] == 1 and pc["mcnemar_exact_p"] == 1.0
