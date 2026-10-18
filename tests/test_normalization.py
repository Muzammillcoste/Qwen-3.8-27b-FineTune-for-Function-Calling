"""Spec L #1-#5: parsing, multi-call retention, malformed rows, required arrays, unknown types."""

from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator

from tooldistill.normalize_xlam import (
    _Ctx,
    convert_type_string,
    normalize_row,
    normalize_rows,
    split_top_level,
)
from tooldistill.schemas import NormalizedRecord, RejectedRecord

from .conftest import FIXTURE_OPTS


def params(rec, tool_name):
    for t in rec.tools:
        if t["function"]["name"] == tool_name:
            return t["function"]["parameters"]
    raise KeyError(tool_name)


def test_string_encoded_tools_convert_to_valid_json_schema(fixture_rows, by_id):
    assert isinstance(fixture_rows[0]["tools"], str)  # source field really is a JSON string
    rec = by_id["fixture:0"]
    p = params(rec, "get_weather")
    Draft202012Validator.check_schema(p)
    assert p["type"] == "object"
    assert p["properties"]["city"]["type"] == "string"
    assert p["required"] == ["city"]
    assert p["additionalProperties"] is False
    assert rec.tools[0]["type"] == "function"
    assert "tools_was_json_string" in rec.notes


def test_two_expected_calls_retained_with_names_and_arguments(by_id):
    rec = by_id["fixture:5"]
    assert rec.call_count == 2
    assert [(c.name, c.arguments) for c in rec.expected_calls] == [
        ("get_weather", {"city": "Oslo"}), ("get_weather", {"city": "Lima"})]
    rec = by_id["fixture:33658"]
    assert [c.name for c in rec.expected_calls] == ["get_order", "sum_numbers"]
    assert rec.expected_calls[1].arguments == {"numbers": [1, 2, 3]}


def test_three_calls_keep_source_order(by_id):
    rec = by_id["fixture:40016"]
    assert [c.name for c in rec.expected_calls] == ["get_order", "get_weather", "sum_numbers"]


def test_malformed_rows_rejected_with_reason_without_stopping(fixture_normalized):
    records, rejected = fixture_normalized
    reasons = {x.reason for x in rejected}
    assert reasons == {"tools_malformed_json", "answers_malformed_json", "query_missing",
                       "answer_item_malformed", "malformed_jsonl_line"}
    assert all(x.detail is not None and x.raw_hash for x in rejected)
    assert len(records) == 30  # every other row still normalized


@pytest.mark.parametrize("row,reason", [
    ("not a dict", "row_not_object"),
    ({"id": 1, "query": "q", "tools": "{}", "answers": "[]"}, "tools_not_list"),
    ({"id": 1, "query": "", "tools": "[]", "answers": "[]"}, "query_empty"),
    ({"id": 1, "query": 3, "tools": "[]", "answers": "[]"}, "query_not_string"),
    ({"id": 1, "query": "q", "tools": "[]"}, "answers_missing"),
    ({"id": 1, "query": "q", "answers": "[]"}, "tools_missing"),
    ({"id": 1, "query": "q", "tools": "[1]", "answers": "[]"}, "tool_not_object"),
    ({"id": 1, "query": "q", "tools": '[{"description": "d"}]', "answers": "[]"},
     "tool_missing_name"),
    ({"id": 1, "query": "q", "tools": "[]", "answers": '[{"name": "f", "arguments": "{bad"}]'},
     "answer_arguments_malformed_json"),
    ({"id": 1, "query": "q", "tools": "[]", "answers": '[{"name": "f", "arguments": [1]}]'},
     "answer_arguments_not_object"),
])
def test_row_level_rejections(row, reason):
    out = normalize_row(row, 0)
    assert isinstance(out, RejectedRecord)
    assert out.reason == reason


def test_required_boolean_becomes_required_array(by_id):
    p = params(by_id["fixture:0"], "get_weather")
    assert p["required"] == ["city"]          # required: true
    assert "units" in p["properties"]         # required: false -> optional, still declared
    assert "units" not in p["required"]


def test_required_string_true_is_coerced_and_noted(by_id):
    rec = by_id["fixture:40018"]
    assert params(rec, "get_weather_v2")["required"] == ["city"]
    assert any(n.startswith("required_coerced_from_string") for n in rec.notes)


def req_state(rec, tool_name, param):
    for entry in rec.metadata["requiredness"]:
        if entry["tool"] == tool_name:
            return entry["params"][param]
    raise KeyError(tool_name)


def test_xlam_optional_suffix_and_default_policy(by_id):
    rec = by_id["fixture:40001"]
    p = params(rec, "search_items")
    # q: no marker -> UNKNOWN (never asserted required); limit/in_stock: suffix; others: default
    assert p["required"] == []
    assert req_state(rec, "search_items", "q")["state"] == "UNKNOWN"
    assert req_state(rec, "search_items", "limit")["inferred_optional_reasons"] == [
        "default_present", "optional_suffix"]
    assert req_state(rec, "search_items", "price_max")["state"] == "OPTIONAL"
    assert req_state(rec, "search_items", "in_stock")["state"] == "OPTIONAL"
    assert p["properties"]["limit"]["type"] == "integer"
    assert p["properties"]["limit"]["default"] == 10
    assert p["properties"]["price_max"]["type"] == "number"
    assert p["properties"]["in_stock"]["type"] == "boolean"
    assert p["properties"]["filters"] == {"type": "object", "description": "extra filters",
                                          "default": {}}


def test_three_state_requiredness_declared_vs_inferred(by_id):
    w = req_state(by_id["fixture:0"], "get_weather", "city")
    assert w == {"state": "REQUIRED", "declared": True, "declared_sources": ["param_required_field"],
                 "inferred_optional_reasons": [], "conflict": []}
    u = req_state(by_id["fixture:0"], "get_weather", "units")
    assert u["state"] == "OPTIONAL" and u["declared"] is False
    opt = req_state(by_id["fixture:40018"], "get_weather_v2", "units")  # Optional[str]
    assert opt["state"] == "OPTIONAL" and opt["declared"] is None
    assert opt["inferred_optional_reasons"] == ["optional_typing"]
    js = by_id["fixture:40002"]
    assert req_state(js, "create_ticket", "title")["declared_sources"] == ["schema_required_list"]
    assert req_state(js, "create_ticket", "priority")["state"] == "OPTIONAL"


def test_contradictory_requiredness_is_unknown_and_flagged(by_id):
    rec = by_id["fixture:40021"]
    info = req_state(rec, "addr_lookup", "postcode")
    assert info["state"] == "UNKNOWN" and info["declared"] is True
    assert info["conflict"] == ["declared_required_but_default_present"]
    assert any(f.startswith("requiredness_conflict:addr_lookup.postcode") for f in rec.flags)
    assert "postcode" not in params(rec, "addr_lookup")["required"]


def test_declarations_disagreeing_is_a_conflict():
    row = {"id": 1, "query": "q", "answers": "[]", "tools": [{
        "name": "f", "description": "d",
        "parameters": {"type": "object", "required": ["a"],
                       "properties": {"a": {"type": "string", "required": False}}}}]}
    rec = normalize_row(row, 0)
    info = rec.metadata["requiredness"][0]["params"]["a"]
    assert info["state"] == "UNKNOWN" and info["conflict"] == ["declarations_disagree"]


def test_original_tool_declarations_preserved(fixture_rows, by_id):
    import json

    rec = by_id["fixture:40001"]
    raw_row = next(r for r in fixture_rows if r.get("id") == 40001)
    assert rec.metadata["source_tools"] == json.loads(raw_row["tools"])


def test_unknown_type_is_flagged_not_cast_to_string(by_id):
    rec = by_id["fixture:33705"]
    cb = params(rec, "register_cb")["properties"]["cb"]
    assert "type" not in cb  # unconstrained, NOT {"type": "string"}
    assert any(f.startswith("unknown_type:register_cb.cb") for f in rec.flags)


def test_missing_parameter_type_is_flagged(by_id):
    rec = by_id["fixture:40014"]
    assert "param_missing_type:translate.text" in rec.flags
    assert params(rec, "translate")["required"] == ["text"]


@pytest.mark.parametrize("type_str,schema,optional", [
    ("int", {"type": "integer"}, False),
    ("integer", {"type": "integer"}, False),
    ("float", {"type": "number"}, False),
    ("str", {"type": "string"}, False),
    ("string", {"type": "string"}, False),
    ("bool", {"type": "boolean"}, False),
    ("list", {"type": "array"}, False),
    ("dict", {"type": "object"}, False),
    ("List[int]", {"type": "array", "items": {"type": "integer"}}, False),
    ("list[str]", {"type": "array", "items": {"type": "string"}}, False),
    ("Dict[str, Any]", {"type": "object"}, False),
    ("Dict[str, int]", {"type": "object", "additionalProperties": {"type": "integer"}}, False),
    ("Union[int, float]", {"anyOf": [{"type": "integer"}, {"type": "number"}]}, False),
    ("Optional[str]", {"anyOf": [{"type": "string"}, {"type": "null"}]}, True),
    ("str, optional", {"type": "string"}, True),
    ("int, default=5", {"type": "integer"}, True),
    ("Tuple[int, int]", {"type": "array", "prefixItems": [{"type": "integer"}] * 2}, False),
    ("Tuple[int, ...]", {"type": "array", "items": {"type": "integer"}}, False),
    ("Literal['a', 'b']", {"enum": ["a", "b"]}, False),
    ("List[Dict[str, Union[int, float]]]",
     {"type": "array", "items": {"type": "object", "additionalProperties":
                                 {"anyOf": [{"type": "integer"}, {"type": "number"}]}}}, False),
])
def test_type_conversion_table(type_str, schema, optional):
    ctx = _Ctx()
    got, reasons = convert_type_string(type_str, ctx, "t.p")
    assert got == schema
    assert bool(reasons) is optional
    assert not ctx.flags
    Draft202012Validator.check_schema(got)


@pytest.mark.parametrize("type_str", ["Callable", "datetime", "Foo[int]", "int, positive"])
def test_unknown_types_and_qualifiers_flagged(type_str):
    ctx = _Ctx()
    convert_type_string(type_str, ctx, "t.p")
    assert ctx.flags


def test_split_top_level_respects_brackets():
    assert split_top_level("Dict[str, int], optional") == ["Dict[str, int]", "optional"]


def test_json_schema_form_parameters(by_id):
    p = params(by_id["fixture:40002"], "create_ticket")
    assert p["required"] == ["title"]
    assert set(p["properties"]) == {"title", "priority"}


def test_parameters_named_type_and_properties_are_not_confused(by_id):
    p = params(by_id["fixture:40011"], "set_item")
    assert set(p["properties"]) == {"type", "properties"}
    assert p["properties"]["type"]["type"] == "string"
    assert p["required"] == ["type"]


def test_dotted_tool_name_preserved(by_id):
    rec = by_id["fixture:40003"]
    assert rec.tools[0]["function"]["name"] == "weather.forecast.v2.get"
    assert rec.expected_calls[0].name == "weather.forecast.v2.get"


def test_arguments_json_string_parsed(by_id):
    rec = by_id["fixture:33701"]
    assert rec.expected_calls[0].arguments == {"order_id": 99}
    assert "answer_arguments_were_json_string" in rec.notes


def test_native_lists_accepted(by_id):
    rec = by_id["fixture:33700"]
    assert "tools_was_native_list" in rec.notes and rec.call_count == 1


def test_already_wrapped_function_tool_is_accepted():
    row = {"id": 1, "query": "q", "answers": "[]",
           "tools": [{"type": "function", "function": {"name": "f", "description": "d",
                                                        "parameters": {}}}]}
    rec = normalize_row(row, 0)
    assert isinstance(rec, NormalizedRecord) and rec.tools[0]["function"]["name"] == "f"


def test_suspicious_markup_flagged_and_text_kept_verbatim(by_id):
    rec = by_id["fixture:40013"]
    assert "suspicious_markup_in_tool_text:lookup" in rec.flags
    assert "<tool_call>" in rec.tools[0]["function"]["description"]  # data, never executed


def test_query_wrapped_in_quotes_is_kept_verbatim():
    q = '"Can you find events for Japan?"'
    rec = normalize_row({"id": 1, "query": q, "tools": "[]", "answers": "[]"}, 0)
    assert rec.query == q
    assert "query_wrapped_in_literal_quotes" in rec.notes


def test_call_structure_distinguishes_parallel_calls_from_multi_turn(by_id):
    assert by_id["fixture:0"].metadata["call_structure"] == "single_call"
    multi = by_id["fixture:40016"].metadata
    assert multi["call_structure"] == "multi_call_single_response"
    assert multi["turn_structure"] == "single_turn"  # one user turn, one assistant response
    assert by_id["fixture:40007"].metadata["call_structure"] == "no_call"


def test_record_roundtrip(by_id):
    rec = by_id["fixture:33658"]
    assert NormalizedRecord.from_dict(rec.to_dict()) == rec


def test_normalization_is_deterministic(fixture_rows):
    a, _ = normalize_rows(fixture_rows, FIXTURE_OPTS)
    b, _ = normalize_rows(fixture_rows, FIXTURE_OPTS)
    assert [r.to_dict() for r in a] == [r.to_dict() for r in b]
