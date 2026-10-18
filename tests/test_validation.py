"""Structural validation (spec E4, F1 #1-2, L #16 label-side). Schema validity != semantic truth."""

from __future__ import annotations

import copy

from jsonschema import Draft202012Validator

from tooldistill.normalize_xlam import normalize_row
from tooldistill.validate import apply_validation, flag_category, flag_kind, validate_record


def _row(tools, answers, query="[TEST] q"):
    import json

    return {"id": 1, "query": query, "tools": json.dumps(tools), "answers": json.dumps(answers)}


ORDER = {"name": "get_order", "description": "d",
         "parameters": {"order_id": {"type": "int", "description": "id", "required": True}}}


def test_all_fixture_tool_schemas_are_valid_json_schema(fixture_normalized):
    records, _ = fixture_normalized
    for r in records:
        for t in r.tools:
            Draft202012Validator.check_schema(t["function"]["parameters"])
        assert not any(f.startswith("tool_schema_invalid") for f in r.flags)


def test_clean_record_has_no_validation_flags(by_id):
    assert validate_record(by_id["fixture:0"]).flags == []
    assert validate_record(by_id["fixture:40016"]).flags == []


def test_hallucinated_tool_in_label_is_flagged(by_id):
    rec = by_id["fixture:40004"]
    assert "call_unknown_tool:get_weather" in rec.flags


def test_missing_required_argument_is_flagged(by_id):
    rec = by_id["fixture:40005"]
    assert "call_args_invalid:get_weather:required" in rec.flags


def test_wrong_type_and_spurious_argument_are_flagged(by_id):
    rec = by_id["fixture:40006"]
    kinds = [f for f in rec.flags if f.startswith("call_args_invalid")]
    assert kinds == ["call_args_invalid:get_order:additionalProperties"]
    assert "schema_argument_type_conflict:get_order.order_id:expected=integer:got=string" \
        in rec.flags
    assert rec.metadata["validation_errors"]
    assert rec.metadata["label_status"] == "invalid"  # spurious arg is a confirmed label error
    assert rec.metadata["strict_schema_valid"] is False


def test_decimal_in_integer_field_is_conflict_not_label_error():
    rec = normalize_row(_row([ORDER], [{"name": "get_order", "arguments": {"order_id": 0.3163}}]), 0)
    apply_validation(rec)
    assert rec.flags == ["schema_argument_type_conflict:get_order.order_id:expected=integer:got=number"]
    assert flag_category(rec.flags[0]) == "declaration_label_conflict"
    assert rec.metadata["label_status"] == "schema_conflict"
    assert rec.metadata["strict_schema_valid"] is False  # strict verdict preserved
    assert rec.expected_calls[0].arguments == {"order_id": 0.3163}  # never repaired


def test_integer_valued_float_is_valid_for_integer_field():
    rec = normalize_row(_row([ORDER], [{"name": "get_order", "arguments": {"order_id": 42.0}}]), 0)
    apply_validation(rec)
    assert rec.metadata["label_status"] == "valid" and rec.metadata["strict_schema_valid"]


def test_no_tool_label_is_flagged_not_dropped(by_id):
    rec = by_id["fixture:40007"]
    assert rec.call_count == 0 and "no_tool_call_label" in rec.flags


def test_invalid_schema_is_flagged():
    rec = normalize_row(_row([ORDER], [{"name": "get_order", "arguments": {"order_id": 1}}]), 0)
    rec.tools[0]["function"]["parameters"]["properties"]["order_id"]["type"] = "not-a-type"
    res = validate_record(rec)
    assert "tool_schema_invalid:get_order" in res.flags


def test_duplicate_identical_calls_flagged():
    call = {"name": "get_order", "arguments": {"order_id": 1}}
    rec = normalize_row(_row([ORDER], [call, call]), 0)
    res = validate_record(rec)
    assert "duplicate_identical_calls" in res.warnings  # ambiguous, not an invalid verdict
    assert res.label_status == "ambiguous"
    assert rec.call_count == 2  # both kept; flagged, not deduplicated silently


def test_missing_unknown_requiredness_arg_is_warning_not_invalid(by_id):
    rec = by_id["fixture:40020"]
    assert rec.metadata["validation_warnings"] == [
        "missing_arg_unknown_requiredness:forecast_lookup.days"]
    assert not any(flag_category(f) == "label_error" for f in rec.flags)
    assert rec.metadata["label_status"] == "ambiguous"


def test_missing_declared_required_arg_is_invalid(by_id):
    assert by_id["fixture:40005"].metadata["label_status"] == "invalid"


def test_conflicting_requiredness_missing_arg_is_only_a_warning(by_id):
    rec = by_id["fixture:40021"]
    assert "missing_arg_unknown_requiredness:addr_lookup.postcode" in \
        rec.metadata["validation_warnings"]
    assert rec.metadata["label_status"] == "ambiguous"


def test_default_type_mismatches_flagged_separately_from_label(by_id):
    rec = by_id["fixture:40021"]
    mism = sorted(f for f in rec.flags if f.startswith("default_type_mismatch"))
    assert mism == [
        "default_type_mismatch:addr_lookup.housenumber:integer:str",  # numeric string "288"
        "default_type_mismatch:addr_lookup.ratio:integer:float",
        "default_type_mismatch:addr_lookup.verbose:boolean:str",
    ]
    assert "default_null_value:addr_lookup.suffix:string" in rec.flags
    assert all(flag_category(f) == "schema_issue" for f in mism)
    # record is not modified: defaults kept verbatim
    props = rec.tools[0]["function"]["parameters"]["properties"]
    assert props["housenumber"]["default"] == "288"


def test_label_status_values(by_id):
    assert by_id["fixture:0"].metadata["label_status"] == "valid"
    assert by_id["fixture:40004"].metadata["label_status"] == "invalid"
    assert by_id["fixture:40007"].metadata["label_status"] == "no_call"


def test_unknown_type_argument_is_unverifiable_warning(by_id):
    rec = by_id["fixture:33705"]
    assert "arg_type_unverifiable:register_cb.cb" in rec.metadata["validation_warnings"]


def test_every_emitted_flag_has_a_category(fixture_normalized):
    from tooldistill.deduplicate import group_records

    records, _ = fixture_normalized
    group_records(records)
    kinds = {flag_kind(f) for r in records for f in r.flags + r.metadata["validation_warnings"]}
    assert {k for k in kinds if flag_category(k) == "other"} == set()


def test_same_named_tools_call_valid_if_any_declaration_accepts():
    as_str = {"name": "lookup", "description": "d",
              "parameters": {"limit": {"type": "str", "description": "l", "required": True}}}
    as_int = {"name": "lookup", "description": "d",
              "parameters": {"limit": {"type": "int", "description": "l", "required": True}}}
    rec = normalize_row(_row([as_int, as_str], [{"name": "lookup", "arguments": {"limit": 5}}]), 0)
    res = validate_record(rec)
    assert not any(f.startswith("call_args_invalid") for f in res.flags)
    bad = normalize_row(_row([as_int, as_str], [{"name": "lookup", "arguments": {"limit": [1]}}]), 0)
    assert not validate_record(bad).strict_schema_valid
    assert "duplicate_tool_names" in rec.flags  # still surfaced for review


def test_validation_does_not_mutate_labels(by_id):
    rec = by_id["fixture:40006"]
    before = copy.deepcopy([c.arguments for c in rec.expected_calls])
    validate_record(rec)
    assert [c.arguments for c in rec.expected_calls] == before


def test_apply_validation_is_idempotent(by_id):
    rec = by_id["fixture:40006"]
    flags = list(rec.flags)
    apply_validation(rec)
    assert rec.flags == flags


def test_flag_kind():
    assert flag_kind("unknown_type:f.x:Foo") == "unknown_type"
