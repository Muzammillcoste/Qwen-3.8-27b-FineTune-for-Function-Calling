"""Structural validation of normalized records (spec E4, F1 components 1-2).

Checks are SYNTACTIC/SCHEMA checks only. Passing them is not evidence of semantic correctness
(whether the call actually answers the user's request); that requires human audit (spec F2).
Nothing in a record is executed.

Three separate outcome channels (approved Phase 1 policy, 2026-10-18):

- LABEL ERRORS   (``validation_flags``, category ``label_error``): the expected call is
  confirmed inconsistent with the declared tools: unknown tool, wrong type, spurious argument,
  missing argument that is DECLARED required.
- LABEL WARNINGS (``validation_warnings``): ambiguous, never an invalid verdict, e.g. a missing
  argument whose requiredness is UNKNOWN, or a value for a parameter whose type is unknown.
- SCHEMA ISSUES  (``validation_flags``, category ``schema_issue``): problems in the tool
  declarations themselves (e.g. ``default_type_mismatch``), independent of label correctness.

Records are never modified or removed because of these results.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .io import canonical_json
from .schemas import NormalizedRecord

VALIDATION_VERSION = "validate-v3"

FLAG_CATEGORIES: dict[str, str] = {
    # label correctness (confirmed structural label errors)
    "call_unknown_tool": "label_error",
    "call_args_invalid": "label_error",  # non-type keywords: additionalProperties, required, enum..
    # declared type vs supplied argument disagree: strict-invalid, NOT a confirmed label error
    "schema_argument_type_conflict": "declaration_label_conflict",
    # label ambiguity (warnings channel)
    "missing_arg_unknown_requiredness": "label_warning",
    "arg_type_unverifiable": "label_warning",
    "duplicate_identical_calls": "label_warning",
    # label structure
    "no_tool_call_label": "label_structure",
    "answers_single_object_wrapped": "label_structure",
    "answer_missing_arguments": "label_structure",
    # tool declarations
    "default_type_mismatch": "schema_issue",
    "default_null_value": "schema_issue",
    "requiredness_conflict": "schema_issue",
    "required_non_boolean": "schema_issue",
    "required_names_not_in_properties": "schema_issue",
    "unknown_type": "schema_issue",
    "unknown_type_qualifier": "schema_issue",
    "empty_type": "schema_issue",
    "param_missing_type": "schema_issue",
    "param_type_not_string": "schema_issue",
    "param_spec_not_object": "schema_issue",
    "parameters_not_object": "schema_issue",
    "param_missing_description": "schema_issue",
    "tool_missing_description": "schema_issue",
    "tool_schema_invalid": "schema_issue",
    "duplicate_tool_names": "schema_issue",
    "no_tools_available": "schema_issue",
    # prompt-injection-like markup (data only, never executed)
    "suspicious_markup_in_tool_text": "safety",
    "suspicious_markup_in_query": "safety",
    # provenance
    "missing_original_id": "provenance",
    "duplicate_original_id": "provenance",
    "teacher_source_unknown": "provenance",
    "id_invalid_type": "provenance",
    # dedup
    "exact_duplicate": "dedup",
    "conflicting_labels_same_input": "dedup",
}

LABEL_STATUSES = ("valid", "ambiguous", "schema_conflict", "invalid", "no_call")


def flag_kind(flag: str) -> str:
    """Strip per-record detail so flags can be counted: 'unknown_type:f.x:Foo' -> 'unknown_type'."""
    return flag.split(":", 1)[0]


def flag_category(flag: str) -> str:
    return FLAG_CATEGORIES.get(flag_kind(flag), "other")


@dataclass
class ValidationResult:
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    label_status: str = "valid"
    strict_schema_valid: bool = True  # strict JSON-Schema verdict, preserved unchanged


def _json_type(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number" if not v.is_integer() else "integer-valued-number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


@lru_cache(maxsize=65536)
def _compiled(schema_json: str) -> tuple[Draft202012Validator | None, str | None]:
    """Meta-validate + compile once per distinct schema (check_schema is ~9 ms; tools repeat)."""
    schema = json.loads(schema_json)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as e:
        return None, e.message[:200]
    return Draft202012Validator(schema), None


def compiled_validator(schema: dict[str, Any]) -> tuple[Draft202012Validator | None, str | None]:
    return _compiled(canonical_json(schema))


def _constraint(schema: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in schema.items() if k not in ("default", "description")}


def _expected(schema: dict[str, Any]) -> str:
    c = _constraint(schema)
    if "type" in c:
        return str(c["type"])
    if "anyOf" in c:
        return "|".join(str(s.get("type", "?")) for s in c["anyOf"])
    return "enum" if "enum" in c else "?"


def check_defaults(rec: NormalizedRecord) -> list[str]:
    """Schema-level: does each declared default satisfy its own parameter type?

    Numeric strings for integer fields (e.g. "288") ARE mismatches. A null default is reported
    separately as ``default_null_value`` (common Python idiom for optional args) unless the type
    allows null. Labels are not involved.
    """
    flags: list[str] = []
    for tool in rec.tools:
        fn = tool["function"]
        for pname, ps in fn["parameters"].get("properties", {}).items():
            if "default" not in ps:
                continue
            c = _constraint(ps)
            if not c:
                continue  # unconstrained (unknown/Any type): cannot judge
            d = ps["default"]
            v, err = compiled_validator(c)
            if v is None:
                continue  # schema itself invalid; flagged elsewhere
            if v.is_valid(d):
                continue
            where = f"{fn['name']}.{pname}"
            if d is None:
                flags.append(f"default_null_value:{where}:{_expected(ps)}")
            else:
                flags.append(f"default_type_mismatch:{where}:{_expected(ps)}:{type(d).__name__}")
    return flags


def _is_unconstrained(schema: dict[str, Any]) -> bool:
    return not _constraint(schema)


def validate_record(rec: NormalizedRecord, max_errors: int = 10) -> ValidationResult:
    res = ValidationResult()
    req_maps = rec.metadata.get("requiredness") or [{} for _ in rec.tools]
    # A row may declare several tools with the same name but different schemas (seen in real
    # xLAM, e.g. id 81). A call is judged against EVERY same-named declaration; the best fit wins.
    candidates: dict[str, list[tuple[Draft202012Validator, dict, dict]]] = {}
    for idx, tool in enumerate(rec.tools):
        fn = tool["function"]
        validator, err = compiled_validator(fn["parameters"])
        if validator is None:
            res.flags.append(f"tool_schema_invalid:{fn['name']}")
            res.errors.append(f"{fn['name']}: {err}")
            continue
        params = (req_maps[idx] or {}).get("params", {}) if idx < len(req_maps) else {}
        candidates.setdefault(fn["name"], []).append((validator, fn["parameters"], params))

    if not rec.expected_calls:
        res.flags.append("no_tool_call_label")

    tool_names = {t["function"]["name"] for t in rec.tools}
    for i, call in enumerate(rec.expected_calls):
        if call.name not in tool_names:
            res.flags.append(f"call_unknown_tool:{call.name}")
            continue
        options = candidates.get(call.name)
        if not options:
            continue  # all same-named schemas invalid; already flagged
        scored = []
        for v, schema, req in options:
            errs = sorted(v.iter_errors(call.arguments), key=lambda e: list(e.path))
            warns = [f"missing_arg_unknown_requiredness:{call.name}.{p}"
                     for p, info in req.items()
                     if info.get("state") == "UNKNOWN" and p not in call.arguments]
            warns += [f"arg_type_unverifiable:{call.name}.{p}"
                      for p in call.arguments
                      if p in schema.get("properties", {})
                      and _is_unconstrained(schema["properties"][p])]
            scored.append((len(errs), len(warns), errs, warns))
        _, _, errs, warns = min(scored, key=lambda s: (s[0], s[1]))
        if errs:
            res.strict_schema_valid = False
            # Declared-type vs supplied-value disagreements (e.g. 0.3163 for an "int" latitude)
            # are declaration/label CONFLICTS: we cannot tell which side is wrong.
            type_errs = [e for e in errs if e.validator == "type"]
            other_errs = [e for e in errs if e.validator != "type"]
            for e in type_errs:
                param = str(e.path[0]) if e.path else "<root>"
                f = (f"schema_argument_type_conflict:{call.name}.{param}:"
                     f"expected={e.validator_value}:got={_json_type(e.instance)}")
                if f not in res.flags:
                    res.flags.append(f)
            if other_errs:
                kinds = sorted({e.validator for e in other_errs})
                res.flags.append(f"call_args_invalid:{call.name}:{','.join(kinds)}")
            for e in errs[:max_errors]:
                res.errors.append(f"call[{i}] {call.name}: {e.message[:200]}")
        for w in warns:
            if w not in res.warnings:
                res.warnings.append(w)

    dup = Counter(canonical_json({"n": c.name, "a": c.arguments}) for c in rec.expected_calls)
    if any(c > 1 for c in dup.values()):
        res.warnings.append("duplicate_identical_calls")

    res.flags.extend(check_defaults(rec))

    if any(flag_category(f) == "label_error" for f in res.flags):
        res.strict_schema_valid = False
    if not rec.expected_calls:
        res.label_status = "no_call"
    elif any(flag_category(f) == "label_error" for f in res.flags):
        res.label_status = "invalid"
    elif any(flag_kind(f) == "schema_argument_type_conflict" for f in res.flags):
        res.label_status = "schema_conflict"
    elif res.warnings:
        res.label_status = "ambiguous"
    else:
        res.label_status = "valid"
    return res


def apply_validation(rec: NormalizedRecord) -> ValidationResult:
    """Validate and merge results into ``rec.metadata`` (idempotent)."""
    res = validate_record(rec)
    for f in res.flags:
        rec.add_flag(f)
    rec.metadata["validation_warnings"] = res.warnings
    rec.metadata["validation_errors"] = res.errors
    rec.metadata["label_status"] = res.label_status
    rec.metadata["strict_schema_valid"] = res.strict_schema_valid
    rec.metadata["validation_version"] = VALIDATION_VERSION
    return res
