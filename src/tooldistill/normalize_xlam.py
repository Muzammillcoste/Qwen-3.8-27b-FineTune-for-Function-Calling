"""Normalize raw xLAM rows into canonical ``NormalizedRecord`` objects (spec E3/E4).

Design rules:
- Field types are inspected, never assumed: ``query``/``tools``/``answers`` may be JSON strings
  or already-parsed objects; ``answers[*].arguments`` may itself be a JSON string.
- Every expected call is kept, in source order. Tool names (including dotted names) are preserved.
- Simple xLAM types (``int``, ``str``, ``List[int]``, ``str, optional`` ...) are converted to JSON
  Schema. Unknown types are FLAGGED and left unconstrained; they are never cast to ``string``.
- A malformed row yields a ``RejectedRecord`` with a reason; it never aborts the dataset.
- Tool text is data only: nothing from a row is ever executed or evaluated.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .io import sha256_obj
from .provenance import DEFAULT_RULES, parse_original_id, teacher_from_id
from .schemas import NormalizedRecord, RejectedRecord, ToolCall

# Markup that could collide with the Qwen chat template / tool-call parser.
SUSPICIOUS_MARKUP = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|</?tool_call>|</?tools>|</?tool_response>",
    re.IGNORECASE,
)

SIMPLE_TYPES: dict[str, dict[str, Any]] = {
    "int": {"type": "integer"},
    "integer": {"type": "integer"},
    "float": {"type": "number"},
    "number": {"type": "number"},
    "double": {"type": "number"},
    "str": {"type": "string"},
    "string": {"type": "string"},
    "bool": {"type": "boolean"},
    "boolean": {"type": "boolean"},
    "list": {"type": "array"},
    "array": {"type": "array"},
    "tuple": {"type": "array"},
    "set": {"type": "array"},
    "dict": {"type": "object"},
    "object": {"type": "object"},
    "none": {"type": "null"},
    "null": {"type": "null"},
}
ANY_TYPES = {"any", "typing.any"}
ARRAY_GENERICS = {"list", "sequence", "set", "iterable", "array"}
DICT_GENERICS = {"dict", "mapping", "object"}
UNION_GENERICS = {"union"}
OPTIONAL_GENERICS = {"optional"}


class RowError(Exception):
    """A row-level failure that makes the row unusable. Carries a machine-readable reason."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass
class NormalizeOptions:
    source_dataset: str = "Salesforce/xlam-function-calling-60k"
    source_revision: str = "UNPINNED"
    id_field: str = "id"
    id_prefix: str = "xlam"
    allow_source_row_index_as_id: bool = False
    additional_properties: bool = False
    teacher_rules: tuple[dict[str, Any], ...] = DEFAULT_RULES


@dataclass
class _Ctx:
    notes: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def flag(self, f: str) -> None:
        if f not in self.flags:
            self.flags.append(f)

    def note(self, n: str) -> None:
        if n not in self.notes:
            self.notes.append(n)


# --------------------------------------------------------------------------- type conversion


def split_top_level(s: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` outside of brackets: 'Dict[str, int], optional' -> 2 parts."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return [p for p in parts if p != ""]


_GENERIC = re.compile(r"^([A-Za-z_][\w.]*)\s*\[(.*)\]$", re.DOTALL)


def _parse_core_type(s: str, ctx: _Ctx, where: str) -> tuple[dict[str, Any], bool]:
    """Return (schema, nullable_optional). ``nullable_optional`` is True for Optional[X]."""
    s = s.strip()
    low = s.lower()
    if low in SIMPLE_TYPES:
        return dict(SIMPLE_TYPES[low]), False
    if low in ANY_TYPES:
        ctx.note(f"any_type_unconstrained:{where}")
        return {}, False
    m = _GENERIC.match(s)
    if m:
        head = m.group(1).lower().split(".")[-1]
        args = split_top_level(m.group(2))
        if head in ARRAY_GENERICS and len(args) == 1:
            items, _ = _parse_core_type(args[0], ctx, where)
            out: dict[str, Any] = {"type": "array"}
            if items:
                out["items"] = items
            return out, False
        if head == "tuple":
            if len(args) == 2 and args[1] == "...":
                items, _ = _parse_core_type(args[0], ctx, where)
                return ({"type": "array", "items": items} if items else {"type": "array"}), False
            prefix = [_parse_core_type(a, ctx, where)[0] for a in args]
            return {"type": "array", "prefixItems": prefix}, False
        if head in DICT_GENERICS and len(args) == 2:
            val, _ = _parse_core_type(args[1], ctx, where)
            out = {"type": "object"}
            if val:
                out["additionalProperties"] = val
            return out, False
        if head in UNION_GENERICS and args:
            subs = [_parse_core_type(a, ctx, where)[0] for a in args]
            has_null = any(x == {"type": "null"} for x in subs)
            if any(x == {} for x in subs):  # Union containing Any/unknown => unconstrained
                return {}, has_null
            return {"anyOf": subs}, has_null
        if head in OPTIONAL_GENERICS and len(args) == 1:
            inner, _ = _parse_core_type(args[0], ctx, where)
            if not inner:
                return {}, True
            return {"anyOf": [inner, {"type": "null"}]}, True
        if head == "literal" and args:
            vals = []
            for a in args:
                a = a.strip()
                try:
                    vals.append(json.loads(a))
                except json.JSONDecodeError:
                    vals.append(a.strip("'\""))
            return {"enum": vals}, False
    ctx.flag(f"unknown_type:{where}:{s[:60]}")
    return {}, False


def convert_type_string(type_str: str, ctx: _Ctx, where: str) -> tuple[dict[str, Any], list[str]]:
    """Convert an xLAM type string. Returns (schema, optional_reasons).

    optional_reasons: "optional_suffix" (``str, optional``), "optional_typing" (``Optional[X]`` /
    ``Union[X, None]``), "default_qualifier" (``int, default=5``).
    """
    parts = split_top_level(type_str)
    if not parts:
        ctx.flag(f"empty_type:{where}")
        return {}, []
    schema, nullable_opt = _parse_core_type(parts[0], ctx, where)
    reasons: list[str] = ["optional_typing"] if nullable_opt else []
    for qual in parts[1:]:
        q = qual.strip().lower()
        if q == "optional":
            reasons.append("optional_suffix")
        elif q.startswith("default"):
            reasons.append("default_qualifier")
        else:
            ctx.flag(f"unknown_type_qualifier:{where}:{qual[:40]}")
    return schema, sorted(set(reasons))


# --------------------------------------------------------------------------- parameter conversion

REQUIRED, OPTIONAL, UNKNOWN = "REQUIRED", "OPTIONAL", "UNKNOWN"


def _required_value(raw: Any, ctx: _Ctx, where: str) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.strip().lower() in {"true", "false"}:
        ctx.note(f"required_coerced_from_string:{where}")
        return raw.strip().lower() == "true"
    ctx.flag(f"required_non_boolean:{where}")
    return None


def classify_requiredness(
    declarations: list[tuple[bool, str]], optional_reasons: list[str], ctx: _Ctx, where: str
) -> dict[str, Any]:
    """Three-state requiredness. Declared and inferred evidence are tracked separately.

    REQUIRED: declared required (explicit bool or required-list membership), no contrary evidence.
    OPTIONAL: declared not-required, or an optional marker / default value is present.
    UNKNOWN : no evidence either way, or contradictory metadata (flagged).
    """
    declared_vals = {v for v, _ in declarations}
    sources = sorted({s for _, s in declarations})
    conflict_reasons: list[str] = []
    if len(declared_vals) > 1:
        conflict_reasons.append("declarations_disagree")
    declared: bool | None = declared_vals.pop() if len(declared_vals) == 1 else None
    if declared is True and optional_reasons:
        conflict_reasons.append("declared_required_but_" + "+".join(optional_reasons))
    if conflict_reasons:
        ctx.flag(f"requiredness_conflict:{where}:{','.join(conflict_reasons)}")
        state = UNKNOWN
    elif declared is True:
        state = REQUIRED
    elif declared is False or optional_reasons:
        state = OPTIONAL
    else:
        state = UNKNOWN
    return {
        "state": state,
        "declared": declared,
        "declared_sources": sources,
        "inferred_optional_reasons": optional_reasons,
        "conflict": conflict_reasons,
    }


def convert_param(
    pname: str, spec: Any, ctx: _Ctx, tool: str, opts: NormalizeOptions
) -> tuple[dict[str, Any], tuple[bool, str] | None, list[str]]:
    """Convert one parameter spec. Returns (schema, explicit_declaration, optional_reasons)."""
    where = f"{tool}.{pname}"
    if not isinstance(spec, dict):
        ctx.flag(f"param_spec_not_object:{where}")
        return {}, None, []
    raw_type = spec.get("type")
    reasons: list[str] = []
    if raw_type is None:
        ctx.flag(f"param_missing_type:{where}")
        schema: dict[str, Any] = {}
    elif isinstance(raw_type, str):
        schema, reasons = convert_type_string(raw_type, ctx, where)
    elif isinstance(raw_type, list) and all(isinstance(t, str) for t in raw_type):
        conv = [convert_type_string(t, ctx, where) for t in raw_type]
        subs = [c[0] for c in conv]
        schema = {"anyOf": subs} if all(subs) else {}
        reasons = sorted({r for c in conv for r in c[1]})
    else:
        ctx.flag(f"param_type_not_string:{where}")
        schema = {}

    # Nested structure carried by JSON-schema-like specs.
    if isinstance(spec.get("properties"), dict):
        nested, _ = convert_parameters(spec, ctx, f"{tool}.{pname}", opts, top_level=False)
        schema = {**schema, **nested} if schema.get("type") in (None, "object") else schema
    if isinstance(spec.get("items"), dict) and schema.get("type") == "array":
        item_schema, _, _ = convert_param("[]", spec["items"], ctx, where, opts)
        if item_schema:
            schema["items"] = item_schema

    desc = spec.get("description")
    if isinstance(desc, str):
        schema["description"] = desc
        if SUSPICIOUS_MARKUP.search(desc):
            ctx.flag(f"suspicious_markup_in_tool_text:{where}")
    elif desc is None:
        ctx.flag(f"param_missing_description:{where}")
    if "default" in spec:
        schema["default"] = spec["default"]
        reasons = sorted(set(reasons) | {"default_present"})
    if isinstance(spec.get("enum"), list):
        schema["enum"] = spec["enum"]
    extra = set(spec) - {"type", "description", "default", "required", "enum", "properties",
                         "items"}
    if extra:
        ctx.note(f"param_extra_keys_preserved_in_source_tools:{where}")

    explicit = None
    if "required" in spec and not isinstance(spec.get("required"), list):
        val = _required_value(spec["required"], ctx, where)
        if val is not None:
            explicit = (val, "param_required_field")
    return schema, explicit, reasons


def convert_parameters(
    params: Any, ctx: _Ctx, tool: str, opts: NormalizeOptions, top_level: bool = True,
    tool_required: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Convert either xLAM mapping form ``{param: {type, description, ...}}`` or a JSON-Schema
    object ``{"type": "object", "properties": {...}, "required": [...]}`` into JSON Schema.

    Returns (schema, requiredness map). The JSON-Schema ``required`` list contains ONLY params
    classified REQUIRED; UNKNOWN params are never asserted to be required.
    """
    if params is None:
        ctx.note(f"parameters_missing:{tool}")
        params = {}
    if not isinstance(params, dict):
        ctx.flag(f"parameters_not_object:{tool}")
        params = {}

    is_json_schema = params.get("type") == "object" and isinstance(params.get("properties"), dict)
    if not is_json_schema and isinstance(params.get("properties"), dict) and not top_level:
        is_json_schema = True  # nested spec with properties but type given elsewhere
    props_in = params["properties"] if is_json_schema else params
    req_lists: list[tuple[set[str], str]] = []
    req_list = params.get("required") if is_json_schema else None
    if isinstance(req_list, list):
        names = {r for r in req_list if isinstance(r, str)}
        missing = names - set(props_in)
        if missing:
            ctx.flag(f"required_names_not_in_properties:{tool}:{sorted(missing)}")
        req_lists.append((names, "schema_required_list"))
    if tool_required is not None:
        req_lists.append(({r for r in tool_required if isinstance(r, str)}, "tool_required_list"))

    properties: dict[str, Any] = {}
    required: list[str] = []
    requiredness: dict[str, dict[str, Any]] = {}
    for pname, spec in props_in.items():
        schema, explicit, reasons = convert_param(pname, spec, ctx, tool, opts)
        properties[pname] = schema
        declarations = [(pname in names, src) for names, src in req_lists]
        if explicit is not None:
            declarations.append(explicit)
        info = classify_requiredness(declarations, reasons, ctx, f"{tool}.{pname}")
        requiredness[pname] = info
        if info["state"] == REQUIRED:
            required.append(pname)

    out: dict[str, Any] = {"type": "object", "properties": properties, "required": required}
    if top_level and not opts.additional_properties:
        out["additionalProperties"] = False
    return out, requiredness


def convert_tool(tool: Any, ctx: _Ctx, opts: NormalizeOptions) -> tuple[dict[str, Any], dict]:
    """Returns (normalized function tool, requiredness map for its top-level params)."""
    if isinstance(tool, dict) and tool.get("type") == "function" and isinstance(
        tool.get("function"), dict
    ):
        ctx.note("tool_already_function_wrapped")
        tool = tool["function"]
    if not isinstance(tool, dict):
        raise RowError("tool_not_object", repr(tool)[:120])
    name = tool.get("name")
    if not isinstance(name, str) or not name.strip():
        raise RowError("tool_missing_name", repr(tool)[:120])
    desc = tool.get("description")
    if not isinstance(desc, str):
        ctx.flag(f"tool_missing_description:{name}")
        desc = ""
    if SUSPICIOUS_MARKUP.search(desc):
        ctx.flag(f"suspicious_markup_in_tool_text:{name}")
    tool_required = tool.get("required") if isinstance(tool.get("required"), list) else None
    params, requiredness = convert_parameters(
        tool.get("parameters"), ctx, name, opts, tool_required=tool_required
    )
    fn = {"type": "function", "function": {"name": name, "description": desc, "parameters": params}}
    return fn, requiredness


# --------------------------------------------------------------------------- field parsing


def parse_json_field(value: Any, field_name: str, ctx: _Ctx) -> Any:
    """Decode a field that may be JSON-encoded text. Records which representation was seen."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as e:
            raise RowError(f"{field_name}_malformed_json", str(e)[:200]) from None
        ctx.note(f"{field_name}_was_json_string")
        return decoded
    ctx.note(f"{field_name}_was_native_{type(value).__name__}")
    return value


def parse_query(value: Any, ctx: _Ctx) -> str:
    if value is None:
        raise RowError("query_missing")
    if not isinstance(value, str):
        raise RowError("query_not_string", type(value).__name__)
    # Query text is kept VERBATIM. Full-data inspection (2026-10-18) found 0/60000 JSON-encoded
    # queries; 12 are plain text wrapped in literal quotes, which we note but never strip.
    q = value
    if q.startswith('"') and q.endswith('"') and len(q) >= 2:
        ctx.note("query_wrapped_in_literal_quotes")
    if not q.strip():
        raise RowError("query_empty")
    if SUSPICIOUS_MARKUP.search(q):
        ctx.flag("suspicious_markup_in_query")
    return q


def parse_answers(value: Any, ctx: _Ctx) -> list[ToolCall]:
    if value is None:
        raise RowError("answers_missing")
    answers = parse_json_field(value, "answers", ctx)
    if isinstance(answers, dict):
        ctx.flag("answers_single_object_wrapped")
        answers = [answers]
    if not isinstance(answers, list):
        raise RowError("answers_not_list", type(answers).__name__)
    calls: list[ToolCall] = []
    for i, a in enumerate(answers):
        if not isinstance(a, dict) or not isinstance(a.get("name"), str) or not a["name"].strip():
            raise RowError("answer_item_malformed", f"index {i}: {repr(a)[:120]}")
        args = a.get("arguments")
        if args is None:
            ctx.flag(f"answer_missing_arguments:{i}")
            args = {}
        elif isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as e:
                raise RowError("answer_arguments_malformed_json", f"index {i}: {e}") from None
            ctx.note("answer_arguments_were_json_string")
        if not isinstance(args, dict):
            raise RowError("answer_arguments_not_object", f"index {i}: {type(args).__name__}")
        extra = set(a) - {"name", "arguments"}
        if extra:
            ctx.note(f"answer_extra_keys:{sorted(extra)}")
        calls.append(ToolCall(name=a["name"], arguments=args))
    return calls


# --------------------------------------------------------------------------- row / dataset


def _resolve_id(
    row: dict[str, Any], row_index: int, opts: NormalizeOptions, ctx: _Ctx
) -> tuple[int | None, str]:
    if opts.id_field in row:
        parsed = parse_original_id(row[opts.id_field])
        if parsed.note:
            (ctx.note if parsed.value is not None else ctx.flag)(parsed.note)
        if parsed.value is not None:
            return parsed.value, f"source_field:{opts.id_field}"
    if opts.allow_source_row_index_as_id:
        ctx.note("original_id_from_source_row_index")
        return row_index, "source_row_index"
    ctx.flag("missing_original_id")
    return None, "missing"


def normalize_row(
    row: Any, row_index: int, opts: NormalizeOptions | None = None
) -> NormalizedRecord | RejectedRecord:
    opts = opts or NormalizeOptions()
    raw_hash = sha256_obj(row)
    raw_id = row.get(opts.id_field) if isinstance(row, dict) else None
    try:
        if not isinstance(row, dict):
            raise RowError("row_not_object", type(row).__name__)
        if "__malformed_line__" in row:
            raise RowError("malformed_jsonl_line", row["__malformed_line__"][:120])
        ctx = _Ctx()
        original_id, id_origin = _resolve_id(row, row_index, opts, ctx)
        query = parse_query(row.get("query"), ctx)
        if row.get("tools") is None:
            raise RowError("tools_missing")
        tools_raw = parse_json_field(row.get("tools"), "tools", ctx)
        if not isinstance(tools_raw, list):
            raise RowError("tools_not_list", type(tools_raw).__name__)
        converted = [convert_tool(t, ctx, opts) for t in tools_raw]
        tools = [c[0] for c in converted]
        requiredness = [{"tool": t["function"]["name"], "params": c[1]}
                        for t, c in zip(tools, converted, strict=True)]
        names = [t["function"]["name"] for t in tools]
        if len(set(names)) != len(names):
            ctx.flag("duplicate_tool_names")
        if not tools:
            ctx.flag("no_tools_available")
        calls = parse_answers(row.get("answers"), ctx)
        extra = sorted(set(row) - {opts.id_field, "query", "tools", "answers"})
        if extra:
            ctx.note(f"extra_source_fields:{extra}")
    except RowError as e:
        return RejectedRecord(row_index, raw_id, e.reason, e.detail, raw_hash)

    teacher = teacher_from_id(original_id, opts.teacher_rules)
    if teacher == "unknown":
        ctx.flag("teacher_source_unknown")
    sample_id = (
        f"{opts.id_prefix}:{original_id}" if original_id is not None
        else f"{opts.id_prefix}:row{row_index}"
    )
    calls_dicts = [{"name": c.name, "arguments": c.arguments} for c in calls]
    normalized_hash = sha256_obj({"query": query, "tools": tools, "expected_calls": calls_dicts})
    return NormalizedRecord(
        sample_id=sample_id,
        source_dataset=opts.source_dataset,
        source_revision=opts.source_revision,
        original_id=original_id,
        id_origin=id_origin,
        source_row_index=row_index,
        teacher_source=teacher,
        query=query,
        tools=tools,
        expected_calls=calls,
        call_count=len(calls),
        raw_hash=raw_hash,
        normalized_hash=normalized_hash,
        metadata={
            # xLAM rows are ONE user turn. "multi-call" = several calls in ONE assistant
            # response (parallel/sequential), NOT a multi-turn conversation.
            "turn_structure": "single_turn",
            "call_structure": ("no_call" if not calls else
                               "single_call" if len(calls) == 1 else "multi_call_single_response"),
            "conversion_notes": ctx.notes,
            "validation_flags": ctx.flags,
            "validation_warnings": [],
            # per tool (aligned with ``tools``): {param: {state, declared, declared_sources,
            # inferred_optional_reasons, conflict}}
            "requiredness": requiredness,
            # original decoded tool declarations, unmodified (spec: preserve original fields)
            "source_tools": tools_raw,
        },
    )


def normalize_rows(
    rows: list[Any], opts: NormalizeOptions | None = None
) -> tuple[list[NormalizedRecord], list[RejectedRecord]]:
    """Normalize a whole source in ORIGINAL order. Duplicate original IDs are flagged and their
    sample_ids disambiguated with the source row index so sample_ids stay unique."""
    opts = opts or NormalizeOptions()
    records: list[NormalizedRecord] = []
    rejected: list[RejectedRecord] = []
    for i, row in enumerate(rows):
        out = normalize_row(row, i, opts)
        (records if isinstance(out, NormalizedRecord) else rejected).append(out)
    id_counts = Counter(r.original_id for r in records if r.original_id is not None)
    for r in records:
        if r.original_id is not None and id_counts[r.original_id] > 1:
            r.add_flag("duplicate_original_id")
            r.sample_id = f"{opts.id_prefix}:{r.original_id}@row{r.source_row_index}"
    return records, rejected
