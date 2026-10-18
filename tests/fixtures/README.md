# Test fixtures

`miniature_xlam.jsonl` is **synthetic structural test data. It is not real xLAM data.** Every query starts with `[FIXTURE]`. IDs are chosen only to exercise provenance boundaries: 0, 33658, 33659, an id above 59999, a missing id, a duplicate id and a digit-string id.

The file has 33 lines:

- 32 JSON rows in xLAM-like shape:
  - Most rows store `tools` and `answers` as JSON strings.
  - One row stores them as native lists.
- 1 deliberately unparseable line.

Regenerate it with:

```bash
python tests/fixtures/build_miniature_xlam.py
```

The rows cover:

- Single, double and triple calls.
- Malformed tools JSON, malformed answers JSON, a missing query and a nameless answer.
- Unknown types.
- xLAM-style type strings: `str, optional`, `List[int]`, `Dict[str, Any]`, `Union`, `Optional` and defaults.
- JSON-Schema-form parameters.
- Dotted tool names.
- Hallucinated label tools, missing required arguments, and wrong-type or extra arguments.
- Empty answers.
- One exact duplicate, one near duplicate, and the same query text with a different label.
- Parameters literally named `type` and `properties`.
- Chat-template markup inside a description.
- A parameter with no type.
- `required` given as the string `"true"`.
- A tool with no description.
