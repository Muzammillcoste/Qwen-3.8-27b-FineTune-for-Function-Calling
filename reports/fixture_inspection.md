# Fixture Inspection: miniature_xlam.jsonl

> **FIXTURE — SYNTHETIC STRUCTURAL TEST DATA, NOT REAL xLAM RECORDS.** Counts below describe the fixture only and are not research findings.

- Source: `miniature_xlam.jsonl`  
- Revision: `local-file`
- Source rows: 33 | normalized: 28 | rejected: 5 | unparseable lines: 1

## Field profile (parseable rows)

| field | present | missing | nulls | python types | JSON-encoded strings |
|---|---:|---:|---:|---|---:|
| `answers` | 32 | 0 | 0 | {'str': 31, 'list': 1} | 30 |
| `id` | 31 | 1 | 0 | {'int': 30, 'str': 1} | 0 |
| `query` | 31 | 1 | 0 | {'str': 31} | 0 |
| `tools` | 32 | 0 | 0 | {'str': 31, 'list': 1} | 30 |

## IDs (field `id`)

- missing: 1, unique: 26, duplicate ids: 1 [33659]
- min/max: 0 / 60005, gaps in range: 59980
- trustworthy for provenance: **False**

## Teacher source (from original IDs only)

NOT COMPUTED — ID field not verified as present, integer and unique.

## Calls and tools

- expected-call count distribution: {0: 1, 1: 24, 2: 2, 3: 1}
- available-tools count distribution: {1: 25, 2: 2, 3: 1}
- rows with >1 available tool: 3
- rows with empty answers (potential no-tool): 1

## Lengths (approximate; NOT Qwen tokens)

- query chars: {'min': 22, 'p50': 27.0, 'p95': 44, 'max': 58, 'mean': 31.8}
- query whitespace tokens: {'min': 3, 'p50': 5.0, 'p95': 8, 'max': 12, 'mean': 5.5}
- tools JSON chars: {'min': 155, 'p50': 294.0, 'p95': 510, 'max': 803, 'mean': 316.4}

## Malformed / rejected rows

- by reason: {'tools_malformed_json': 1, 'answers_malformed_json': 1, 'query_missing': 1, 'answer_item_malformed': 1, 'malformed_jsonl_line': 1}
  - row 6 id=33702: `tools_malformed_json` Expecting property name enclosed in double quotes: line 1 column 51 (char 50)
  - row 7 id=33703: `answers_malformed_json` Expecting value: line 1 column 37 (char 36)
  - row 8 id=33704: `query_missing` 
  - row 29 id=40017: `answer_item_malformed` index 0: {'arguments': {'order_id': 3}}
  - row 32 id=None: `malformed_jsonl_line` {this line is not valid json

## Validation flags (record counts per flag kind)

- `duplicate_original_id`: 2
- `call_args_invalid`: 2
- `teacher_source_unknown`: 2
- `unknown_type`: 1
- `call_unknown_tool`: 1
- `no_tool_call_label`: 1
- `suspicious_markup_in_tool_text`: 1
- `missing_original_id`: 1
- `param_missing_type`: 1
- `tool_missing_description`: 1

## Conversion notes

- `tools_was_json_string`: 27
- `answers_was_json_string`: 27
- `required_inferred_optional_from_default`: 2
- `tools_was_native_list`: 1
- `answers_was_native_list`: 1
- `answer_arguments_were_json_string`: 1
- `required_inferred_no_marker`: 1
- `any_type_unconstrained`: 1
- `id_coerced_from_string`: 1
- `required_coerced_from_string`: 1

## Task domain

Not inferred: no reliable domain classifier is used (spec E2 forbids faking one).

## First 10 normalized records (truncated)

### `fixture:0` teacher=`deepseek-v2-chat` calls=1

- query: [FIXTURE] What is the weather in Paris?
- tools: ['get_weather']
- expected_calls: `[{"arguments":{"city":"Paris"},"name":"get_weather"}]`
- flags: []

### `fixture:5` teacher=`deepseek-v2-chat` calls=2

- query: [FIXTURE] Weather in Oslo and in Lima, please.
- tools: ['get_weather']
- expected_calls: `[{"arguments":{"city":"Oslo"},"name":"get_weather"},{"arguments":{"city":"Lima"},"name":"get_weather"}]`
- flags: []

### `fixture:33658` teacher=`deepseek-v2-chat` calls=2

- query: [FIXTURE] Get order 42 and sum 1, 2 and 3.
- tools: ['get_order', 'sum_numbers']
- expected_calls: `[{"arguments":{"order_id":42},"name":"get_order"},{"arguments":{"numbers":[1,2,3]},"name":"sum_numbers"}]`
- flags: []

### `fixture:33659@row3` teacher=`mixtral-8x22b-instruct` calls=1

- query: [FIXTURE] Fetch order number 7.
- tools: ['get_order']
- expected_calls: `[{"arguments":{"order_id":7},"name":"get_order"}]`
- flags: ['duplicate_original_id']

### `fixture:33700` teacher=`mixtral-8x22b-instruct` calls=1

- query: [FIXTURE] Sum 10 and 20.
- tools: ['sum_numbers']
- expected_calls: `[{"arguments":{"numbers":[10,20]},"name":"sum_numbers"}]`
- flags: []

### `fixture:33701` teacher=`mixtral-8x22b-instruct` calls=1

- query: [FIXTURE] Order 99 details?
- tools: ['get_order']
- expected_calls: `[{"arguments":{"order_id":99},"name":"get_order"}]`
- flags: []

### `fixture:33705` teacher=`mixtral-8x22b-instruct` calls=1

- query: [FIXTURE] Register a callback.
- tools: ['register_cb']
- expected_calls: `[{"arguments":{"cb":"f"},"name":"register_cb"}]`
- flags: ['unknown_type:register_cb.cb:Callable[[int], int]']

### `fixture:40001` teacher=`mixtral-8x22b-instruct` calls=1

- query: [FIXTURE] Search for red shoes under 50.
- tools: ['search_items']
- expected_calls: `[{"arguments":{"price_max":50.0,"q":"red shoes"},"name":"search_items"}]`
- flags: []

### `fixture:40002` teacher=`mixtral-8x22b-instruct` calls=1

- query: [FIXTURE] Open a ticket titled Login bug.
- tools: ['create_ticket']
- expected_calls: `[{"arguments":{"title":"Login bug"},"name":"create_ticket"}]`
- flags: []

### `fixture:40003` teacher=`mixtral-8x22b-instruct` calls=1

- query: [FIXTURE] 3-day forecast for Rome.
- tools: ['weather.forecast.v2.get']
- expected_calls: `[{"arguments":{"city":"Rome","days":3},"name":"weather.forecast.v2.get"}]`
- flags: []

