# Phase 1 Data Audit — Salesforce xLAM Function Calling 60K

Factual description of the data only. No selection, no quality score, no training. Schema/label checks are syntactic; they do not establish semantic correctness.

## 1. Source and reproducibility

| item | value |
|---|---|
| repo_id | `Salesforce/xlam-function-calling-60k` |
| revision | `26d14ebfe18b1f7b524bd39b404b50af5dc97866` |
| file | `xlam_function_calling_60k.json` |
| row_count | `60000` |
| raw_sha256 | `4ef5c6f0dc552f2231f93f5853a9ef431e9e806d7aa514d0f6b615606ce576c6` |
| downloaded_utc | `2026-10-18T16:04:02+00:00` |
| license | `CC-BY-4.0` |
| code commit | `uncommitted` |
| normalized.jsonl sha256 | `1ea2824da323291d1643fac7158783abd06f01028892134d9ce44ad966af9774` |

## 2. Totals and parse failures

- source rows: **60000**; normalized: **60000**; rejected (parse failures): **0**
- rejected by reason: none
- exact duplicates excluded from eligible pool: 400; eligible: 59600

## 3. IDs and teacher source (DeepSeek vs Mixtral)

- id field trustworthy: **True** (missing 0, duplicates 0, min 0, max 59999, gaps 0)
- card check computed: True, **match: True** — Teacher-source counts match the dataset card.

| teacher | records | share |
|---|---:|---:|
| deepseek-v2-chat | 33659 | 56.10% |
| mixtral-8x22b-instruct | 26341 | 43.90% |

## 4. Requiredness (parameter level, three-state)

- parameters: 362482; records with ≥1 UNKNOWN parameter: 18348 (30.58%)

| state | params | share |
|---|---:|---:|
| REQUIRED | 0 | 0.00% |
| OPTIONAL | 270314 | 74.57% |
| UNKNOWN | 92168 | 25.43% |

Evidence (a parameter can have several inferred reasons):

| evidence | params |
|---|---:|
| inferred:default_present | 260944 |
| inferred:optional_suffix | 114906 |
| no_evidence | 92168 |
| inferred:default_qualifier | 98 |

- contradictory metadata: none found

## 5. Label validity: confirmed invalid vs schema conflict vs ambiguous

`invalid` = confirmed structural label error (unknown tool, spurious argument, missing DECLARED-required argument, enum violation). `schema_conflict` = only declared-type vs supplied-value disagreements (`schema_argument_type_conflict`, e.g. a decimal in an `int` field): strictly schema-invalid, but NOT a confirmed semantic label error. `ambiguous` = warnings only. Records are kept either way.

- strict JSON-Schema valid: 58644 (97.74%)

| label_status | records | share |
|---|---:|---:|
| valid | 58179 | 96.97% |
| ambiguous | 465 | 0.78% |
| schema_conflict | 1351 | 2.25% |
| invalid | 5 | 0.01% |
| no_call | 0 | 0.00% |

By teacher:

| teacher | valid | ambiguous | schema_conflict | invalid | no_call |
|---|---:|---:|---:|---:|---:|
| deepseek-v2-chat | 32836 (97.55%) | 219 (0.65%) | 604 (1.79%) | 0 (0.00%) | 0 (0.00%) |
| mixtral-8x22b-instruct | 25343 (96.21%) | 246 (0.93%) | 747 (2.84%) | 5 (0.02%) | 0 (0.00%) |

- `call_args_invalid` by JSON-Schema keyword: {'additionalProperties': 5}

Warnings (ambiguous channel):

| warning | occurrences | records |
|---|---:|---:|
| missing_arg_unknown_requiredness | 276 | 275 |
| arg_type_unverifiable | 215 | 215 |
| duplicate_identical_calls | 38 | 38 |

## 6. All flags by category

| flag | category | occurrences | records |
|---|---|---:|---:|
| default_type_mismatch | schema_issue | 63361 | 27561 |
| schema_argument_type_conflict | declaration_label_conflict | 2595 | 1351 |
| unknown_type | schema_issue | 593 | 593 |
| exact_duplicate | dedup | 400 | 400 |
| duplicate_tool_names | schema_issue | 260 | 260 |
| conflicting_labels_same_input | dedup | 87 | 87 |
| call_args_invalid | label_error | 5 | 5 |

### Default-type mismatches (schema issue; separate from label correctness)

| expected / got | params |
|---|---:|
| expected integer / got str | 52318 |
| expected boolean / got str | 2782 |
| expected integer / got float | 2745 |
| expected array / got float | 1268 |
| expected number / got bool | 1242 |
| expected array / got str | 590 |
| expected array / got int | 526 |
| expected string / got int | 521 |
| expected string / got bool | 460 |
| expected number / got str | 398 |
| expected integer / got bool | 397 |
| expected boolean / got int | 72 |
| expected integer / got list | 28 |
| expected string / got list | 14 |

- unknown type strings (top 20): {'Callable[[float], float]': 593}

## 7. Structure

- expected-call count by teacher: {'deepseek-v2-chat': {1: 16708, 2: 14854, 3: 1692, 4: 346, 5: 30, 6: 18, 7: 4, 8: 1, 10: 2, 11: 2, 23: 1, 30: 1}, 'mixtral-8x22b-instruct': {1: 11753, 2: 10568, 3: 3005, 4: 710, 5: 157, 6: 76, 7: 16, 8: 11, 9: 12, 10: 17, 11: 2, 12: 6, 14: 2, 19: 1, 20: 2, 24: 1, 26: 1, 52: 1}}
- available-tools count: {1: 15410, 2: 12726, 3: 12655, 4: 13078, 5: 2427, 6: 1464, 7: 1108, 8: 1132}

## 8. Duplicates and near-duplicates

- algorithm `dedup-v1`, params {'group_same_query_text': True, 'near_duplicate_jaccard': 0.8, 'oversized_blocks': []}
- exact duplicates: **400**; groups: 52742; largest group: 72
- union edges by type: {'exact': 400, 'near_duplicate': 5142, 'same_original_id': 0, 'same_query_text': 1716}
- group size histogram (10 = ≥10): {1: 50062, 2: 1773, 3: 443, 4: 148, 5: 77, 6: 41, 7: 24, 8: 13, 9: 17, 10: 144}
- input sets with conflicting labels: 37
- exact-duplicate examples:
  - xlam:851 == xlam:480: List the top albums globally for the week ending March 31, 2023.
  - xlam:934 == xlam:864: Can you provide the hourly forecast for New York City tomorrow?
  - xlam:1074 == xlam:29: Is the year 2000 a leap year?
- near-duplicate (2-member group) examples:
  - xlam:35315: Project the growth of an investment with an initial amount of $20,000, an annual addition of $3,000, over a...
    xlam:50831: Project the growth of an investment with an initial amount of $10,000, an annual addition of $2,000, over a...
  - xlam:9451: Fetch detailed metadata for the TikTok video at 'https://www.tiktok.com/@charlidamelio/video/71374239659825...
    xlam:19956: Fetch metadata for the TikTok video at https://www.tiktok.com/@charlidamelio/video/7137423965982592302.
  - xlam:46877: What are the results of rolling 4 regular dice? What games are associated with E-Sports event with ID 123?
    xlam:55161: What are the results of rolling 4 regular dice? What games are associated with E-Sports event with ID 134?
- largest groups:
  - grp:7938dd712d6b7ce6 size 72: ['Find all pairs of integers in the list [1, 2, 3, 4, 5] that sum up to 5.', 'Find all pairs of integers in the list [3, 7, 1, 9, 2, 6] that sum up to 10.', 'Find all pairs of integers in the list [1, 2, 3, 4, 5] that sum up to 5.', 'Identify all pairs of numbers in the list [1, 2, 3, 4, 5] that sum up to 5.']
  - grp:b1656b01638244fd size 70: ['Find the next greater element for each number in the list [4, 5, 2, 25].', 'What is the next greater element for each element in the list [3, 1, 4, 1, 5, 9]?', 'Find the next greater element for each number in the list [1, 3, 2, 4].', 'Find the next greater element for each number in the list [5, 3, 2, 10, 6, 9, 8]']
  - grp:d7f2c527d1a410b6 size 68: ['Calculate the area of a triangle with a base of 10 units and a height of 5 units.', 'Calculate the area of a triangle with a base of 10 units and a height of 5 units.', 'Determine the area of a triangle with a base of 7.5 cm and a height of 3.2 cm.', 'What is the area of a triangle with a base of 12.5 cm and a height of 8 cm?']

## 9. Token lengths

Method: `qwen_tokenizer_json_raw_segments_v1 (Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c0, tokenizer.json)`. Raw segments only — chat-template overhead NOT included; totals are lower bounds of rendered length.

| segment | min | p50 | p90 | p95 | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| query | 4 | 28 | 55 | 67 | 98 | 468 | 32.2 |
| tools | 70 | 339 | 704 | 865 | 1110 | 2556 | 378.7 |
| calls | 14 | 48 | 86 | 102 | 149 | 2287 | 53.1 |
| total | 98 | 427 | 798 | 951 | 1205 | 2598 | 463.9 |

- records with total above N tokens: {'512': '21952 (36.59%)', '1024': '1897 (3.16%)', '1536': '100 (0.17%)', '2048': '22 (0.04%)', '4096': '0 (0.00%)'}
- total by teacher: {'deepseek-v2-chat': {'n': 33659, 'min': 98, 'p50': 428, 'p90': 828, 'p95': 986, 'p99': 1239, 'max': 2598, 'mean': 470.6}, 'mixtral-8x22b-instruct': {'n': 26341, 'min': 98, 'p50': 425, 'p90': 769, 'p95': 901, 'p99': 1142, 'max': 2485, 'mean': 455.5}}
- total by label_status: {'ambiguous': {'n': 465, 'min': 148, 'p50': 448, 'p90': 957, 'p95': 995, 'p99': 1131, 'max': 1275, 'mean': 502.8}, 'invalid': {'n': 5, 'min': 549, 'p50': 613, 'p90': 615, 'p95': 615, 'p99': 615, 'max': 1716, 'mean': 820.4}, 'schema_conflict': {'n': 1351, 'min': 110, 'p50': 579, 'p90': 1022, 'p95': 1186, 'p99': 1543, 'max': 2190, 'mean': 622.1}, 'valid': {'n': 58179, 'min': 98, 'p50': 424, 'p90': 789, 'p95': 942, 'p99': 1193, 'max': 2598, 'mean': 459.9}}

## 10. Splits (group-aware, seed 42) and leakage audit

| split | n | teacher | calls | label_status |
|---|---:|---|---|---|
| train | 47690 | {'deepseek-v2-chat': 26698, 'mixtral-8x22b-instruct': 20992} | {'multi': 25143, 'single': 22547} | {'valid': 46219, 'schema_conflict': 1071, 'ambiguous': 395, 'invalid': 5} |
| val | 5955 | {'deepseek-v2-chat': 3320, 'mixtral-8x22b-instruct': 2635} | {'multi': 3144, 'single': 2811} | {'valid': 5792, 'schema_conflict': 130, 'ambiguous': 33} |
| test | 5955 | {'deepseek-v2-chat': 3328, 'mixtral-8x22b-instruct': 2627} | {'single': 2813, 'multi': 3142} | {'valid': 5788, 'schema_conflict': 141, 'ambiguous': 26} |

- leakage audit ok: **True** 

## 11. Examples for every warning/error category

### `arg_type_unverifiable` (label_warning)

- `xlam:380` — `arg_type_unverifiable:numerical_derivative.function`  
  query: Estimate the derivative of the function f(x) = x^2 at x = 3 using numerical methods.  
  [{"name": "numerical_derivative", "arguments": {"function": "lambda x: x ** 2", "x": 3}}]
- `xlam:439` — `arg_type_unverifiable:numerical_derivative.function`  
  query: Estimate the derivative of the function f(x) = 3x^3 at x = 2. Also, calculate the monthly mortgage payment for a loan of $150,000 with an...  
  [{"name": "numerical_derivative", "arguments": {"function": "lambda x: 3 * x ** 3", "x": 2}}, {"name": "monthly_mortgage_payment", "arguments": {"annual_inte...
- `xlam:525` — `arg_type_unverifiable:numerical_derivative.function`  
  query: Calculate the derivative of the function h(x) = 2x^4 at x = 1, and find the average of these numbers: 2, 4, 6, 8.  
  [{"name": "numerical_derivative", "arguments": {"function": "lambda x: 2 * x ** 4", "x": 1}}, {"name": "average", "arguments": {"numbers": [2, 4, 6, 8]}}]

### `call_args_invalid` (label_error)

- `xlam:41175` — `call_args_invalid:web_search:additionalProperties`  
  query: Search for the top 10 recipes for vegetarian pasta and find information about the postal code '10001' using the Postali API.  
  call[0] web_search: Additional properties are not allowed ('limit', 'query', 'related_keywords' were unexpected)
- `xlam:43169` — `call_args_invalid:timezone:additionalProperties`  
  query: Can you tell me the timezone for the location with coordinates (40.7128° N, 74.0060° W), and provide me with a compact JSON response?  
  call[0] timezone: Additional properties are not allowed ('c', 'lat', 'lon' were unexpected)
- `xlam:51204` — `call_args_invalid:web_search:additionalProperties`  
  query: Search for the latest news about artificial intelligence and find information about the postal code '94043' using the Postali API.  
  call[0] web_search: Additional properties are not allowed ('limit', 'query', 'related_keywords' were unexpected)

### `conflicting_labels_same_input` (dedup)

- `xlam:1259` — `conflicting_labels_same_input`  
  query: Can you fetch the treasure of the day without a JSONP callback?  
  call[0] today: None is not of type 'string'
- `xlam:2028` — `conflicting_labels_same_input`  
  query: Retrieve all dog breeds from Japan and the details of the Bulldog breed.
- `xlam:2614` — `conflicting_labels_same_input`  
  query: Reverse the phrase 'Hello World' and count the frequency of each word in the sentence 'The quick brown fox jumps over the lazy dog'.

### `default_type_mismatch` (schema_issue)

- `xlam:2` — `default_type_mismatch:t3ma.v_factor:integer:float`  
  query: What is the T3MA for 'ETH/BTC' using a 1h interval and a time period of 14?
- `xlam:3` — `default_type_mismatch:get_animes.year_greater:integer:str`  
  query: List titles originally aired on networks '1' and '8', released after 2010, sorted by release date in descending order.
- `xlam:5` — `default_type_mismatch:get_product.is_id:integer:str`  
  query: Fetch details for product 456789 with locale 'es_ES'.

### `duplicate_identical_calls` (label_warning)

- `xlam:2833` — `duplicate_identical_calls`  
  query: I want to check the login status for 'JaneSmith' with password 'securepass456' using all available login functions.  
  [{"name": "loginuser", "arguments": {"password": "securepass456", "username": "JaneSmith"}}, {"name": "loginuser", "arguments": {"password": "securepass456",...
- `xlam:8266` — `duplicate_identical_calls`  
  query: I need to log in with 'user3' and 'userpass3' on both systems, and fetch product 66666 details.  
  [{"name": "loginuser", "arguments": {"password": "userpass3", "username": "user3"}}, {"name": "loginuser", "arguments": {"password": "userpass3", "username":...
- `xlam:15947` — `duplicate_identical_calls`  
  query: Log in with 'demo' and 'demopass' on both, and get product 88888's info.  
  [{"name": "loginuser", "arguments": {"password": "demopass", "username": "demo"}}, {"name": "loginuser", "arguments": {"password": "demopass", "username": "d...

### `duplicate_tool_names` (schema_issue)

- `xlam:81` — `duplicate_tool_names`  
  query: Could you provide me with the details of the top 5 hospitals in Douala, Cameroon, and the top 5 banks in Dar es Salaam, Tanzania? Also, c...
- `xlam:136` — `duplicate_tool_names`  
  query: Calculate the distance in miles between Kampala (latitude 0.3163, longitude 32.5825) and Jinja (latitude 0.4463, longitude 33.2041).  
  call[0] measure_distance: 0.3163 is not of type 'integer'; call[0] measure_distance: 0.4463 is not of type 'integer'
- `xlam:152` — `duplicate_tool_names`  
  query: Calculate the distance between New York (40.7128° N, 74.0060° W) and London (51.5074° N, 0.1278° W) in miles.  
  call[0] measure_distance: 40.7128 is not of type 'integer'; call[0] measure_distance: 51.5074 is not of type 'integer'

### `exact_duplicate` (dedup)

- `xlam:851` — `exact_duplicate`  
  query: List the top albums globally for the week ending March 31, 2023.
- `xlam:934` — `exact_duplicate`  
  query: Can you provide the hourly forecast for New York City tomorrow?
- `xlam:1074` — `exact_duplicate`  
  query: Is the year 2000 a leap year?

### `missing_arg_unknown_requiredness` (label_warning)

- `xlam:65` — `missing_arg_unknown_requiredness:calculate_electric_field.permitivity`  
  query: What is the electric field intensity at 3 meters from a 2 Coulomb charge, and at 4 meters from a 3 Coulomb charge?  
  [{"name": "calculate_electric_field", "arguments": {"charge": 2, "distance": 3}}, {"name": "calculate_electric_field", "arguments": {"charge": 3, "distance":...
- `xlam:218` — `missing_arg_unknown_requiredness:calculate_electric_field.permitivity`  
  query: Calculate the electric field produced by a charge of 1 Coulomb at a distance of 1 meter.  
  [{"name": "calculate_electric_field", "arguments": {"charge": 1, "distance": 1}}]
- `xlam:279` — `missing_arg_unknown_requiredness:calculate_electric_field.permitivity`  
  query: Calculate the electric field at 10 meters from a charge of 2 Coulombs.  
  [{"name": "calculate_electric_field", "arguments": {"charge": 2, "distance": 10}}]

### `schema_argument_type_conflict` (declaration_label_conflict)

- `xlam:53` — `schema_argument_type_conflict:is_valid_sudoku.board:expected=string:got=integer`  
  query: Check if a Sudoku board is valid and draw 1 card. The board has numbers filled in a way that each row, column, and 3x3 box should have nu...  
  call[0] is_valid_sudoku: 5 is not of type 'string'; call[0] is_valid_sudoku: 3 is not of type 'string'
- `xlam:130` — `schema_argument_type_conflict:ec.penalty:expected=integer:got=number`  
  query: Generate a QR code for the URL 'https://www.example.com' with a size of 7, and analyze the password 'Secure123' with a penalty of 0.3 and...  
  call[1] ec: 0.3 is not of type 'integer'
- `xlam:136` — `schema_argument_type_conflict:measure_distance.lat1:expected=integer:got=number`  
  query: Calculate the distance in miles between Kampala (latitude 0.3163, longitude 32.5825) and Jinja (latitude 0.4463, longitude 33.2041).  
  call[0] measure_distance: 0.3163 is not of type 'integer'; call[0] measure_distance: 0.4463 is not of type 'integer'

### `unknown_type` (schema_issue)

- `xlam:280` — `unknown_type:numerical_derivative.function:Callable[[float], float]`  
  query: If the initial bacterial population is 1000 and the growth rate is 0.02 per minute, what will the population be after 1 hour, assuming a ...
- `xlam:380` — `unknown_type:numerical_derivative.function:Callable[[float], float]`  
  query: Estimate the derivative of the function f(x) = x^2 at x = 3 using numerical methods.
- `xlam:439` — `unknown_type:numerical_derivative.function:Callable[[float], float]`  
  query: Estimate the derivative of the function f(x) = 3x^3 at x = 2. Also, calculate the monthly mortgage payment for a loan of $150,000 with an...

