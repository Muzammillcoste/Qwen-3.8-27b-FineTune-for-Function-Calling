# Dataset Inspection: xlam_function_calling_60k.json

- Source: `xlam_function_calling_60k.json`  
- Revision: `local-file`
- Source rows: 60000 | normalized: 60000 | rejected: 0 | unparseable lines: 0

## Field profile (parseable rows)

| field | present | missing | nulls | python types | JSON-encoded strings |
|---|---:|---:|---:|---|---:|
| `answers` | 60000 | 0 | 0 | {'str': 60000} | 60000 |
| `id` | 60000 | 0 | 0 | {'int': 60000} | 0 |
| `query` | 60000 | 0 | 0 | {'str': 60000} | 0 |
| `tools` | 60000 | 0 | 0 | {'str': 60000} | 60000 |

## IDs (field `id`)

- missing: 0, unique: 60000, duplicate ids: 0 []
- min/max: 0 / 59999, gaps in range: 0
- trustworthy for provenance: **True**

## Teacher source (from original IDs only)

{'deepseek-v2-chat': 33659, 'mixtral-8x22b-instruct': 26341}

## Calls and tools

- expected-call count distribution: {1: 28461, 2: 25422, 3: 4697, 4: 1056, 5: 187, 6: 94, 7: 20, 8: 12, 9: 12, 10: 19, 11: 4, 12: 6, 14: 2, 19: 1, 20: 2, 23: 1, 24: 1, 26: 1, 30: 1, 52: 1}
- available-tools count distribution: {1: 15410, 2: 12726, 3: 12655, 4: 13078, 5: 2427, 6: 1464, 7: 1108, 8: 1132}
- rows with >1 available tool: 44590
- rows with empty answers (potential no-tool): 0

## Lengths (approximate; NOT Qwen tokens)

- query chars: {'min': 16, 'p50': 100.0, 'p95': 220, 'max': 1330, 'mean': 111.9}
- query whitespace tokens: {'min': 1, 'p50': 17.0, 'p95': 39, 'max': 253, 'mean': 19.6}
- tools JSON chars: {'min': 253, 'p50': 1252.0, 'p95': 3160, 'max': 9344, 'mean': 1401.2}

## Malformed / rejected rows

- by reason: {}

## Validation flags (occurrences per flag kind)

- `default_type_mismatch`: 63361
- `call_args_invalid`: 1384
- `unknown_type`: 593
- `duplicate_tool_names`: 260

## Conversion notes

- `tools_was_json_string`: 60000
- `answers_was_json_string`: 60000
- `query_wrapped_in_literal_quotes`: 12

## Task domain

Not inferred: no reliable domain classifier is used (spec E2 forbids faking one).

## First 10 normalized records (truncated)

### `xlam:0` teacher=`deepseek-v2-chat` calls=2

- query: Where can I find live giveaways for beta access and games?
- tools: ['live_giveaways_by_type']
- expected_calls: `[{"arguments":{"type":"beta"},"name":"live_giveaways_by_type"},{"arguments":{"type":"game"},"name":"live_giveaways_by_type"}]`
- flags: []

### `xlam:1` teacher=`deepseek-v2-chat` calls=1

- query: I need to understand the details of the Ethereum blockchain for my cryptocurrency project. Can you fetch the details for 'ethereum'?
- tools: ['peers', 'web_chain_details']
- expected_calls: `[{"arguments":{"chain_slug":"ethereum"},"name":"web_chain_details"}]`
- flags: []

### `xlam:2` teacher=`deepseek-v2-chat` calls=1

- query: What is the T3MA for 'ETH/BTC' using a 1h interval and a time period of 14?
- tools: ['t3ma', 'stock_v2_get_profile']
- expected_calls: `[{"arguments":{"interval":"1h","symbol":"ETH/BTC","time_period":14},"name":"t3ma"}]`
- flags: ['default_type_mismatch:t3ma.v_factor:integer:float']

### `xlam:3` teacher=`deepseek-v2-chat` calls=1

- query: List titles originally aired on networks '1' and '8', released after 2010, sorted by release date in descending order.
- tools: ['get_animes', 'list_titles']
- expected_calls: `[{"arguments":{"network_ids":"1,8","release_date_start":20110101,"sort_by":"release_date_desc"},"name":"list_titles"}]`
- flags: ['default_type_mismatch:get_animes.year_greater:integer:str', 'default_type_mismatch:get_animes.year_less:integer:str', 'default_type_mismatch:get_animes.limit:integer:str', 'default_type_mismatch:get_animes.sort:boolean:str', 'default_type_mismatch:get_animes.offset:integer:str', 'default_type_mismatch:get_animes.year_equal:integer:str', 'default_type_mismatch:list_titles.limit:integer:str', 'default_type_mismatch:list_titles.page:integer:str', 'default_type_mismatch:list_titles.release_date_start:integer:str', 'default_type_mismatch:list_titles.release_date_end:integer:str']

### `xlam:4` teacher=`deepseek-v2-chat` calls=1

- query: Fetch the competitor standings for the recently concluded stage 98765.
- tools: ['stagecompetitorstandings']
- expected_calls: `[{"arguments":{"is_id":98765},"name":"stagecompetitorstandings"}]`
- flags: []

### `xlam:5` teacher=`deepseek-v2-chat` calls=1

- query: Fetch details for product 456789 with locale 'es_ES'.
- tools: ['get_product', 'product_id']
- expected_calls: `[{"arguments":{"is_id":456789,"locale":"es_ES"},"name":"product_id"}]`
- flags: ['default_type_mismatch:get_product.is_id:integer:str', 'default_type_mismatch:product_id.is_id:integer:str']

### `xlam:6` teacher=`deepseek-v2-chat` calls=2

- query: Identify the ID of 'Pulp Fiction' and search for torrents related to it.
- tools: ['get_id', 'search_torrents', 'basic_info']
- expected_calls: `[{"arguments":{"q":"Pulp Fiction"},"name":"get_id"},{"arguments":{"keywords":"Pulp Fiction","quantity":10},"name":"search_torrents"}]`
- flags: ['default_type_mismatch:search_torrents.quantity:integer:str', 'default_type_mismatch:search_torrents.page:integer:str']

### `xlam:7` teacher=`deepseek-v2-chat` calls=2

- query: Can you find the time zone for a Canadian postal code G2J and a latitude-longitude pair of 48.8567,2.3508?
- tools: ['time_zone_api']
- expected_calls: `[{"arguments":{"q":"G2J"},"name":"time_zone_api"},{"arguments":{"q":"48.8567,2.3508"},"name":"time_zone_api"}]`
- flags: []

### `xlam:8` teacher=`deepseek-v2-chat` calls=2

- query: Determine the peak of the sequence [44, 46, 48, 50, 52] and the light travel time for 25 light years.
- tools: ['find_peak_element', 'light_travel_time']
- expected_calls: `[{"arguments":{"nums":[44,46,48,50,52]},"name":"find_peak_element"},{"arguments":{"distance_in_light_years":25},"name":"light_travel_time"}]`
- flags: []

### `xlam:9` teacher=`deepseek-v2-chat` calls=1

- query: Is 'https://www.apple.com' available in the Wayback Machine on September 9, 2015?
- tools: ['availability']
- expected_calls: `[{"arguments":{"timestamp":"20150909","url":"https://www.apple.com"},"name":"availability"}]`
- flags: []

