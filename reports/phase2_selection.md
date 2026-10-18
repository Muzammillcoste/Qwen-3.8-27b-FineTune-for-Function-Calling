# Phase 2 — Candidate Subsets and Selection Audit

Descriptive analysis only. No training, no model weights (tokenizer files only), no quality score, no semantic-correctness claim. Framing: **Option A — data selection, diversity, coverage and quantity.** Option B (label noise) deferred.

## 1. Primary pool (TRAIN ∩ single-call ∩ non-duplicate)

| set | n | teacher | label_status | strict valid % |
| --- | --- | --- | --- | --- |
| train_single_call | 22476 | {'deepseek-v2-chat': 13147, 'mixtral-8x22b-instruct': 9329} | {'ambiguous': 183, 'invalid': 1, 'schema_conflict': 499, 'valid': 21793} | 97.78 |
| val_single_call | 2811 | {'deepseek-v2-chat': 1633, 'mixtral-8x22b-instruct': 1178} | {'ambiguous': 4, 'schema_conflict': 56, 'valid': 2751} | 98.01 |
| test_single_call | 2813 | {'deepseek-v2-chat': 1642, 'mixtral-8x22b-instruct': 1171} | {'ambiguous': 7, 'schema_conflict': 65, 'valid': 2741} | 97.69 |

- train row = eligible pool AFTER excluding 71 records whose full chat-template render exceeds 1536 tokens (their assistant target would be truncated); val/test rows are unfiltered.
- multi-call records preserved but unused (per split): {'train': 25143, 'val': 3144, 'test': 3142}. Multi-call = several calls in ONE assistant response; every xLAM record is a single user turn (no multi-turn data).
- pool groups: 17379; records in multi-member (near-)duplicate groups: 6564; largest group 72
- distinct called functions in pool: 2520; single-call TEST records whose function never appears in the pool: 2.92%
- rendered-length tertile cuts (train pool): [401, 635]

## 2. Final chat-template token lengths (real Qwen template)

- chat template sha256 `64f85b198065d0fb…`; prompt is exact token prefix of full render: 28171/28171; every expected call rendered verbatim: 28171/28171

| set | max_len | truncated | truncated % | prompt alone >= max_len |
| --- | --- | --- | --- | --- |
| train_pool | 1024 | 1540 | 6.83 | 1233 |
| train_pool | 1536 | 71 | 0.315 | 65 |
| train_pool | 2048 | 12 | 0.053 | 12 |
| val | 1024 | 164 | 5.834 | 130 |
| val | 1536 | 5 | 0.178 | 5 |
| val | 2048 | 1 | 0.036 | 1 |
| test | 1024 | 168 | 5.972 | 132 |
| test | 1536 | 13 | 0.462 | 11 |
| test | 2048 | 3 | 0.107 | 3 |

Right-truncation always cuts the assistant target (it is last). Records over the limit must be excluded from training or the limit raised; never train on cut targets.

## 3. Comparison tables

### Sources and tokens

| metric | pool | random_1k | schema_consistent_1k | diversity_1k | random_3k |
| --- | --- | --- | --- | --- | --- |
| records | 22476 | 1000 | 1000 | 1000 | 3000 |
| DeepSeek % | 58.49 | 58.5 | 58.5 | 58.5 | 58.53 |
| rendered tokens (sum) | 12495300 | 554547 | 553874 | 555718 | 1671748 |
| target tokens (sum) | 844921 | 37969 | 36960 | 39223 | 112167 |
| tokens mean | 555.9 | 554.5 | 553.9 | 555.7 | 557.2 |
| tokens p50 | 517 | 514 | 513 | 512 | 521 |
| tokens p90 | 924 | 914 | 903 | 925 | 932 |
| tokens max | 1535 | 1469 | 1469 | 1535 | 1510 |

### Difficulty proxies

| metric | pool | random_1k | schema_consistent_1k | diversity_1k | random_3k |
| --- | --- | --- | --- | --- | --- |
| available tools | {'t1': 6276, 't2-3': 8216, 't4+': 7984} | {'t1': 280, 't2-3': 363, 't4+': 357} | {'t1': 280, 't2-3': 363, 't4+': 357} | {'t1': 280, 't2-3': 363, 't4+': 357} | {'t1': 840, 't2-3': 1094, 't4+': 1066} |
| call args | {'a0-1': 10073, 'a2': 5824, 'a3+': 6579} | {'a0-1': 447, 'a2': 258, 'a3+': 295} | {'a0-1': 447, 'a2': 258, 'a3+': 295} | {'a0-1': 447, 'a2': 258, 'a3+': 295} | {'a0-1': 1346, 'a2': 775, 'a3+': 879} |
| length tertile | {'L0': 7464, 'L1': 7516, 'L2': 7496} | {'L0': 333, 'L1': 332, 'L2': 335} | {'L0': 333, 'L1': 332, 'L2': 335} | {'L0': 333, 'L1': 332, 'L2': 335} | {'L0': 998, 'L1': 1002, 'L2': 1000} |
| mean tools | 2.939 | 2.915 | 2.928 | 2.897 | 2.942 |
| mean args | 1.985 | 1.976 | 1.964 | 1.991 | 1.978 |
| list/dict arg % | 15.55 | 15.2 | 15.2 | 15.2 | 15.43 |

### Quality warnings (descriptive; not semantic correctness)

| metric | pool | random_1k | schema_consistent_1k | diversity_1k | random_3k |
| --- | --- | --- | --- | --- | --- |
| label_status | {'ambiguous': 183, 'invalid': 1, 'schema_conflict': 499, 'valid': 21793} | {'ambiguous': 8, 'schema_conflict': 30, 'valid': 962} | {'valid': 1000} | {'ambiguous': 11, 'schema_conflict': 27, 'valid': 962} | {'ambiguous': 28, 'schema_conflict': 58, 'valid': 2914} |
| strict valid % | 97.78 | 97.0 | 100.0 | 97.3 | 98.07 |
| schema-consistent % | 96.67 | 95.7 | 100.0 | 96.2 | 96.8 |
| type-conflict records | 499 | 30 | 0 | 27 | 58 |
| records with warnings | 197 | 11 | 0 | 12 | 31 |
| default-mismatch records % | 47.77 | 46.6 | 45.8 | 48.1 | 47.17 |
| UNKNOWN-param records % | 34.75 | 34.7 | 34.7 | 34.7 | 34.8 |

### Coverage and diversity

| metric | pool | random_1k | schema_consistent_1k | diversity_1k | random_3k |
| --- | --- | --- | --- | --- | --- |
| distinct functions | 2520 | 604 | 605 | 756 | 1185 |
| distinct arg signatures | 4154 | 661 | 659 | 777 | 1414 |
| distinct dedup groups | 17379 | 930 | 935 | 1000 | 2628 |
| function entropy (bits) | 10.022 | 8.924 | 8.925 | 9.266 | 9.526 |
| max records / function | 377 | 13 | 15 | 13 | 48 |
| test records w/ function seen % | 97.08 | 54.0 | 53.86 | 58.02 | 73.12 |
| query word types | 9430 | 2248 | 2266 | 4192 | 3871 |
| query type/token | 0.0338 | 0.1849 | 0.1862 | 0.3411 | 0.1035 |
| mean pairwise query Jaccard (lower = more diverse) | 0.0743 | 0.0738 | 0.0746 | 0.055 | 0.0736 |

### Balance (standardized mean difference; |SMD| < 0.1 = balanced)

| arm | reference | rendered_total | n_tools | n_args | n_params_called_tool | n_unknown_params_called_tool | n_structured_args |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_1k | vs_pool | -0.005 | -0.014 | -0.008 | 0.006 | -0.01 | 0.003 |
| random_1k | vs_random_1k | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| schema_consistent_1k | vs_pool | -0.008 | -0.006 | -0.018 | 0.014 | -0.007 | 0.003 |
| schema_consistent_1k | vs_random_1k | -0.003 | 0.008 | -0.01 | 0.008 | 0.003 | 0.0 |
| diversity_1k | vs_pool | -0.001 | -0.024 | 0.005 | 0.029 | 0.02 | -0.003 |
| diversity_1k | vs_random_1k | 0.004 | -0.011 | 0.012 | 0.023 | 0.03 | -0.006 |
| random_3k | vs_pool | 0.005 | 0.002 | -0.006 | 0.001 | -0.003 | 0.002 |
| random_3k | vs_random_1k | 0.01 | 0.016 | 0.002 | -0.005 | 0.007 | -0.001 |

- teacher share deviation vs pool: {'random_1k': 0.0001, 'schema_consistent_1k': 0.0001, 'diversity_1k': 0.0001, 'random_3k': 0.0004}

### Overlap between arms

| pair | intersection | Jaccard |
| --- | --- | --- |
| random_1k~schema_consistent_1k | 957 | 0.9175 |
| random_1k~diversity_1k | 46 | 0.0235 |
| random_1k~random_3k | 1000 | 0.3333 |
| schema_consistent_1k~diversity_1k | 42 | 0.0215 |
| schema_consistent_1k~random_3k | 1000 | 0.3333 |
| diversity_1k~random_3k | 130 | 0.0336 |

## 4. Schema-consistent arm: does it differ meaningfully from random_1k?

- pool schema-inconsistent: 748 (3.33%, 95% CI (3.1, 3.57)) by status {'schema_conflict': 499, 'ambiguous': 183, 'valid': 65, 'invalid': 1}
- inconsistent records in random_1k: 43; in schema_consistent_1k: 0
- records differing between the two arms: **43 (4.3%)**; gate threshold 5.0% -> meaningful: **False**
- Schema consistency is NOT evidence of higher semantic correctness; only the blinded human audit can test that.

## 5. Manifests

| arm | n | algorithm | sample_ids_sha256 | rendered tokens |
| --- | --- | --- | --- | --- |
| random_1k | 1000 | stratified_random_nested_in_3k | `47dde02a9206f174…` | 554547 |
| schema_consistent_1k | 1000 | stratified_random_same_order_skip_inconsistent | `cbcfffc1ecf6f291…` | 553874 |
| diversity_1k | 1000 | stratified_greedy_coverage | `4ef8b42aaab5eba6…` | 555718 |
| random_3k | 3000 | stratified_random | `907d17f2ddc61908…` | 1671748 |

Full ID lists: `reports/phase2_manifests/<arm>.json`. Selection params: 150 strata, seed 42.

## 6. Blinded semantic audit sample

- 148 unique items; strata (an item can serve several arms): {'arm:diversity_1k': 40, 'arm:schema_consistent_1k': 40, 'diagnostic:schema_conflict': 20, 'diagnostic:ambiguous': 10, 'arm:random_1k': 40}
- annotator sheet and hidden key in `data/audits/semantic_audit/` (gitignored; contains xLAM text). Protocol: `docs/SEMANTIC_AUDIT_PROTOCOL.md`.

## 7. Training-cost estimates (tokens measured; throughput ASSUMED)

| arm | n | steps/epoch (eff. batch 8) | tokens/epoch | hours/epoch @ assumed tok/s |
| --- | --- | --- | --- | --- |
| random_1k | 1000 | 125 | 554547 | {'150': 1.03, '300': 0.51, '600': 0.26} |
| schema_consistent_1k | 1000 | 125 | 553874 | {'150': 1.03, '300': 0.51, '600': 0.26} |
| diversity_1k | 1000 | 125 | 555718 | {'150': 1.03, '300': 0.51, '600': 0.26} |
| random_3k | 3000 | 375 | 1671748 | {'150': 3.1, '300': 1.55, '600': 0.77} |

- evaluation: 2813 single-call test records, mean prompt 505.2 tokens, mean reference completion 36.3 tokens; hours per model (batch 1) at assumed generation speed: {'8': 3.54, '15': 1.89, '25': 1.13}
