# Dataset card: ToolDistill-Lab derived xLAM splits

**Status: NOT YET PRODUCED.** The full xLAM data has not been downloaded, because access is gated and Phase 1 is not approved. Every count below is either taken from the upstream card (marked *card*) or still pending (marked *TBD*). Nothing here has been measured on real data yet.

## Source

| Item | Value |
|---|---|
| Upstream | Salesforce/xlam-function-calling-60k |
| Revision | `26d14ebfe18b1f7b524bd39b404b50af5dc97866` (observed 2026-10-18; confirm in Phase 1) |
| File | `xlam_function_calling_60k.json` (card config `dataset`, split `train`) |
| License | CC BY 4.0, gated; research use only per card |
| Rows | 60,000 (*card*) / TBD (measured) |
| Teacher mix | DeepSeek-V2-Chat ids 0–33658 = 33,659; Mixtral-8x22B-Instruct = 26,341 (*card*) / TBD |
| Card correctness claim | >95% correct on a 600-sample human evaluation (*card*). This is a sampling estimate, not per-row truth. |

## Upstream structure (from card; to be verified)

- The fields `query`, `tools` and `answers` are stored as **JSON strings** (card: "can be parsed with `json.loads`").
- `tools` holds items with `name`, `description` and `parameters`. In the card example, each parameter has `type` (`int`, `list`, `float`...), `description` and a boolean `required`.
- `answers` holds items with `name` and `arguments`. There can be multiple calls per row.
- **ID field:** the card text refers to "id 0 to id 33658", but the card's example object shows only `query`, `tools` and `answers`, with **no `id` key**. Phase 1 must confirm the real ID field before teacher provenance is trusted. The setting `configs/data.yaml: ids.allow_source_row_index_as_id` defaults to `false`. It may only be enabled by a human after verifying that the raw file order is the original id order.

### Observed on the first 100 rows (ids 0–99, authenticated stream, 2026-10-18)

- Row keys are `answers`, `id`, `query`, `tools`. **`id` is present**, int, unique and contiguous.
- `tools` and `answers` are JSON strings in 100/100 rows. `query` is plain text.
- Parameters have `{type, description[, default]}`. There is **no `required` key**, which contradicts the card example. Types use Python-style strings such as `str, optional` and `List[int]`.
- This sample covers only DeepSeek and is not representative of the full data.

## Derived artifacts (Phase 1, measured 2026-10-18)

The raw file has SHA256 `4ef5c6f0dc552f2231f93f5853a9ef431e9e806d7aa514d0f6b615606ce576c6` and 60,000 rows. It is read-only under `data/raw/<revision>/`. The full audit is in `reports/phase1_audit.md`.

| Artifact | Count |
|---|---|
| normalized records | 60,000 (all IDs 0–59999 present, unique) |
| rejected rows | 0 |
| teacher mix | DeepSeek-V2-Chat 33,659 / Mixtral-8x22B 26,341 (**matches card**) |
| exact duplicates excluded | 400 → 59,600 eligible |
| groups / largest group | 52,742 / 72 (template families, e.g. "find all pairs that sum to N") |
| train / val / test records | 47,690 / 5,955 / 5,955 (leakage audit passes) |
| single / multi-call (all) | 28,461 single; 31,539 multi (max 52 calls) |
| no-call records | 0. **xLAM has no no-tool examples.** |
| requiredness (362,482 params) | REQUIRED 0, OPTIONAL 270,314 (inferred), UNKNOWN 92,168. **No parameter declares `required`.** |
| label status | valid 58,179 (96.97%), ambiguous 465 (0.78%), invalid 1,356 (2.26%) |
| default_type_mismatch | 63,361 params in 27,561 records (schema issue, not label) |
| tokens (Qwen, raw segments) | total p50 427, p95 951, p99 1205, max 2598; 3.16% > 1024 |

## Exclusions policy

- Rows that cannot be parsed are rejected. Each rejection is logged in `rejected.jsonl` with a reason.
- Exact duplicates are excluded. Each exclusion is logged in `excluded_exact_duplicates.jsonl`, with the canonical row kept.
- Rows with validation flags are **kept and flagged** at this stage. Eligibility rules, such as a single-call restriction, are decided after Phase 1 inspection and before selection, and they are documented here.
