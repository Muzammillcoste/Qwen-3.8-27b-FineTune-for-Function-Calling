# Selection methods (Phase 2, implemented: `select-v2`)

**Framing (approved 2026-10-18): Option A.** This is a study of data selection, diversity, coverage and quantity. Its arms are **not** claims about semantic label quality. There is no quality score. Option B (controlled label noise) is deferred and not implemented.

Code: `src/tooldistill/select.py`; orchestration in `src/tooldistill/phase2.py`; command `python scripts/audit_selection.py --config configs/research.yaml`. Results: `reports/phase2_selection.md`. Manifests: `reports/phase2_manifests/<arm>.json`.

## Primary pool

A record is eligible when all of the following hold. The rules are applied in this order, and the pool contains only TRAIN records (`build_pool` raises on any val/test record).

1. The record is in the TRAIN split (group-aware split from Phase 1, seed 42).
2. It is single-call: exactly one expected call. Multi-call records are preserved untouched for later work. "Multi-call" means several calls in one assistant response; xLAM has no multi-turn conversations.
3. It is not an exact duplicate.
4. Its full Qwen chat-template render is at most `max_rendered_tokens` (1,536). Longer records would have their assistant target truncated. This rule uses train lengths only.

Schema conflicts, ambiguous records and declaration issues stay in the pool. Nothing is repaired.

## Shared strata (all arms)

`stratum = teacher × tools_bucket × args_bucket × length_tertile × structured_args × tool_style`

| Feature | Definition |
|---|---|
| teacher | `deepseek-v2-chat` / `mixtral-8x22b-instruct` (from the original ID) |
| tools_bucket | available tools: 1 / 2–3 / 4+ |
| args_bucket | arguments in the call: 0–1 / 2 / 3+ |
| length_tertile | rendered-token tertile; cut points computed on the TRAIN pool |
| structured_args | does the call contain any list/dict argument value? |
| tool_style | does the called tool have any UNKNOWN-requiredness parameter (no default, no marker)? |

In `select-v1`, the last two features were absent. Balance diagnostics on the train pool then showed `diversity_1k` drifting:

- structured-argument records (SMD −0.36);
- UNKNOWN-parameter tools (SMD −0.49).

Both features were added before any training. No validation, test or BFCL information was used.

**Quotas.** Quotas are allocated in proportion to stratum size using largest remainder, with ties broken deterministically. The 1k quotas are capped by the 3k quotas, so `random_1k ⊂ random_3k`.

## Arms

| Arm | Algorithm |
|---|---|
| `random_3k` | Per stratum, a seeded random order (`select-v2|seed|random|stratum`); take the first q3 records. |
| `random_1k` | The first q1 records of the **same** order. It is nested in `random_3k`. |
| `schema_consistent_1k` | The same order, skipping records that are not schema-consistent. It differs from `random_1k` only where `random_1k` drew an inconsistent record. **Schema-consistent** means strict JSON-Schema valid, `label_status == valid` (no label warnings), and not in a conflicting-label input set. Declaration-only issues (`default_type_mismatch`, …) are deliberately ignored. This is NOT a semantic-correctness claim. |
| `diversity_1k` | Greedy coverage within strata (see below), using a separate seeded order. |

### Diversity features (exactly these)

The arm uses round-robin over strata. In each stratum it scans a window of 400 candidates in seeded order and takes the candidate with the lexicographically largest score:

1. **Unseen called function.** True if no selected record calls this function yet.
2. **Unseen argument signature.** True if no selected record has this `function(sorted argument names)` signature yet.
3. **Query word novelty.** The share of the query's lowercase non-numeric word types not yet covered.
4. **Earlier position** in the seeded order, used as the tie-break.

There is one hard constraint: **never two records from the same dedup group.** No embeddings, model or held-out data are used. The result is deterministic given the seed.

### Constraints and checks

- The teacher share of every arm is within ±0.02 of the pool, otherwise selection stops. All 1k arms have identical stratum counts.
- Selection stops with an explicit error if a stratum lacks enough candidates.
- The same seed and pool give identical ID lists and SHA256 hashes. A different seed changes the random arms but not the algorithm definitions.

## Binding constraints (approved 2026-10-18)

- Requiredness is three-state, and UNKNOWN is never penalized.
- `default_type_mismatch` and `schema_argument_type_conflict` describe the tool declaration or a declaration–label conflict. They are not confirmed semantic label errors.
- No record is modified or discarded because of ambiguous metadata, and no quality score is built on uncertain metadata.
- **Schema-arm gate.** `schema_consistent_1k` is kept only if it differs from `random_1k` in at least 5% of its examples. Otherwise removing the arm is recommended rather than running a weak experiment.

## Audit outputs

- Comparison tables in `reports/phase2_selection.md`:
  - quality warnings, coverage, diversity, difficulty, sources and tokens;
  - SMD balance vs the pool and vs `random_1k`;
  - pairwise overlap and Jaccard.
- The blinded semantic-audit sample and protocol: `docs/SEMANTIC_AUDIT_PROTOCOL.md`.
