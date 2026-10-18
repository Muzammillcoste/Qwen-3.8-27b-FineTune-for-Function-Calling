# Evaluation plan (Phase 2 design; nothing evaluated yet)

## 1. Internal held-out xLAM (primary)

| Item | Plan |
|---|---|
| Test set | **single-call TEST split** (group-aware, seed 42), sealed until the evaluation pipeline is frozen |
| Development | single-call VAL split only |
| Secondary (later) | multi-call test records (preserved, unused in the primary experiment) |
| Prompt | the same Qwen chat-template rendering as training (`format_qwen.render`, template sha256 in the manifests) |
| Decoding | `do_sample=False`, `max_new_tokens=256`, batch size and stop token `<|im_end|>` fixed across runs |
| Parsing | defensive parsing of `<tool_call>` blocks: JSON parse, no `eval`, no tool execution |
| Primary metric | exact structured-call accuracy (tool name + canonical argument key/values) |
| Secondary metrics | parse rate; tool-name accuracy; per-argument value accuracy; schema-valid predicted-call rate; hallucinated-tool rate; extra/missing-call rate; latency |
| Statistics | paired bootstrap 95% CI over test examples; McNemar on exact accuracy. With one training seed, these CIs cover test-sample uncertainty only. |
| Breakdowns | by teacher source, available-tool bucket, argument bucket and length tertile. Also report whether the called function was seen in the training arm. Measured coverage: only 2.92% of single-call test records call a function absent from the whole train pool, but 46% are unseen by `random_1k`, 42% by `diversity_1k` and 27% by `random_3k`. The internal test is therefore mostly in-distribution at the pool level; **truly unseen-tool generalization is measured mainly by BFCL.** |

**Scoring caveat.** Reference labels contain about 2% schema-conflict records (for example, a decimal in an `int` field). Score them against the reference value exactly as given. Also report accuracy with schema-conflict test records excluded, as a sensitivity check.

## 2. BFCL (external, single-turn)

The category files present on the HF dataset at pinned revision `61fc0608` (v3 naming) were enumerated from the official listing. They have **not** been downloaded.

| Category | Use | Reason |
|---|---|---|
| `simple` (Python) | **primary external** | single call, one available function: matches the training format |
| `multiple` | **primary external** | single call chosen from several functions: matches xLAM distractors |
| `live_simple`, `live_multiple` | **secondary external** | user-contributed, less contaminated, single call |
| `irrelevance`, `live_irrelevance` | **diagnostic only** | no-tool behaviour; **xLAM has 0 no-tool examples, so this is not a trained capability** |
| `parallel`, `parallel_multiple`, `live_parallel*` | excluded from the primary analysis | multi-call; possible follow-up with multi-call data |
| `multi_turn_*` | excluded | multi-turn; not in scope (xLAM is single-turn) |
| `java`, `javascript`, `sql`, `rest`, `exec_*`, `chatable` | excluded | language- or execution-specific, outside the training distribution |

### Protocol requirements (Phase 3)

- Install `bfcl-eval`. PyPI 2026.3.23 was observed; pin the version.
- Enumerate category names **from the installed package**. The package may bundle a newer data version (v4) than the HF v3 files. Use whatever the official evaluator uses and record the version.
- Use the **official AST evaluator** through a local model handler that runs the same Qwen template and decoding. If integration is not feasible, report internal metrics only and state that BFCL is future work. Never substitute a home-grown checker under the BFCL name.
- Label results **"BFCL subset (categories, version, N)"**. They are never an official leaderboard score.
- **Never train on BFCL, and never tune selection on BFCL.**
- **Contamination check.** Run exact and near-duplicate query matching (the same `dedup-v1` normalization) between the selected training arms and the BFCL evaluation items, and report the overlap. Public BFCL may still have been in Qwen's pretraining data; this is a stated limitation.

## 3. Evaluation compute

Generation cost may exceed training cost: a few thousand test prompts × 5–6 models. Estimates are in `reports/phase2_selection.md` §7. Options to decide in Phase 3, with **no model change**:

- batched generation with left padding; or
- a fixed, seeded, stratified evaluation subset of the test split for every model (for example 1,000 records), with the full test set kept for the final models only.

Any evaluation subset is chosen once, before any model is evaluated.
