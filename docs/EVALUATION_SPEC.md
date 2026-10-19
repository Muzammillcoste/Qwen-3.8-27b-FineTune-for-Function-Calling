# Evaluation specification: `eval-v2` (FROZEN 2026-10-19)

This spec was frozen **before** any experimental arm was evaluated. It was designed using only the 100-example Phase 3A development sample from the **validation** split, and no test data. Any later change requires a new version number and the researcher's approval, and must be reported as a deviation.

Code: `src/tooldistill/evaluate.py` (scoring), `src/tooldistill/stats.py` (statistics), `src/tooldistill/compare.py` (run comparison; it rescores raw outputs, so every run is scored by the same frozen code).

## 1. Evaluation sets

| Set | n | Use | Location |
|---|---:|---|---|
| `val_sample_100` | 100 | **Development only.** It was inspected to design this spec. | Kaggle bundle |
| `val_eval_400` | 400 | **Initial comparison** of base / `random_1k` / `diversity_1k`. It is disjoint from the dev 100. | Kaggle bundle; manifest `reports/eval_manifests/val_eval_400.json` |
| `test_eval_1000` | 1,000 | **Final comparison.** It is **sealed**: not in the Kaggle bundle, and `eval_test` is refused while `configs/eval.yaml: test.sealed: true`. | `data/eval_sealed/` (local, gitignored); manifest `reports/eval_manifests/test_eval_1000.json` |

Construction rules, shared by both sets:

- Records are single-call and non-duplicate.
- The rendered prompt plus target is at most 1,536 tokens, the same exclusion used for training.
- **One record per dedup group**, so correlated near-duplicates don't inflate N.
- **Group-disjoint from TRAIN** (asserted).
- Stratified using the shared select-v2 strata, seed 42.

The sample-ID SHA256 and file SHA256 are in each manifest.

## 2. Metrics

| Metric | Role | Definition |
|---|---|---|
| **Strict exact structured-call match** | **PRIMARY** | Parse succeeds, and the multiset of (name, arguments) equals the reference after canonicalization. Canonicalization is JSON numeric equality only (`5 == 5.0`). Strings are compared exactly: `"BTCUSD" ≠ "BTCUSDT"`, `"Ford" ≠ "ford"`, `"US" ≠ "United States"`. Numerically different values are never equal: `34.05 ≠ 34.0522`. |
| **Schema-aware structured match** | secondary | Same as strict, except that an argument equal to its parameter's **declared default** counts as equivalent to omitting it, **only if that default is valid under the parameter's own declared type** (and agrees across same-named tools). Never equivalent: a malformed default (`"10"` for an integer); `null` vs omission; equivalence that comes only from the description text (e.g. "Defaults to False" with no schema default); any coercion or case-folding. |
| Tool-name accuracy | secondary | The multiset of predicted names equals the reference names. |
| Argument value accuracy | secondary | For name-matched single calls: correct reference arguments ÷ all reference arguments (pooled). |
| Parse success | secondary | At least one well-formed `<tool_call>` block and no malformed block. JSON only; nothing is ever `eval`-ed or executed. |
| Schema-valid prediction, hallucinated-tool rate, text outside blocks | diagnostic | — |

On the Phase 3A dev sample, the schema-aware rule changed exactly one item (an `outputsize: 30` that equals a valid default). It is deliberately narrow.

## 3. Statistics (paired: same examples for every model)

- **Decision test:** two-sided **exact McNemar** on discordant pairs, for the primary metric.
- **Descriptive interval:** a **paired bootstrap 95% CI** of the accuracy difference: 10,000 resamples, seed 42, percentile method. It is unreliable when there are few discordant pairs. On the dev sample, 5 discordant pairs gave a CI excluding 0 while McNemar p = 0.0625, so **the CI never overrides McNemar**.
- **Single-model accuracy:** Wilson 95% interval.
- **Primary contrast:** `diversity_1k − random_1k` on `val_eval_400`. Contrasts against the base are secondary.
- **One training seed.** All intervals describe evaluation-sample uncertainty only, not training-run variance. Effects must not be described as causal or general without additional seeds.

## 4. Coverage analysis (separates function coverage from other diversity effects)

Every evaluation item is categorised by whether its reference function appears in each arm's training set:

| Category | val_eval_400 | test_eval_1000 |
|---|---:|---:|
| seen_by_both | 151 | 387 |
| seen_by_random_1k_only | 52 | 124 |
| seen_by_diversity_1k_only | 60 | 162 |
| seen_by_neither | 137 | 327 |

These counts were computed before any arm was evaluated.

- Accuracy and `diversity − random` (with McNemar) are reported per category.
- A diversity gain concentrated in `seen_by_diversity_1k_only` suggests a **coverage** effect.
- A gain in `seen_by_both` or `seen_by_neither` suggests **other** effects, such as redundancy reduction or generalization.

Category cells are small, so these results are exploratory.

## 5. Mismatch review rubric (frozen before new models are evaluated)

Failures are classified for interpretation. **Dataset labels are never rewritten**, and scores always use the original labels.

| Class | Rule |
|---|---|
| **Definite model mistake** | The reference is consistent with the query and tool description, and the prediction is not: wrong value or ordering, ignores a value given in the query, invalid identifier format, missing `<tool_call>` tags. |
| **Possible label error** | The reference itself is unsupported or invalid: an invented ID or placeholder not derivable from the query, a code expression instead of a value, or a format contradicting the tool's documented convention. |
| **Genuinely ambiguous** | The query or tool under-specifies the answer and both forms are defensible: naming, casing or code conventions, precision of world-knowledge values, explicit vs omitted optional arguments, how free text is split across parameters. |

Applied to the 18 base-model failures on the dev sample, the counts were 5 definite / 4 possible label / 9 ambiguous (`reports/phase3a_base_mismatch_review.md`). This classification is provisional, by Claude, and needs the researcher's confirmation.

**Interpretation:** strict exact match understates semantic correctness, and that applies equally to every arm. Comparisons are paired, so this mostly cancels in the arm contrast.

## 6. Decoding and prompt (identical for every model)

| Setting | Value |
|---|---|
| Prompt | Real Qwen chat template (sha256 `64f85b19…`), no custom system prompt. The prompt equals the training prefix (verified). |
| Decoding | Greedy (`do_sample=False`), `max_new_tokens=256`, stop at `<|im_end|>`, batch size 1 |
| Adapter path | Every adapter is evaluated as a freshly loaded 4-bit base plus `PeftModel` adapter (not merged), the same path for every arm |

## 7. Limitations (must appear in every report)

- **Possible pretraining exposure (contamination).** xLAM 60K and BFCL are public, and similar function-calling data is widely available. The base model `Qwen3-4B-Instruct-2507` **may have seen xLAM or similar examples during pretraining or post-training.** We can't verify or rule this out, and we **make no claim of zero prior exposure.** The base model's high strict accuracy (82% on the dev sample) is consistent with this, but doesn't prove it. Because all arms share the same base and evaluation set, prior exposure affects absolute scores more than the paired arm contrast. It could, however, compress the room for improvement (a ceiling effect).
- **Scope.** Results cover single-call, single-turn function calling. There is no claim about multi-turn tool use or no-tool (abstention) decisions, since xLAM has no no-tool examples.
- **Label noise.** Reference labels contain errors and ambiguity (§5). The decision metric is strict, on the original labels.
- **Test protocol.** The test set stays sealed until the researcher explicitly unseals it.
