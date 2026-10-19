# Experiment log

Append-only. Record every phase action, deviation and decision here. Never record results that were not actually observed.

---

## 2026-10-18 — Phase 0: scaffold and source verification

**Done**

- Created the repository scaffold, configs, docs, the `tooldistill` package, CLI and scripts.
- Implemented and tested:
  - normalization: JSON-string or native fields, multi-call, type conversion, required policy, reject-with-reason;
  - validation;
  - provenance from IDs;
  - dedup and grouping;
  - group-aware stratified split;
  - leakage audit;
  - manifests;
  - gated-access handling.
- Created the synthetic fixture with 33 lines. It is clearly labelled and is not xLAM data.
- Inspected the fixture into `reports/fixture_inspection.md`.
- Test results: see the Phase 0 summary in chat / the review checklist.

**Source verification (metadata only, no files downloaded)**

- xLAM: `gated=auto`, CC BY 4.0, revision `26d14ebf...`.
  - The only data file is `xlam_function_calling_60k.json`. The card maps config `dataset` to split `train`.
  - The unauthenticated access check returned HTTP 401 (GatedRepoError).
  - There is no HF token on this machine.
- BFCL: HF dataset not gated (Apache-2.0); `bfcl-eval` 2026.3.23 on PyPI; upstream GitHub path confirmed live.
- Qwen3-4B-Instruct-2507: not gated (Apache-2.0). Revision observed; **weights not downloaded**.

**Deviations from spec D:** see README "Layout and documented deviations".

**Open issues carried to Phase 1**

1. The card example shows no `id` field. The ID-based teacher provenance is unverified until the raw file is inspected.
2. Required-status representation in the real data is unknown. The card shows a boolean `required`; other shapes such as `str, optional` and `default` are also supported. Inferred counts must be reviewed.
3. Near-duplicate grouping cost: pairwise comparison within call-name blocks. Runtime on 60k rows must be measured.
4. Eligibility rules (for example a single-call restriction) are still undecided. Decide after Phase 1 inspection.
5. Kaggle requirements are not pinned (Phase 3). The BFCL category list has not been enumerated (Phase 3).
6. The code license has not been chosen.

**No training, no model download, no API calls, no research results.**

---

## 2026-10-18 — Phase 0: authenticated 100-row inspection (permitted by spec K, Phase 0)

- The researcher accepted the HF conditions and logged in. `scripts/check_access.py` returned `ok` with the pinned revision.
- `inspect_xlam.py --sample 100` streamed the first 100 rows. They are **ids 0–99, all DeepSeek, not representative**. Output: `reports/dataset_inspection.md`. The HF cache grew by about 20 KB; there was no full download.
- **The `id` field exists** (int, unique, contiguous 0–99). This resolves the card-example ambiguity for this sample; full-data verification is still required.
- **Real parameter format:** every parameter has `{type, description}` or `{type, description, default}` (584 parameters). There is **no `required` key anywhere**, unlike the card example. Required status is therefore 100% inferred:
  - 281 parameters were inferred optional (they have a default or an `optional` marker);
  - 93 were inferred required (no marker).
- Type strings seen include: `str`, `str, optional`, `int`, `int, optional`, `float`, `bool, optional`, `List[int]`, `List`, `Dict`, `List[Union[int, float]]`, `List[Tuple[float, float]]`, `List[List[str]]`. All of them converted without unknown-type flags.
- Call counts: 1 call = 50, 2 calls = 45, 3 calls = 5. No empty answers.
- **Bug fixed:** row 81 declares two different tools that are both named `facilities_lookup`. The validator used to check only the last one. Calls are now valid if any same-named declaration accepts them. A test was added.
- Remaining flags are genuine data issues:
  - id 53: the sudoku `board` is typed `List[List[str]]`, but the label contains ints.
  - id 65: `permitivity` has no default, so it was inferred required, but it is omitted from the label. Its description says "default is 8.854e-12".
- Also observed: some defaults have the wrong type (id 81: `housenumber` is int with default `"288"`; id 65: an int parameter with default `8.854e-12`). These are not flagged yet.

---

## 2026-10-18 — Phase 1 (approved with required changes)

**Approved policy changes, now implemented and tested (156 tests pass, 3 skipped)**

- Requiredness is three-state: REQUIRED, OPTIONAL, UNKNOWN. Declared and inferred evidence are tracked separately, and conflicts are detected.
- A missing UNKNOWN argument produces the warning `missing_arg_unknown_requiredness`, not an invalid verdict.
- `default_type_mismatch` and `default_null_value` are recorded as schema issues, separate from label correctness.
- Every record gets a label status: `valid`, `ambiguous`, `invalid` or `no_call`.
- The original tool declarations are preserved in `metadata.source_tools`.

**Fixes made during the run**

- Queries wrapped in literal quotes (12 rows) were previously JSON-decoded. That stripped the user's quotes. Queries are now kept verbatim.
- Compiled JSON-Schema validators are now cached, because `check_schema` took 96% of the runtime.
- The inspection report heading was corrected from "records" to "occurrences".

**Commands**

```
download_xlam
inspect_xlam --sample 60000
prepare_dataset --seed 42
audit_dataset
```

For token counting, the audit downloaded only Qwen `tokenizer.json` (11.4 MB, at the pinned revision). No model weights were downloaded.

**Measured results**

- The raw SHA256 is `4ef5c6f0…`.
- 0 parse failures. The IDs are intact, and the teacher counts **match the card exactly**.
- No parameter in the corpus declares `required`. Requiredness is therefore 0% REQUIRED, 74.6% OPTIONAL (inferred) and 25.4% UNKNOWN.
- Label status: valid 96.97%, ambiguous 0.78%, invalid 2.26%.
  - The invalid rate is 1.79% for DeepSeek and 2.85% for Mixtral.
  - 763 of the 1,356 invalid records are "expected integer / got float". These are most likely errors in the *declaration*, for example latitude declared as `int`, not label errors.
- Invalid records are longer: median 579 tokens, against 424 for valid records. This is a difficulty confound.
- There are 400 exact duplicates, 1,716 edges from identical query text and 5,142 near-duplicate edges. 37 input sets have conflicting labels.
- **There are zero no-tool examples.**
- Tokens (raw, before chat-template overhead): p95 is 951 and p99 is 1205. 3.16% of records exceed 1024 tokens.

**No selection, no quality score, no Qwen weights, no training.**

---

## 2026-10-18 - Phase 1 accepted; Phase 1 follow-ups; Phase 2 (selection and preparation only)

**Researcher decisions:**

- Adopt Option A framing (data selection, diversity, coverage and quantity).
- Defer Option B (label noise).
- The primary experiment uses single-call records only.
- Add `schema_argument_type_conflict`.
- Create a git branch.

**Phase 1 follow-ups (committed as `60bbe79` on `research/phase1-audit`; author: the researcher only; not pushed)**

- Introduced `schema_argument_type_conflict` (validate-v3). Type disagreements are no longer confirmed label errors. The strict verdict is kept in `strict_schema_valid`, and the new label status is `schema_conflict`.
- Re-audit results: 5 confirmed structural label errors (all spurious arguments, all Mixtral) and 1,351 schema-conflict records.
- Records now carry `call_structure`, which distinguishes several calls in one response from multi-turn conversations. xLAM is entirely single-turn.

**Phase 2 (not committed; awaiting review)**

- Rendered 28,171 single-call records with the real Qwen template. Only tokenizer and config files were downloaded (15.9 MB), with no weights and no torch. The prompt is an exact token prefix in 28,171 of 28,171 records, and every call renders verbatim.
- Truncation at 1024 tokens affects 6.8% of the train pool, at 1536 it affects 0.32%, and at 2048 it affects 0.05%. The 71 train records longer than 1536 were made ineligible before selection.
- select-v1 balance diagnostics showed `diversity_1k` drifting on structured arguments and tool style. Strata were extended to 6 features (select-v2, 150 strata), using train-only information. After that, every |SMD| is at most 0.03.
- **The schema arm fails the gate.** `schema_consistent_1k` differs from `random_1k` in only 43 of 1,000 records (4.3%), below the 5% threshold. Removing the arm is recommended.
- `diversity_1k` vs `random_1k`: 756 vs 604 distinct functions, 4,192 vs 2,248 query word types, and pairwise query Jaccard 0.055 vs 0.074. These metrics are partly circular, because they are the features the arm optimizes.
- The blinded audit sample has 148 items (3 arms × 40 + 30 diagnostic). The sheet and key are in `data/audits/semantic_audit/`.
- Docs: `SELECTION_METHODS`, `SEMANTIC_AUDIT_PROTOCOL`, `EVALUATION_PLAN`, `TRAINING_PROTOCOL`.
- 177 tests pass, and 1 test is skipped (the Phase 3 metrics placeholder).

**Not done:** no training, no Qwen weights, no Kaggle GPU use, no paid APIs, no synthetic teacher data.

---

## 2026-10-18 - Phase 2 approved with changes; Phase 3A built (Kaggle run pending)

**Decisions:**

- `schema_consistent_1k` is dropped; its manifest is kept.
- The primary comparison is `random_1k` vs `diversity_1k`.
- `random_3k` and `random_1k` for 3 epochs are follow-up runs.
- 1,536 tokens is approved for the pilot, with consistent exclusion of longer records.
- A small fixed validation sample is used first. The test split and BFCL are reserved.
- The scope is single-call only.

**Git:** Phase 2 was committed as `7fdb2e8` on `research/phase1-audit`. The researcher is the sole author, and it was not pushed.

**Phase 3A (not committed; awaiting the Kaggle run and review)**

- Pins:
  - Kaggle: transformers 5.18.0, peft 0.21.2, accelerate 1.15.0, bitsandbytes 0.50.2, tokenizers 0.23.3; torch is Kaggle's preinstalled build.
  - Local: CPU torch 2.14.1 is used for mechanics tests only.
- Under transformers 5.18 the template sha is identical to the Phase 2 manifest, and the rendered lengths of all 200 bundle records are identical.
- Code:
  - `train.py`: 4-bit NF4, fp16, LoRA on 7 projection modules, paged AdamW 8-bit, monitor callback, NaN stop, interrupt/resume.
  - `infer.py`: greedy, left-padded generation.
  - `evaluate.py`: defensive parser and metrics.
  - `hardware.py`: preflight.
  - `phase3a.py`: orchestration and report.
  - `kaggle_export.py`: the bundle.
  - Notebook rebuilt for MODE preflight / format_check / baseline / phase3a.
- Bundle:
  - 100 pilot TRAIN records (label_status: 96 valid, 3 schema_conflict, 1 ambiguous);
  - 100 fixed VALIDATION records (99 valid, 1 schema_conflict);
  - 20 format-check IDs.
- Tests: 199 passed, including CPU mechanics on a tiny randomly initialised Qwen3 model (no real weights):
  - loss mask;
  - padding mask;
  - **interrupt + resume into a fresh model reproduces uninterrupted LoRA weights**;
  - adapter save/reload;
  - NaN stop;
  - preflight failure without a GPU or with too little memory.
- **NOT yet measured** (requires the Kaggle run): GPU memory, training speed, baseline accuracy, pilot results.

---

## 2026-10-19 - Phase 3A results; Phase 3B prepared (primary runs NOT executed)

**Phase 3A, measured on Kaggle** (`reports/phase3a_feasibility.md`)

- Training at 246.5 tokens/s, with peak VRAM of 9.0 GB.
- No NaN.
- True resume into a fresh model worked.
- Strict accuracy on the dev 100: base 82, pilot 77 (McNemar p = 0.0625). The pilot result is not a finding.

**Git**

- The researcher committed Phase 3A as `f87cb30`. That commit also contains `results/phase3a/`: the outputs zip with adapter weights, and the prediction files with validation text.
- The researcher chose to leave the history as it is.
- `.gitignore` now excludes results, zips, adapters, prediction files, extracted outputs and benchmark data.

**Researcher decisions**

- Keep the learning rate at 1e-4, with no sweep.
- The primary runs are `random_1k` and `diversity_1k`, 1 epoch each.

**Phase 3B (uncommitted)**

- eval-v2 is frozen: strict primary; a schema-aware secondary metric (valid declared defaults only); McNemar as the decision test, with a paired bootstrap CI as descriptive.
- The 18 dev mismatches were reviewed: 5 model / 4 label / 9 ambiguous. This is provisional.
- `val_eval_400` is fixed and disjoint from the dev 100. `test_eval_1000` is fixed and SEALED.
- Both sets use one record per group and are group-disjoint from TRAIN.
- The two arms are comparable: 125 steps each, tokens +0.2%, supervised tokens +3.4%, identical teacher share.
- Coverage categories were precomputed.
- `experiment.py` provides the Kaggle runner, `compare.py` the analysis, and `stats.py` the statistics.
- The notebook was rebuilt for Phase 3B.
- Estimated cost of the primary runs: about 3.1 GPU hours.
- 215 tests pass.
