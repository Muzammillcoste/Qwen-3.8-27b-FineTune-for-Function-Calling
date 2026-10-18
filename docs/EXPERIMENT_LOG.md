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
