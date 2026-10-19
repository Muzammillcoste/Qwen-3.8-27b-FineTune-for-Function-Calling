# Research protocol

**Status:** pre-registered design (Phase 0). No data has been processed and no model has been trained or evaluated. The authoritative source is the master spec v2.0. This document restates the binding decisions so a reviewer can check later phases against them.

## 1. Title and framing

The working title is *Quality and Diversity Versus Quantity in Small Language Model Tool-Calling Adaptation: A Controlled Data-Selection Study*.

xLAM is already filtered: the authors report more than 95% correctness on a 600-example human sample. So this study reliably measures **selection policies and sample size**. The word "quality" is used for a true label-quality effect only if independent human annotation shows real quality differences between subsets (feasibility gate, §5).

## 2. Research questions

| ID | Question | Comparison |
|---|---|---|
| RQ1 | Does a transparent selection policy beat uniform sampling at equal N? | `quality_1k` vs `random_1k` |
| RQ2 | Does diversity coverage add beyond quality? | `quality_diverse_1k` vs `quality_1k` |
| RQ3 | Can 1k selected examples match 3k random ones (fixed-epoch)? | `quality_1k` vs `random_3k` |
| RQ4 | Do findings survive token, step and runtime accounting? | secondary budget-matched control, if affordable |
| RQ5 | Do independent human quality labels predict gains? | audit of ≥100 examples |
| RQ6 | Do gains generalize? | internal held-out test + BFCL subset |

Null results are valid outcomes.

## 3. Data protocol (implemented in Phase 0 code)

1. **Access.** Use only authorized HF access at a pinned revision. A 401/403 response is nonfatal and shows manual steps. There is no scraping.
2. **Inspect before normalizing.** Inspect field types (JSON strings vs native values), ID field existence and uniqueness, call-count distribution and malformed rows. The output goes to `reports/dataset_inspection.md`.
3. **Normalize** to `tooldistill.normalized.v1`:
   - Every expected call is kept, in order.
   - Tool names are preserved, including dotted names.
   - Types are converted to JSON Schema. Unknown types are flagged and never cast to string.
   - Required status follows this order:
     1. an explicit `required` value;
     2. the `, optional` marker or `Optional[...]`;
     3. a `default` makes the parameter optional, and this is noted as inferred;
     4. otherwise the parameter is required, and this is noted as inferred.
   - Malformed rows are rejected with a logged reason. They never abort the run.
4. **Validate.** Run JSON Schema validity checks on tool parameters and validate label arguments against them. Flag hallucinated label tools, duplicate identical calls and empty answers. These checks are syntactic only. They are not semantic correctness.
5. **Provenance.** The teacher is derived **only from the original ID** (0–33658 is DeepSeek-V2-Chat, 33659–59999 is Mixtral-8x22B-Instruct). It is never derived from a shuffled position. If IDs are missing, duplicated or non-integer, teacher counts are not computed. On full data, counts must match the card (33,659 / 26,341). Otherwise the mismatch is reported.
6. **Deduplicate and group before splitting.** Groups are built with union-find over these edges:
   - identical exact key (normalized query + tool signatures + calls);
   - identical original ID;
   - identical normalized query text;
   - token-set Jaccard ≥ 0.8 between queries with the same expected-call-name multiset.

   Exact duplicates are excluded from the eligible pool. Near duplicates are grouped, not deleted.
7. **Split.** Whole groups go to one split. The target is 80/10/10, seeded (default 42) and stratified by teacher source and call-count bucket. Splitting is deterministic and invariant to input order.
8. **Leakage audit.** No group, exact key, normalized query, sample ID or original ID may appear in more than one split. Any violation is a hard error.
9. **Sealed test.** Validation and test are never used to set quality thresholds, diversity centres or hyperparameters. The test split stays sealed until the evaluation pipeline is frozen.

## 4. Experiment matrix

| Run | N | Rule | Status |
|---|---:|---|---|
| `base_0` | 0 | untuned base | planned, mandatory |
| `random_1k` | 1000 | uniform from eligible train, seed 42 | planned, mandatory |
| `quality_1k` | 1000 | frozen quality heuristic | planned, if gate passes |
| `random_3k` | 3000 | uniform from the same pool | planned, if affordable |
| `quality_diverse_1k` | 1000 | quality + diversity | planned, desirable |

Each run is an independent LoRA adapter trained from the same frozen base. Adapters are never chained. Controlled settings are listed in spec G4.

## 5. Feasibility gate (Phase 2, human decision)

The gate requires all of the following:

- A 20-bin histogram of the quality proxy.
- A comparison of candidate subsets on proxy, token length, number of tools, call count and teacher mix.
- At least 100 blinded human-reviewed examples, 50 per candidate subset, with semantic-correctness labels entered by a human. Code never fills these labels.

If the quality separation is unclear, the study proceeds only as data-selection research, after human approval.

## 6. Metrics and statistics (Phase 3–5)

- **Primary metric:** exact structured-call accuracy.
- **Secondary metrics:**
  - parse rate
  - tool-name set accuracy
  - argument correctness
  - schema-valid predicted-call rate
  - hallucinated-tool rate
  - latency
  - training cost
- **Statistics:** paired bootstrap over test examples, plus McNemar. With a single training seed, intervals reflect test-sample uncertainty only.
- **BFCL:** results come only from the official evaluator. They are labelled with subset, version and N, and are never presented as a leaderboard rank.

## 7. Known limitations (to carry into the report)

- The corpus is prefiltered, so quality differences may be small.
- The proxy is imperfect.
- The two teachers are mixed.
- The data may contain no no-tool examples, so abstention is not a trained capability.
- BFCL may be contaminated in pretraining.
- **Possible prior exposure to xLAM.** The base model may have seen xLAM or similar public function-calling data during pre- or post-training. This cannot be ruled out, and no claim of zero exposure is made. The base already scores 82% strict on the dev sample, so a ceiling effect is possible. See `docs/EVALUATION_SPEC.md` §7.
- Free-GPU limits may force fewer seeds.
- Domain shift between xLAM and BFCL.
