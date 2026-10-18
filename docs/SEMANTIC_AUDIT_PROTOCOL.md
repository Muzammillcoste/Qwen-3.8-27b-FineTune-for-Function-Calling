# Blinded semantic-quality audit protocol (Phase 2 design; labels by humans only)

**Purpose.** Measure what the code cannot: whether each expected call is a *semantically correct* answer to its query. The audit also tests whether the schema-consistency filter tracks semantic correctness at all. Code never fills these labels, and nothing in this file is a result.

## Sample (built by `scripts/audit_selection.py`)

| Stratum | Size | Drawn from |
|---|---:|---|
| `arm:random_1k` | 40 | random_1k, split by teacher in proportion to the arm |
| `arm:schema_consistent_1k` | 40 | schema_consistent_1k, split the same way |
| `arm:diversity_1k` | 40 | diversity_1k, split the same way |
| `diagnostic:schema_conflict` | 20 | pool records with `schema_argument_type_conflict` that are in none of the 1k arms |
| `diagnostic:ambiguous` | 10 | pool records with label warnings only that are in none of the 1k arms |

- **Overlap between arms.** random_1k and schema_consistent_1k share most records. An item drawn for several arms is shown only once, and the hidden key lists every arm it belongs to. The unique item count is therefore about 120–150.
- **random_3k.** Its extra 2,000 records come from the same stratified-random procedure, so the random_1k sample stands in for random_3k. A dedicated random_3k sample would add little.

## Files

- **`data/audits/semantic_audit/annotation_sheet.csv`.** This is what annotators see. It has an opaque `audit_id`, the query, the available tools (signatures and descriptions) and the expected call. **It shows no arm, teacher, validation flag or label status.** Rows are shuffled.
- **`data/audits/semantic_audit/audit_key.csv`.** This is the hidden mapping. The researcher keeps it until all labels are frozen.
- Both files contain xLAM text and are gitignored. Do not share them publicly.

## Rubric (fill one row per item)

| Column | Values | Rule |
|---|---|---|
| `semantic_correct` | `Y` / `N` / `U` | **Y**: a careful human would issue exactly this call (right tool, argument values faithful to the query, nothing essential missing). **N**: it would not. **U**: you cannot judge, because the query is ambiguous or the tool description is insufficient. |
| `error_type` | `wrong_tool`, `wrong_arg_value`, `missing_needed_arg`, `extra_arg`, `unfaithful_to_query`, `query_ambiguous`, `other` | Only when the label is `N` or `U`. |
| `severity` | `minor` / `major` | **minor**: formatting or units; a reasonable API would still work. **major**: the call does the wrong thing. |
| `notes`, `annotator_id` | free text | — |

Judge the call against the **query and the tool description**, not against the declared JSON types. A decimal latitude passed to a parameter declared `int` is semantically correct if the query asks for that latitude; record the type issue in `notes`. Do not execute any tool.

## Procedure

1. Annotator A labels all items.
2. Annotator B independently labels a random subset of at least 30 items (if a second person is available). Compute Cohen's κ on `semantic_correct`, and adjudicate disagreements.
3. Freeze the labels, then join them with `audit_key.csv`.

## Analysis plan (fixed before labelling)

- **Semantic error rate per arm** (N / (Y+N)), with Wilson 95% CIs, reported overall and by teacher.
- **schema_consistent_1k vs random_1k.** Compute the difference in semantic error rate. With about 40 items per arm, only very large differences are detectable. Report the result as an estimate, not a test of superiority.
- **Diagnostic stratum.** Is the semantic error rate of schema-conflict records higher than that of arm records? If it is not, the schema filter does not track semantic quality, and that is evidence for dropping the schema arm.
- Report the `U` share and the error-type distribution.
- No result from this audit may be used to change selection after test evaluation.
