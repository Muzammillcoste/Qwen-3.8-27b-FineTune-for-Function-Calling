# Selection methods (Phase 2 — planned, not implemented)

Nothing in this file has been implemented or run. It records the constraints that Phase 2 must follow (spec F1–F3, G1).

## Inputs

- The **train split only**, after grouping and splitting. Validation, test and BFCL results are never inspected.
- The eligibility rules are fixed and documented in `DATASET_CARD.md` before any subset is drawn.

## Quality proxy (`quality_proxy_score`, with a `score_version`)

The score has four named components:

1. **Tool identity integrity.** Every called function exists in the available tools.
2. **Argument and schema validity.** Arguments type-check, required arguments are present, and there are no spurious arguments.
3. **Grounding signal.** Literal argument values appear in the query where this can be checked. This is a noisy heuristic.
4. **Consistency.** No contradictory or unexplained duplicate calls, and no incomplete annotations.

### Binding constraints approved 2026-10-18 (researcher)

- **Requiredness is three-state.** A parameter is REQUIRED, OPTIONAL or UNKNOWN; declared and inferred evidence are tracked separately.
- **UNKNOWN requiredness is never penalized** in the primary quality score. A missing argument whose requiredness is UNKNOWN is a *warning* (`missing_arg_unknown_requiredness`), not an invalid label.
- **`default_type_mismatch` and other schema issues describe the tool declaration, not the label.** They must not be counted as label errors. They also must not enter the primary score unless this is separately justified and approved.
- **No record is modified or discarded** because of ambiguous metadata.
- **No quality score may be built on uncertain metadata.**

Two things are recorded **separately** and never folded into "quality":

- difficulty or complexity, such as number of tools, calls and arguments, and length;
- novelty or duplication.

Weights and thresholds are frozen **before** training. The score must not reward short or simple examples by construction.

## Arms

| Arm | Algorithm |
|---|---|
| `random_1k`, `random_3k` | Seeded uniform sampling from eligible train. |
| `quality_1k` | Top-ranked by the frozen proxy. Teacher mix is matched within the tolerance in `configs/research.yaml`. |
| `quality_diverse_1k` | Quality-filtered pool, then stratified coverage over teacher, call count, tool count, argument count, query-length bin and API family. TF-IDF + seeded k-means or farthest-point selection is optional. It must be a distinct algorithm from `quality_1k`; report the overlap and Jaccard. |

## Audit outputs (Phase 2)

- Score histogram (20 bins) and the valid/invalid percentage for each component.
- For each arm: token-length, tool-count, call-count and teacher-mix distributions.
- A blinded annotation CSV for at least 100 human labels. Semantic correctness is never auto-filled.
- An immutable manifest for each arm: sorted sample IDs plus SHA256.
