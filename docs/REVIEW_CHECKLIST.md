# Human review checklist

## Phase 0 → Phase 1 gate (current)

- [ ] Scope summary and open issues read (see EXPERIMENT_LOG.md, Phase 0 entry).
- [ ] `pytest -q` passes locally; skipped tests are only the Phase 2/3 placeholders.
- [ ] `ruff check .` passes.
- [ ] Field parsing reviewed against `reports/fixture_inspection.md`. The report is clearly labelled as fixture.
- [ ] Required-status policy accepted, or changed in `configs/data.yaml`:
  - an explicit `required` value wins;
  - otherwise a `, optional` marker or `Optional[...]` makes the parameter optional;
  - otherwise a `default` makes it optional (noted as inferred);
  - otherwise it is required (noted as inferred).
- [ ] ID-field risk understood: the card example shows no `id` key. Phase 1 must verify it.
- [ ] License, access and gating instructions read (LICENSE_NOTICES.md, KAGGLE_GUIDE.md §1).
- [ ] HF conditions accepted and `hf auth login` done (manual), or Phase 1 deferred.
- [ ] Phase 1 command approved explicitly.

## Phase 1 → Phase 2

- [ ] `reports/dataset_inspection.md` reviewed: real field types, ID field, malformed counts, call distribution.
- [ ] Teacher counts match the card (33,659 / 26,341), or the mismatch is explained.
- [ ] `DATASET_CARD.md` filled with measured counts; exclusions documented.
- [ ] Split counts, teacher mix and call buckets reviewed; leakage audit passes.

## Phase 2 → Phase 3

- [ ] Quality proxy frozen (`score_version`) before training.
- [ ] Score distributions show meaningful spread, or the narrower framing is accepted.
- [ ] ≥100 blinded human labels entered by a human; agreement analysed.
- [ ] Teacher mix of the subsets is within tolerance; overlap and Jaccard between arms reported.

## Phase 3 → Phase 4

- [ ] Preflight log reviewed (GPU, versions, model SHA, data hashes).
- [ ] Pilot results:
  - [ ] peak VRAM
  - [ ] tokens/s
  - [ ] truncation ≤ 5%
  - [ ] loss mask verified on real tokens
  - [ ] estimated runtime under quota
- [ ] Run order and GPU budget approved.

## Phase 4 → Phase 5

- [ ] Every run has a manifest, adapter, trainer state and logs.
- [ ] No selection rule changed after seeing the test set.
- [ ] Anomalies explained.
