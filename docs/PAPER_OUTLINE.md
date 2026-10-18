# Paper outline (skeleton — no results exist)

Each section is labelled *planned* until it is backed by actual runs.

1. **Abstract** (planned). Write it last, from actual results only.
2. **Introduction.**
   - Data efficiency for small-LM tool calling.
   - Selection vs quantity.
   - Why existing teacher data is used ($0 budget, reproducibility).
3. **Related work.**
   - APIGen/xLAM (arXiv:2406.18518).
   - BFCL.
   - Data selection and curation for SFT.
   - QLoRA.
4. **Data.**
   - xLAM provenance, with the mixed teachers.
   - Normalization, dedup and grouping, group-aware split, leakage audit.
   - Exclusions.
5. **Methods.**
   - Selection policies: random, quality proxy, quality + diversity.
   - Frozen proxy definition.
   - Training setup and controlled settings.
6. **Feasibility gate.** Proxy distributions and the human audit (≥100 labels).
7. **Results** (planned).
   - Internal held-out results.
   - BFCL subset (labelled with version, categories and N).
   - Paired tests.
   - Cost accounting.
8. **Analysis** (planned).
   - At least 20 categorized failure cases.
   - Difficulty confounds.
   - Teacher-mix effects.
9. **Limitations.** Carry over everything from RESEARCH_PROTOCOL §7, plus anything observed.
10. **Reproduction appendix.** Manifests, hashes, commands and versions.

Do not claim novelty, publishability or acceptance without peer review.
