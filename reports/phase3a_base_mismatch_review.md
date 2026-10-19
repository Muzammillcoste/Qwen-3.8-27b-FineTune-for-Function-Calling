# Phase 3A base-model mismatch review (dev sample, validation split)

- **Scope.** The 18 strict-match failures of the untuned base model on the 100-example development sample.
- **Status.** The classification is **provisional, by Claude**, and needs the researcher's confirmation.
- **Rubric.** `docs/EVALUATION_SPEC.md` §5.
- **Labels.** Never rewritten; scores use the original labels.

| sample_id | class | reason (short) |
|---|---|---|
| xlam:14160 | definite model mistake | `bottom_left` and `top_right` swapped; the reference correctly uses the south-west point as bottom-left |
| xlam:39814 | definite model mistake | slug given with a space ("invisible friends"); slugs are space-free (the reference matches) |
| xlam:56853 | definite model mistake | ignored the identifier stated in the query and used the schema's default URL |
| xlam:37676 | definite model mistake (format) | correct call content, but output without `<tool_call>` tags, so it fails to parse |
| xlam:53770 | definite model mistake | "ALIBABA" is not a ticker (the reference "BABA" with region HK is itself questionable) |
| xlam:28885 | possible label error | reference `series_id 98765` is not derivable from the query (invented); the prediction used the schema default |
| xlam:34111 | possible label error | reference uses the display name; the tool's documented IDs look like `washington_post` (the prediction matches) |
| xlam:1622 | possible label error | reference argument is a Python expression string; the prediction is a concrete list |
| xlam:48812 | possible label error | reference `match_id` is a placeholder; the base model asked for clarification (no-tool behaviour is out of scope) |
| xlam:35924 | ambiguous | "Bitcoin" on Bybit: BTCUSD (schema default) vs BTCUSDT (reference); both are valid markets |
| xlam:37286 | ambiguous | explicit `descending: false`; equivalent only by description text, so not schema-equivalent |
| xlam:22293 | ambiguous | the query supplies no text; the reference uses the literal phrase and the prediction invents a sentence |
| xlam:22722 | ambiguous | Los Angeles coordinates at different precision (no coordinates in the query) |
| xlam:33666 | ambiguous | explicit `league_ids: null` for an optional parameter (the prediction is also schema-invalid) |
| xlam:52782 | ambiguous | country "US" vs "United States"; the format is unspecified |
| xlam:22667 | ambiguous | casing: "Ford/Mustang" vs "ford/mustang" (the schema examples are lowercase) |
| xlam:22758 | ambiguous | integer-typed latitude for Sydney: −34 (reference) vs −33; plus an explicit elevation default (that default is type-invalid) |
| xlam:3193 | ambiguous | how "1960s" is split between the query and the filter text; plus an explicit `number` (that default is type-invalid) |

**Counts:** 5 definite model mistakes · 4 possible label errors · 9 genuinely ambiguous.

**Implications.**

- Strict exact match is a conservative lower bound on semantic correctness.
- Label noise exists even in records whose validation status is `valid`, so the blinded semantic audit remains important.
- None of the ambiguous cases is resolved by the schema-aware metric, because none involves a *valid* declared default. This is intended: equivalence only through explicit schema semantics.
