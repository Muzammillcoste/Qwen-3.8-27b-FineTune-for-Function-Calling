# Compute-aware training protocol (Phase 2 design; nothing trained)

All numbers marked **measured** come from `reports/phase2_selection.md`, using the real Qwen chat template and tokenizer. Numbers marked **estimate** are back-of-envelope figures from the model config (`Qwen3-4B-Instruct-2507`: 36 layers, hidden size 2560, 32/8 attention/KV heads, MLP 9728, vocabulary 151,936, tied embeddings, trained in bf16). The Phase 3 pilot replaces every estimate with a measurement.

## 1. Controlled settings (identical for every arm)

The following are identical for every arm:

- base model and revision;
- tokenizer and chat-template sha256;
- prompt serialization (no custom system prompt; the template's tool block);
- 4-bit NF4 with double quantization;
- LoRA r = 8, alpha = 16, dropout = 0.05, all-linear excluding the output head;
- learning rate 1e-4 with cosine schedule and warmup ratio 0.03;
- the `paged_adamw_8bit` optimizer;
- per-device batch 1 × gradient accumulation 8 (effective batch 8);
- `max_seq_length` 1536;
- completion-only loss;
- seed 42;
- evaluation set and decoding.

Each arm is an **independent adapter from the same frozen base**. Adapters are never chained.

## 2. Comparisons

| Comparison | Schedule | Updates | Training tokens (measured) | Interpretation |
|---|---|---|---|---|
| `random_1k` vs `diversity_1k` (± `schema_consistent_1k`) | 1 epoch each | 125 steps each | 554.5k / 555.7k (/ 553.9k) | **Primary, fair.** Equal N, steps and tokens (within 0.3%), and matched strata. |
| `random_1k` vs `random_3k`, fixed epoch | 1 epoch each | 125 vs 375 | 554.5k vs 1,671.7k | Practical data-efficiency trade-off. 3k gets 3× the compute. It is **not** evidence that quantity beats selection at equal compute. |
| `random_1k` vs `random_3k`, **compute-matched** (secondary) | 1k × 3 epochs vs 3k × 1 epoch | 375 vs 375 | ≈1,663.6k vs 1,671.7k (seen) | Equal optimizer steps and about equal tokens seen. Differs in unique examples (1k seen 3× vs 3k seen 1×). Report unique tokens and repeated exposures explicitly. |
| `diversity_1k` × 3 epochs vs `random_3k` (optional) | 3 vs 1 epochs | 375 vs 375 | ≈1,667.2k vs 1,671.7k | Does coverage-selected 1k at equal compute match 3k random? |

**Coverage confound.** `random_3k` covers 1,185 distinct functions, against 604 for `random_1k` and 756 for `diversity_1k`. Quantity also buys coverage, so any 1k-vs-3k result must be read together with these coverage numbers.

The cosine schedule always spans the run's own total steps. Warmup is 3% of each run's steps, so 1k and 3k runs have identical schedule *shapes*. Always report N, unique tokens, seen tokens, steps, epochs, wall-clock time and peak VRAM.

## 3. Sequence length and truncation (measured)

| Max length | Train pool truncated | Val | Test |
|---|---|---|---|
| 1024 | 6.83% (1,540; for 1,233 the prompt alone is ≥ 1024) | 5.83% | 5.97% |
| **1536** | **0.32% (71)** | 0.18% | 0.46% |
| 2048 | 0.05% | 0.04% | 0.11% |

The assistant target is the final segment of every record, so right-truncation always destroys it.

- **Proposal: a 1,536-token pilot.** The 71 train-pool records longer than 1536 are made ineligible before selection, so no arm contains a truncated target. The longest selected record is 1,535 tokens.
- 1024 is **not acceptable**: it would truncate more than 5% of records, beyond the spec's stop rule.
- At inference, test prompts are never truncated (the context limit is far above 1536). The 0.46% of test records over 1536 are evaluated normally and flagged as over-length in breakdowns.

## 4. T4 memory risks (estimates; the pilot must measure)

| Component | Estimate at seq 1536 | Basis |
|---|---|---|
| 4-bit linear weights | ~1.9 GB | ~3.63 B linear params × 0.5 B + quantization constants |
| Embedding / LM head (tied, fp16, not quantized) | ~0.78 GB | 151,936 × 2560 × 2 B |
| LoRA params + grads + 8-bit Adam | ~0.2 GB | ~16.5 M LoRA params (r = 8, all-linear, 36 layers) |
| Checkpointed activations | ~0.3–0.6 GB | 1536 × 2560 × 2 B per layer × 36, plus recompute of one layer |
| **Logits** | **~1.4–2.0 GB peak** | 1536 × 151,936 × (2 B fp16 + 4 B fp32 upcast in the loss) + grad |
| CUDA context / allocator slack | ~0.5–1.0 GB | — |
| **Total** | **~5–7 GB of ~15 GB usable** | It fits on paper. The main risks are not capacity. |

Risks to test in the pilot:

1. **No bf16 on T4.** T4 (sm_75) lacks bf16, so compute runs in fp16 while Qwen was trained in bf16. Activation overflow could produce NaN or inf losses. Mitigations: keep LoRA weights in fp32, use the fp16 GradScaler, and log NaN/inf counts. If the problem persists, *propose* alternatives; never switch silently.
2. **The logits dominate peak memory.** Only about 37 target tokens per record carry loss (measured mean completion ≈ 37 tokens), so computing the LM head only on target positions would save about 1.5 GB if needed.
3. **No FlashAttention-2 on sm_75.** Use PyTorch SDPA (memory-efficient kernel). Throughput will be lower than on modern GPUs.
4. **Two T4s are not 32 GB pooled.** Use one GPU and avoid an accidental DataParallel / accelerate multi-GPU launch.
5. **Quota.** Kaggle's free GPU quota is about 30 h/week, with per-session limits. Verify both live; they change.

## 5. Cost estimates (tokens measured; throughput assumed 150 / 300 / 600 tok/s)

| Run | Steps | Tokens | Hours (150 / 300 / 600 tok/s) |
|---|---|---|---|
| pilot_100 | ~13 | ~56k | 0.10 / 0.05 / 0.03 |
| each 1k arm, 1 epoch | 125 | ~555k | 1.03 / 0.51 / 0.26 |
| random_3k, 1 epoch | 375 | 1.67M | 3.10 / 1.55 / 0.77 |
| 1k × 3 epochs (compute-matched) | 375 | ~1.66M | ~3.1 / 1.5 / 0.8 |

**Evaluation may cost more than training.** The test set has 2,813 single-call records, with mean prompt ≈ 505 tokens and mean target ≈ 36 tokens. With batch-1 generation at an assumed 8 / 15 / 25 tok/s, that is **≈ 3.5 / 1.9 / 1.1 hours per model**. For 5–6 evaluated models this is 6–20 GPU hours. Decide in Phase 3:

- batched generation; or
- one fixed, seeded, stratified evaluation subset (for example 1,000 records) for all models, chosen before any evaluation;

plus the full test set only for the final comparison models.
