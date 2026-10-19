# Phase 3B readiness: two primary runs (NOT executed; awaiting approval)

## Frozen configuration

These settings are unchanged from the Phase 3A pilot. `configs/train_qlora.yaml` has `frozen: true`.

- **Model:** `Qwen/Qwen3-4B-Instruct-2507` at revision `cdbee75f…`.
- **Quantization:** 4-bit NF4 with double quantization, fp16 compute.
- **LoRA:** r 8, alpha 16, dropout 0.05 on q/k/v/o/gate/up/down; the LM head is excluded.
- **Sequence:** max length 1,536; micro-batch 1 × grad accumulation 8.
- **Optimizer:** learning rate 1e-4, cosine schedule, 3% warmup, paged AdamW 8-bit, max grad norm 1.0.
- **Training:** 1 epoch, seed 42.
- **Checkpoints:** every 25 steps. This does not affect the training math.

Each arm starts from the **original base checkpoint** with a **fresh LoRA**. The LoRA initialisation is seeded identically for both arms (tested). There is no warm start from `pilot_100` or from the other arm.

## Arm comparability

Measured with the real tokenizer and the exact training-example builder (`reports/phase3b_arm_comparability.json`).

| | random_1k | diversity_1k | check |
|---|---:|---:|---|
| examples | 1,000 | 1,000 | |
| optimizer steps (1 epoch) | 125 | 125 | equal ✔ |
| total tokens | 554,547 | 555,718 | +0.21% ✔ (≤ 5%) |
| supervised (target) tokens | 36,969 | 38,223 | +3.39% ✔ (≤ 5%) |
| DeepSeek share | 0.585 | 0.585 | ✔ (≤ 0.02) |
| max record tokens | 1,469 | 1,535 | ✔ (≤ 1,536) |
| distinct functions | 604 | 756 | the intended difference |
| mean tools / mean args | 2.915 / 1.976 | 2.897 / 1.991 | |
| sample-ID SHA256 (Phase 2 manifests) | `47dde02a…` | `4ef8b42a…` | unchanged ✔ |

Difficulty and length balance from Phase 2: every |SMD| ≤ 0.03. Strata counts are identical across the two arms.

**Remaining difference to disclose:** `diversity_1k` has 3.4% more supervised tokens (1,254), because its calls are slightly longer. This is within tolerance and is reported.

## Fixed evaluation sets

| Set | n | sample-ID SHA256 | file SHA256 | status |
|---|---:|---|---|---|
| `val_eval_400` | 400 | `1e0922140c339d2eba839ce64dfc9f5823d19cf100cd61ffb052fa6ad8f7098b` | `bdb1d44ac629da89d0c2f17db44d70c4fa2947a88f143421c6caf62037653ba7` | in the Kaggle bundle |
| `test_eval_1000` | 1,000 | `6066f7d3cf53a82bc404a1571f2cc3f430268eef0666274429539a9e28638ce4` | `c04187cd449673e3d487af9728553250431d07adda698051d23f5af947d9854e` | **SEALED** (local only) |

Full ID lists and per-item coverage categories are in `reports/eval_manifests/`.

## Commands

Local, already run. Re-running produces identical files:

```bash
python scripts/export_phase3b.py        # eval sets, arm checks, Kaggle bundle + project zip
```

Kaggle: one job per session, using `notebooks/kaggle_train_eval.ipynb`. **Edit Cell 1 only, then Run All.**

| Session | Cell 1 | What runs |
|---|---|---|
| 0 (optional, ~3 min) | `MODE="preflight"`, `RUN_ID="random_1k"` | environment, hashes, template, loss mask for the arm |
| 1 | `MODE="eval_base_val"`, `RUN_ID=None` | untuned base on `val_eval_400` |
| 2 | `MODE="train_eval"`, `RUN_ID="random_1k"` | train 125 steps, then evaluate the fresh-loaded adapter on `val_eval_400` |
| 3 | `MODE="train_eval"`, `RUN_ID="diversity_1k"` | same as session 2, for the other arm |

Local analysis after downloading the three output zips:

```bash
python scripts/summarize_results.py \
  --run base=results/phase4/base/predictions_base_val400.jsonl \
  --run random_1k=results/phase4/random_1k/predictions_random_1k_val400.jsonl \
  --run diversity_1k=results/phase4/diversity_1k/predictions_diversity_1k_val400.jsonl
```

## GPU-hour estimates

These are estimates derived from the **measured** Phase 3A rates (training 246.5 tokens/s; generation 3.07 s/example for the base and 5.24 s/example for an adapter).

| Job | Training | Evaluation | Overhead (install, download, loads) | Total |
|---|---:|---:|---:|---:|
| eval_base_val (400) | — | 0.34 h | ~0.1 h | **~0.45 h** |
| train_eval random_1k | 0.62 h | 0.58 h | ~0.12 h | **~1.3 h** |
| train_eval diversity_1k | 0.63 h | 0.58 h | ~0.12 h | **~1.3 h** |
| **Phase 4 primary total** | | | | **~3.1 h** |
| Later, sealed test_eval_1000 (base + 2 adapters) | — | 0.85 + 2 × 1.46 h | ~0.3 h | ~4.1 h |
| Later follow-ups (random_3k; random_1k × 3 epochs), each | ~1.9 h | ~0.6 h (val) | ~0.1 h | ~2.6 h |

**Kaggle persistence:** checkpoints live in `/kaggle/working`, which is lost if a session dies. A ~1.3 h job fits comfortably in one session. If a session dies, rerun the job from scratch: training is seeded and deterministic up to CUDA kernel nondeterminism.
