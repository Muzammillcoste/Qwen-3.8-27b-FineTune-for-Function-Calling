# ToolDistill-Lab — Codex Master Specification (v2.0)

**Research focus:** Data-selection quality, diversity, and quantity in *small language model tool-calling supervised fine-tuning*  
**Prepared:** 18 October 2026  
**Status:** DESIGN ONLY. No dataset has been downloaded, no experiment has been run, and no results have been observed.  
**Primary researcher:** Human researcher using Codex as an implementation assistant.  
**Operating constraint:** Local development and data preparation; **free Kaggle GPUs** for training; **$0 teacher API budget**.  
**Canonical file:** This file supersedes `SLM_Tool_Calling_Distillation_Codex_Master_Spec.md` v1.0 on all conflicts. Do not combine old teacher-generation instructions with this version.

> **Instruction for Codex:** Read the full document before making changes. Follow phases and approval gates exactly. DO NOT generate artificial research results or download a model / start GPU training unless the current phase permits it. This is a research project, not a quick demo. Prefer tested, transparent, small scripts to a complex framework.

---

## A. Decision memo — what changed and why

### A1. Final decision for the first study

**Use the existing [Salesforce xLAM Function Calling 60K](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) for TRAINING and [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard) for EXTERNAL EVALUATION.**

- Do **not** use Groq, DeepSeek, Gemini, OpenAI, Claude, or another hosted LLM to generate this study's training labels.
- Do **not** invent a new CRM/ERP dataset as a prerequisite. Generic function calling, not CRM-only calling, is the main study domain because xLAM covers many API types.
- No teacher API is required. xLAM already contains synthetic teacher-produced function-call answers from **DeepSeek-V2-Chat** (original IDs 0–33658) and **Mixtral-8x22B-Instruct** (remaining IDs), according to the Salesforce dataset card. Preserve source-model provenance wherever IDs make it possible.
- This is **offline, response-based teacher-data distillation / supervised fine-tuning (SFT)**; it is **not** knowledge distillation from teacher logits or hidden states, and no new teacher inference occurs.
- xLAM is already heavily verified: creators report **over 95% correctness** on a 600-example reviewed sample. As a result, we may not find enough meaningful label-quality differences for the original quality-versus-quantity hypothesis. There is an explicit research-feasibility gate before full training.
- BFCL is a benchmark; never train on or tune selection using its evaluation labels. Use its official scoring where feasible; partial BFCL categories are not an official overall leaderboard score.
- Important: The xLAM dataset does **not** guarantee suitable no-tool / ambiguity examples. Do not claim improvements on abstention based on this corpus; evaluate that separately as a diagnostic or introduce a separately disclosed supplementary corpus in a follow-on study.

### A2. Revised, defensible research title

**Primary working title:** *Quality and Diversity Versus Quantity in Small Language Model Tool-Calling Adaptation: A Controlled Data-Selection Study*.

**Alternative only if the quality gate succeeds:** *Can Higher-Quality Synthetic Function-Calling Examples Offset Lower Training Volume?*

A scientifically honest framing matters: with a prefiltered public corpus we can reliably study **selection policies** (and their proxies for quality) and sample size; we cannot automatically claim to have measured genuine correctness quality without independently verifying the selected and control sets.

### A3. Research questions

1. **Equal examples:** Does a transparent selection policy improve held-out function-call accuracy relative to uniform sampling of the same size and source pool?
2. **Equal examples:** Does including diversity coverage help beyond a quality heuristic?
3. **Different sizes:** Can a carefully selected 1,000-example set match or outperform 3,000 randomly selected examples from the same eligible pool?
4. **Budget control:** Do findings persist after comparing training-token counts, gradient updates and actual runtime fairly?
5. **Validation:** Do independently annotated quality differences predict gains? If not, present study as data selection, not data correctness.
6. **Generalization:** Does any gain show up on a truly held-out internal xLAM split and selected BFCL single-turn categories?

**Null results are valid.** Do not assume that quality must beat quantity.

### A4. The scientific risks (must be visible in final report)

| Risk | What Codex must do |
|---|---|
| xLAM is already prefiltered | Show full quality-score distribution; require evidence of meaningful differences |
| Selection heuristic rewards easier examples | Match or report difficulty proxies, example lengths, number of tools and multi-call rates |
| Different teachers created xLAM rows | Preserve original ID; track original-model source; stratify/source-match across subsets |
| Scores measured by code are imperfect correctness proxies | Separate syntax/schema checks from *semantic correctness*; hand-audit a sample |
| Bigger data implies more training tokens / compute | Report primary fixed-epoch comparison AND secondary budget-matched control if feasible |
| xLAM may contain near duplicates | Split/group before selecting; detect contamination; log unresolved overlap |
| BFCL differs from xLAM | Report external domain shift; only selected BFCL categories, never an official BFCL overall rank |
| Public BFCL could have been seen during original model training | No claim of guaranteed uncontaminated test; include limitations |
| 4B QLoRA may not fit or finish on free Kaggle | Run timed 100-example pilot, preflight and automatic safe resume; never silently switch base model |
| Source may have zero no-tool labels | Do not describe no-tool cases as trained unless actually added and disclosed |
| Multiple runs are expensive | Stage runs by priority; never claim statistical reproducibility from one training seed |

---

## B. Sources, licenses and access

### B1. Salesforce training source

- Dataset: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
- Authors/paper: *APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets* (2024), https://arxiv.org/abs/2406.18518
- Card license: **CC BY 4.0**. Follow original **access acknowledgement, conditions, required attribution, and current documentation**, not just the license badge.
- Dataset repo is publicly discoverable but **gated behind acceptance of conditions**. Human must open the HF page, accept, and authenticate if needed. The script must catch 401/403 errors and explain what to do; never scrape around gates.
- Example columns described by the card: `query`, `tools`, `answers`; JSON-serialized fields (`tools`, `answers`, possibly `query`) must be parsed according to **inspected actual types**, never assumed. An `id` is used by the source dataset and should be preserved when present.
- Each `tools` item has `name`, `description`, and a `parameters` mapping, often with `type`, `description`, `required` fields. This is **NOT necessarily JSON Schema** and must be converted for the Qwen chat template carefully.
- Each `answers` entry has `name` and `arguments`; **multiple expected calls can exist**. Do not silently keep only the first call.
- Card: 33,659 original rows (IDs 0–33658) generated using DeepSeek-V2-Chat; remainder Mixtral-8x22B-Instruct. Record which original field identifies the ID and verify its interpretation on inspection; don't infer source by local DataFrame index after shuffling.
- The card reports 60,000 rows and three verification stages; its stated 95%+ correctness estimate is a property of the dataset's sampling study, NOT the true correctness of every specific example.

### B2. BFCL evaluation source

- Official benchmark and versions: https://gorilla.cs.berkeley.edu/leaderboard
- Official dataset: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
- Evaluator and instructions: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard (confirm active upstream path during Phase 0), and https://pypi.org/project/bfcl-eval/
- Only test on the applicable **non-live single-turn** categories that the evaluation harness supports on local models. Prioritize simple, multiple-function, parallel only if parser handles it, and irrelevance if independently appropriate.
- BFCL categories and formats may evolve. **Pin official benchmark dataset version and evaluator commit/package**, enumerate exact category names from installed version, and record tested IDs. Do not invent category names or hardcode old counts without inspecting them.
- Do not report public leaderboard ranking unless official full benchmark protocol was completed. A subset score must be labeled **BFCL subset, protocol/version and N**.
- If the official evaluator cannot be integrated without infrastructure beyond budget, report internal held-out metrics and make BFCL support a future extension; **do not pretend a home-grown JSON checker equals BFCL**.

### B3. Qwen student and training libraries

- Base: `Qwen/Qwen3-4B-Instruct-2507` — https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- Maintain exact model revision/commit and tokenization/chat-template version in run manifests.
- Qwen's tool template uses `<tool_call>...` blocks and its `apply_chat_template(..., tools=...)` format. **Use the model's actual tokenizer template, not an invented one**.
- Hugging Face TRL: https://huggingface.co/docs/trl/sft_trainer
- Tool datasets: https://huggingface.co/docs/trl/dataset_formats
- HF PEFT/QLoRA: https://huggingface.co/docs/peft/developer_guides/quantization
- Bitsandbytes quantization: https://huggingface.co/docs/transformers/quantization/bitsandbytes
- Kaggle notebook help: https://www.kaggle.com/docs/notebooks
- Kaggle efficient GPU use: https://www.kaggle.com/docs/efficient-gpu-usage
- Kaggle P100 retirement (2026): https://www.kaggle.com/product-announcements/735239

### B4. Free hardware and API budget

- Kaggle free notebooks may offer T4 x2 (each GPU ~16 GB); **memory does not become one pooled 32 GB device**.
- Nominal quotas and sessions are subject to change and account availability; record live device availability and actual `nvidia-smi` output at runtime. Never guarantee a fixed time-to-train.
- The project's API spend limit is **USD 0**: no external model teacher endpoint calls at any phase. Hugging Face dataset/model downloads and Kaggle notebook use are separate from model inference API spending.
- May require internet-enabled Kaggle runtime to download public packages/model; if unavailable, stage HF cache in a Kaggle dataset, respecting original access rules.
- Store only **LoRA adapters, results and configs**, not merged multi-GB full weights by default.

---

## C. Main research architecture

```text
                       LOCAL MACHINE (CPU, no GPU)
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Download xLAM from authorized Hugging Face access                 │
 │            ↓ inspect raw schema + ID/source                         │
 │  Normalize → validate tool specs / labels → deduplicate             │
 │            ↓ group-aware train/val/test split BEFORE selection       │
 │  EDA + independent annotation + quality feasibility gate            │
 │            ↓ four candidate subset manifests                        │
 │  Render Qwen-native tools/messages → token-length analysis          │
 │            ↓ export preprocessed JSONL + manifest + checksums       │
 └─────────────────────────────────────────────────────────────────────┘
                                │
                       Kaggle Dataset upload
                                │
                                ▼
                       KAGGLE FREE GPU
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Thin notebook imports scripts from project ZIP/repo/dataset        │
 │  Hardware preflight → base inference smoke test                     │
 │            ↓ 100-example QLoRA training benchmark                   │
 │  Researcher approves GPU budget / experimental sequence             │
 │            ↓ train independent LoRA adapters for each subset        │
 │            ↓ local held-out evaluation, same decoding config        │
 │            ↓ BFCL external evaluation on feasible subset           │
 │  CSV metrics + per-example predictions + adapters + saved logs      │
 └─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                       LOCAL ANALYSIS / REPORT
             paired test-set comparisons, cost, limitations
```

**No hosted database, deployment API, microservices, web dashboard, vector DB or frontend are required.** A CLI-first research repository is preferable.

---

## D. Repository layout — required

```text
tooldistill-lab/
├── README.md
├── LICENSE_NOTICES.md
├── .gitignore
├── pyproject.toml
├── requirements-local.txt
├── requirements-kaggle.txt
├── configs/
│   ├── data.yaml
│   ├── research.yaml
│   ├── train_qlora.yaml
│   ├── eval.yaml
│   └── experiments.yaml
├── docs/
│   ├── RESEARCH_PROTOCOL.md
│   ├── DATASET_CARD.md
│   ├── SELECTION_METHODS.md
│   ├── KAGGLE_GUIDE.md
│   ├── REVIEW_CHECKLIST.md
│   ├── EXPERIMENT_LOG.md
│   └── PAPER_OUTLINE.md
├── src/tooldistill/
│   ├── __init__.py
│   ├── schemas.py
│   ├── io.py
│   ├── provenance.py
│   ├── normalize_xlam.py
│   ├── validate.py
│   ├── deduplicate.py
│   ├── split.py
│   ├── quality.py
│   ├── diversity.py
│   ├── select.py
│   ├── format_qwen.py
│   ├── hf_download.py
│   ├── manifests.py
│   ├── hardware.py
│   ├── train.py
│   ├── infer.py
│   ├── evaluate.py
│   ├── bfcl_adapter.py
│   ├── stats.py
│   └── cli.py
├── scripts/
│   ├── inspect_xlam.py
│   ├── download_xlam.py
│   ├── prepare_dataset.py
│   ├── audit_selection.py
│   ├── export_kaggle.py
│   ├── kaggle_preflight.py
│   ├── train_adapter.py
│   ├── eval_adapter.py
│   └── summarize_results.py
├── notebooks/
│   └── kaggle_train_eval.ipynb
├── tests/
│   ├── fixtures/
│   │   └── miniature_xlam.jsonl
│   ├── test_normalization.py
│   ├── test_validation.py
│   ├── test_leakage.py
│   ├── test_provenance.py
│   ├── test_selection.py
│   ├── test_chat_template.py
│   ├── test_metrics.py
│   └── test_budget_guards.py
├── data/              # gitignored; never commit large source files
│   ├── raw/
│   ├── normalized/
│   ├── splits/
│   ├── selected/
│   ├── kaggle_upload/
│   └── audits/
├── artifacts/         # gitignored; checkpoints/adapter outputs
├── results/
│   ├── predictions/
│   ├── tables/
│   └── plots/
└── reports/
    └── study_report.md
```

Codex may simplify module boundaries if the number of files introduces overhead, but must document any deviation. Everything must work on Windows local development as well as Linux Kaggle, avoid hardcoded POSIX-specific paths, and never use personal absolute paths.

---

## E. Dataset preparation, provenance and leakage controls

### E1. Access workflow

1. Human accepts the terms at the original Salesforce xLAM HF repository and authenticates using the standard HF CLI if gated.
2. Download dataset through `datasets.load_dataset("Salesforce/xlam-function-calling-60k", revision=<pinned_commit>)`, after inspecting available split names. Do **not** assume there is a usable predefined test split.
3. If HTTP 401/403, exit with a precise nonfatal message linking to the dataset card. No unauthenticated scraping.
4. Record repo ID, commit SHA, source split, row count, raw-file SHA256, download timestamp, license/attribution requirements, and any visible dataset acknowledgement.
5. Keep raw dataset unchanged and read-only; write derived files elsewhere. Never publish gated raw data to an unrestricted Kaggle dataset without verifying distribution rights and terms. If necessary, keep datasets private and attach them to the research notebook privately.

### E2. Inspect before normalizing

Create `reports/dataset_inspection.md` containing:

- Actual dataset schema, first 10 parsed records, field types and null counts.
- Verified row count and ID uniqueness; min/max IDs and missing IDs.
- Original teacher source counts by original ID, **only after verifying that the source row ID field is real**.
- Single versus multiple `answers` distribution.
- Calls with multiple available tools; per-record API/schema token lengths.
- Plain-text/JSON-encoded fields; malformed JSON counts with sample record IDs.
- Number of examples where names or argument schemas do not validate.
- Task-domain frequency if reliably inferable; do not fake a domain classifier.
- Potential no-tool/empty-answer records (do not assume they exist).
- Short inspection output that is safe for terminal display, no dumping all 60k records.

### E3. Canonical normalized record

Represent one source record as a typed Python structure, for example:

```json
{
  "sample_id": "xlam:12345",
  "source_dataset": "Salesforce/xlam-function-calling-60k",
  "source_revision": "ACTUAL_COMMIT_SHA",
  "original_id": 12345,
  "teacher_source": "deepseek-v2-chat",
  "query": "Get the order details for order 42",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_order",
        "description": "Get an order by its ID",
        "parameters": {
          "type": "object",
          "properties": {"order_id": {"type": "integer"}},
          "required": ["order_id"],
          "additionalProperties": false
        }
      }
    }
  ],
  "expected_calls": [
    {"name": "get_order", "arguments": {"order_id": 42}}
  ],
  "source_group_id": "canonical-hash-from-normalized-query-and-api-family",
  "call_count": 1,
  "quality_proxy": null,
  "split": null,
  "metadata": {"conversion_notes": [], "validation_flags": []}
}
```

**This is an illustrative record, not verified literal xLAM data.** Keep actual IDs, schema data and call order intact. Original fully-qualified names with dots should not be renamed silently. Store raw-normalized hash pair to verify provenance.

### E4. Schema conversion rules

- Map original simple types: `int`→`integer`; `float`→`number`; `str`/`string`→`string`; `bool`→`boolean`; `list`→`array` (requires valid `items` if available); `dict`→`object` (children if available). Unknown type → flag for review, do not silently force `string`.
- For each argument: transfer original description and required status; `required` is a list of parameter names, not per-property Boolean.
- Keep optional arguments optional; do not manufacture missing required values.
- Use current JSON Schema validator for *schema validity* and `jsonschema` to validate answer arguments. JSON Schema correctness is **not** proof of semantic correctness.
- Names may be namespaced or dotted. Preserve them and test that the Qwen tokenizer/chat template renders them correctly.
- Multi-call `answers` retain their order; primary experiment may restrict to single-call samples, but record all exclusions and reasons. This is a choice to make after inspecting distribution.
- Tool schemas are user-controlled input for the model; prevent prompt-injection markup in descriptions from affecting validation/code execution. Never execute any sourced tool, only validate its declaration.

### E5. Dedup and split BEFORE selecting

**Order of operations is mandatory:** normalize -> deduplicate/group -> split -> fit selection policy on eligible train -> choose subsets. Do not select full corpus then split (leaks related examples).

- Exact duplication: canonical JSON of normalized query + tool signatures + expected calls.
- Near duplication: normalized query text and tool signature; implement a deterministic fingerprint/similarity method. Flag and group likely paraphrase families, do not blindly delete semantically different labels.
- Group ID uses normalized request, identical expected function(s) and API/tool-family fingerprint; store grouping algorithm version and collision diagnostics.
- Train/validation/internal test: target **80/10/10 percent of eligible data** via group-aware deterministic seeded split, with source- and broad-difficulty stratification where feasible. Actual counts may differ.
- Do not use validation/test to set quality thresholds or compute diversity centers. Final test must remain sealed until the model/eval pipeline is stable. Validation only for development.
- Reserve an optional *tool-family-held-out* challenge split **only if feasible**; this is a distinct generalization measure, not the default split.
- Audit exact + near duplicates across train/val/test and between xLAM and chosen BFCL subset when technically practical. If overlap exists, exclude or flag; disclose limitations.
- Human-approved seed default 42. Log dataset checksum; fail if loaded revision changes without re-audit.

---

## F. Quality, diversity and human audit

### F1. Define what 'quality' means BEFORE running experiments

A data-quality score is a **proxy** with components:

1. **Tool identity integrity:** called function present in available tools.
2. **Argument/schema validity:** expected arguments type-check; required fields present, no spurious fields unless allowed.
3. **Explicit grounding signals:** string/integer argument values plausibly grounded in query text where checkable, but synonyms/inference make this only a noisy heuristic.
4. **Consistency:** no contradictory calls, unexplained duplicate calls, incomplete annotations.
5. **Complexity coverage / difficulty:** record independently; do NOT automatically treat long/complex examples as low-quality.
6. **Deduplication:** record separately as diversity/novelty rather than incorrectly equating rare examples with truthful ones.

Use a `quality_proxy_score` plus named subcomponents and `score_version`. Establish threshold and weights *before seeing evaluation results*. `quality_1k` must not be chosen using test performance.

**Don't reward only simple examples.** A shorter clean query is not intrinsically higher quality than a multi-argument example. A strict JSON schema check may eliminate almost none of already-verified xLAM; codex must not manufacture artificial ranking from arbitrary punctuation/length.

### F2. Required research feasibility gate

After full source inspection (before GPU):

- Produce 20-bin quality score histogram; percentages valid/invalid per proxy.
- Compare candidate `random_1k` and `quality_1k` proxy distributions, token-length distributions, number of tools, call count, and original teacher source.
- Perform **at least 100 human-reviewed examples**, preferably 50 from each candidate subset, randomized and blind to subset where practical. Audit actual semantic correctness and rubric agreement; Codex cannot invent these labels.
- If independent quality differences are insignificant/unclear, **do not call the comparison a proven quality experiment**. Proceed only as transparent data-selection/coverage research after human approval, or introduce a justified separate dataset in v3.
- The user may approve a narrower study even if no quality separation emerges; Codex must surface the limitation plainly.

### F3. Diversity selection

Implement transparent selection (no paid embeddings required):

- Basic method: stratify by source teacher, call count, tools count, arguments count, query length bins, API namespace/family; within strata prefer high quality and de-duplicated examples.
- Optionally `scikit-learn` TF-IDF embedding + seeded k-means or farthest-point selection on a manageable eligible candidate pool. Do not add heavyweight sentence-embedding models unless CPU runtime supports them and the method is justified.
- Deterministic and auditable; store decisions and quality/diversity score for selected and unselected rows.
- `quality_1k` and `quality_diverse_1k` must be distinct algorithms; show overlap and Jaccard index.
- Source-model ratio must be matched within reasonable tolerances. If matching source and difficulty causes insufficient samples, stop and report rather than violating constraints invisibly.

---

## G. Experiment matrix, comparisons and resource gates

### G1. Required runs

| Run key | Unique selected training examples | Rule | Status |
|---|---:|---|---|
| `base_0` | 0 | Untuned `Qwen/Qwen3-4B-Instruct-2507` | MANDATORY |
| `random_1k` | 1,000 | Random from eligible train, seed 42 | MANDATORY |
| `quality_1k` | 1,000 | Fixed quality heuristic, composition checked | MANDATORY if research gate passes |
| `quality_diverse_1k` | 1,000 | Quality + diversity | Desirable; after budget check |
| `random_3k` | 3,000 | Random from same source train pool | MANDATORY for quantity conclusion if affordable |

**Do not describe these as four different base models.** They are separate independent **LoRA adapter training runs** initialized from the same frozen base. Never train `quality_1k` on top of the `random_1k` adapter or carry optimizer state between runs.

### G2. 100-example pilot before real training

- First local lightweight unit tests with fixture rows.
- In Kaggle test **1 normal example** end-to-end: render tool schema, prompt, expected target, loss mask; produce baseline prediction, parse and score.
- Then train one 100-example **pilot** adapter, with short max sequence (e.g., 1,024–1,536) to test QLoRA memory and gradient flow; pilot is **not a research finding**.
- Measure max GPU VRAM, wall-clock time, step/s, actual tokens processed, disk and output artifacts.
- Stop if CUDA OOM, broken label mask, >5% unintended truncation, or runtime implausible under current quota; only proceed after fixes and approval.
- If 4B remains unworkable on T4 after reducing sequence length, microbatch, LoRA rank and attention implementation, **propose** smaller student (e.g., Qwen3 1.7B) as a new approved experiment. Never silently change model ID.

### G3. Comparison protocol — how to avoid misleading conclusions

**Primary:** `random_1k` vs `quality_1k` vs `quality_diverse_1k`, same example count and same optimization hyperparameters/epochs, but report exact training token totals and training steps.

**Quantity test:** compare `quality_1k` vs `random_3k` with a clearly labeled *fixed-epoch* schedule (more data means more updates/compute); report that result as a practical data-efficiency trade-off, NOT proof that quality intrinsically beats quantity at equal compute.

**Secondary compute-controlled sensitivity:** if budget permits, match training-token exposure / optimizer steps using deterministic resampling or matched sample weights; keep examples and repeated-exposure accounting explicit. This does not erase all differences but narrows confounding. No requirement to do 20+ runs on free Kaggle.

Training budget priorities when GPU time is scarce:

1. `base_0` + 100-sample pilot.
2. `random_1k` + `quality_1k` **if quality gate passed**.
3. `random_3k` (needed for quality-vs-quantity contrast).
4. `quality_diverse_1k` (ablation).
5. Repeat essential comparisons with 2–3 seeds **only if resources allow**; if not, explicitly limit causal/generalization claims. Test-set bootstrap intervals alone do not measure seed-to-seed training variability.

### G4. Controlled settings

All runs have same base model revision, tokenizer template revision, quant settings, LoRA target modules, lr, optimizer, input truncation policy, inference parameters, target label serialization, evaluation sets, primary seed and reporting code. Difference = selected examples (and documented sample-size change for random_3k).

- Do not swap teacher sources between arms without source matching.
- Compute effective sequence lengths **after** tools are rendered; extremely large tool schemas may consume budget before any target tokens.
- Save every train subset's IDs to an immutable manifest and SHA256. This is crucial for repeatability.

---

## H. QLoRA training specification — Kaggle free tier

### H1. Base configuration (defaults, subject to pilot)

```yaml
model_name: Qwen/Qwen3-4B-Instruct-2507
model_revision: PIN_AFTER_DOWNLOAD
seed: 42
method: qlora_sft
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: bfloat16_if_supported_else_float16
lora:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules: all-linear  # verify supported module names and exclude inappropriate output heads
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 0.0001
  num_train_epochs: 1
  max_seq_length: 1024  # raise to 1536/2048 only if feasibility and no significant label truncation
  gradient_checkpointing: true
  optimizer: paged_adamw_8bit
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  save_steps: 100
  logging_steps: 10
  eval_strategy: steps_or_epoch_as_supported_by_pinned_trl
  packing: false
  mixed_precision: auto
  completion_only_loss: true
inference:
  do_sample: false
  max_new_tokens: 256
  batch_size: 1
```

These are *starting suggestions*, not guaranteed optimal values or compatibility. Codex must test actual APIs for the pinned TRL/PEFT/transformers versions. Do not copy unsupported keys directly into Python config. Use only one GPU by default, avoid unintended data parallel / accelerate distributed launches, and avoid simultaneously loading a second full 4B model in the same VRAM.

### H2. Correct training target and tokenizer template

- Convert xLAM tool descriptions into schema format expected by Qwen `tokenizer.apply_chat_template(..., tools=[...])`.
- Represent prompt as system instruction + user query + tools; assistant output must contain **one or multiple canonical tool calls** using native Qwen `<tool_call>` tags from the installed tokenizer template, not a new arbitrary JSON syntax.
- Preserve xLAM labels exactly in intended function identity and arguments (up to valid canonical JSON formatting). Verify that each expected call appears when rendered.
- The native Qwen template may accept `messages` with `assistant.tool_calls`; test with a real tokenizer (CPU, no model weights needed) on 3 fixture records including multi-call.
- Use completion-only (or assistant-only) loss; confirm `labels == -100` over prompt/tool documentation and labels active on assistant tool-call tokens. A successful `Trainer` start does not prove the loss mask is correct.
- Check EOS token e.g. `<|im_end|>` against actual template; avoid truncating before completing assistant targets.
- No hidden chain-of-thought / reasoning trace generation or labeling; Qwen3-4B-Instruct-2507 is instruction-oriented and should be used in its documented output mode.

### H3. Robustness, logging, safety

- Preflight `nvidia-smi`, CUDA, GPU name/free memory, RAM, disk, package versions, HF model SHA, dataset hash, batch/sequence settings; print concise report before allocating GPU.
- Log train steps, eval loss, throughput, elapsed seconds, peak VRAM, checkpoint size, count/tokens/examples encountered, sample truncation counts.
- Save **adapters**, trainer state, tokenizer/config references, manifest and reproducibility notes; no full model merge unless approved after study.
- All seeds fixed across `random`, `numpy`, `torch`, tokenizer/data sampler, and selection algorithm; note that full determinism is not guaranteed on every CUDA kernel.
- Fail and checkpoint gracefully on quota/session end when feasible; `resume_from_checkpoint` must actually restore optimizer/scheduler/random state. Re-loading a saved LoRA weight without trainer state is NOT an exact resume.
- Kaggle `/kaggle/input` is read-only and `/kaggle/working` may be ephemeral across sessions. Export outputs through Notebook Save Version / artifacts download or attached dataset in compliance with content licenses.
- Do not store tokens/secrets in `.ipynb`, logs or committed `.env`.

---

## I. Evaluation (metrics and category exclusions)

### I1. Evaluate without any model fine-tuning first

Evaluate `base_0` on the internal held-out test and, if feasible, selected BFCL categories using the **same prompt serialization and inference settings** as adapters. Keep fixed max_new_tokens and use deterministic decoding (`do_sample=False`); record truncation/unparsed outputs.

### I2. Metrics

| Metric | Definition | Important caveat |
|---|---|---|
| JSON/tool-block parse rate | Fraction of predictions parseable into one or more tool calls | High parse rate is not task success |
| Tool-name set accuracy | Exact expected tool names; optionally order aware | Distinguish missing/extra calls |
| Exact structured-call accuracy (primary) | All expected calls and argument key/value structures match after principled canonicalization | Don't equate JSON string formatting with semantic correctness |
| Required argument correctness | Per parameter (type + value) on matched tools | Only meaningful on applicable samples |
| Schema-valid predicted-call rate | Generated call matches supplied JSON schema | Can be valid but wrong for user request |
| Hallucinated-tool-call rate | Called a function absent from available tools | Report separately for call/no-call categories |
| No-tool/abstention error rate | On BFCL irrelevance (or separately verified no-tool tests) | **Not a trained capability in xLAM-only setup** |
| Eval inference length / latency | Mean or p50/p95 with hardware disclosure | Useful resource comparison, not accuracy |
| Training costs | Actual GPU runtime, tokens and max VRAM | No dollars if Kaggle free, but compute isn't free in scientific terms |

### I3. Parsing and ground truth

- Parse native Qwen `<tool_call>` blocks defensively; support multiple blocks.
- Normalize numeric JSON types only where semantic equivalence is well defined; don't coerce different strings or silently execute Python expressions.
- Never `eval` generated code, call external APIs, or execute tool descriptions. The evaluator is offline.
- For known-no-tool evaluation, distinguish direct answer from unrecognized malformed tool markup; include out-of-set tool calls as errors.
- For BFCL use its **official AST/execution evaluator** when applicable and clearly state category/version/IDs; do not replace it with a custom matcher and call the number an official BFCL score.
- Save per-example predictions (`sample_id`, `run_id`, expected, parsed, error category, evaluation version), not just aggregate accuracy.

### I4. Reporting + statistics

- Per arm report N train, unique tokens, total seen tokens, steps, epochs, teacher mix, quality-score distribution, % single/multi calls, time and VRAM.
- Main accuracy report: numerator/denominator, absolute percentage-point deltas, paired bootstrap confidence intervals over test examples where feasible; paired McNemar for exact accuracy with caveats.
- If only 1 model training seed, intervals describe uncertainty across sampled test examples, NOT variability from different training runs.
- Inspect at least 20 comparative failure cases, categorize selection vs argument error, invalid output, target ambiguity and cross-domain mismatch.
- Keep all negative findings and protocol deviations; no retroactive dataset tuning using final test results.
- Distinguish exploration/pilot from confirmatory experiment; never assert causality from observational subset correlations alone.

---

## J. Kaggle notebook and upload workflow

### J1. Local-first

The project source should be developed locally (Codex). No GPU needed for schema conversion, quality scoring, splitting, source analysis and exporting training subsets. Local internet + HF user access may be needed once for gated xLAM.

### J2. Export

`python scripts/export_kaggle.py --config configs/data.yaml --out data/kaggle_upload` should produce:

- `train_random_1k.jsonl`, `train_quality_1k.jsonl`, `train_quality_diverse_1k.jsonl`, `train_random_3k.jsonl` (only after selection gate).
- `val.jsonl`, `test.jsonl`, source provenance/ID lists and quality feature table.
- `manifest.json` with dataset SHA, source commit, count per arm, random seed, code commit, schema-v, render-template version, quality/selection parameters.
- `README_DATA.md` documenting format, license, gated-data handling and split restrictions.
- Small `project_source.zip` or a pinned Git repo snapshot for Kaggle, excluding private raw data/secrets.

### J3. Notebook design

`notebooks/kaggle_train_eval.ipynb` must be thin (the actual work lives in `src/`):

1. Cell: setup/requirements (verify compatible versions; never uncontrolled upgrades mid-study).
2. Cell: confirm GPU, memory, disk, dataset mounts and environment.
3. Cell: define/verify paths; test reading sample JSONL and source checksums.
4. Cell: load Qwen tokenizer, inspect native tool chat template and one loss mask.
5. Cell: run base-only evaluation on 10 fixture samples (not substantive scoring).
6. Cell: optional 100-example training pilot.
7. Cell: **manual run_id parameter**; one adapter training run per session by default.
8. Cell: eval adapter on validation/test, output predictions and metrics.
9. Cell: export `/kaggle/working/outputs/` as zipped artifacts, manifest, checkpoint references.

Never automatically execute all four training arms in one notebook run. Provide a clear human input `RUN_ID` and `MODE={preflight,pilot,train,eval}`. If Kaggle session times out, use checkpoint/resume.

### J4. Expected user commands (Codex must implement or update docs)

```bash
# Local: only after human accepts xLAM dataset conditions
python scripts/inspect_xlam.py --source Salesforce/xlam-function-calling-60k --sample 100

# Local: download authorized dataset and produce reproducible metadata
python scripts/download_xlam.py --config configs/data.yaml

# Local: normalization, group splitting, quality scoring, subset candidates
python scripts/prepare_dataset.py --config configs/data.yaml --seed 42

# Local: quality/separation audit; STOP for researcher's approval
python scripts/audit_selection.py --config configs/research.yaml

# Local: export input files for Kaggle only after approval
python scripts/export_kaggle.py --config configs/data.yaml

# Kaggle notebook (not local CLI by default): run MODE=preflight, then pilot
# Then RUN_ID=random_1k or another approved experiment
```

**These commands are the intended interface. Codex must create them and verify help/tests. They do not already exist.**

---

## K. Phases and mandatory human approval gates

### Phase 0 — requirements + source verification (**start here**)

- Read THIS master spec, inspect repository if existing, summarize scope and open ambiguities without blocking on questions that can be investigated.
- Create repo scaffolding/configs, pinned optional requirements after compatibility checks; unit tests, no GPU code execution.
- Add source/license/provenance documentation and research protocol.
- Implement dataset loader that can inspect schema from a tiny local fixture even when HF is gated.
- Attempt authenticated xLAM inspection only if access already exists; **never ask for or print access tokens**. If gated, show manual acceptance/login instructions and continue with fixtures.
- Show 20 normalized example records or synthetic **structural test fixtures clearly labeled as fixtures**, schema summary and test results.
- **STOP:** request human approval. No 60k bulk preprocessing, training or model download unless specifically approved.

### Phase 1 — full dataset integrity and split

- With HF access, download source; preserve full raw fields/revision metadata.
- Parse and normalize, validate JSON schema and answer format, dedup/group, split with audit.
- Create initial source analysis, field distribution and high-level task stats.
- Document exclusions + evidence.
- **STOP:** human reviews `dataset_inspection.md`, `DATASET_CARD.md`, split counts, source mix.

### Phase 2 — selection algorithms + quality feasibility gate

- Implement transparent quality proxy and diversity; create four candidate subsets with strict source stratification.
- Show actual score distributions, candidate source/domain/difficulty/token mixes.
- Prepare blinded annotation CSV/template for >=100 human labels; do not auto-fill semantic correctness.
- Compare independent audit results once the human provides them.
- Clearly recommend proceed / narrow title / redesign. If no meaningful quality spread, **do not call experiment 'true label-quality effect'**.
- **STOP:** human approval of subset strategy before training.

### Phase 3 — Kaggle export + GPU feasibility pilot

- Build thin notebook, Python CLI/library helpers, baseline inference and evaluation, adapter training; dry-run test with fixtures.
- Export Kaggle data with hashes (private when required).
- User uploads repo snapshot and prepared data to Kaggle notebook, enables a free GPU, runs `MODE=preflight` then `MODE=pilot`.
- Show max VRAM, measured throughput, pilot loss mask, actual truncated % and estimated runtime under quota.
- **STOP:** user approves cost/feasibility; if not viable suggest smaller study.

### Phase 4 — approved research runs

- Evaluate untuned base.
- Train independent adapters on same pinned model and selected subsets, according to priority G3.
- Validate and checkpoint each run; save actual step/token accounting.
- No change to selection rules after seeing test set.
- User supervises GPU sessions; Codex provides run/resume/eval commands and checks logs.
- **STOP:** review metrics and unresolved anomalies before paper writing.

### Phase 5 — evaluation and reporting

- Report held-out and external BFCL subset where feasible, with exact version and selected IDs.
- Produce paired metrics, resource plots, failure breakdown, honest limitations, and full run manifest.
- Write `reports/study_report.md` with Abstract, Introduction, Related Work, Methods, Results, Limitations, Conclusion and Reproduction Appendix.
- Label anything not actually run as **planned**, not completed.
- Do not automatically claim publishability, novelty, professor acceptance or research significance.

---

## L. Unit tests, acceptance criteria and failure policy

All of these must be tested before GPU training:

1. A fixture row with a string-encoded `tools` list parses correctly and converts to valid function JSON Schema.
2. A fixture row with **two expected tool calls** retains both, names and arguments.
3. A fixture row with malformed input fails with a logged reason, without terminating the whole dataset.
4. `required=True/False` correctly becomes JSON Schema `required` array.
5. Unknown types and partially absent parameters are explicitly flagged (never silently miscast).
6. Original sample IDs survive every transformation, split, filter and export.
7. DeepSeek/Mixtral provenance is derived from original IDs only and matches source card counts on full data if IDs intact.
8. Exact / near-duplicate grouping prevents leakage across train/validation/test.
9. Same seed + same source hash yields identical selected sample ID lists and manifest hashes.
10. Different seed changes random selection but not quality algorithm definition.
11. Quality score does not inspect held-out test labels or BFCL results.
12. Quality+diversity policy is demonstrably different from quality-only, or flagged.
13. Each split/subset cannot exceed eligible group count and adheres to source-mix constraints; errors are explicit.
14. Qwen chat template renders recognized `<tool_call>` output and correct tool documentation.
15. Completion loss mask excludes user/tool spec text and includes actual assistant target tokens.
16. Output parser rejects hallucinated tool names; multiple calls scored without losing extras.
17. Simulated zero-GPU and OOM preflight fail safely, with no baseline-score fabrication.
18. Kaggle notebook runs in **preflight** mode without accidentally starting training.
19. Every exported file has count, license acknowledgment, schema version and SHA manifest.
20. No API credentials, fake research results, external calls to arbitrary tools, or user/customer data are committed.

`pytest -q` must pass locally with only fixtures; `ruff check .` or equivalent lint must pass. Fail fast on unverifiable data claims, license/access conflicts, OOM, corrupted metrics, empty splits, invalid label masks and contamination. Never force the experiment to produce a favorable result.

---

## M. Research notes and methodological objections Codex must surface

1. **Existing data vs generating fresh data:** for a first $0 study, existing is superior for cost, reproducibility and comparison. Creating custom teacher data can be an **extension** if the xLAM quality range is too narrow; do not add it without new approval and methodological justification.
2. **Quality selection vs true data correctness:** formal validators catch schema issues but usually cannot tell if the selected call actually answers the user's meaning. Independent human annotation is needed to support the term 'quality'.
3. **Why QLoRA?** 4-bit base weights reduce GPU memory; LoRA adapter parameters receive gradients. The base model is frozen, so we don't need to save merged full weights.
4. **Why not 60k?** Free Kaggle constraints; research question is data efficiency at 1k/3k. Additional sizes can be added only after measured costs.
5. **Teacher terminology:** xLAM source rows were originally synthesized using two larger models; **no new teacher model or teacher inference is involved**. Saying 'teacher = DeepSeek' alone misrepresents the mixed data provenance.
6. **Synthetic teacher data rights:** preserve CC BY attribution and source access conditions; separate teacher original terms from dataset publication rights. Do not assert legal conclusions beyond documented dataset terms.
7. **No tool execution:** this study tests choosing tools and arguments, not whether real Salesforce/QuickBooks integrations function. Do not use or request production credentials.
8. **Model capability saturation:** original Qwen may already perform very well on simple tools, so a 'small gain' may be noisy. Better holdout selection and evaluation are more valuable than forcing difficult examples into labels.
9. **Paper novelty:** straightforward 1k-vs-3k comparisons are not inherently novel. Publishability requires well-motivated controls, analysis of failure mechanisms, honest replication and ideally a genuinely new insight.
10. **Professor outreach:** completed, independently verified experiments plus a GitHub repo and a methods report can support an MSc application. Do not claim a paper is accepted before peer review.

---

## N. Bib and practical reading (human researcher)

1. Salesforce AI Research: [APIGen / xLAM dataset](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) — read dataset card and access terms first.
2. Liu et al., [APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets](https://arxiv.org/abs/2406.18518) — understand how the current corpus was filtered.
3. UC Berkeley: [BFCL Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard) — benchmark categories and proper scoring.
4. Qwen: [Qwen3-4B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) — model/chat template/usage.
5. HF: [TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer) and [tool-calling datasets](https://huggingface.co/docs/trl/dataset_formats).
6. HF: [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization) — QLoRA setup and limitations.
7. Kaggle: [notebooks](https://www.kaggle.com/docs/notebooks) and [efficient GPU use](https://www.kaggle.com/docs/efficient-gpu-usage).

---

## O. FINAL EXECUTION RULE: first Codex session

**First Codex session MUST execute Phase 0 ONLY.** Use the accompanying prompt from the chat or the `Codex_Phase0_First_Prompt.txt` file. Phase 0 is approved for implementation; data scale-up, training, remote API generation and paper-results claims are not.

**Human check before Phase 1:** inspect source summary, field parsing, license/access instructions, a short fixture preview, unit-test log and exact Phase 1 proposed command. Then approve explicitly.
