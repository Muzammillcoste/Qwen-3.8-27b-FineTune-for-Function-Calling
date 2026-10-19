# Kaggle workflow guide

Phase 0 prepares this workflow but does not run it. Nothing may run on Kaggle until Phase 3 is approved.

## 1. Gated dataset access

Do these steps yourself, once, on your local machine.

1. Sign in at https://huggingface.co.
2. Open https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k.
3. Read the card and the access conditions: *"Acknowledge to follow corresponding license and cite APIGen to access the repository"*.
4. Fill in first name, last name, country and affiliation, then accept.
5. Create a **read-scoped** token at https://huggingface.co/settings/tokens.
6. Log in locally with `hf auth login`. With an older CLI, use `huggingface-cli login`. The token is stored in your user HF cache, outside this repo. Never paste it into code, notebooks, configs, chat or logs.
7. Verify access with `python scripts/check_access.py`.
   - Exit code `0` means access is granted.
   - Exit code `3` means access was not granted, and the command prints the manual steps again.

## 2. Local preparation (Phases 1–2, after approval)

```bash
python scripts/download_xlam.py --config configs/data.yaml        # raw file, read-only, SHA256 recorded
python scripts/prepare_dataset.py --config configs/data.yaml --seed 42
python scripts/audit_selection.py --config configs/research.yaml  # Phase 2; STOP for approval
python scripts/export_kaggle.py --config configs/data.yaml        # Phase 3
```

## 3. Phase 3A run instructions (feasibility only)

### 3.1 Build the bundle locally

```bash
python notebooks/build_notebook.py
python scripts/export_kaggle.py        # -> data/kaggle_upload/{data/, project/project_source.zip}
```

| Output | Contents |
|---|---|
| `data/kaggle_upload/data/` | `pilot_train_100.jsonl` (100 TRAIN records, stratified random, **not an arm**), `val_sample_100.jsonl` (a fixed seeded VALIDATION sample, used for scoring only), `manifest.json` (hashes, IDs, template sha, model revision), `README_DATA.md` |
| `data/kaggle_upload/project/project_source.zip` | `src`, `configs`, `scripts`, `notebooks`, `tests`, `reports/phase2_manifests`, `requirements-kaggle.txt`. It contains **no data**. |
| `data/audits/val_sample_label_check.csv` | Optional independent human check of the 100 validation labels (Y/N/U). It stays local. |

### 3.2 Create two PRIVATE Kaggle datasets

Go to kaggle.com → *Datasets* → *New Dataset*.

1. **`tooldistill-data`.** Upload the 4 files from `data/kaggle_upload/data/` and set visibility to **Private**. The data is xLAM-derived from a gated source, so never make it public.
2. **`tooldistill-project`.** Upload `project_source.zip`; Kaggle extracts it. Set it to **Private**.

### 3.3 Create the notebook

1. *Code* → *New Notebook* → *File* → *Import Notebook* → `notebooks/kaggle_train_eval.ipynb`.
2. In the right panel, set:
   - **Accelerator:** GPU T4 x2. Only GPU 0 is used.
   - **Internet:** On. The Qwen weights (about 8 GB, not gated) download from Hugging Face. No token is needed, so never paste one.
   - **Persistence:** none needed.
3. Use *Add Input* to attach both private datasets.

### 3.4 Run in order (edit only Cell 1)

| `MODE` | What runs | Approx. time | Stop and check |
|---|---|---|---|
| `preflight` | GPU, versions, data hashes, template sha, loss mask, eval-prompt = train-prefix. **No model download.** | ~2 min | `status: preflight_ok`; GPU is a T4, not bf16; `supervised_fraction` |
| `format_check` | Load the 4-bit base model; generate on 20 val examples; parse the `<tool_call>` output | ~15–25 min (includes the download) | parse rate, malformed outputs in `predictions_base_format20.jsonl` |
| `phase3a` | format check + 100-example val baseline + 100-example pilot (interrupted at step 6, resumed in a fresh model) + adapter reload + pilot eval on the same 100 + projections | ~45–90 min (estimate) | `report.md` |

After each run, Cell 6 writes `/kaggle/working/phase3a_outputs.zip`. It contains `report.json`, `report.md`, `predictions_*.jsonl`, `pip_freeze.txt` and the final adapter (about 66 MB; checkpoints are deleted). Download it, or *Save Version*. Then place it in `results/phase3a/` locally and send me `report.md`.

### 3.5 If something fails

Stop and report. Do **not** change the model, sequence length, LoRA rank or batch size to make it fit; that needs approval.

| Symptom | Where | Meaning |
|---|---|---|
| `NoGPUError` | preflight | The accelerator is not enabled. |
| `InsufficientGPUMemoryError` | preflight | Less than 12 GB free; restart the session. |
| `RuntimeError: chat template sha … prompt inconsistency` | data checks | Wrong tokenizer revision or transformers version. |
| `status: cuda_oom` | report | Records peak reserved memory, then stops. **Diagnose; never auto-reduce.** |
| `NonFiniteLossError` | pilot | A NaN/Inf loss or gradient norm under fp16. Training stops and the steps are logged. |
| pip resolver warnings | Cell 4 | Kaggle's preinstalled packages may complain. The pinned versions in `pip_freeze.txt` are what counts. |

## 4. Phase 3B/4 primary runs (prepared; run ONLY after approval)

1. Rebuild the bundle locally: `python scripts/export_phase3b.py`. It adds the following to `data/kaggle_upload/data/` and refreshes `project_source.zip`:
   - `val_eval_400.jsonl`
   - `train_random_1k.jsonl`
   - `train_diversity_1k.jsonl`
   - an updated `manifest.json`

   The sealed test set is **never** added.
2. On Kaggle, create a **New Version** of each private dataset:
   - **`tooldistill-data`:** upload all files from `data/kaggle_upload/data/`.
   - **`tooldistill-project`:** upload the new zip.
3. Import the new `notebooks/kaggle_train_eval.ipynb`, attach both datasets (latest versions), set GPU T4 x2 and Internet On, and restart the session.
4. Run one job per session. In each session, edit Cell 1 and then Run All:
   - `MODE="eval_base_val"`, `RUN_ID=None`
   - `MODE="train_eval"`, `RUN_ID="random_1k"`
   - `MODE="train_eval"`, `RUN_ID="diversity_1k"`

   Each session writes `/kaggle/working/<MODE>_<RUN_ID>_outputs.zip`. Download it and unzip it into `results/phase4/`, which is gitignored.
5. Locally, run `python scripts/summarize_results.py` with the three prediction files (see `reports/phase3b_readiness.md`).

## 5. Later phases (not approved yet)

- Train the 1k/3k arms one per session, using `RUN_ID`.
- The notebook never loops over arms.
- Resume from checkpoints restores optimizer, scheduler and RNG state. This is verified in a CPU test and exercised in the Phase 3A pilot.

## 5. Hardware notes

- Free-tier Kaggle GPUs (for example T4 ×2) are **two separate ~16 GB devices**, not 32 GB pooled. Use one GPU by default.
- P100 has been retired (2026 announcement). Record live `nvidia-smi` output for each run. Quotas change, so no fixed time-to-train is promised.
- If 4B QLoRA does not fit on T4 even after reducing sequence length, micro-batch and rank, **propose** a smaller student as a new approved experiment. Never switch silently.
