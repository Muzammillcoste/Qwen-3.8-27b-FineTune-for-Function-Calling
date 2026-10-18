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

## 3. Upload to Kaggle (Phase 3)

- **Data.** Create a **private** Kaggle dataset from `data/kaggle_upload/`. It contains the subset JSONL files, `val`/`test`, `manifest.json` and `README_DATA.md`. Never make xLAM-derived data public without verifying the redistribution terms.
- **Code.** Create a second private dataset from a `project_source.zip` or a pinned git snapshot. Exclude `data/`, `.venv/` and any secrets.
- In the notebook, attach both datasets. They appear read-only under `/kaggle/input/...`. Outputs go to `/kaggle/working/outputs`, which is ephemeral; persist it with *Save Version*.

## 4. Notebook (`notebooks/kaggle_train_eval.ipynb`)

The notebook is thin; all logic lives in `src/`. Edit **only** Cell 1:

```python
MODE = "preflight"   # preflight | pilot | train | eval
RUN_ID = None        # e.g. "random_1k" — one run per session
```

- Run `preflight` first. It checks `nvidia-smi`, versions, disk and data checksums, and starts no training.
- Then run `pilot`, which trains on 100 examples to measure VRAM, throughput and the loss mask. **The pilot is not a finding.**
- Then run the approved `train` runs one at a time, followed by `eval`.
- The notebook never loops over all arms.
- If a session times out, resume from the checkpoint. This restores optimizer, scheduler and RNG state.

## 5. Hardware notes

- Free-tier Kaggle GPUs (for example T4 ×2) are **two separate ~16 GB devices**, not 32 GB pooled. Use one GPU by default.
- P100 has been retired (2026 announcement). Record live `nvidia-smi` output for each run. Quotas change, so no fixed time-to-train is promised.
- If 4B QLoRA does not fit on T4 even after reducing sequence length, micro-batch and rank, **propose** a smaller student as a new approved experiment. Never switch silently.
