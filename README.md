# ToolDistill-Lab

Research code for **"Quality and Diversity Versus Quantity in Small Language Model Tool-Calling
Adaptation: A Controlled Data-Selection Study"**.

- Training source: [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) (CC BY 4.0, **gated**).
- External evaluation (planned): selected BFCL single-turn categories with the official evaluator.
- Student (planned): `Qwen/Qwen3-4B-Instruct-2507`, QLoRA adapters on free Kaggle GPUs.
- Teacher API budget: **USD 0**. There is no new teacher generation. xLAM rows were already produced by
  DeepSeek-V2-Chat and Mixtral-8x22B-Instruct. This is offline response-based SFT, not logit distillation.

The canonical specification is [ToolDistill_Lab_Codex_Master_Spec_v2_Existing_Data.md](ToolDistill_Lab_Codex_Master_Spec_v2_Existing_Data.md).

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffold, normalization/validation/provenance/dedup/split code, fixtures, tests, docs | done (approved) |
| 1 | Authorized download, full integrity, three-state requiredness, dedup, split, data audit | done (accepted; branch `research/phase1-audit`) |
| 2 | Single-call pool, chat-template lengths, candidate subsets, blinded audit design, eval/training plans | done (approved with changes) |
| 3A | Kaggle feasibility: notebook, QLoRA pilot, baseline on 100 val | **built; Kaggle run pending** (`docs/KAGGLE_GUIDE.md` §3) |
| 4 | Approved adapter runs | not started |
| 5 | Evaluation and report | not started |

**No experiment has been run, and this repository contains no research results.**

## Quick start (local, CPU)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements-local.txt
pip install -e .
pytest -q
ruff check .

# Inspect the SYNTHETIC fixture (works without HF access)
python scripts/inspect_xlam.py --source tests/fixtures/miniature_xlam.jsonl
# Run normalize -> validate -> dedup -> split -> leakage audit on the fixture
python scripts/prepare_dataset.py --input tests/fixtures/miniature_xlam.jsonl --out data/normalized/fixture --fixture
```

## Real data (Phase 1, after approval)

xLAM is gated. To get access:

1. Accept the conditions on the dataset page.
2. Run `hf auth login` (never put tokens in code, notebooks or chat).
3. Run `python scripts/check_access.py`.

The full steps are in [docs/KAGGLE_GUIDE.md](docs/KAGGLE_GUIDE.md#1-gated-dataset-access).

```bash
python scripts/inspect_xlam.py --source Salesforce/xlam-function-calling-60k --sample 100
python scripts/download_xlam.py --config configs/data.yaml
python scripts/prepare_dataset.py --config configs/data.yaml --seed 42
python scripts/audit_dataset.py --config configs/data.yaml   # needs Qwen tokenizer.json only (~11 MB, no weights)
python scripts/audit_selection.py --config configs/research.yaml   # Phase 2; tokenizer files only
```

## Pipeline order (mandatory, spec E5)

`normalize -> validate -> deduplicate/group -> split (group-aware, seeded) -> leakage audit -> [Phase 2: selection fitted on train only]`

## Layout and documented deviations from spec D

- The project root is this git repository. There is no nested `tooldistill-lab/` folder.
- Added modules:
  - `src/tooldistill/inspection.py`: builds the inspection report.
  - `src/tooldistill/pipeline.py`: the prepare pipeline.
  - `src/tooldistill/audit.py` and `token_lengths.py`: the Phase 1 data audit (`scripts/audit_dataset.py`, `tests/test_audit.py`).
  - `src/tooldistill/phase2.py`, `analysis.py`, `manual_audit.py`: Phase 2 orchestration, comparison tables and the blinded audit sample.
  - Diversity selection lives in `select.py`. `diversity.py` and `quality.py` remain stubs; Option A has no quality score.
- Added scripts:
  - `scripts/check_access.py`
  - `tests/fixtures/build_miniature_xlam.py`: the fixture generator.
  - `notebooks/build_notebook.py`: the notebook generator.
- Added tests: `tests/test_access.py` (mocked gated-access handling) and `tests/conftest.py`.
- Raw download uses `huggingface_hub.hf_hub_download` of the original JSON file at a pinned revision. This keeps the file byte-exact so its SHA256 is meaningful. `datasets.load_dataset(..., streaming=True)` is used for the 100-row inspection.
- Later-phase modules exist only as stubs that raise `PhaseNotImplemented`:
  - `quality`, `diversity`, `select`, `format_qwen`, `hardware`, `train`, `infer`, `evaluate`, `bfcl_adapter`, `stats`.
  - Their CLI commands exit with code 2.
- `tests/test_selection.py`, `test_chat_template.py` and `test_metrics.py` are skipped placeholders until their phase. Phase 0 does not fake them.
- Ruff line length is 110.

See [LICENSE_NOTICES.md](LICENSE_NOTICES.md) for data/model licensing and attribution.
