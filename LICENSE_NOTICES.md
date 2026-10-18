# License and attribution notices

Facts below were checked on **2026-10-18** from public metadata and dataset/model cards only. No files were downloaded. They are not legal advice. Re-check them before any redistribution.

## Code

The license for this repository's code has **not yet been chosen** by the researcher. That decision is open.

## Salesforce xLAM Function Calling 60K (training source)

- Repository: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
- Revision observed: `26d14ebfe18b1f7b524bd39b404b50af5dc97866` (last modified 2025-01-24)
- License on the card: **CC BY 4.0**
- Access is **gated** (`gated: auto`). Users must log in and accept the gate: *"Acknowledge to follow corresponding license and cite APIGen to access the repository"*. The gate form asks for first name, last name, country and affiliation.
- An unauthenticated access check from this machine returned **HTTP 401 (GatedRepoError)**.
- The card restricts intended use: *"This release is for research purposes only in support of an academic paper."* It also refers to the Salesforce AUP and AI AUP.
- Required citation (APIGen): Liu et al., 2024, *APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets*, arXiv:2406.18518.
- Teacher-model provenance stated on the card: ids 0–33658 come from DeepSeek-V2-Chat, and the remaining rows from Mixtral-8x22B-Instruct. Those models' own terms are separate from the dataset's publication license. This project makes no legal claim about them.

### Handling rules for this project

- Never commit raw or derived xLAM data. `data/` is gitignored.
- Raw data stays unchanged and read-only.
- Kaggle uploads of xLAM-derived files must be **private** datasets attached only to the researcher's notebooks. Distribution rights must be verified before any publication.
- Every exported file carries the license acknowledgement in its manifest (`src/tooldistill/manifests.py`).

## BFCL (planned external evaluation)

- Dataset: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard. Revision observed: `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda`. Not gated. Card license: Apache-2.0.
- Evaluator: PyPI `bfcl-eval`, latest observed version `2026.3.23` (Apache 2.0). Upstream: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard (path confirmed live).
- Never used for training or for tuning selection.

## Qwen3-4B-Instruct-2507 (planned student)

- https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507. Revision observed: `cdbee75f17c01a7cc42f958dc650907174af0554`. Not gated. Card license: Apache-2.0.
- **No weights downloaded** in Phase 0.
