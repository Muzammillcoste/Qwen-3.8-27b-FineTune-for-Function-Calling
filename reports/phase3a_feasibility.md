# Phase 3A feasibility results (MEASURED on Kaggle, 2026-10-19)

The values below are aggregates of `results/phase3a/extracted/report.json`. The raw outputs are not version-controlled. The 100-example pilot is a feasibility check, **not an experimental finding**.

| Item | Measured |
|---|---|
| Hardware | 1 × Tesla T4 (cc 7.5, 15.64 GB), torch 2.11.0+cu128, transformers 5.18.0, bitsandbytes 0.50.2. `bf16_supported=True` is torch's emulation flag; training ran in **fp16** as configured. |
| Data / mask checks | template sha matches; eval prompt = training prefix; supervised tokens 3,601 / 56,954 (6.3%) |
| Model load | 48.4 s; 2.65 GB VRAM after load |
| **Training** | 13 steps, 56,954 non-padding tokens in 231.0 s = **246.5 tokens/s**, mean step 17.8 s |
| **Peak VRAM** | **9.00 GB** allocated / 9.75 GB reserved (at most 1,447 tokens per example) |
| Stability | no NaN/Inf; gradient norms 0.13–2.10 (clip 1.0); loss 0.0009–0.29 (it started low: 0.07) |
| Resume | interrupted at step 6, resumed into a fresh model to 13/13; optimizer, scheduler and RNG restored |
| Adapter reload | OK (66 MB) |
| Base model, dev 100 (eval-v2) | strict 82/100; schema-aware 82; parse 98; tool name 98; argument value 166/184 |
| Pilot adapter, dev 100 | strict 77/100; schema-aware 78; parse 98; tool name 98; argument value 164/184 |
| Paired, pilot − base (strict) | −0.05; McNemar p = 0.0625 (5 vs 0 discordant); bootstrap CI [−0.10, −0.01] (descriptive only) |
| Generation | base 3.07 s/example (11.8 new tokens/s); adapter (unmerged LoRA) 5.24 s/example (6.9 tokens/s) |
| Total wall time | 19.9 min |
