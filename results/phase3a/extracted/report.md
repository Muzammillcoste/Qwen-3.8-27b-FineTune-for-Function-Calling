# Phase 3A report — status `phase3a_done`

Feasibility only; pilot numbers are NOT experimental findings.

- GPU: Tesla T4 (cc 7.5), total 15.64 GB, free 15.52 GB, bf16 True; torch 2.11.0+cu128, transformers 5.18.0, bitsandbytes 0.50.2
- data/mask checks: template sha `64f85b198065`, 100 pilot examples, supervised tokens 3601 (6.3% of tokens), eval prompt == train prefix: True
- model load: 48.4 s, VRAM after load 2.653 GB

| eval | exact acc | parse | tool name | schema-valid pred | hallucinated | s/example | new tok/s |
|---|---|---|---|---|---|---|---|
| format_check (20 val, base) | 17/20 | 1.0 | 1.0 | 1.0 | 0.0 | 3.143 | 11.6 |
| baseline (100 val, base) | 82/100 | 0.98 | 0.98 | 0.97 | 0.0 | 3.068 | 11.75 |
| pilot adapter (100 val) | 77/100 | 0.98 | 0.98 | 0.97 | 0.0 | 5.236 | 6.89 |

- pilot training: steps 13 (interrupted at 6, resumed to 13/13; optimizer/scheduler/RNG restored: True), 56954 tokens in 231.0 s = **246.5 tok/s**, mean step 17.77 s, **peak VRAM 9.003 GB** (reserved 9.748 GB)
- losses: [0.07040952146053314, 0.12314662337303162, 0.03968114033341408, 0.0044118305668234825, 0.0009123687632381916, 0.03140239790081978, 0.19678260385990143, 0.03354790434241295, 0.10038555413484573, 0.03895076364278793, 0.02647886984050274, 0.20267440378665924, 0.2944651246070862]
- grad norms: [1.5818939208984375, 1.5208404064178467, 0.6618050932884216, 0.3510267734527588, 0.13419131934642792, 1.7400903701782227, 1.5659030675888062, 2.095270872116089, 0.7330570220947266, 0.6513251066207886, 0.4762282073497772, 1.6584314107894897, 1.8973608016967773]
- paired exact-match (base vs pilot, same 100): {'both_correct': 77, 'only_first': 5, 'only_second': 0, 'neither': 18, 'discordant': 5, 'mcnemar_exact_p': 0.0625}

**Projections (ESTIMATES from measured pilot throughput; pilot train 246.5 non-pad tokens/s; 3.07 s/example generation (batch 1))**
- training hours: {'random_1k (1 ep)': 0.62, 'diversity_1k (1 ep)': 0.63, 'random_3k (1 ep)': 1.88, 'random_1k x3 ep (compute-matched)': 1.87}
- eval hours per model by #examples: {'100': 0.09, '1000': 0.85, '2813': 2.4}

- total wall time: 19.9 min
