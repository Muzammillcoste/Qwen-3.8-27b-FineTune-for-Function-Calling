"""Paired uncertainty for two models scored on the SAME examples (spec I4).

- Paired bootstrap 95% CI of the accuracy difference (resample examples with replacement).
- Exact two-sided McNemar test on discordant pairs.
- Wilson 95% interval for a single accuracy.
With one training seed these describe test-sample uncertainty only, not training variance.
"""

from __future__ import annotations

import math
import random
from typing import Any

STATS_VERSION = "stats-v1"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict[str, Any]:
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    d = only_a + only_b
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(d, i) for i in range(k + 1)) / 2 ** d) if d else 1.0
    return {"only_a": only_a, "only_b": only_b, "discordant": d, "p_value": round(p, 5)}


def paired_bootstrap_ci(a: list[bool], b: list[bool], n_boot: int = 10_000, seed: int = 42,
                        alpha: float = 0.05) -> dict[str, Any]:
    """CI for mean(b) - mean(a) over paired examples (b minus a, in accuracy points 0..1)."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and equal length")
    n = len(a)
    diffs = [int(y) - int(x) for x, y in zip(a, b, strict=True)]
    rng = random.Random(f"{STATS_VERSION}|{seed}")
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"diff": round(sum(diffs) / n, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "n": n, "n_boot": n_boot, "seed": seed}


def paired_report(a: list[bool], b: list[bool], **kw) -> dict[str, Any]:
    n = len(a)
    return {"acc_a": round(sum(a) / n, 4), "acc_a_wilson": wilson(sum(a), n),
            "acc_b": round(sum(b) / n, 4), "acc_b_wilson": wilson(sum(b), n),
            "bootstrap": paired_bootstrap_ci(a, b, **kw), "mcnemar": mcnemar_exact(a, b)}
