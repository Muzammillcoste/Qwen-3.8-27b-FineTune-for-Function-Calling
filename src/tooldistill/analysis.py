"""Phase 2 comparison tables: quality warnings, coverage, diversity, difficulty, sources, tokens.

Descriptive only. Nothing here is a quality score or an outcome measure.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from itertools import combinations
from typing import Any

from .select import args_bucket, length_bin, tools_bucket

CONTINUOUS = ("rendered_total", "n_tools", "n_args", "n_params_called_tool",
              "n_unknown_params_called_tool", "n_structured_args")


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    return statistics.fmean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def smd(a: list[float], b: list[float]) -> float:
    """Standardized mean difference (pooled SD). |SMD| < 0.1 is the usual 'balanced' rule."""
    ma, sa = _mean_sd(a)
    mb, sb = _mean_sd(b)
    pooled = math.sqrt((sa ** 2 + sb ** 2) / 2)
    return 0.0 if pooled == 0 else (ma - mb) / pooled


def entropy(counter: Counter) -> float:
    n = sum(counter.values())
    return -sum((c / n) * math.log2(c / n) for c in counter.values() if c)


def mean_pairwise_jaccard(word_sets: list[set[str]], max_pairs: int = 200_000,
                          seed: int = 42) -> float:
    pairs = list(combinations(range(len(word_sets)), 2))
    if len(pairs) > max_pairs:
        pairs = random.Random(seed).sample(pairs, max_pairs)
    tot = 0.0
    for i, j in pairs:
        a, b = word_sets[i], word_sets[j]
        u = len(a | b)
        tot += (len(a & b) / u) if u else 1.0
    return tot / len(pairs) if pairs else 0.0


def describe(ids: list[str], feats: dict[str, dict[str, Any]], cuts: list[int],
             heldout_functions: Counter, completion: dict[str, int]) -> dict[str, Any]:
    F = [feats[i] for i in ids]
    n = len(F)
    teacher = Counter(f["teacher"] for f in F)
    fn = Counter(f["function"] for f in F)
    words_all = [w for f in F for w in f["query_words"]]
    totals = sorted(f["rendered_total"] for f in F)
    held_total = sum(heldout_functions.values())
    return {
        "n": n,
        "teacher_deepseek_pct": round(100 * teacher["deepseek-v2-chat"] / n, 2),
        "tokens_sum_total": sum(totals),
        "tokens_sum_completion": sum(completion[i] for i in ids),
        "tokens_mean": round(statistics.fmean(totals), 1),
        "tokens_p50": totals[n // 2],
        "tokens_p90": totals[int(0.9 * (n - 1))],
        "tokens_max": totals[-1],
        "tools_bucket": dict(sorted(Counter(tools_bucket(f["n_tools"]) for f in F).items())),
        "args_bucket": dict(sorted(Counter(args_bucket(f["n_args"]) for f in F).items())),
        "length_bin": dict(sorted(Counter(length_bin(f["rendered_total"], cuts)
                                          for f in F).items())),
        "mean_n_tools": round(statistics.fmean(f["n_tools"] for f in F), 3),
        "mean_n_args": round(statistics.fmean(f["n_args"] for f in F), 3),
        "structured_arg_pct": round(100 * sum(f["n_structured_args"] > 0 for f in F) / n, 2),
        # quality warnings (descriptive)
        "label_status": dict(sorted(Counter(f["label_status"] for f in F).items())),
        "strict_schema_valid_pct": round(100 * sum(f["strict_schema_valid"] for f in F) / n, 2),
        "schema_consistent_pct": round(100 * sum(f["schema_consistent"] for f in F) / n, 2),
        "type_conflict_records": sum(f["has_type_conflict"] for f in F),
        "warning_records": sum(f["n_warnings"] > 0 for f in F),
        "default_mismatch_records_pct": round(100 * sum(f["has_default_mismatch"] for f in F) / n, 2),
        "unknown_param_records_pct": round(
            100 * sum(f["n_unknown_params_called_tool"] > 0 for f in F) / n, 2),
        # coverage / diversity
        "distinct_functions": len(fn),
        "distinct_arg_signatures": len({f["arg_signature"] for f in F}),
        "distinct_groups": len({f["group"] for f in F}),
        "function_entropy_bits": round(entropy(fn), 3),
        "max_records_one_function": max(fn.values()),
        "heldout_records_function_seen_pct": round(
            100 * sum(c for f_, c in heldout_functions.items() if f_ in fn) / held_total, 2),
        "query_word_types": len(set(words_all)),
        "query_type_token_ratio": round(len(set(words_all)) / max(len(words_all), 1), 4),
        "mean_pairwise_query_jaccard": round(mean_pairwise_jaccard([f["query_words"] for f in F]),
                                             4),
    }


def balance_vs(ids: list[str], ref_ids: list[str], feats) -> dict[str, float]:
    return {k: round(smd([feats[i][k] for i in ids], [feats[i][k] for i in ref_ids]), 3)
            for k in CONTINUOUS}


def overlap(a: list[str], b: list[str]) -> dict[str, Any]:
    sa, sb = set(a), set(b)
    return {"intersection": len(sa & sb), "jaccard": round(len(sa & sb) / len(sa | sb), 4)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 2), round(100 * (c + h), 2))
