"""Phase 2 candidate subsets (spec F3, G1; researcher decisions 2026-10-18).

Primary pool = TRAIN split ∩ single-call ∩ not an exact duplicate. Selection never sees
validation/test records (enforced) and uses one fixed seed.

All 1k arms share the SAME proportional stratum quotas, so they are matched on teacher source,
difficulty proxies and rendered token length by construction:

    stratum = teacher × tools_bucket × args_bucket × length_tertile × structured_args × tool_style

- ``random_3k``           seeded random order per stratum, first q3 records.
- ``random_1k``           first q1 records of the SAME order (so random_1k ⊂ random_3k).
- ``schema_consistent_1k`` same order, skipping records that are not schema-consistent: the arm
                           differs from random_1k ONLY where random_1k drew an inconsistent record.
- ``diversity_1k``        greedy coverage within strata (features below), separate seeded order.

Schema-consistent := strict JSON-Schema valid AND label_status == "valid" (no label warnings)
AND not in a conflicting-label input set. Declaration-only issues (e.g. default_type_mismatch)
are NOT used: they describe the tool declaration, not the label. It is NOT a semantic-correctness
claim.

Diversity features (exactly these; documented in docs/SELECTION_METHODS.md):
  1. called function name            (prefer functions not yet selected)
  2. argument-key signature          (function + sorted argument names; prefer unseen)
  3. query word novelty              (share of non-numeric word types not yet covered)
  4. dedup group                     (never two records from the same source group)
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .deduplicate import normalize_query_text
from .io import sha256_text
from .schemas import NormalizedRecord

SELECTION_VERSION = "select-v2"
ARMS = ("random_1k", "schema_consistent_1k", "diversity_1k", "random_3k")
_DIGITS = re.compile(r"^\d+$")


class SelectionError(RuntimeError):
    pass


# --------------------------------------------------------------------------- features


def tools_bucket(n: int) -> str:
    return "t1" if n <= 1 else ("t2-3" if n <= 3 else "t4+")


def args_bucket(n: int) -> str:
    return "a0-1" if n <= 1 else ("a2" if n == 2 else "a3+")


def is_schema_consistent(rec: NormalizedRecord) -> bool:
    m = rec.metadata
    return (bool(m.get("strict_schema_valid")) and m.get("label_status") == "valid"
            and "conflicting_labels_same_input" not in rec.flags)


def record_features(rec: NormalizedRecord, rendered_total: int | None = None) -> dict[str, Any]:
    call = rec.expected_calls[0] if rec.expected_calls else None
    called_tool_params = 0
    unknown_params = 0
    for t, req in zip(rec.tools, rec.metadata.get("requiredness", []), strict=False):
        if call and t["function"]["name"] == call.name:
            called_tool_params = len(t["function"]["parameters"].get("properties", {}))
            unknown_params = sum(1 for i in req["params"].values() if i["state"] == "UNKNOWN")
    args = call.arguments if call else {}
    words = {w for w in normalize_query_text(rec.query).split() if not _DIGITS.match(w)}
    return {
        "sample_id": rec.sample_id,
        "teacher": rec.teacher_source,
        "group": rec.source_group_id,
        "function": call.name if call else None,
        "arg_signature": (call.name + "(" + ",".join(sorted(args)) + ")") if call else None,
        "n_tools": len(rec.tools),
        "n_args": len(args),
        "n_params_called_tool": called_tool_params,
        "n_unknown_params_called_tool": unknown_params,
        "n_structured_args": sum(isinstance(v, (list, dict)) for v in args.values()),
        "query_words": words,
        "rendered_total": rendered_total,
        "schema_consistent": is_schema_consistent(rec),
        "label_status": rec.metadata.get("label_status"),
        "strict_schema_valid": bool(rec.metadata.get("strict_schema_valid")),
        "n_warnings": len(rec.metadata.get("validation_warnings", [])),
        "has_default_mismatch": any(f.startswith("default_type_mismatch") for f in rec.flags),
        "has_type_conflict": any(f.startswith("schema_argument_type_conflict")
                                 for f in rec.flags),
    }


# --------------------------------------------------------------------------- pool + strata


def build_pool(records: list[NormalizedRecord]) -> list[NormalizedRecord]:
    """Primary pool. Refuses any non-train record (selection must never see val/test)."""
    bad = [r.sample_id for r in records if r.split != "train"]
    if bad:
        raise SelectionError(f"selection pool must contain TRAIN records only; got {bad[:5]}")
    return [r for r in records if r.call_count == 1 and r.duplicate_of is None]


def length_bins(values: list[int], n_bins: int = 3) -> list[int]:
    """Tertile cut points computed on the TRAIN pool only."""
    s = sorted(values)
    return [s[int(len(s) * k / n_bins)] for k in range(1, n_bins)]


def length_bin(v: int, cuts: list[int]) -> str:
    for i, c in enumerate(cuts):
        if v < c:
            return f"L{i}"
    return f"L{len(cuts)}"


def stratum_key(f: dict[str, Any], cuts: list[int]) -> str:
    """select-v2 strata. v1 used only the first four parts; balance diagnostics on the TRAIN
    pool showed diversity_1k drifting on structured arguments and tool style (UNKNOWN-requiredness
    params), so both were added. No validation/test information is used."""
    return "|".join([f["teacher"], tools_bucket(f["n_tools"]), args_bucket(f["n_args"]),
                     length_bin(f["rendered_total"], cuts),
                     "struct" if f["n_structured_args"] > 0 else "flat",
                     "unk" if f["n_unknown_params_called_tool"] > 0 else "known"])


def allocate(sizes: dict[str, int], total: int, caps: dict[str, int] | None = None) -> dict[str, int]:
    """Largest-remainder proportional allocation, deterministic ties (by key), with caps."""
    caps = caps or sizes
    n = sum(sizes.values())
    if total > sum(min(sizes[k], caps[k]) for k in sizes):
        raise SelectionError(f"cannot allocate {total} from capacity {sum(caps.values())}")
    exact = {k: total * v / n for k, v in sizes.items()}
    q = {k: min(int(exact[k]), caps[k]) for k in sizes}
    order = sorted(sizes, key=lambda k: (-(exact[k] - int(exact[k])), k))
    while sum(q.values()) < total:
        progressed = False
        for k in order:
            if sum(q.values()) >= total:
                break
            if q[k] < caps[k]:
                q[k] += 1
                progressed = True
        if not progressed:
            raise SelectionError("allocation stalled; capacity exhausted")
    return q


def _seeded_order(ids: list[str], seed: int, salt: str) -> list[str]:
    ids = sorted(ids)
    random.Random(f"{SELECTION_VERSION}|{seed}|{salt}").shuffle(ids)
    return ids


# --------------------------------------------------------------------------- arms


@dataclass
class Subset:
    name: str
    sample_ids: list[str]
    algorithm: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def ids_sha256(self) -> str:
        return sha256_text("\n".join(sorted(self.sample_ids)))


def select_all(feats: dict[str, dict[str, Any]], seed: int, n_small: int = 1000,
               n_large: int = 3000) -> tuple[dict[str, Subset], dict[str, Any]]:
    """Build the four arms from a pool's feature table (keyed by sample_id)."""
    if len(feats) < n_large:
        raise SelectionError(f"pool has {len(feats)} records < {n_large}")
    cuts = length_bins([f["rendered_total"] for f in feats.values()])
    strata: dict[str, list[str]] = defaultdict(list)
    for sid, f in feats.items():
        strata[stratum_key(f, cuts)].append(sid)
    sizes = {k: len(v) for k, v in strata.items()}
    q3 = allocate(sizes, n_large)
    q1 = allocate(sizes, n_small, caps=q3)  # q1 <= q3 per stratum => random_1k ⊂ random_3k

    rand3, rand1, sc1 = [], [], []
    for k in sorted(strata):
        order = _seeded_order(strata[k], seed, "random|" + k)
        rand3 += order[: q3[k]]
        rand1 += order[: q1[k]]
        consistent = [s for s in order if feats[s]["schema_consistent"]]
        if len(consistent) < q1[k]:
            raise SelectionError(f"stratum {k}: {len(consistent)} consistent < quota {q1[k]}")
        sc1 += consistent[: q1[k]]

    div1 = _diversity(feats, strata, q1, seed)
    params = {"seed": seed, "selection_version": SELECTION_VERSION, "length_cuts": cuts,
              "n_strata": len(strata), "strata_sizes": sizes, "quota_1k": q1, "quota_3k": q3}
    subsets = {
        "random_1k": Subset("random_1k", rand1, "stratified_random_nested_in_3k"),
        "schema_consistent_1k": Subset("schema_consistent_1k", sc1,
                                       "stratified_random_same_order_skip_inconsistent"),
        "diversity_1k": Subset("diversity_1k", div1, "stratified_greedy_coverage"),
        "random_3k": Subset("random_3k", rand3, "stratified_random"),
    }
    for s in subsets.values():
        if len(set(s.sample_ids)) != len(s.sample_ids):
            raise SelectionError(f"{s.name}: duplicate sample ids")
    return subsets, params


def _diversity(feats, strata, quotas, seed, window: int = 400) -> list[str]:
    remaining = {k: _seeded_order(v, seed, "diversity|" + k) for k, v in strata.items()}
    left = dict(quotas)
    seen_fn: set[str] = set()
    seen_sig: set[str] = set()
    seen_groups: set[str] = set()
    covered: set[str] = set()
    chosen: list[str] = []
    while any(v > 0 for v in left.values()):
        for k in sorted(strata):  # round-robin so early strata don't monopolise novelty
            if left[k] <= 0:
                continue
            best, best_score, best_pos = None, None, None
            for pos, sid in enumerate(remaining[k][:window]):
                f = feats[sid]
                if f["group"] in seen_groups:
                    continue
                w = f["query_words"]
                novelty = len(w - covered) / len(w) if w else 0.0
                score = (f["function"] not in seen_fn, f["arg_signature"] not in seen_sig,
                         round(novelty, 6), -pos)
                if best_score is None or score > best_score:
                    best, best_score, best_pos = sid, score, pos
            if best is None:  # window exhausted by group conflicts: widen to whole stratum
                cands = [s for s in remaining[k] if feats[s]["group"] not in seen_groups]
                if not cands:
                    raise SelectionError(f"diversity: stratum {k} exhausted")
                best = cands[0]
                best_pos = remaining[k].index(best)
            remaining[k].pop(best_pos)
            f = feats[best]
            seen_fn.add(f["function"])
            seen_sig.add(f["arg_signature"])
            seen_groups.add(f["group"])
            covered |= f["query_words"]
            chosen.append(best)
            left[k] -= 1
    return chosen


def check_teacher_balance(subsets: dict[str, Subset], feats: dict[str, dict[str, Any]],
                          max_dev: float) -> dict[str, float]:
    pool = Counter(f["teacher"] for f in feats.values())
    pool_share = pool["deepseek-v2-chat"] / sum(pool.values())
    devs = {}
    for name, s in subsets.items():
        c = Counter(feats[i]["teacher"] for i in s.sample_ids)
        devs[name] = round(c["deepseek-v2-chat"] / len(s.sample_ids) - pool_share, 4)
        if abs(devs[name]) > max_dev:
            raise SelectionError(f"{name}: teacher share deviates {devs[name]} > {max_dev}")
    return devs
