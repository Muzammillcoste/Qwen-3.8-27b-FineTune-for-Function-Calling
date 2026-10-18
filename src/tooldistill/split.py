"""Group-aware, stratified, seeded train/val/test split + leakage audit (spec E5).

Whole groups are assigned to one split, so exact/near duplicates can never straddle splits.
Selection policies (Phase 2) must only ever see the ``train`` split.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .deduplicate import normalize_query_text
from .io import sha256_text
from .schemas import NormalizedRecord

ALGORITHM_VERSION = "split-v1"
SPLITS = ("train", "val", "test")


class EmptySplitError(RuntimeError):
    pass


class LeakageError(RuntimeError):
    pass


def call_count_bucket(n: int) -> str:
    return "none" if n == 0 else ("single" if n == 1 else "multi")


def _group_stratum(members: list[NormalizedRecord], stratify_by: list[str]) -> str:
    parts = []
    for key in stratify_by:
        if key == "teacher_source":
            vals = {m.teacher_source for m in members}
            parts.append(vals.pop() if len(vals) == 1 else "mixed")
        elif key == "call_count_bucket":
            parts.append(call_count_bucket(max(m.call_count for m in members)))
        else:
            raise ValueError(f"unsupported stratify_by key: {key}")
    return "|".join(parts)


def assign_splits(
    records: list[NormalizedRecord],
    ratios: dict[str, float],
    seed: int,
    stratify_by: list[str] | None = None,
    min_records_per_split: int = 1,
) -> dict[str, Any]:
    """Assign ``rec.split`` in place. Returns a summary. Deterministic for (records, seed)."""
    if abs(sum(ratios.values()) - 1.0) > 1e-9 or set(ratios) != set(SPLITS):
        raise ValueError(f"ratios must cover {SPLITS} and sum to 1: {ratios}")
    stratify_by = stratify_by if stratify_by is not None else ["teacher_source", "call_count_bucket"]
    groups: dict[str, list[NormalizedRecord]] = defaultdict(list)
    for r in records:
        if r.source_group_id is None:
            raise ValueError(f"{r.sample_id}: group before splitting (run deduplicate first)")
        groups[r.source_group_id].append(r)

    strata: dict[str, list[str]] = defaultdict(list)
    for gid, members in groups.items():
        strata[_group_stratum(members, stratify_by)].append(gid)

    for stratum in sorted(strata):
        gids = sorted(strata[stratum])
        random.Random(f"{ALGORITHM_VERSION}|{seed}|{stratum}").shuffle(gids)
        counts = dict.fromkeys(SPLITS, 0)
        total = 0
        for gid in gids:
            size = len(groups[gid])
            total += size
            # Largest deficit vs. target after adding this group; ties broken by SPLITS order.
            target = max(SPLITS, key=lambda s: ratios[s] * total - counts[s])
            counts[target] += size
            for m in groups[gid]:
                m.split = target

    split_counts = Counter(r.split for r in records)
    empty = [s for s in SPLITS if split_counts.get(s, 0) < min_records_per_split]
    if empty:
        raise EmptySplitError(
            f"splits below minimum size {min_records_per_split}: {empty}; "
            f"counts={dict(split_counts)}, groups={len(groups)}"
        )
    return summarize_splits(records, seed, ratios, stratify_by)


def summarize_splits(
    records: list[NormalizedRecord], seed: int, ratios: dict[str, float], stratify_by: list[str]
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "algorithm_version": ALGORITHM_VERSION,
        "seed": seed,
        "ratios_target": ratios,
        "stratify_by": stratify_by,
        "splits": {},
    }
    for s in SPLITS:
        members = [r for r in records if r.split == s]
        ids = sorted(r.sample_id for r in members)
        out["splits"][s] = {
            "n_records": len(members),
            "n_groups": len({r.source_group_id for r in members}),
            "teacher_source": dict(sorted(Counter(r.teacher_source for r in members).items())),
            "call_count_bucket": dict(
                sorted(Counter(call_count_bucket(r.call_count) for r in members).items())
            ),
            "sample_ids_sha256": sha256_text("\n".join(ids)),
        }
    return out


@dataclass
class LeakageReport:
    ok: bool
    violations: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "violations": self.violations}


def audit_leakage(records: list[NormalizedRecord]) -> LeakageReport:
    """Check that no group / exact key / normalized query / sample id / original id crosses splits."""
    keys = {
        "source_group_id": lambda r: r.source_group_id,
        "exact_dup_key": lambda r: r.exact_dup_key,
        "normalized_query": lambda r: normalize_query_text(r.query),
        "original_id": lambda r: None if r.original_id is None else str(r.original_id),
    }
    violations: dict[str, list[str]] = {}
    unassigned = [r.sample_id for r in records if r.split not in SPLITS]
    if unassigned:
        violations["unassigned_split"] = unassigned[:50]
    sid_counts = Counter(r.sample_id for r in records)
    dup_sids = [s for s, c in sid_counts.items() if c > 1]
    if dup_sids:
        violations["duplicate_sample_id"] = dup_sids[:50]
    for name, fn in keys.items():
        seen: dict[str, set[str]] = defaultdict(set)
        for r in records:
            k = fn(r)
            if k is not None:
                seen[k].add(r.split)
        bad = sorted(k for k, s in seen.items() if len(s) > 1)
        if bad:
            violations[name] = [b[:80] for b in bad[:50]]
    return LeakageReport(ok=not violations, violations=violations)


def assert_no_leakage(records: list[NormalizedRecord]) -> LeakageReport:
    rep = audit_leakage(records)
    if not rep.ok:
        raise LeakageError(f"cross-split leakage detected: {list(rep.violations)}")
    return rep
