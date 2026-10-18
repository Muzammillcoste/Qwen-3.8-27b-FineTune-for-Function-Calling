"""Exact and near-duplicate grouping BEFORE splitting (spec E5).

Group edges (union-find):
1. identical exact key = canonical JSON of normalized query text + tool signatures + expected calls;
2. identical original source id (duplicate ids are a data defect; keep them together);
3. identical normalized query text (optional, default on) regardless of labels;
4. near-duplicate queries: token-set Jaccard >= threshold within the same expected-call-name block.

Groups are never used to delete semantically different labels. Only exact duplicates are marked
``duplicate_of`` (the canonical member is the lowest original id / source row).
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .io import canonical_json, sha256_obj, sha256_text
from .schemas import NormalizedRecord

ALGORITHM_VERSION = "dedup-v1"
_TOKEN = re.compile(r"\w+", re.UNICODE)


def normalize_query_text(q: str) -> str:
    q = unicodedata.normalize("NFKC", q).casefold()
    return " ".join(_TOKEN.findall(q))


def tool_signature(tool: dict[str, Any]) -> dict[str, Any]:
    fn = tool["function"]
    props = fn["parameters"].get("properties", {})
    return {
        "name": fn["name"],
        "params": sorted((k, canonical_json(v.get("type", v.get("anyOf", "")))) for k, v in props.items()),
        "required": sorted(fn["parameters"].get("required", [])),
    }


def exact_key(rec: NormalizedRecord) -> str:
    return sha256_obj(
        {
            "query": normalize_query_text(rec.query),
            "tools": sorted(canonical_json(tool_signature(t)) for t in rec.tools),
            "calls": [{"name": c.name, "arguments": c.arguments} for c in rec.expected_calls],
        }
    )


def label_free_key(rec: NormalizedRecord) -> str:
    """Query + tools only; used to detect conflicting labels for the same input."""
    return sha256_obj(
        {
            "query": normalize_query_text(rec.query),
            "tools": sorted(canonical_json(tool_signature(t)) for t in rec.tools),
        }
    )


def call_block(rec: NormalizedRecord) -> tuple[str, ...]:
    names = sorted(c.name for c in rec.expected_calls)
    return tuple(names) if names else ("<no_call>",)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[max(ra, rb)] = min(ra, rb)
        return True


@dataclass
class DedupReport:
    algorithm_version: str
    n_records: int
    n_exact_duplicates: int
    n_groups: int
    largest_group: int
    edges: dict[str, int]
    conflicting_label_sets: list[list[str]] = field(default_factory=list)
    largest_blocks: list[tuple[str, int]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def group_records(
    records: list[NormalizedRecord],
    near_duplicate_jaccard: float = 0.8,
    group_same_query_text: bool = True,
    max_block_pairwise: int = 5000,
) -> DedupReport:
    """Assign ``source_group_id``, ``exact_dup_key`` and ``duplicate_of`` in place."""
    n = len(records)
    uf = _UnionFind(n)
    edges = {"exact": 0, "same_original_id": 0, "same_query_text": 0, "near_duplicate": 0}
    norm_q = [normalize_query_text(r.query) for r in records]
    tokens = [set(q.split()) for q in norm_q]

    by_exact: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        r.exact_dup_key = exact_key(r)
        by_exact[r.exact_dup_key].append(i)
    for idxs in by_exact.values():
        for j in idxs[1:]:
            edges["exact"] += uf.union(idxs[0], j)

    by_id: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        if r.original_id is not None:
            by_id[r.original_id].append(i)
    for idxs in by_id.values():
        for j in idxs[1:]:
            edges["same_original_id"] += uf.union(idxs[0], j)

    if group_same_query_text:
        by_q: dict[str, list[int]] = defaultdict(list)
        for i, q in enumerate(norm_q):
            by_q[q].append(i)
        for idxs in by_q.values():
            for j in idxs[1:]:
                edges["same_query_text"] += uf.union(idxs[0], j)

    blocks: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        blocks[call_block(r)].append(i)
    oversized = []
    for key, idxs in blocks.items():
        if len(idxs) > max_block_pairwise:
            oversized.append(("|".join(key), len(idxs)))
        # Exhaustive pairwise with a length filter (|A|/|B| >= t is necessary for J >= t).
        order = sorted(idxs, key=lambda i: len(tokens[i]))
        for a_pos, a in enumerate(order):
            la = len(tokens[a])
            for b in order[a_pos + 1:]:
                lb = len(tokens[b])
                if lb and la / lb < near_duplicate_jaccard:
                    break
                if jaccard(tokens[a], tokens[b]) >= near_duplicate_jaccard:
                    edges["near_duplicate"] += uf.union(a, b)

    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comps[uf.find(i)].append(i)
    for members in comps.values():
        anchor = min(records[i].sample_id for i in members)
        gid = "grp:" + sha256_text(anchor)[:16]
        for i in members:
            records[i].source_group_id = gid
            records[i].metadata["dedup_version"] = ALGORITHM_VERSION

    n_dups = 0
    for idxs in by_exact.values():
        if len(idxs) < 2:
            continue
        canon = min(idxs, key=lambda i: (
            records[i].original_id if records[i].original_id is not None else 10**12,
            records[i].source_row_index,
        ))
        for i in idxs:
            if i != canon:
                records[i].duplicate_of = records[canon].sample_id
                records[i].add_flag("exact_duplicate")
                n_dups += 1

    conflicts: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        conflicts[label_free_key(r)].append(i)
    conflicting = []
    for idxs in conflicts.values():
        if len({records[i].exact_dup_key for i in idxs}) > 1:
            conflicting.append(sorted(records[i].sample_id for i in idxs))
            for i in idxs:
                records[i].add_flag("conflicting_labels_same_input")

    sizes = [len(m) for m in comps.values()]
    top_blocks = sorted(((("|".join(k)), len(v)) for k, v in blocks.items()),
                        key=lambda kv: (-kv[1], kv[0]))[:10]
    return DedupReport(
        algorithm_version=ALGORITHM_VERSION,
        n_records=n,
        n_exact_duplicates=n_dups,
        n_groups=len(comps),
        largest_group=max(sizes) if sizes else 0,
        edges=edges,
        conflicting_label_sets=sorted(conflicting),
        largest_blocks=top_blocks,
        params={
            "near_duplicate_jaccard": near_duplicate_jaccard,
            "group_same_query_text": group_same_query_text,
            "oversized_blocks": oversized,
        },
    )
