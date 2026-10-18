from __future__ import annotations

import random
from pathlib import Path

import pytest

from tooldistill.io import read_records
from tooldistill.normalize_xlam import NormalizeOptions, normalize_rows
from tooldistill.validate import apply_validation

FIXTURE = Path(__file__).parent / "fixtures" / "miniature_xlam.jsonl"
FIXTURE_OPTS = NormalizeOptions(source_dataset="fixture:miniature_xlam", source_revision="fixture",
                                id_prefix="fixture")


@pytest.fixture
def fixture_rows():
    return read_records(FIXTURE)


@pytest.fixture
def fixture_normalized(fixture_rows):
    records, rejected = normalize_rows(fixture_rows, FIXTURE_OPTS)
    for r in records:
        apply_validation(r)
    return records, rejected


@pytest.fixture
def by_id(fixture_normalized):
    records, _ = fixture_normalized
    return {r.sample_id: r for r in records}


def synthetic_rows(n_families: int = 300, seed: int = 7) -> list[dict]:
    """SYNTHETIC rows for split/leakage tests: families of paraphrases + exact duplicates.

    Each family shares a tool and near-identical query wording, so a correct grouping must keep
    each family inside one split. IDs span both teacher-source ranges.
    """
    import json

    rng = random.Random(seed)
    rows, next_id = [], 0
    for f in range(n_families):
        tool = {"name": f"fam{f}_tool", "description": "SYNTHETIC",
                "parameters": {"x": {"type": "int", "description": "x", "required": True}}}
        base = f"[SYNTH] please run family {f} operation with value"
        variants = [base, base + " now", "please " + base][: rng.randint(1, 3)]
        n_calls = 2 if f % 7 == 0 else 1
        for v in variants:
            answers = [{"name": tool["name"], "arguments": {"x": f}}] * n_calls
            rid = next_id if f % 2 == 0 else 33659 + next_id
            rows.append({"id": rid, "query": v, "tools": json.dumps([tool]),
                         "answers": json.dumps(answers)})
            next_id += 1
        if f % 10 == 0:  # exact duplicate with a new id
            dup = dict(rows[-1])
            dup["id"] = 50000 + f
            rows.append(dup)
    return rows
