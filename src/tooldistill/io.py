"""File IO helpers: JSON/JSONL reading, canonical JSON, hashing, config loading.

All paths are handled with pathlib so the code runs on Windows and Linux (Kaggle).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    """Resolve a config-relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8 preserved."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(resolve(path), "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON array file or a JSONL file. Row order is preserved (it is provenance).

    JSONL lines that fail to parse are returned as ``{"__malformed_line__": <text>}`` so the
    normalizer can reject them with a logged reason instead of aborting the whole dataset.
    """
    p = resolve(path)
    text = p.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{p}: expected a JSON array of records")
        return data
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            row = {"__malformed_line__": line[:500]}
        rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(resolve(path), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
