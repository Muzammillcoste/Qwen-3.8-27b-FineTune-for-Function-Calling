"""Reproducibility manifests: file SHA256s, counts, schema version, code commit, license notice."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import REPO_ROOT, resolve, sha256_file
from .schemas import SCHEMA_VERSION

LICENSE_ACKNOWLEDGEMENT = (
    "Derived from Salesforce/xlam-function-calling-60k (CC BY 4.0; gated: users must accept the "
    "dataset's access conditions and cite APIGen, Liu et al. 2024, arXiv:2406.18518). "
    "Research use only per the dataset card. Do not redistribute publicly without verifying terms."
)


def code_commit() -> str:
    """Current git commit, suffixed with '+dirty' if the tree has uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
            check=True,
        ).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def _count_lines(path: Path) -> int | None:
    if path.suffix != ".jsonl":
        return None
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def build_manifest(files: dict[str, str | Path], **extra: Any) -> dict[str, Any]:
    entries = {}
    for name, p in sorted(files.items()):
        path = resolve(p)
        entries[name] = {
            "path": path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT)
            else path.name,
            "sha256": sha256_file(path),
            "n_records": _count_lines(path),
            "bytes": path.stat().st_size,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": code_commit(),
        "license_acknowledgement": LICENSE_ACKNOWLEDGEMENT,
        "files": entries,
        **extra,
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", "utf-8")
    return p
