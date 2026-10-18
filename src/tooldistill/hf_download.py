"""Authorized Hugging Face access for the gated xLAM dataset (spec B1, E1).

Never scrapes around the gate, never prints or logs tokens. On 401/403 returns a nonfatal status
with manual instructions. The raw file is downloaded byte-for-byte at a pinned revision so its
SHA256 can be recorded, then made read-only.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import read_records, resolve, sha256_file

MANUAL_STEPS = """\
The dataset {repo_id} is GATED on Hugging Face (CC BY 4.0 + access acknowledgement).
Manual steps (do these yourself; never paste your token into chat, code, notebooks or logs):
  1. Sign in at https://huggingface.co and open {card_url}
  2. Read the card and the access conditions ("Acknowledge to follow corresponding license and
     cite APIGen"), fill in the requested fields, and accept.
  3. Create a READ-scoped access token at https://huggingface.co/settings/tokens
  4. Authenticate locally:   hf auth login        (older CLI: huggingface-cli login)
     The token is stored in your HF cache directory, outside this repository.
  5. Re-run:  python scripts/inspect_xlam.py --source {repo_id} --sample 100
"""


@dataclass
class AccessStatus:
    ok: bool
    status: str  # ok | gated_not_authenticated | gated_not_accepted | not_found | network_error |
    #              missing_dependency
    repo_id: str
    message: str
    token_present: bool
    resolved_revision: str | None = None
    gated: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _card_url(repo_id: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}"


def check_access(repo_id: str, revision: str | None = None) -> AccessStatus:
    try:
        from huggingface_hub import HfApi, get_token
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    except ImportError:
        return AccessStatus(False, "missing_dependency", repo_id,
                            "pip install -r requirements-local.txt", False)
    token_present = get_token() is not None
    steps = MANUAL_STEPS.format(repo_id=repo_id, card_url=_card_url(repo_id))
    api = HfApi()
    try:
        meta = api.dataset_info(repo_id, revision=revision, token=False)
        gated, sha = getattr(meta, "gated", None), meta.sha
    except GatedRepoError:  # subclass of RepositoryNotFoundError: must be caught first
        gated, sha = True, None
    except RepositoryNotFoundError:
        return AccessStatus(False, "not_found", repo_id, f"{repo_id}@{revision} not found",
                            token_present)
    except (HfHubHTTPError, OSError) as e:
        return AccessStatus(False, "network_error", repo_id, f"{type(e).__name__}: {e}"[:300],
                            token_present)
    try:
        from huggingface_hub import auth_check

        auth_check(repo_id, repo_type="dataset")
    except GatedRepoError as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        # No token => not logged in. Token present => conditions not accepted (or token invalid).
        status = "gated_not_accepted" if token_present else "gated_not_authenticated"
        return AccessStatus(False, status, repo_id, f"HTTP {code}: access denied.\n{steps}",
                            token_present, sha, gated)
    except HfHubHTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code in (401, 403):
            return AccessStatus(False, "gated_not_authenticated", repo_id,
                                f"HTTP {code}.\n{steps}", token_present, sha, gated)
        return AccessStatus(False, "network_error", repo_id, str(e)[:300], token_present, sha,
                            gated)
    return AccessStatus(True, "ok", repo_id, "access granted", token_present, sha, gated)


def stream_sample(repo_id: str, revision: str, n: int, split: str = "train") -> list[dict]:
    """Read the first ``n`` source rows via streaming (no full download). Requires access."""
    from datasets import load_dataset

    ds = load_dataset(repo_id, revision=revision, split=split, streaming=True)
    return [dict(row) for _, row in zip(range(n), ds, strict=False)]


def download_raw(
    repo_id: str, revision: str, filename: str, raw_dir: str | Path, license_id: str
) -> dict[str, Any]:
    """Download the raw source file unchanged at a pinned revision and write metadata.

    Raises ``PermissionError`` with manual steps if access is not granted.
    """
    status = check_access(repo_id, revision)
    if not status.ok:
        raise PermissionError(status.message)
    if status.resolved_revision != revision:
        raise RuntimeError(
            f"Pinned revision {revision} resolved to {status.resolved_revision}; re-audit required."
        )
    from huggingface_hub import hf_hub_download

    out_dir = resolve(raw_dir) / revision
    out_dir.mkdir(parents=True, exist_ok=True)
    local = Path(hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision,
                                 local_dir=out_dir))
    os.chmod(local, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)  # raw stays read-only
    rows = read_records(local)
    meta = {
        "repo_id": repo_id,
        "revision": revision,
        "file": filename,
        "split": "train",
        "row_count": len(rows),
        "raw_sha256": sha256_file(local),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "license": license_id,
        "acknowledgement": "Gated: accept access conditions; cite APIGen (arXiv:2406.18518).",
        "card_url": _card_url(repo_id),
    }
    (out_dir / "raw_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
    return meta
