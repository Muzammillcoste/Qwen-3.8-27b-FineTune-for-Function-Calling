"""Gated-access handling (spec B1, E1): 401/403 is nonfatal, explains manual steps, no token leak.

All HF calls are mocked: these tests make no network requests.
"""

from __future__ import annotations

from types import SimpleNamespace

import huggingface_hub
from huggingface_hub.errors import GatedRepoError

from tooldistill import hf_download
from tooldistill.cli import main

FAKE_TOKEN = "hf_" + "x" * 34  # never a real token


def _patch(monkeypatch, token, auth_exc=None):
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: token)

    class FakeApi:
        def dataset_info(self, repo_id, revision=None, token=None):
            return SimpleNamespace(sha=revision or "abc", gated="auto")

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    def fake_auth_check(repo_id, repo_type=None, token=None):
        if auth_exc:
            raise auth_exc

    monkeypatch.setattr(huggingface_hub, "auth_check", fake_auth_check)


def _gated(code):
    return GatedRepoError("gated", response=SimpleNamespace(status_code=code, headers={},
                                                            request=None))


def test_unauthenticated_gated_repo_is_nonfatal_with_steps(monkeypatch):
    _patch(monkeypatch, None, _gated(401))
    st = hf_download.check_access("Salesforce/xlam-function-calling-60k", "rev")
    assert not st.ok and st.status == "gated_not_authenticated"
    assert "hf auth login" in st.message and "accept" in st.message
    assert st.token_present is False


def test_token_without_accepted_conditions(monkeypatch):
    _patch(monkeypatch, FAKE_TOKEN, _gated(403))
    st = hf_download.check_access("Salesforce/xlam-function-calling-60k", "rev")
    assert st.status == "gated_not_accepted"
    assert FAKE_TOKEN not in st.message and FAKE_TOKEN not in str(st.to_dict())


def test_access_ok(monkeypatch):
    _patch(monkeypatch, FAKE_TOKEN)
    st = hf_download.check_access("Salesforce/xlam-function-calling-60k", "rev")
    assert st.ok and st.resolved_revision == "rev"


def test_inspect_cli_exits_3_and_never_prints_token(monkeypatch, capsys):
    _patch(monkeypatch, FAKE_TOKEN, _gated(403))
    rc = main(["inspect", "--source", "Salesforce/xlam-function-calling-60k", "--sample", "5"])
    out = capsys.readouterr()
    assert rc == 3
    assert FAKE_TOKEN not in out.out + out.err
    assert "gated_not_accepted" in out.err


def test_download_refuses_without_access(monkeypatch, capsys):
    _patch(monkeypatch, None, _gated(401))
    rc = main(["download"])
    assert rc == 3
    assert "Manual steps" in capsys.readouterr().err
