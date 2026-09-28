"""Offline recovery tests; no SSH, credentials, or network."""
import hashlib
import io
import json
import sys
import urllib.request

import pytest

from scripts.fetch_release_remote import REMOTE


def specification(tmp_path):
    data = b"verified release bytes"
    asset = {"name": "weights.pt", "size": len(data),
             "sha256": hashlib.sha256(data).hexdigest(), "url": "https://unused.invalid"}
    return data, {"directory": str(tmp_path), "assets": [asset]}


def run_remote(spec, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(spec)))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: pytest.fail("unexpected network"))
    exec(REMOTE, {})


def test_complete_partial_recovers_without_http_416(tmp_path, monkeypatch, capsys):
    data, spec = specification(tmp_path)
    partial = tmp_path / "weights.pt.partial"
    partial.write_bytes(data)
    run_remote(spec, monkeypatch)
    assert not partial.exists()
    assert (tmp_path / "weights.pt").read_bytes() == data
    assert json.loads(capsys.readouterr().out)["verified"] is True


@pytest.mark.parametrize("extra", [0, 1])
def test_bad_complete_or_oversized_partial_fails_closed(tmp_path, monkeypatch, extra):
    data, spec = specification(tmp_path)
    (tmp_path / "weights.pt.partial").write_bytes(b"x" * (len(data) + extra))
    with pytest.raises(ValueError, match="partial checksum/size mismatch"):
        run_remote(spec, monkeypatch)
    assert not (tmp_path / "weights.pt").exists()


def test_existing_verified_asset_is_not_redownloaded(tmp_path, monkeypatch):
    data, spec = specification(tmp_path)
    (tmp_path / "weights.pt").write_bytes(data)
    run_remote(spec, monkeypatch)
    assert (tmp_path / "weights.pt").read_bytes() == data
