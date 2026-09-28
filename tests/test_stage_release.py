import hashlib
import io

import pytest

from scripts import stage_release


PAYLOAD = b"pinned release fixture"
SHA = hashlib.sha256(PAYLOAD).hexdigest()
URL = "https://example.org/releases/model-v1/weights.safetensors"


def test_cache_needs_no_network_and_installs_only_requested_file(tmp_path, monkeypatch):
    cache, dest = tmp_path / "cache", tmp_path / "image" / "weights.safetensors"
    cache.write_bytes(PAYLOAD)
    monkeypatch.setattr(stage_release, "urlopen", lambda *a, **k: pytest.fail("unexpected network"))
    stage_release.stage_asset(cache, dest, URL, SHA)
    assert dest.read_bytes() == PAYLOAD
    assert list(dest.parent.iterdir()) == [dest]


def test_missing_cache_downloads_fixed_url_with_checksum(tmp_path, monkeypatch):
    calls = []
    def fetch(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(PAYLOAD)
    monkeypatch.setattr(stage_release, "urlopen", fetch)
    dest = tmp_path / "weights"
    stage_release.stage_asset(tmp_path / "missing", dest, URL, SHA)
    assert calls == [(URL, 60)] and dest.read_bytes() == PAYLOAD


@pytest.mark.parametrize("cached", [False, True])
def test_bad_checksum_keeps_existing_destination(tmp_path, monkeypatch, cached):
    cache, dest = tmp_path / "cache", tmp_path / "image" / "weights"
    dest.parent.mkdir()
    dest.write_bytes(b"original")
    if cached:
        cache.write_bytes(b"bad")
    monkeypatch.setattr(stage_release, "urlopen", lambda *a, **k: io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        stage_release.stage_asset(cache, dest, URL, SHA)
    assert dest.read_bytes() == b"original"
    assert list(dest.parent.iterdir()) == [dest]


def test_non_https_source_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="HTTPS"):
        stage_release.stage_asset(tmp_path / "missing", tmp_path / "dest", "http://example.org/w", SHA)
