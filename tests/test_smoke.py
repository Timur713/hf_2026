import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from reid.infer.smoke import _resolve_model, _resolve_verifier, main, run_smoke


def _write_input(path: Path, prefix: str, count: int):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "x", "y", "w", "h"])
        for index in range(count):
            writer.writerow(["{}{}".format(prefix, index), 0, 0, 1, 1])


def test_missing_model_is_an_explicit_skip(tmp_path, capsys):
    images = tmp_path / "images"
    images.mkdir()
    _write_input(tmp_path / "test_query.csv", "q", 25)
    _write_input(tmp_path / "test_gallery.csv", "g", 25)
    for prefix in ("q", "g"):
        for index in range(25):
            (images / "{}{}.jpg".format(prefix, index)).write_bytes(b"smoke")

    assert run_smoke(
        tmp_path,
        n_images=50,
        require_runner=False,
        model_dir=tmp_path / "missing-model",
    ) == 0
    assert "SKIP" in capsys.readouterr().out


def test_strict_smoke_fails_without_model(tmp_path, capsys):
    images = tmp_path / "images"
    images.mkdir()
    _write_input(tmp_path / "test_query.csv", "q", 25)
    _write_input(tmp_path / "test_gallery.csv", "g", 25)
    for prefix in ("q", "g"):
        for index in range(25):
            (images / "{}{}.jpg".format(prefix, index)).write_bytes(b"smoke")

    assert run_smoke(
        tmp_path,
        n_images=50,
        require_runner=True,
        model_dir=tmp_path / "missing-model",
    ) == 2
    assert "FAIL" in capsys.readouterr().err


def _fusion_bundle(path):
    for i in range(2):
        member = path / "components" / str(i)
        member.mkdir(parents=True)
        (member / "weights.safetensors").write_bytes(b"unit-test-placeholder")
        (member / "model_config.yaml").write_text("name: smoke-member\n")
    (path / "fusion_config.json").write_text(json.dumps({
        "schema_version": 1, "name": "smoke-fusion",
        "candidate_policy": "gallery_z", "candidate_threshold": 6.0,
        "rerank_mode": "standardized", "rerank_weight": 0.15,
        "components": [{"path": f"components/{i}", "similarity_std": 0.1} for i in range(2)],
    }))
    return path


def test_explicit_fusion_resolves_without_top_level_c3(tmp_path):
    bundle = _fusion_bundle(tmp_path / "fusion")
    assert not (bundle / "model_config.yaml").exists()
    assert _resolve_model(bundle) == bundle
    (bundle / "components/1/weights.safetensors").unlink()
    with pytest.raises(ValueError, match="missing C3 component"):
        _resolve_model(bundle)


def test_default_lookup_prefers_fusion_to_legacy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bundle = _fusion_bundle(tmp_path / "artifacts/models/fusion")
    legacy = tmp_path / "artifacts/models/final"
    legacy.mkdir(parents=True)
    (legacy / "model_config.yaml").write_text("{}")
    (legacy / "weights.safetensors").write_bytes(b"test")
    assert _resolve_model(None) == Path("artifacts/models/fusion")
    assert _resolve_model(legacy) == legacy
    # A broken final pair must not silently smoke the old checkpoint.
    (bundle / "components/1/weights.safetensors").unlink()
    with pytest.raises(ValueError, match="missing C3 component"):
        _resolve_model(None)
    (bundle / "fusion_config.json").unlink()
    with pytest.raises(FileNotFoundError):
        _resolve_model(None)


def test_default_lookup_falls_back_only_when_fusion_is_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _resolve_model(None) is None
    legacy = tmp_path / "artifacts/models/final"
    legacy.mkdir(parents=True)
    (legacy / "model_config.yaml").write_text("{}")
    (legacy / "weights.safetensors").write_bytes(b"test")
    assert _resolve_model(None) == Path("artifacts/models/final")


def test_fusion_smoke_requires_bundled_verifier_but_allows_override(tmp_path):
    bundle = _fusion_bundle(tmp_path / "fusion")
    with pytest.raises(ValueError, match="missing smoke verifier"):
        _resolve_verifier(bundle, None)
    bundled = bundle / "verifier.pt"
    bundled.write_bytes(b"bundled")
    assert _resolve_verifier(bundle, None) == bundled
    override = tmp_path / "other-verifier.pt"
    override.write_bytes(b"override")
    assert _resolve_verifier(bundle, override) == override
    assert _resolve_verifier(tmp_path / "legacy", None) is None


@pytest.mark.parametrize("target", ["smoke", "smoke-strict", "determinism", "plate-mask-control"])
def test_make_targets_honor_explicit_model_and_verifier(target):
    result = subprocess.run(
        ["make", "-n", target, "MODEL=/legacy/model", "VERIFIER=/legacy/verifier.pt"],
        check=True, capture_output=True, text=True,
    )
    assert '--model "/legacy/model"' in result.stdout
    assert '--verifier "/legacy/verifier.pt"' in result.stdout


@pytest.mark.parametrize("target", ["determinism", "plate-mask-control"])
def test_make_release_controls_default_to_pair(target):
    result = subprocess.run(["make", "-n", target], check=True, capture_output=True, text=True)
    assert '--model "artifacts/models/fusion"' in result.stdout
    assert '--verifier "artifacts/models/fusion/verifier.pt"' in result.stdout


def test_fusion_smoke_dispatches_fifty_images_and_verifier_then_validates(tmp_path, monkeypatch, capsys):
    """Dispatch/C6 regression; model extraction itself is tested separately."""
    images = tmp_path / "images"
    images.mkdir()
    for prefix, split in (("q", "query"), ("g", "gallery")):
        _write_input(tmp_path / f"test_{split}.csv", prefix, 25)
        for i in range(25):
            (images / f"{prefix}{i}.jpg").write_bytes(b"placeholder")
    bundle = _fusion_bundle(tmp_path / "fusion")
    verifier = bundle / "verifier.pt"
    verifier.write_bytes(b"test-verifier")
    calls = []

    def child_run(command, check):
        assert check is True
        assert command[1:3] == ["-m", "reid.infer.run"]
        opts = dict(zip(command[3::2], command[4::2]))
        assert opts["--model"] == str(bundle)
        assert opts["--verifier"] == str(verifier)
        assert opts["--images"] == str(images)
        for key in ("--query", "--gallery"):
            with Path(opts[key]).open() as handle:
                assert len(list(csv.DictReader(handle))) == 25
        out = Path(opts["--out"])
        with (out / "submission.csv").open("w", newline="") as handle:
            csv.writer(handle).writerows([[f"q{q}", *[f"g{i}" for i in range(10)]] for q in range(25)])
        np.save(out / "embeddings.npy", np.tile(np.array([[1.0, 0.0]], np.float32), (50, 1)))
        (out / "candidates.csv").write_text("query_id,gallery_id,confidence\n")
        calls.append(command)

    monkeypatch.setattr("reid.infer.smoke.subprocess.run", child_run)
    # No explicit verifier flag: the final image must exercise its own verifier.
    assert main(["--data-root", str(tmp_path), "--model", str(bundle),
                 "--require-runner"]) == 0
    assert len(calls) == 1
    assert "PASS: end-to-end inference and C6 validation passed for 50 images" in capsys.readouterr().out
