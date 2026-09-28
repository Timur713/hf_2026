import json
import subprocess

import numpy as np
import pytest

from scripts.audit_research_protocol import (
    SEED, bench_probe, build_report, fusion_probe, junk_probe, split_probe,
    verifier_window_probe, write_report, source_metadata,
)


def test_fusion_formulas_have_a_reproducible_counterexample():
    first = fusion_probe(np.random.default_rng(SEED))
    assert first == fusion_probe(np.random.default_rng(SEED))
    assert first["different_top1_queries"] > 0
    assert first["different_top10_order_queries"] > 0
    assert first["kind"] == "synthetic_counterexample_not_model_quality"


def test_verifier_normalization_and_window_effects_are_separate():
    result = verifier_window_probe(np.random.default_rng(SEED))
    assert result["normalization_only_different_top1_queries"] > 0
    assert result["normalization_only_different_top10_order_queries"] > 0
    assert result["expanded_window_different_top10_order_queries"] > 0


def test_train_ids_are_shared_by_even_odd_query_split():
    result = split_probe()
    assert len(result["folds"]) == 5
    for fold in result["folds"]:
        assert fold["overlapping_vehicle_count"] > 0
        assert fold["overlapping_vehicle_count"] <= fold["union_vehicle_count"]
        assert fold["gallery_rows"] == fold["unique_gallery_vehicle_camera_pairs"]
        assert fold["duplicate_gallery_vehicle_camera_groups"] == 0
        assert fold["max_frames_per_gallery_vehicle_camera"] == 1


def test_top10_junk_counterexample_uses_canonical_evaluator():
    result = junk_probe()
    assert result["strict_top10_map10"] == 0.0
    assert result["long_list_map10"] == 0.1


def test_research_bench_matches_canonical_evaluator():
    result = bench_probe(np.random.default_rng(SEED))
    assert result["absolute_difference"] <= result["tolerance"]
    assert result["kind"] == "synthetic_scores_on_train_ids_not_model_quality"


def test_report_has_reproduction_metadata_and_serializes(tmp_path):
    report = build_report()
    assert report["seed"] == SEED
    assert report["source_commit"] is None or len(report["source_commit"]) == 40
    assert "splits/folds.csv" in report["source_file_sha256"]
    assert all(len(sha) == 64 for sha in report["source_file_sha256"].values())
    assert "no trained-model quality" in report["scope"]
    out = tmp_path / "new" / "report.json"
    write_report(out, report)
    assert json.loads(out.read_text()) == report


@pytest.mark.parametrize("failure", [FileNotFoundError("git"), subprocess.CalledProcessError(128, "git")])
def test_source_metadata_without_git_is_explicitly_unknown(monkeypatch, failure):
    def unavailable(*args, **kwargs):
        raise failure
    monkeypatch.setattr(subprocess, "check_output", unavailable)
    report = source_metadata()
    assert report["source_commit"] is None
    assert report["tracked_worktree_dirty"] is None
    assert all(len(sha) == 64 for sha in report["source_file_sha256"].values())


def test_output_is_never_overwritten(tmp_path):
    out = tmp_path / "report.json"
    write_report(out, {"existing": True})
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        write_report(out, {"replacement": True})
    assert out.read_bytes() == before


def test_symlink_target_is_never_overwritten(tmp_path):
    target = tmp_path / "target.json"
    write_report(target, {"existing": True})
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(FileExistsError):
        write_report(alias, {"replacement": True})
    assert json.loads(target.read_text()) == {"existing": True}


def test_nonfinite_report_does_not_create_output(tmp_path):
    out = tmp_path / "report.json"
    with pytest.raises(ValueError):
        write_report(out, {"bad": float("nan")})
    assert not out.exists()
