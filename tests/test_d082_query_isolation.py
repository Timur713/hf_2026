from argparse import Namespace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from reid.infer.run import InputRow, read_inputs, write_outputs
from reid.post.search import SearchResult
from scripts import check_d082_query_isolation as check
from scripts.check_query_independence import write_inputs


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    queries = [
        InputRow(f"q{i}", images / f"q{i}.jpg", (0, 0, 10, 10)) for i in range(8)
    ]
    gallery = [
        InputRow(f"g{i}", images / f"g{i}.jpg", (0, 0, 10, 10)) for i in range(12)
    ]
    for row in [*queries, *gallery]:
        row.image_path.write_bytes(b"fixture, never decoded")
    query_csv, gallery_csv = tmp_path / "query.csv", tmp_path / "gallery.csv"
    write_inputs(query_csv, queries)
    write_inputs(gallery_csv, gallery)
    args = Namespace(
        model=tmp_path / "model",
        images=images,
        query=query_csv,
        gallery=gallery_csv,
        out=tmp_path / "out",
        d082_manifest_sha256="a" * 64,
        device="cpu",
        batch_size=4,
        gallery_limit=None,
        strict_batch1=False,
    )
    events = []

    def gate(model, pin, qids, gids):
        assert model == args.model and pin == "a" * 64
        assert qids == [
            r.image_id for r in queries
        ]  # Entire supplied CSV, not only prefix.
        assert gids == [r.image_id for r in gallery]
        events.append("gate")
        return SimpleNamespace(identity={"fold": 0, "kind": "unit-test-only-runtime"})

    monkeypatch.setattr(check, "load_d082_runtime", gate)
    return args, events


def fake_pipeline(events, mutation=None):
    def run(**kwargs):
        assert events[0] == "gate"
        assert kwargs["candidate_threshold"] is None
        assert kwargs["d082_manifest_sha256"] == "a" * 64
        assert "extractor" not in kwargs and "verifier" not in kwargs
        queries = read_inputs(kwargs["query_csv"], kwargs["images_dir"])
        gallery = read_inputs(kwargs["gallery_csv"], kwargs["images_dir"])
        name = kwargs["out_dir"].name
        events.append((name, kwargs["batch_size"]))
        qe = np.asarray(
            [[1.0, int(r.image_id[1:]) + 1.0, 2.0, 0.0] for r in queries],
            dtype=np.float32,
        )
        qe /= np.linalg.norm(qe, axis=1, keepdims=True)
        ge = np.asarray(
            [[1.0, i + 1.0, 1.0, 0.0] for i in range(len(gallery))], dtype=np.float32
        )
        ge /= np.linalg.norm(ge, axis=1, keepdims=True)
        ranking = [[g.image_id for g in gallery[:10]] for _ in queries]
        scores = np.full(len(queries), 7.0, dtype=np.float64)
        changed = name == "replaced_neighbours"
        for index, row in enumerate(queries):
            if row.image_id not in ("q0", "q1"):
                continue
            if (changed and mutation == "embedding") or (
                name == "batch1" and mutation == "batch-shape"
            ):
                qe[index, -1] = np.float32(1e-8)
            if changed and mutation == "signed-zero":
                qe[index, -1] = np.float32(-0.0)
            if changed and mutation == "ranking":
                ranking[index][:2] = ranking[index][:2][::-1]
            if changed and mutation == "confidence":
                scores[index] += 1e-7
            if changed and mutation == "acceptance":
                scores[index] = 5.0
        if changed and mutation == "gallery":
            ge[0, -1] = 1e-8
        result = SearchResult(ranking, [row[0] for row in ranking], scores)
        write_outputs(kwargs["out_dir"], queries, gallery, qe, ge, 6.6, result)

    return run


def test_actual_pipeline_interface_fixed_shape_order_and_artifact_proofs(
    fixture, monkeypatch
):
    args, events = fixture
    monkeypatch.setattr(check, "run_pipeline", fake_pipeline(events))
    result = check.execute(args)
    assert events == [
        "gate",
        ("baseline", 4),
        ("replaced_neighbours", 4),
        ("reversed_replacement", 4),
    ]
    assert result["passed"] and result["fixed_shape_isolation_passed"]
    assert result["strict_batch_shape"]["executed"] is False
    assert result["strict_batch_shape"]["passed"] is None
    assert result["gallery_count"] == 12 and result["gallery_limit"] is None
    assert result["anchor_ids"] == ["q0", "q1"]
    assert result["variants"]["baseline"]["query_ids"] == ["q0", "q1", "q2", "q3"]
    assert result["variants"]["replaced_neighbours"]["query_ids"] == [
        "q0",
        "q1",
        "q4",
        "q5",
    ]
    assert result["variants"]["reversed_replacement"]["query_ids"] == [
        "q5",
        "q4",
        "q1",
        "q0",
    ]
    for record in result["variants"].values():
        assert record["c6"]["validated"]
        assert record["comparison"]["anchor_embeddings_bitwise_equal"]
        assert record["gallery_embeddings_bitwise_equal"]
        assert set(record["output_sha256"]) == {
            "submission.csv",
            "embeddings.npy",
            "candidates.csv",
        }
        assert record["comparison"]["compared_queries"] == 2
    assert json.loads((args.out / "report.json").read_text()) == result
    with pytest.raises(FileExistsError):
        check.execute(args)


@pytest.mark.parametrize(
    "mutation",
    ["embedding", "signed-zero", "ranking", "confidence", "acceptance", "gallery"],
)
def test_no_tolerance_hides_changed_anchor_or_gallery(fixture, monkeypatch, mutation):
    args, events = fixture
    monkeypatch.setattr(check, "run_pipeline", fake_pipeline(events, mutation))
    result = check.execute(args)
    assert not result["passed"] and not result["fixed_shape_isolation_passed"]
    record = result["variants"]["replaced_neighbours"]
    if mutation == "gallery":
        assert not record["gallery_embeddings_bitwise_equal"]
    else:
        assert not record["comparison"]["passed"]


def test_batch1_failure_is_preserved_separately_and_fails_overall(fixture, monkeypatch):
    args, events = fixture
    args.strict_batch1 = True
    args.gallery_limit = 10
    monkeypatch.setattr(check, "run_pipeline", fake_pipeline(events, "batch-shape"))
    result = check.execute(args)
    assert result["fixed_shape_isolation_passed"]
    assert not result["strict_batch_shape"]["passed"]
    assert not result["passed"]
    assert events[-1] == ("batch1", 1)
    assert result["variants"]["batch1"]["comparison"]["compared_queries"] == 4
    assert result["gallery_count"] == 10


@pytest.mark.parametrize("limit", [0, 9, True, 13])
def test_invalid_gallery_prefix_rejected_before_runtime(fixture, monkeypatch, limit):
    args, events = fixture
    args.gallery_limit = limit
    monkeypatch.setattr(
        check, "run_pipeline", lambda **kwargs: pytest.fail("must not run")
    )
    with pytest.raises(ValueError, match="gallery-limit"):
        check.execute(args)
    assert not events and not args.out.exists()


def test_allowlist_failure_stops_before_forward_or_output(fixture, monkeypatch):
    args, events = fixture

    def reject(*args, **kwargs):
        raise ValueError("test/control2 forbidden")

    monkeypatch.setattr(check, "load_d082_runtime", reject)
    monkeypatch.setattr(
        check, "run_pipeline", lambda **kwargs: pytest.fail("must not run")
    )
    with pytest.raises(ValueError, match="test/control2"):
        check.execute(args)
    assert not args.out.exists()


def test_cli_requires_pin_and_preserves_nonzero_failure_exit(fixture, monkeypatch):
    args, _ = fixture
    argv = [
        value
        for name in ("model", "images", "query", "gallery", "out")
        for value in (f"--{name}", str(getattr(args, name)))
    ]
    with pytest.raises(SystemExit) as error:
        check.main(argv)
    assert error.value.code == 2
    monkeypatch.setattr(check, "execute", lambda args: {"passed": False})
    assert check.main([*argv, "--d082-manifest-sha256", "a" * 64]) == 1
