"""Opt-in D082 main-run routing and independently pinned development gates."""

import csv
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from reid.infer import d082, run
from reid.infer.fusion import load_fusion_config
from tests.test_d082_bundle import digest, prepare, write_json

pytest_plugins = ("tests.test_d082_bundle",)
from tests.test_infer_pipeline import FakeExtractor, FakeVerifier, _write_input


@pytest.fixture
def runtime_case(case, monkeypatch):
    prepare(case)
    monkeypatch.setattr(
        d082, "SOURCE_MANIFEST_SHA256", digest(case["package"] / "MANIFEST.json")
    )
    return case


@pytest.fixture(autouse=True)
def restore_logger():
    logger = logging.getLogger("reid")
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    yield
    for handler in logger.handlers[:]:
        if handler not in handlers:
            logger.removeHandler(handler)
            handler.close()
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def repin(root, edit=None):
    path = root / "ranker_runtime.json"
    data = json.loads(path.read_text())
    if edit is not None:
        edit(data)
    for name, record in data["files"].items():
        asset = root / name
        record.update(sha256=digest(asset), bytes=asset.stat().st_size)
    write_json(path, data)
    return digest(path)


def load(case, pin=None, queries=None, galleries=None, verifier=None):
    root = case["out"]
    return d082.load_d082_runtime(
        root,
        pin or digest(root / "ranker_runtime.json"),
        ["q0", "q1", "q2"] if queries is None else queries,
        [f"g{i}" for i in range(12)] if galleries is None else galleries,
        verifier,
    )


def test_real_compact_head_loads_exact_bundle_and_development_subset(runtime_case):
    root = runtime_case["out"]
    runtime = load(runtime_case, queries=["q2", "q0"], galleries=["g1", "g0"])
    cfg = load_fusion_config(root)
    assert runtime.fold == 0
    assert runtime.manifest_sha256 == digest(root / "ranker_runtime.json")
    assert runtime.verifier_path == root / "verifier.pt"
    assert runtime.scales == tuple(c["similarity_std"] for c in cfg["components"])
    assert runtime.identity["recipe"] == dict(
        residual_weight=0.5, verifier_weight=0.15, candidate_threshold=6.6
    )
    assert 0 < runtime.weight_bytes < 2_000_000_000
    assert load(runtime_case, verifier=root).verifier_path == runtime.verifier_path
    assert (
        load(runtime_case, verifier=root / "verifier.pt").verifier_path
        == runtime.verifier_path
    )


@pytest.mark.parametrize("pin", ["", "a" * 63, "A" * 64, "g" * 64, "0" * 64, 42])
def test_bad_manifest_pin_fails_before_head_load(runtime_case, monkeypatch, pin):
    monkeypatch.setattr(
        d082, "DevelopmentGraphScorer", lambda *a, **k: pytest.fail("head loaded")
    )
    with pytest.raises(ValueError, match="SHA"):
        d082.load_d082_runtime(runtime_case["out"], pin, ["q0"], ["g0"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("fold", 2),
        ("fold", True),
        ("fold", "0"),
        ("fold", -1),
        ("schema_version", True),
        ("schema_version", 2),
        ("kind", "all_data"),
        ("source_manifest_sha256", "0" * 64),
        (
            "recipe",
            dict(residual_weight=0.5, verifier_weight=0.15, candidate_threshold=6.0),
        ),
        (
            "recipe",
            dict(residual_weight=1, verifier_weight=0.15, candidate_threshold=6.6),
        ),
        ("weight_bytes", 2_000_000_001),
    ],
)
def test_modified_pinned_metadata_still_fails_semantic_gates(
    runtime_case, field, value
):
    pin = repin(runtime_case["out"], lambda data: data.update({field: value}))
    with pytest.raises(ValueError):
        load(runtime_case, pin)


@pytest.mark.parametrize(
    "queries,galleries",
    [
        (["outside-development"], ["g0"]),
        (["q0"], ["outside-development"]),
        ([], ["g0"]),
        (["q0"], []),
        (["q0", "q0"], ["g0"]),
        (["q0"], ["g0", "g0"]),
        (["g0"], ["q0"]),
    ],
)
def test_input_allowlist_rejects_outside_empty_duplicate_or_swapped_ids(
    runtime_case, queries, galleries
):
    with pytest.raises(ValueError, match="allowlist"):
        load(runtime_case, queries=queries, galleries=galleries)


def test_allowlist_cannot_include_labels_or_overlap(runtime_case):
    root = runtime_case["out"]
    path = root / "development_ids.json"
    ids = json.loads(path.read_text())
    write_json(path, dict(ids, camera_id=[1]))
    with pytest.raises(ValueError, match="IDs only"):
        load(runtime_case, repin(root))
    ids["gallery_ids"].append("q0")
    write_json(path, ids)
    with pytest.raises(ValueError, match="overlap"):
        load(runtime_case, repin(root))


@pytest.mark.parametrize(
    "name",
    ["verifier.pt", "components/0/weights.safetensors", "head/model_config.json"],
)
def test_tampered_asset_without_repin_is_rejected(runtime_case, name):
    path = runtime_case["out"] / name
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="integrity"):
        load(runtime_case)


@pytest.mark.parametrize("mode", ["extra", "missing", "symlink"])
def test_exact_file_set_and_no_links(runtime_case, mode):
    root = runtime_case["out"]
    path = root / "verifier.pt"
    if mode == "extra":
        (root / "unlisted.pt").write_bytes(b"x")
    elif mode == "missing":
        path.unlink()
    else:
        path.unlink()
        path.symlink_to(runtime_case["verifier"])
    with pytest.raises(ValueError):
        load(runtime_case)


@pytest.mark.parametrize(
    "name", ["components/0/training_receipt.json", "verifier_training_receipt.json"]
)
def test_outer_or_sealed_training_membership_rejected(runtime_case, name):
    root = runtime_case["out"]
    path = root / name
    receipt = json.loads(path.read_text())
    receipt["actual_train_folds"] = [1, 2, 3, 4]
    write_json(path, receipt)
    with pytest.raises(ValueError, match="OOF provenance"):
        load(runtime_case, repin(root))


def test_fusion_threshold_and_external_verifier_cannot_override_recipe(runtime_case):
    root = runtime_case["out"]
    with pytest.raises(ValueError, match="inside its pinned bundle"):
        load(runtime_case, verifier=runtime_case["verifier"])
    path = root / "fusion_config.json"
    cfg = json.loads(path.read_text())
    cfg["candidate_threshold"] = 6.0
    write_json(path, cfg)
    with pytest.raises(ValueError, match="frozen recipe"):
        load(runtime_case, repin(root))


def test_duplicate_metadata_key_is_not_silently_last_wins(runtime_case):
    root = runtime_case["out"]
    path = root / "ranker_runtime.json"
    path.write_text(path.read_text().replace('"fold": 0', '"fold": 0, "fold": 0'))
    with pytest.raises(ValueError, match="duplicate"):
        load(runtime_case, digest(path))


class ComponentExtractor:
    def __init__(self, ids, blocks, cfg=None):
        self.indices = {name: i for i, name in enumerate(ids)}
        self.blocks = blocks
        self.cfg = cfg or {}
        self.calls = []

    def extract_batch(self, items):
        pytest.fail(
            "D082 reconstructed original components from legacy fused embeddings"
        )

    def extract_components_batch(self, items):
        self.calls.append([Path(path).stem for path, _ in items])
        indices = [self.indices[Path(path).stem] for path, _ in items]
        return tuple(block[indices] for block in self.blocks)


def rows(ids):
    return [run.InputRow(i, Path(f"{i}.jpg"), (0, 0, 1, 1)) for i in ids]


def test_component_export_twice_numpy_l2_exact_and_no_batch_fitting():
    rng = np.random.default_rng(20261017)
    ids = [f"q{i}" for i in range(7)]
    blocks = tuple(rng.normal(size=(7, dim)).astype(np.float32) for dim in (17, 19))
    blocks = tuple(
        block / np.linalg.norm(block, axis=1, keepdims=True) for block in blocks
    )
    once = tuple(
        block / np.linalg.norm(block, axis=1, keepdims=True) for block in blocks
    )
    expected = tuple(
        block / np.linalg.norm(block, axis=1, keepdims=True) for block in once
    )
    assert any(not np.array_equal(a, b) for a, b in zip(once, expected))
    extractor = ComponentExtractor(ids, blocks)
    for batch_size in (1, 3, 7):
        actual = d082.extract_component_rows(extractor, rows(ids), batch_size)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
    reversed_output = d082.extract_component_rows(extractor, rows(ids[::-1]), 3)
    for a, b in zip(reversed_output, expected):
        np.testing.assert_array_equal(a[::-1], b)


@pytest.mark.parametrize(
    "mode", ["dtype", "nan", "zero", "short", "empty_dim", "one_component"]
)
def test_invalid_component_outputs_fail_closed(mode):
    class Bad:
        def extract_components_batch(self, items):
            block = np.ones((len(items), 4), dtype=np.float32)
            if mode == "dtype":
                block = block.astype(np.float64)
            if mode == "nan":
                block[0, 0] = np.nan
            if mode == "zero":
                block.fill(0)
            if mode == "short":
                block = block[:-1]
            if mode == "empty_dim":
                block = block[:, :0]
            return (block,) if mode == "one_component" else (block, block)

    with pytest.raises(ValueError):
        d082.extract_component_rows(Bad(), rows(["q0"]), 1)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_component_batch_size_does_no_forward(batch_size):
    with pytest.raises(ValueError, match="positive integer"):
        d082.extract_component_rows(None, rows(["q0"]), batch_size)


def setup_cli(case, tmp_path):
    qids, gids = [f"q{i}" for i in range(3)], [f"g{i}" for i in range(12)]
    images = tmp_path / "images"
    images.mkdir()
    for image_id in qids + gids:
        (images / f"{image_id}.jpg").touch()
    _write_input(tmp_path / "query.csv", qids)
    _write_input(tmp_path / "gallery.csv", gids)
    args = [
        "--images",
        str(images),
        "--query",
        str(tmp_path / "query.csv"),
        "--gallery",
        str(tmp_path / "gallery.csv"),
        "--model",
        str(case["out"]),
        "--out",
        str(tmp_path / "results"),
        "--device",
        "cpu",
        "--batch-size",
        "2",
    ]
    return qids, gids, args


@pytest.mark.parametrize("masked", [False, True])
def test_main_opt_in_uses_original_components_graph_and_pinned_verifier(
    runtime_case, tmp_path, monkeypatch, masked
):
    qids, gids, args = setup_cli(runtime_case, tmp_path)
    root = runtime_case["out"]
    cfg = load_fusion_config(root)
    blocks = tuple(np.concatenate([q, g]) for q, g in runtime_case["originals"])
    extractor = ComponentExtractor(qids + gids, blocks, cfg)
    constructor_args = []

    def create_extractor(model, device, *, fusion_internal_batch_size):
        constructor_args.append((model, device, fusion_internal_batch_size))
        return extractor

    if masked:
        monkeypatch.setattr(
            "reid.infer.plate_mask.PlateMaskedExtractor", create_extractor
        )
        monkeypatch.setattr(
            run, "Extractor", lambda *a: pytest.fail("unmasked extractor used")
        )
    else:
        monkeypatch.setattr(run, "Extractor", create_extractor)
    verifier_paths = []

    def verifier(path, device):
        verifier_paths.append((path, device))
        return FakeVerifier()

    if masked:
        monkeypatch.setattr("reid.infer.plate_mask.PlateMaskedPairVerifier", verifier)
        monkeypatch.setattr(
            run, "PairVerifier", lambda *a: pytest.fail("unmasked verifier used")
        )
    else:
        monkeypatch.setattr(run, "PairVerifier", verifier)
    thresholds = []
    write_outputs = run.write_outputs

    def write(*args, **kwargs):
        thresholds.append(args[5])
        return write_outputs(*args, **kwargs)

    monkeypatch.setattr(run, "write_outputs", write)
    monkeypatch.setattr(
        run, "rank_embeddings", lambda *a, **k: pytest.fail("legacy route used")
    )
    extra = ["--d082-plate-mask-control"] if masked else []
    assert (
        run.main(
            [
                *args,
                "--d082-manifest-sha256",
                digest(root / "ranker_runtime.json"),
                *extra,
            ]
        )
        == 0
    )
    assert constructor_args == [(root, "cpu", 4)]
    assert verifier_paths == [(root / "verifier.pt", "cpu")]
    assert thresholds == [6.6]
    assert [name for batch in extractor.calls for name in batch] == qids + gids
    result = tmp_path / "results"
    assert np.load(result / "embeddings.npy", allow_pickle=False).shape == (15, 3072)
    ranking = list(csv.reader((result / "submission.csv").open()))
    assert [row[0] for row in ranking] == qids
    assert all(len(row) == 11 for row in ranking)


def test_main_default_never_loads_d082_or_components(
    runtime_case, tmp_path, monkeypatch
):
    qids, gids, args = setup_cli(runtime_case, tmp_path)
    # Ordinary legacy/fusion artifact, not a D082 bundle accidentally lacking
    # its explicit pin. The latter must fail closed, tested separately below.
    args += ["--model", str(tmp_path / "ordinary-model")]
    rng = np.random.default_rng(4)
    vectors = {i: rng.normal(size=7) for i in qids + gids}
    monkeypatch.setattr(run, "Extractor", lambda *a: FakeExtractor(vectors))
    monkeypatch.setattr(
        d082, "load_d082_runtime", lambda *a: pytest.fail("default used D082")
    )
    assert run.main(args) == 0
    assert np.load(tmp_path / "results/embeddings.npy", allow_pickle=False).shape == (
        15,
        7,
    )
    assert run._parser().parse_args(args).d082_manifest_sha256 is None


@pytest.mark.parametrize("bundle,masked", [(True, False), (False, True), (True, True)])
def test_d082_bundle_or_mask_control_requires_explicit_pin(
    runtime_case, tmp_path, monkeypatch, bundle, masked
):
    _, _, args = setup_cli(runtime_case, tmp_path)
    if not bundle:
        args += ["--model", str(tmp_path / "ordinary-model")]
    if masked:
        args += ["--d082-plate-mask-control"]
    monkeypatch.setattr(run, "Extractor", lambda *a: pytest.fail("extractor loaded"))
    assert run.main(args) == 1
    assert not (tmp_path / "results").exists()


def test_mask_control_rejects_injected_extractor(runtime_case, tmp_path):
    _, _, args = setup_cli(runtime_case, tmp_path)
    opts = run._parser().parse_args(args)
    with pytest.raises(ValueError, match="own masked extractor"):
        run.run_pipeline(
            opts.images,
            opts.query,
            opts.gallery,
            opts.out,
            opts.model,
            "cpu",
            2,
            None,
            extractor=object(),
            d082_manifest_sha256=digest(runtime_case["out"] / "ranker_runtime.json"),
            d082_plate_mask_control=True,
        )


def test_d082_rejects_injected_verifier(runtime_case, tmp_path):
    _, _, args = setup_cli(runtime_case, tmp_path)
    opts = run._parser().parse_args(args)
    with pytest.raises(ValueError, match="pinned verifier"):
        run.run_pipeline(
            opts.images,
            opts.query,
            opts.gallery,
            opts.out,
            opts.model,
            "cpu",
            2,
            None,
            verifier=FakeVerifier(),
            d082_manifest_sha256=digest(runtime_case["out"] / "ranker_runtime.json"),
        )


def test_main_rejects_non_development_input_before_model_load(
    runtime_case, tmp_path, monkeypatch
):
    _, _, args = setup_cli(runtime_case, tmp_path)
    (tmp_path / "images/not-in-development.jpg").touch()
    _write_input(tmp_path / "query.csv", ["not-in-development"])
    monkeypatch.setattr(run, "Extractor", lambda *a: pytest.fail("extractor loaded"))
    assert (
        run.main(
            [
                *args,
                "--d082-manifest-sha256",
                digest(runtime_case["out"] / "ranker_runtime.json"),
            ]
        )
        == 1
    )
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "override",
    [
        ["--candidate-threshold", "6.0"],
        ["--rerank-weight", ".25"],
        ["--rerank-mode", "raw"],
    ],
)
def test_main_rejects_recipe_override_before_loading_models(
    runtime_case, tmp_path, monkeypatch, override
):
    _, _, args = setup_cli(runtime_case, tmp_path)
    monkeypatch.setattr(run, "Extractor", lambda *a: pytest.fail("model loaded"))
    assert (
        run.main(
            [
                *args,
                "--d082-manifest-sha256",
                digest(runtime_case["out"] / "ranker_runtime.json"),
                *override,
            ]
        )
        == 1
    )
    assert not (tmp_path / "results").exists()
