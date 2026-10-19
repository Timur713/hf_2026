"""Compact, CPU-only bundle acceptance and fail-closed adversarial fixtures."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.torch import save_file
import torch
import yaml

from reid.model.list_ranker import CandidateGraphRanker
from scripts import prepare_d082_development_bundle as bundle
from scripts.prepare_fusion_artifact import gallery_similarity_std
from scripts.replay_fusion import load_components


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def refresh(case, monkeypatch):
    package = case["package"]
    manifest = case["manifest"]
    manifest["files"] = {
        str(p.relative_to(package)): digest(p)
        for p in sorted(package.rglob("*"))
        if p.is_file() and p.name != "MANIFEST.json"
    }
    write_json(package / "MANIFEST.json", manifest)
    monkeypatch.setattr(
        bundle, "PACKAGE_MANIFEST_SHA256", digest(package / "MANIFEST.json")
    )


@pytest.fixture
def case(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    monkeypatch.setattr(bundle, "REFERENCE_SPLITS", reference)
    rng = np.random.default_rng(20261017)
    qids, gids = [f"q{i}" for i in range(3)], [f"g{i}" for i in range(12)]
    for role, ids in (("query", qids), ("gallery", gids)):
        text = "image_id,x,y,w,h\n" + "".join(f"{i},0,0,1,1\n" for i in ids)
        (package / "splits").mkdir(exist_ok=True)
        (package / "splits" / f"fold0_{role}.csv").write_text(text)
        (reference / f"fold0_{role}.csv").write_text(text)
    (package / "splits/folds.csv").write_text(
        "image_id,vehicle_id,camera_id,fold,role\n"
    )
    (reference / "folds.csv").write_bytes((package / "splits/folds.csv").read_bytes())
    supplied, upstream, originals = {}, [], []
    for name, arch, size in (
        ("dino", "dinov2_vitb14", 392),
        ("clip", "clip_vitb16", 288),
    ):
        root = package / "fold0"
        root.mkdir(exist_ok=True)
        path = tmp_path / f"{name}.safetensors"
        path.write_bytes(f"mock-{name}-weight-bytes".encode())
        supplied[f"{name}_weights"] = path
        cfg = dict(
            arch=arch,
            input_size=[size, size],
            pool="cls_top16",
            embed_dim=1536,
            train=dict(folds="1,3,4"),
        )
        (root / f"{name}_config.yaml").write_text(yaml.safe_dump(cfg))
        receipt = dict(
            fold=0,
            actual_train_folds=[1, 3, 4],
            excluded_folds=[0, 2],
            weights_sha256=digest(path),
            config_sha256=digest(root / f"{name}_config.yaml"),
        )
        write_json(root / f"{name}_training_receipt.json", receipt)
        q, g = (
            rng.normal(size=(3, 1536)).astype(np.float32),
            rng.normal(size=(12, 1536)).astype(np.float32),
        )
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        originals.append((q, g))
        np.savez(
            root / f"{name}.npz",
            query_ids=qids,
            gallery_ids=gids,
            query_emb=q,
            gallery_emb=g,
        )
        export = dict(
            receipt,
            embedding_path="full-frame",
            dump_sha256=digest(root / f"{name}.npz"),
            query_split_sha256=digest(package / "splits/fold0_query.csv"),
            gallery_split_sha256=digest(package / "splits/fold0_gallery.csv"),
        )
        write_json(root / f"{name}_export_receipt.json", export)
        upstream.append(
            dict(
                fold=0,
                component=name,
                bytes=path.stat().st_size,
                weights_sha256=digest(path),
            )
        )
    verifier = tmp_path / "verifier.pt"
    # Intentionally not a pickle: packaging must copy and SHA-check, never load.
    verifier.write_bytes(b"opaque-pinned-verifier-bytes")
    supplied["verifier"] = verifier
    write_json(
        package / "fold0/verifier_training_receipt.json",
        dict(
            validation_fold=0,
            actual_train_folds=[1, 3, 4],
            excluded_folds=[0, 2],
            weights_sha256=digest(verifier),
            options=dict(input=392),
            backbone_matches_pinned_initialization_exactly=True,
            pair_inputs=[dict(fold=f) for f in (1, 3, 4)],
        ),
    )
    upstream.append(
        dict(
            fold=0,
            component="verifier",
            bytes=verifier.stat().st_size,
            weights_sha256=digest(verifier),
        )
    )
    head = package / "fold0/head"
    head.mkdir()
    torch.manual_seed(4)
    save_file(CandidateGraphRanker().state_dict(), head / "weights.safetensors")
    write_json(
        head / "model_config.json",
        dict(
            features=13,
            hidden=64,
            depth=2,
            outer=0,
            actual_supervised_folds=[1, 3, 4],
            excluded_folds=[0, 2],
        ),
    )
    nested = []
    for view in (1, 3, 4):
        parent = dict(
            actual_train_folds=[f for f in (1, 3, 4) if f != view],
            excluded_folds=sorted([0, 2, view]),
            weights_sha256="a" * 64,
            config_sha256="b" * 64,
        )
        nested.append(
            dict(view=view, provenance=[dict(receipt=parent, export=parent)] * 2)
        )
    write_json(
        head / "training_receipt.json",
        dict(
            actual_train_folds=[1, 3, 4],
            excluded_folds=[0, 2],
            weights_sha256=digest(head / "weights.safetensors"),
            config_sha256=digest(head / "model_config.json"),
            nested_supervision=nested,
        ),
    )
    manifest = dict(
        recipe="z(hub)+.15*z(verifier_logit)+.5*(head_score-z(hub))",
        heads="depth2/hidden64/features13",
        fold2="sealed",
        upstream_weights=upstream,
        folds={str(f): {} for f in (0, 1, 3, 4)},
    )
    manifest["folds"]["0"] = dict(
        dumps=["fold0/dino.npz", "fold0/clip.npz"], head="fold0/head"
    )
    result = dict(
        package=package,
        manifest=manifest,
        originals=originals,
        out=tmp_path / "out",
        **supplied,
    )
    refresh(result, monkeypatch)
    return result


def prepare(case):
    return bundle.prepare_bundle(
        case["package"],
        0,
        case["dino_weights"],
        case["clip_weights"],
        case["verifier"],
        case["out"],
    )


def test_exact_manifest_and_fixed_original_scales(case):
    result = prepare(case)
    out = case["out"]
    runtime = json.loads((out / "ranker_runtime.json").read_text())
    assert result["ranker_runtime_sha256"] == digest(out / "ranker_runtime.json")
    assert runtime["source_manifest_sha256"] == digest(
        case["package"] / "MANIFEST.json"
    )
    assert runtime["kind"] == "d082_development_runtime"
    assert runtime["fold"] == 0
    assert runtime["recipe"] == dict(
        residual_weight=0.5, verifier_weight=0.15, candidate_threshold=6.6
    )
    assert len(runtime["files"]) == 13
    assert "ranker_runtime.json" not in runtime["files"]
    for name, record in runtime["files"].items():
        assert record == dict(
            sha256=digest(out / name), bytes=(out / name).stat().st_size
        )
    assert runtime["weight_bytes"] == sum(
        v["bytes"]
        for k, v in runtime["files"].items()
        if k.endswith((".safetensors", ".pt"))
    )
    ids = json.loads((out / "development_ids.json").read_text())
    assert ids == dict(
        query_ids=[f"q{i}" for i in range(3)], gallery_ids=[f"g{i}" for i in range(12)]
    )
    arrays = load_components(
        [case["package"] / f"fold0/{n}.npz" for n in ("dino", "clip")],
        ids["query_ids"],
        ids["gallery_ids"],
    )
    fusion = json.loads((out / "fusion_config.json").read_text())
    assert fusion["components"] == [
        dict(path=f"components/{i}", similarity_std=gallery_similarity_std(g))
        for i, (_, g) in enumerate(arrays)
    ]
    assert fusion["candidate_threshold"] == 6.6
    assert fusion["rerank_weight"] == 0.15
    assert (out / "verifier.pt").read_bytes() == case["verifier"].read_bytes()


@pytest.mark.parametrize("fold", [2, 5, -1, True, "0", "all"])
def test_sealed_fold_and_all_data_rejected_before_reading(case, fold):
    with pytest.raises(ValueError, match="fold2/all-data"):
        bundle.prepare_bundle(
            Path("absent"), fold, Path("x"), Path("y"), Path("z"), case["out"]
        )


@pytest.mark.parametrize("role", ["dino_weights", "clip_weights", "verifier"])
def test_wrong_supplied_weight_sha(case, role):
    case[role].write_bytes(b"changed")
    with pytest.raises(ValueError, match="weight SHA mismatch"):
        prepare(case)
    assert not case["out"].exists()


def test_manifest_pin_not_self_trusted(case):
    with (case["package"] / "MANIFEST.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="MANIFEST SHA"):
        prepare(case)


def test_changed_pinned_member(case):
    (case["package"] / "fold0/dino_config.yaml").write_text("changed: yes\n")
    with pytest.raises(ValueError, match="package SHA mismatch"):
        prepare(case)


@pytest.mark.parametrize(
    "name", ["../outside", "/etc/passwd", "fold0/../outside", "fold0\\outside"]
)
def test_unsafe_manifest_paths(case, monkeypatch, name):
    case["manifest"]["files"][name] = "0" * 64
    write_json(case["package"] / "MANIFEST.json", case["manifest"])
    monkeypatch.setattr(
        bundle, "PACKAGE_MANIFEST_SHA256", digest(case["package"] / "MANIFEST.json")
    )
    with pytest.raises(ValueError, match="unsafe package path"):
        prepare(case)


def test_symlink_member_rejected(case):
    source = case["package"] / "fold0/dino_config.yaml"
    moved = case["package"].parent / "same-config.yaml"
    source.rename(moved)
    source.symlink_to(moved)
    with pytest.raises(ValueError, match="symlink"):
        prepare(case)


@pytest.mark.parametrize("component", ["dino", "clip", "verifier"])
def test_outer_or_sealed_fold_in_training_rejected(case, monkeypatch, component):
    path = case["package"] / f"fold0/{component}_training_receipt.json"
    receipt = json.loads(path.read_text())
    receipt["actual_train_folds"] = [1, 2, 3, 4]
    write_json(path, receipt)
    refresh(case, monkeypatch)
    with pytest.raises(ValueError, match="OOF provenance"):
        prepare(case)


def test_nested_head_leak_rejected(case, monkeypatch):
    path = case["package"] / "fold0/head/training_receipt.json"
    receipt = json.loads(path.read_text())
    receipt["nested_supervision"][0]["provenance"][0]["receipt"][
        "actual_train_folds"
    ] = [0, 3, 4]
    write_json(path, receipt)
    refresh(case, monkeypatch)
    with pytest.raises(ValueError, match="nested encoder"):
        prepare(case)


def test_relabelled_dump_rejected(case, monkeypatch):
    path = case["package"] / "fold0/dino.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {k: archive[k] for k in archive.files}
    values["query_ids"] = values["query_ids"][::-1]
    np.savez(path, **values)
    export_path = case["package"] / "fold0/dino_export_receipt.json"
    export = json.loads(export_path.read_text())
    export["dump_sha256"] = digest(path)
    write_json(export_path, export)
    refresh(case, monkeypatch)
    with pytest.raises(ValueError, match="C4 IDs/order"):
        prepare(case)


def test_output_existing_empty_directory_not_replaced(case):
    case["out"].mkdir()
    with pytest.raises(FileExistsError):
        prepare(case)
    assert list(case["out"].iterdir()) == []


def test_atomic_no_replace_handles_concurrent_empty_output(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "data").write_bytes(b"ok")
    with pytest.raises(FileExistsError):
        bundle._publish_no_replace(source, target)
    assert source.exists() and not list(target.iterdir())


def test_copied_bytes_rechecked_and_no_partial_output(case, monkeypatch):
    original = bundle.shutil.copyfile

    def corrupt(source, destination):
        result = original(source, destination)
        if destination.name == "verifier.pt":
            destination.write_bytes(b"concurrent mutation")
        return result

    monkeypatch.setattr(bundle.shutil, "copyfile", corrupt)
    with pytest.raises(ValueError, match="copied bytes changed"):
        prepare(case)
    assert not case["out"].exists()
    assert not list(case["out"].parent.glob(".out-*"))


def test_weight_budget_preflight(case, monkeypatch):
    monkeypatch.setattr(bundle, "MAX_WEIGHT_BYTES", 10)
    with pytest.raises(ValueError, match="exceed 2 GB"):
        prepare(case)
    assert not case["out"].exists()


def test_duplicate_json_key_rejected():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        bundle._json(b'{"fold":0,"fold":2}')


def test_query_values_cannot_change_fusion_scales(case, monkeypatch):
    first = prepare(case)
    first_cfg = (Path(first["path"]) / "fusion_config.json").read_bytes()
    for name in ("dino", "clip"):
        path = case["package"] / f"fold0/{name}.npz"
        with np.load(path, allow_pickle=False) as archive:
            values = {k: archive[k] for k in archive.files}
        values["query_emb"] = -values["query_emb"][::-1]
        np.savez(path, **values)
        export_path = case["package"] / f"fold0/{name}_export_receipt.json"
        export = json.loads(export_path.read_text())
        export["dump_sha256"] = digest(path)
        write_json(export_path, export)
    refresh(case, monkeypatch)
    case["out"] = case["out"].with_name("out-queries-changed")
    prepare(case)
    assert (case["out"] / "fusion_config.json").read_bytes() == first_cfg


def test_modified_original_split_rejected(case):
    (bundle.REFERENCE_SPLITS / "fold0_query.csv").write_text(
        "image_id,x,y,w,h\nother,0,0,1,1\n"
    )
    with pytest.raises(ValueError, match="changed the original fold split"):
        prepare(case)


def test_wrong_export_receipt_rejected(case, monkeypatch):
    path = case["package"] / "fold0/dino_export_receipt.json"
    export = json.loads(path.read_text())
    export["embedding_path"] = "cache"
    write_json(path, export)
    refresh(case, monkeypatch)
    with pytest.raises(ValueError, match="pinned full-frame provenance"):
        prepare(case)


def test_wrong_upstream_size_rejected(case, monkeypatch):
    case["manifest"]["upstream_weights"][0]["bytes"] += 1
    refresh(case, monkeypatch)
    with pytest.raises(ValueError, match="manifest source"):
        prepare(case)


def test_sealed_package_flag_rejected(case, monkeypatch):
    case["manifest"]["fold2"] = "opened"
    refresh(case, monkeypatch)
    with pytest.raises(ValueError, match="fold2 remains sealed"):
        prepare(case)


def test_no_pickle_deserialization(case, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("bundle creation must not deserialize pickle")

    monkeypatch.setattr(torch, "load", forbidden)
    prepare(case)


def test_output_symlink_parent_cannot_modify_source(case):
    alias = case["package"].parent / "package-alias"
    alias.symlink_to(case["package"], target_is_directory=True)
    case["out"] = alias / "new-bundle"
    with pytest.raises(ValueError, match="must not modify the source"):
        prepare(case)
