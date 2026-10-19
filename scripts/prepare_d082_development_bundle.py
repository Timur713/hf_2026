"""Package pinned D082 OOF weights for opt-in, full-frame development runs.

No model training, image forward, query statistics, pickle loading, threshold
selection or release approval. Fold2/all-data are deliberately unsupported.
The original C4 gallery fixes fusion scales; runtime must not re-fit them.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
from scripts.hashing import file_digest
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile

import numpy as np
import yaml

from reid.eval.validate_outputs import _read_input_ids
from reid.infer.list_ranker import DevelopmentGraphScorer
from scripts.prepare_fusion_artifact import gallery_similarity_std
from scripts.replay_fusion import load_components


# Independently checked against the immutable A handoff / GitHub archive digest.
PACKAGE_MANIFEST_SHA256 = (
    "fee3e21ec7312847e3abf9098206122df10738bfb36c633e1f15fd53a6c725e8"
)
REFERENCE_SPLITS = Path(__file__).resolve().parents[1] / "splits"
MAX_WEIGHT_BYTES = 2_000_000_000
DEVELOPMENT_FOLDS = (0, 1, 3, 4)


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def _json(data: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(data, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("metadata must be a JSON object")
    return value


def _digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _folds(value: object, expected: list[int]) -> bool:
    return (
        isinstance(value, list)
        and all(type(v) is int for v in value)
        and value == expected
    )


def _package_path(package: Path, name: str) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or PurePosixPath(name).is_absolute()
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise ValueError(f"unsafe package path: {name!r}")
    path = package / name
    # Even internal symlinks are not accepted as immutable package members.
    if any(
        part.is_symlink() for part in [path, *path.parents] if part != package.parent
    ):
        raise ValueError(f"symlink in package path: {name}")
    if not path.resolve().is_relative_to(package) or not path.is_file():
        raise ValueError(f"missing/outside package file: {name}")
    return path


def _verified_bytes(package: Path, files: dict, name: str) -> bytes:
    if name not in files or not _digest(files[name]):
        raise ValueError(f"unlisted/unpinned package file: {name}")
    data = _package_path(package, name).read_bytes()
    if hashlib.sha256(data).hexdigest() != files[name]:
        raise ValueError(f"package SHA mismatch: {name}")
    return data


def verify_package(package: Path) -> tuple[dict, str]:
    """Validate the pinned archive index before interpreting its contents."""
    manifest_bytes = _package_path(package, "MANIFEST.json").read_bytes()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if digest != PACKAGE_MANIFEST_SHA256:
        raise ValueError("source MANIFEST SHA differs from the pinned D082 handoff")
    manifest = _json(manifest_bytes)
    if (
        manifest.get("fold2") != "sealed"
        or set(manifest.get("folds", {})) != {"0", "1", "3", "4"}
        or manifest.get("recipe")
        != "z(hub)+.15*z(verifier_logit)+.5*(head_score-z(hub))"
        or manifest.get("heads") != "depth2/hidden64/features13"
    ):
        raise ValueError(
            "not the frozen development D082 package; fold2 remains sealed"
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("package requires a nonempty files SHA index")
    for name, expected in files.items():
        if not _digest(expected) or sha(_package_path(package, name)) != expected:
            raise ValueError(f"package SHA mismatch: {name}")
    return manifest, digest


def _publish_no_replace(source: Path, destination: Path) -> None:
    """Atomic directory rename that cannot overwrite even an empty destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        args = (os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        args = (
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )  # NOREPLACE
    else:
        raise OSError("atomic no-replace publication requires Linux or macOS")
    rename.restype = ctypes.c_int
    if rename(*args) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def prepare_bundle(
    package: Path,
    fold: int,
    dino_weights: Path,
    clip_weights: Path,
    verifier: Path,
    out: Path,
) -> dict:
    if type(fold) is not int or fold not in DEVELOPMENT_FOLDS:
        raise ValueError("only development folds0/1/3/4; fold2/all-data are sealed")
    package, out = Path(package).resolve(), Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite bundle: {out}")
    out = out.parent.resolve() / out.name
    if out.is_relative_to(package):
        raise ValueError("output must not modify the source package")
    manifest, manifest_sha = verify_package(package)
    files, prefix = manifest["files"], f"fold{fold}"
    allowed = [f for f in DEVELOPMENT_FOLDS if f != fold]
    excluded = sorted([fold, 2])
    spec = manifest["folds"][str(fold)]
    if (
        spec.get("dumps") != [f"{prefix}/dino.npz", f"{prefix}/clip.npz"]
        or spec.get("head") != f"{prefix}/head"
    ):
        raise ValueError("unexpected component/head paths or order")
    ids = {}
    for role in ("query", "gallery"):
        name = f"fold{fold}_{role}.csv"
        split_bytes = _verified_bytes(package, files, f"splits/{name}")
        if hashlib.sha256(split_bytes).hexdigest() != sha(REFERENCE_SPLITS / name):
            raise ValueError("source package changed the original fold split")
        ids[f"{role}_ids"] = _read_input_ids(package / "splits" / name)
    if (
        not ids["query_ids"]
        or len(ids["gallery_ids"]) < 10
        or set(ids["query_ids"]) & set(ids["gallery_ids"])
    ):
        raise ValueError("invalid query/gallery IDs")
    folds_bytes = _verified_bytes(package, files, "splits/folds.csv")
    if hashlib.sha256(folds_bytes).hexdigest() != sha(REFERENCE_SPLITS / "folds.csv"):
        raise ValueError("source package changed original fold assignments")

    sources: list[tuple[Path, str, str]] = []
    weight_bytes = 0
    supplied = {
        "dino": Path(dino_weights),
        "clip": Path(clip_weights),
        "verifier": Path(verifier),
    }
    for name, weight_path in supplied.items():
        receipt_name = f"{prefix}/{name}_training_receipt.json"
        receipt = _json(_verified_bytes(package, files, receipt_name))
        if (
            not _folds(receipt.get("actual_train_folds"), allowed)
            or not _folds(receipt.get("excluded_folds"), excluded)
            or receipt.get("fold", receipt.get("validation_fold")) != fold
        ):
            raise ValueError(f"{name} receipt violates OOF provenance")
        digest = receipt.get("weights_sha256")
        if (
            not _digest(digest)
            or not weight_path.is_file()
            or sha(weight_path) != digest
        ):
            raise ValueError(f"provided {name} weight SHA mismatch")
        upstream = [
            entry
            for entry in manifest.get("upstream_weights", [])
            if entry.get("fold") == fold and entry.get("component") == name
        ]
        if (
            len(upstream) != 1
            or upstream[0].get("weights_sha256") != digest
            or type(upstream[0].get("bytes")) is not int
            or upstream[0]["bytes"] != weight_path.stat().st_size
        ):
            raise ValueError(f"{name} weight differs from manifest source")
        weight_bytes += weight_path.stat().st_size
        if name == "verifier":
            if (
                receipt.get("options", {}).get("input") != 392
                or receipt.get("backbone_matches_pinned_initialization_exactly")
                is not True
                or sorted(p.get("fold", -1) for p in receipt.get("pair_inputs", []))
                != allowed
            ):
                raise ValueError("verifier input/provenance does not match D082")
            sources.extend(
                [
                    (weight_path, "verifier.pt", digest),
                    (
                        package / receipt_name,
                        "verifier_training_receipt.json",
                        files[receipt_name],
                    ),
                ]
            )
            continue
        index = 0 if name == "dino" else 1
        cfg_name, export_name = (
            f"{prefix}/{name}_config.yaml",
            f"{prefix}/{name}_export_receipt.json",
        )
        cfg_bytes = _verified_bytes(package, files, cfg_name)
        cfg = yaml.safe_load(cfg_bytes)
        export = _json(_verified_bytes(package, files, export_name))
        if (
            not isinstance(cfg, dict)
            or cfg.get("arch") != ("dinov2_vitb14" if index == 0 else "clip_vitb16")
            or cfg.get("input_size") != ([392, 392] if index == 0 else [288, 288])
            or cfg.get("pool") != "cls_top16"
            or cfg.get("embed_dim") != 1536
            or cfg.get("train", {}).get("folds") != ",".join(map(str, allowed))
            or receipt.get("config_sha256") != files[cfg_name]
            or export.get("config_sha256") != files[cfg_name]
            or export.get("weights_sha256") != digest
            or not _folds(export.get("actual_train_folds"), allowed)
            or export.get("embedding_path") != "full-frame"
            or export.get("dump_sha256") != files.get(f"{prefix}/{name}.npz")
            or export.get("query_split_sha256") != files[f"splits/fold{fold}_query.csv"]
            or export.get("gallery_split_sha256")
            != files[f"splits/fold{fold}_gallery.csv"]
        ):
            raise ValueError(
                f"{name} config/export does not match pinned full-frame provenance"
            )
        sources.extend(
            [
                (weight_path, f"components/{index}/weights.safetensors", digest),
                (
                    package / cfg_name,
                    f"components/{index}/model_config.yaml",
                    files[cfg_name],
                ),
                (
                    package / receipt_name,
                    f"components/{index}/training_receipt.json",
                    files[receipt_name],
                ),
            ]
        )

    head_prefix = f"{prefix}/head"
    # Existing safe adapter checks exact architecture and every nested OOF parent.
    DevelopmentGraphScorer(
        package / head_prefix,
        outer_fold=fold,
        weights_sha256=files[f"{head_prefix}/weights.safetensors"],
        config_sha256=files[f"{head_prefix}/model_config.json"],
        receipt_sha256=files[f"{head_prefix}/training_receipt.json"],
    )
    for name in ("weights.safetensors", "model_config.json", "training_receipt.json"):
        source_name = f"{head_prefix}/{name}"
        sources.append((package / source_name, f"head/{name}", files[source_name]))
    weight_bytes += (package / head_prefix / "weights.safetensors").stat().st_size
    if weight_bytes > MAX_WEIGHT_BYTES:
        raise ValueError("combined inference weights exceed 2 GB")

    # Same C4 normalization and exact static gallery used by the accepted replay.
    # No query values enter this scale calculation; nothing is fit on a subset.
    arrays = load_components(
        [package / path for path in spec["dumps"]], ids["query_ids"], ids["gallery_ids"]
    )
    if any(q.shape[1] != 1536 or g.shape[1] != 1536 for q, g in arrays):
        raise ValueError("C4 dimensions do not match DINO392 + CLIP288 configs")
    scales = [gallery_similarity_std(g) for _, g in arrays]
    if any(not np.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("invalid fixed original gallery scales")
    fusion = dict(
        schema_version=1,
        name=f"d082-development-fold{fold}",
        components=[
            dict(path=f"components/{i}", similarity_std=s) for i, s in enumerate(scales)
        ],
        candidate_policy="gallery_z",
        candidate_threshold=6.6,
        rerank_mode="standardized",
        rerank_weight=0.15,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as temp:
        staged = Path(temp) / "bundle"
        staged.mkdir()
        records = {}
        for source, name, expected in sources:
            destination = staged / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            actual = sha(destination)
            if actual != expected:
                raise ValueError(f"copied bytes changed after preflight: {name}")
            records[name] = dict(sha256=actual, bytes=destination.stat().st_size)
        for name, document in (
            ("fusion_config.json", fusion),
            ("development_ids.json", ids),
        ):
            destination = staged / name
            destination.write_text(
                json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            records[name] = dict(
                sha256=sha(destination), bytes=destination.stat().st_size
            )
        actual_weight_bytes = sum(
            r["bytes"]
            for p, r in records.items()
            if p.endswith((".safetensors", ".pt"))
        )
        if (
            actual_weight_bytes != weight_bytes
            or actual_weight_bytes > MAX_WEIGHT_BYTES
        ):
            raise ValueError("copied weight budget differs from verified source")
        runtime = dict(
            schema_version=1,
            kind="d082_development_runtime",
            fold=fold,
            recipe=dict(
                residual_weight=0.5, verifier_weight=0.15, candidate_threshold=6.6
            ),
            source_manifest_sha256=manifest_sha,
            files=records,
            weight_bytes=actual_weight_bytes,
        )
        runtime_path = staged / "ranker_runtime.json"
        runtime_path.write_text(
            json.dumps(runtime, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        runtime_sha = sha(runtime_path)
        # Recheck exact packaged bytes immediately before atomic publication.
        if any(
            sha(staged / name) != record["sha256"] for name, record in records.items()
        ):
            raise ValueError("staged bundle changed before publication")
        _publish_no_replace(staged, out)
    return {
        "path": str(out),
        "ranker_runtime_sha256": runtime_sha,
        "weight_bytes": actual_weight_bytes,
        "scope": "OOF development only; not release acceptance",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=DEVELOPMENT_FOLDS, required=True)
    parser.add_argument("--dino-weights", type=Path, required=True)
    parser.add_argument("--clip-weights", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_bundle(
                args.package,
                args.fold,
                args.dino_weights,
                args.clip_weights,
                args.verifier,
                args.out,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
