"""Pinned, opt-in D082 development runtime. No fitting or evaluator imports.

This first integration deliberately accepts only OOF0/1/3/4 artifacts and their
development ID allowlists. It cannot be mistaken for an all-data release or used
to open control2. The unchanged default path still implements the submitted C5.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import yaml

from reid.infer.fusion import load_fusion_config
from reid.infer.list_ranker import DevelopmentGraphScorer
from reid.infer.observability import file_sha256
from reid.post.ranker import PreparedRankerGallery


SOURCE_MANIFEST_SHA256 = (
    "fee3e21ec7312847e3abf9098206122df10738bfb36c633e1f15fd53a6c725e8"
)
FILES = {
    "fusion_config.json",
    "development_ids.json",
    "verifier.pt",
    "verifier_training_receipt.json",
    "head/weights.safetensors",
    "head/model_config.json",
    "head/training_receipt.json",
    *{
        f"components/{i}/{name}"
        for i in (0, 1)
        for name in (
            "model_config.yaml",
            "weights.safetensors",
            "training_receipt.json",
        )
    },
}
RECIPE = dict(residual_weight=0.5, verifier_weight=0.15, candidate_threshold=6.6)


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate D082 JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("D082 metadata must be an object")
    return value


def _folds(value, expected):
    return (
        isinstance(value, list)
        and all(type(f) is int for f in value)
        and value == expected
    )


@dataclass(frozen=True)
class D082Runtime:
    model_dir: Path
    manifest_sha256: str
    fold: int
    scales: tuple[float, float]
    scorer: DevelopmentGraphScorer
    weight_bytes: int

    @property
    def verifier_path(self):
        return self.model_dir / "verifier.pt"

    @property
    def identity(self):
        return dict(
            kind="d082_development_runtime",
            fold=self.fold,
            manifest_sha256=self.manifest_sha256,
            weight_bytes=self.weight_bytes,
            recipe=RECIPE,
            source_manifest_sha256=SOURCE_MANIFEST_SHA256,
        )

    def prepare(self, components, gallery_rows):
        return PreparedRankerGallery(
            components,
            [r.image_id for r in gallery_rows],
            [(r.image_path, r.bbox) for r in gallery_rows],
            self.scales,
            ranker=self.scorer,
            residual_weight=0.5,
        )


def load_d082_runtime(
    model_dir: Path,
    manifest_sha256: str,
    query_ids,
    gallery_ids,
    verifier_path: Path | None = None,
) -> D082Runtime:
    root = model_dir.resolve()
    if (
        not isinstance(manifest_sha256, str)
        or re.fullmatch("[0-9a-f]{64}", manifest_sha256) is None
    ):
        raise ValueError("--d082-manifest-sha256 requires a full lowercase SHA256 pin")
    path = root / "ranker_runtime.json"
    if not path.resolve().is_relative_to(root) or file_sha256(path) != manifest_sha256:
        raise ValueError("D082 runtime manifest SHA/path mismatch")
    data = _json(path)
    required = {
        "schema_version",
        "kind",
        "fold",
        "recipe",
        "source_manifest_sha256",
        "files",
        "weight_bytes",
    }
    if (
        set(data) != required
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or data["kind"] != "d082_development_runtime"
    ):
        raise ValueError("unsupported D082 runtime manifest schema")
    fold = data["fold"]
    if type(fold) is not int or fold not in (0, 1, 3, 4):
        raise ValueError("only D082 development0/1/3/4; fold2 remains sealed")
    if (
        data["recipe"] != RECIPE
        or any(type(v) not in (int, float) for v in data["recipe"].values())
        or data["source_manifest_sha256"] != SOURCE_MANIFEST_SHA256
    ):
        raise ValueError(
            "not the pinned D082 recipe/source; no threshold or weight retuning"
        )
    records = data["files"]
    if not isinstance(records, dict) or set(records) != FILES:
        raise ValueError("D082 artifact must contain the exact 13 pinned runtime files")
    actual_files = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() or p.is_symlink()
    }
    if actual_files != FILES | {"ranker_runtime.json"}:
        raise ValueError("unlisted, missing or linked files in D082 artifact")
    for relative, record in records.items():
        asset = root / relative
        if (
            not isinstance(record, dict)
            or set(record) != {"sha256", "bytes"}
            or not isinstance(record["sha256"], str)
            or re.fullmatch("[0-9a-f]{64}", record["sha256"]) is None
            or type(record["bytes"]) is not int
            or record["bytes"] <= 0
            or asset.is_symlink()
            or not asset.resolve().is_relative_to(root)
            or asset.stat().st_size != record["bytes"]
            or file_sha256(asset) != record["sha256"]
        ):
            raise ValueError(f"D082 file integrity mismatch: {relative}")
    total = sum(
        r["bytes"] for n, r in records.items() if n.endswith((".pt", ".safetensors"))
    )
    if (
        type(data["weight_bytes"]) is not int
        or total != data["weight_bytes"]
        or total > 2_000_000_000
    ):
        raise ValueError("D082 total model weight budget mismatch/exceeds2GB")
    ids = _json(root / "development_ids.json")
    if set(ids) != {"query_ids", "gallery_ids"}:
        raise ValueError("development allowlists must contain IDs only")
    for key, supplied in (("query_ids", query_ids), ("gallery_ids", gallery_ids)):
        allowed = ids[key]
        if (
            not isinstance(allowed, list)
            or any(not isinstance(i, str) or not i for i in allowed)
            or len(allowed) != len(set(allowed))
            or not supplied
            or len(supplied) != len(set(supplied))
            or not set(supplied) <= set(allowed)
        ):
            raise ValueError(
                f"inputs outside OOF development allowlist: {key}; test/control2 forbidden"
            )
    if set(ids["query_ids"]) & set(ids["gallery_ids"]):
        raise ValueError("development query/gallery allowlists overlap")
    expected = [f for f in (0, 1, 3, 4) if f != fold]
    for index, arch, side in ((0, "dinov2_vitb14", 392), (1, "clip_vitb16", 288)):
        prefix = f"components/{index}"
        receipt = _json(root / prefix / "training_receipt.json")
        config = yaml.safe_load((root / prefix / "model_config.yaml").read_text())
        if (
            not _folds(receipt.get("actual_train_folds"), expected)
            or receipt.get("weights_sha256")
            != records[f"{prefix}/weights.safetensors"]["sha256"]
            or receipt.get("config_sha256")
            != records[f"{prefix}/model_config.yaml"]["sha256"]
            or not isinstance(config, dict)
            or config.get("arch") != arch
            or config.get("input_size") != [side, side]
            or config.get("pool") != "cls_top16"
        ):
            raise ValueError(f"D082 component{index} recipe/OOF provenance mismatch")
    vr = _json(root / "verifier_training_receipt.json")
    if (
        not _folds(vr.get("actual_train_folds"), expected)
        or vr.get("weights_sha256") != records["verifier.pt"]["sha256"]
    ):
        raise ValueError("D082 verifier OOF provenance mismatch")
    if verifier_path is not None:
        given = (
            verifier_path / "verifier.pt" if verifier_path.is_dir() else verifier_path
        )
        if given.resolve() != root / "verifier.pt":
            raise ValueError("D082 must use the verifier inside its pinned bundle")
    cfg = load_fusion_config(root)
    if (
        len(cfg["components"]) != 2
        or [c["path"] for c in cfg["components"]] != ["components/0", "components/1"]
        or cfg["candidate_threshold"] != 6.6
        or cfg.get("rerank_weight") != 0.15
    ):
        raise ValueError("D082 fusion config differs from frozen recipe")
    scorer = DevelopmentGraphScorer(
        root / "head",
        outer_fold=fold,
        weights_sha256=records["head/weights.safetensors"]["sha256"],
        config_sha256=records["head/model_config.json"]["sha256"],
        receipt_sha256=records["head/training_receipt.json"]["sha256"],
    )
    return D082Runtime(
        root,
        manifest_sha256,
        fold,
        tuple(c["similarity_std"] for c in cfg["components"]),
        scorer,
        total,
    )


def extract_component_rows(extractor, rows, batch_size):
    """Mirror A export→C4 load arithmetic; never reconstruct blocks from fusion.

    A full-frame export uses extract_rows (NumPy L2 after Torch L2); D082's C4
    loader applies a second NumPy L2. Both operations are retained explicitly
    to avoid an unmeasured numerical change when moving from replay to images.
    """
    if type(batch_size) is not int or batch_size <= 0 or not rows:
        raise ValueError(
            "component extraction needs rows and a positive integer batch size"
        )
    chunks = [[], []]
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        blocks = extractor.extract_components_batch(
            [(r.image_path, r.bbox) for r in batch]
        )
        if len(blocks) != 2:
            raise ValueError("D082 needs exactly two original component embeddings")
        for index, block in enumerate(blocks):
            if (
                not isinstance(block, np.ndarray)
                or block.dtype != np.float32
                or block.ndim != 2
                or len(block) != len(batch)
                or block.shape[1] == 0
                or not np.isfinite(block).all()
            ):
                raise ValueError("extractor returned invalid D082 components")
            chunks[index].append(block)
    results = []
    for parts in chunks:
        result = np.concatenate(parts)
        for _ in range(2):
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or np.any(norms <= 0):
                raise ValueError("extractor returned zero/nonfinite component norm")
            result = result / norms
        results.append(result)
    return tuple(results)
