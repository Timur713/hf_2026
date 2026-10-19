"""SHA-pinned all-data D082 candidate format; NOT a final acceptance gate.

This validates supplied provenance statements, not the historical training run.
No development promotion, calibration, download, evaluation or fold2 access.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import stat
from urllib.parse import urlsplit

import numpy as np
from safetensors.torch import load
import torch
import yaml

from reid.infer.d082 import RECIPE
from reid.infer.fusion import load_fusion_config
from reid.infer.observability import file_sha256
from reid.model.list_ranker import CandidateGraphRanker
from reid.post.ranker import PreparedRankerGallery

MANIFEST = "d082_production.json"
FOLDS = [0, 1, 2, 3, 4]
MAX_METADATA = 1024 * 1024
MAX_WEIGHTS = 2_000_000_000
FILES = frozenset(
    {
        "fusion_config.json",
        "recipe_receipt.json",
        "verifier.pt",
        "verifier_config.json",
        "verifier_training_receipt.json",
        "head/weights.safetensors",
        "head/model_config.json",
        "head/training_receipt.json",
        *{
            f"components/{i}/{n}"
            for i in (0, 1)
            for n in (
                "model_config.yaml",
                "weights.safetensors",
                "training_receipt.json",
            )
        },
    }
)
DIRECTORIES = frozenset({"components", "components/0", "components/1", "head"})
ARCHES = ("dinov2_vitb14", "clip_vitb16")
SIDES = (392, 288)
VERIFIER_CONFIG = dict(
    arch="dinov2_vits14", input=392, head="concat", trained_on="folds [0, 1, 2, 3, 4]"
)
GRAPH_CONFIG = dict(
    features=13,
    hidden=64,
    depth=2,
    scope="all_data",
    actual_train_folds=FOLDS,
    excluded_folds=[],
)


def digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected full lowercase SHA256")
    return value


def commit(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("provenance requires a full Git commit")
    return value


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate metadata key: {key}")
        result[key] = value
    return result


class _Yaml(yaml.SafeLoader):
    pass


def _yaml_mapping(loader, node):
    return _unique(
        [
            (loader.construct_object(k), loader.construct_object(v))
            for k, v in node.value
        ]
    )


_Yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping)


def metadata(path):
    with path.open("rb") as handle:
        raw = handle.read(MAX_METADATA + 1)
    if len(raw) > MAX_METADATA:
        raise ValueError("metadata exceeds 1 MiB")
    try:
        value = (
            yaml.load(raw, Loader=_Yaml)
            if path.suffix == ".yaml"
            else json.loads(
                raw,
                object_pairs_hook=_unique,
                parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)),
            )
        )
    except (yaml.YAMLError, UnicodeError, RecursionError) as exc:
        raise ValueError("invalid production metadata") from exc
    if not isinstance(value, dict):
        raise ValueError("production metadata must be an object")
    return value


def _equal(value, expected):
    # JSON structural comparison that rejects bool-as-int and NaN/Infinity.
    return json.dumps(value, sort_keys=True, allow_nan=False) == json.dumps(
        expected, sort_keys=True
    )


def check_tree(root, *, manifest_required=True):
    root = Path(root).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)) or not root.is_dir():
        raise ValueError("bundle must be a real directory without symlink ancestors")
    files = set()
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode) and name in DIRECTORIES:
            continue
        if not stat.S_ISREG(mode) or name not in FILES | {MANIFEST}:
            raise ValueError(f"unlisted, linked or special bundle entry: {name}")
        files.add(name)
    if files != (FILES | {MANIFEST} if manifest_required else FILES):
        raise ValueError("production bundle must contain the exact all-data file set")
    return root


def _public_init(data):
    if not isinstance(data, dict) or set(data) != {
        "kind",
        "url",
        "revision",
        "sha256",
        "author",
    }:
        raise ValueError("public initialization requires kind/url/revision/SHA/author")
    digest(data["sha256"])
    revision = commit(data["revision"])
    url = urlsplit(data["url"])
    if (
        data["kind"] != "public"
        or not isinstance(data["author"], str)
        or not data["author"].strip()
        or url.scheme != "https"
        or url.netloc != "huggingface.co"
        or url.query
        or url.fragment
        or re.fullmatch(
            r"/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/resolve/"
            + revision
            + r"/[A-Za-z0-9._/-]+\.safetensors",
            url.path,
        )
        is None
        or any(p in (".", "..") for p in url.path.split("/"))
    ):
        raise ValueError(
            "initialization must pin an immutable public HF safetensors revision"
        )


def _receipt(data, role, folds, excluded, dataset_sha, weights=None, config=None):
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or type(data.get("schema_version")) is not int
        or data.get("role") != role
        or data.get("scope") != ("all_data" if folds == FOLDS else "nested_oof")
        or not _equal(data.get("actual_train_folds"), folds)
        or not _equal(data.get("excluded_folds"), excluded)
        or data.get("dataset_manifest_sha256") != dataset_sha
    ):
        raise ValueError(
            f"{role} receipt violates actual training fold/scope/dataset provenance"
        )
    commit(data.get("source_commit"))
    if folds == FOLDS and any(
        data.get(k) is not None for k in ("outer", "validation_fold", "held_out_fold")
    ):
        raise ValueError("all-data receipt cannot retain an OOF held-out fold")
    for key, expected in (("weights_sha256", weights), ("config_sha256", config)):
        digest(data.get(key))
        if expected is not None and data[key] != expected:
            raise ValueError(f"{role} receipt {key} mismatch")
    if role != "graph":
        _public_init(data.get("initialization"))


def _provenance(root, data):
    records, dataset = data["files"], data["dataset_manifest_sha256"]
    for i, (arch, side) in enumerate(zip(ARCHES, SIDES)):
        prefix = f"components/{i}"
        cfg = metadata(root / prefix / "model_config.yaml")
        receipt = metadata(root / prefix / "training_receipt.json")
        _receipt(
            receipt,
            f"encoder{i}",
            FOLDS,
            [],
            dataset,
            records[f"{prefix}/weights.safetensors"]["sha256"],
            records[f"{prefix}/model_config.yaml"]["sha256"],
        )
        required = dict(
            arch=arch,
            input_size=[side, side],
            pool="cls_top16",
            embed_dim=1536,
            bbox_pad=0.0,
            resize="stretch",
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            neck="bnneck",
            infer_dtype="fp16",
            tta_flip=False,
        )
        if any(not _equal(cfg.get(k), v) for k, v in required.items()):
            raise ValueError(
                f"encoder{i} differs from frozen architecture/preprocessing"
            )
        train, public = cfg.get("train", {}), cfg.get("pretrained", {})
        init = receipt["initialization"]
        if (
            not isinstance(train, dict)
            or train.get("folds") != "all"
            or train.get("excluded_folds") != []
            or not isinstance(public, dict)
            or train.get("init_from") is not None
            or not _equal(cfg.get("parts", 0), 0)
            or any(
                public.get(k) != init[k]
                for k in ("url", "revision", "sha256", "author")
            )
        ):
            raise ValueError(
                f"encoder{i} config contradicts all-data/public-init receipt"
            )
        if "commit" in train and (
            not isinstance(train["commit"], str)
            or re.fullmatch(r"[0-9a-f]{7,40}", train["commit"]) is None
            or not receipt["source_commit"].startswith(train["commit"])
        ):
            raise ValueError(f"encoder{i} config/receipt training commit mismatch")
    vc = metadata(root / "verifier_config.json")
    if not _equal(vc, VERIFIER_CONFIG):
        raise ValueError(
            "verifier config differs from frozen input392/concat/all-data recipe"
        )
    vr = metadata(root / "verifier_training_receipt.json")
    _receipt(
        vr,
        "verifier",
        FOLDS,
        [],
        dataset,
        records["verifier.pt"]["sha256"],
        records["verifier_config.json"]["sha256"],
    )
    mining = vr.get("mining")
    if (
        not isinstance(mining, dict)
        or mining.get("source") != "public_init_only"
        or not _equal(mining.get("input_folds"), FOLDS)
        or mining.get("dataset_manifest_sha256") != dataset
        or not isinstance(mining.get("encoders"), list)
        or len(mining["encoders"]) != 2
    ):
        raise ValueError("verifier needs train-only public-init mining provenance")
    digest(mining.get("pairs_sha256"))
    for i, source in enumerate(mining["encoders"]):
        if (
            not isinstance(source, dict)
            or source.get("arch") != ARCHES[i]
            or source.get("actual_train_folds") != []
        ):
            raise ValueError(
                "verifier mining encoders must be public-only, not leaked fine-tunes"
            )
        _public_init(source.get("initialization"))
    if vr.get("plate_mask_training") is not True:
        raise ValueError(
            "verifier must declare plate-mask training; visual coverage still needs acceptance"
        )
    gc = metadata(root / "head/model_config.json")
    if not _equal(gc, GRAPH_CONFIG):
        raise ValueError("graph config must be depth2/hidden64/features13 all-data")
    gr = metadata(root / "head/training_receipt.json")
    _receipt(
        gr,
        "graph",
        FOLDS,
        [],
        dataset,
        records["head/weights.safetensors"]["sha256"],
        records["head/model_config.json"]["sha256"],
    )
    if gr.get("initialization") != {"kind": "random"}:
        raise ValueError("graph must start randomly, never from an OOF head")
    views = gr.get("nested_supervision")
    if (
        not isinstance(views, list)
        or len(views) != 5
        or any(not isinstance(v, dict) or type(v.get("view")) is not int for v in views)
        or sorted(v["view"] for v in views) != FOLDS
    ):
        raise ValueError(
            "all-data graph requires five separately audited OOF supervision views"
        )
    for view in views:
        heldout = view["view"]
        expected = [f for f in FOLDS if f != heldout]
        parents = view.get("encoders")
        if not isinstance(parents, list) or len(parents) != 2:
            raise ValueError(
                "each graph view requires two nested encoder receipts/exports"
            )
        for i, parent in enumerate(parents):
            if not isinstance(parent, dict):
                raise ValueError("nested provenance must be an object")
            receipt, export = parent.get("receipt", {}), parent.get("export", {})
            _receipt(receipt, f"encoder{i}", expected, [heldout], dataset)
            if (
                not isinstance(export, dict)
                or export.get("weights_sha256") != receipt["weights_sha256"]
                or export.get("config_sha256") != receipt["config_sha256"]
                or not _equal(export.get("actual_train_folds"), expected)
                or type(export.get("view")) is not int
                or export["view"] != heldout
                or export.get("embedding_path") != "full_frame_c3_fixed4"
            ):
                raise ValueError(
                    "graph nested export/receipt mismatch or self-view leakage"
                )
            digest(export.get("features_sha256"))
    fc = load_fusion_config(root)
    if (
        len(fc["components"]) != 2
        or [c["path"] for c in fc["components"]] != ["components/0", "components/1"]
        or fc["candidate_policy"] != "gallery_z"
        or fc["candidate_threshold"] != 6.6
        or fc.get("rerank_weight") != 0.15
        or fc["rerank_mode"] != "standardized"
    ):
        raise ValueError("fusion differs from fixed D082 recipe")
    recipe = metadata(root / "recipe_receipt.json")
    commit(recipe.get("source_commit"))
    digest(recipe.get("recipe_freeze_sha256"))
    if (
        not _equal(recipe.get("recipe"), RECIPE)
        or recipe.get("scale_source") != "training_only"
        or not _equal(recipe.get("scale_folds"), FOLDS)
        or recipe.get("dataset_manifest_sha256") != dataset
        or recipe.get("fusion_config_sha256") != records["fusion_config.json"]["sha256"]
        or recipe.get("encoder_weights_sha256")
        != [records[f"components/{i}/weights.safetensors"]["sha256"] for i in (0, 1)]
    ):
        raise ValueError("fusion scales/recipe lack matching train-only provenance")
    digest(recipe.get("scale_inputs_sha256"))
    return fc


def inspect_bundle(root: Path, manifest_sha256: str) -> dict:
    try:
        return _inspect_bundle(root, manifest_sha256)
    except (TypeError, KeyError, AttributeError, RecursionError) as exc:
        raise ValueError("invalid production manifest/provenance structure") from exc


def _inspect_bundle(root: Path, manifest_sha256: str) -> dict:
    """Bounded integrity + provenance validation before any model deserialize."""
    digest(manifest_sha256)
    root = check_tree(root)
    if (root / MANIFEST).stat().st_size > MAX_METADATA or file_sha256(
        root / MANIFEST
    ) != manifest_sha256:
        raise ValueError("production manifest SHA/size mismatch")
    data = metadata(root / MANIFEST)
    if (
        set(data)
        != {
            "schema_version",
            "kind",
            "files",
            "weight_bytes",
            "recipe",
            "dataset_manifest_sha256",
        }
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or data["kind"] != "d082_production_candidate"
        or not _equal(data["recipe"], RECIPE)
    ):
        raise ValueError(
            "unsupported production schema/recipe; development promotion forbidden"
        )
    digest(data["dataset_manifest_sha256"])
    records = data["files"]
    if not isinstance(records, dict) or set(records) != FILES:
        raise ValueError("production manifest has unexpected file set")
    total = 0
    for name, record in records.items():
        if not isinstance(record, dict) or set(record) != {"sha256", "bytes"}:
            raise ValueError("invalid production file record")
        digest(record["sha256"])
        weight = name.endswith((".pt", ".safetensors"))
        if (
            type(record["bytes"]) is not int
            or not 0 < record["bytes"] <= (MAX_WEIGHTS if weight else MAX_METADATA)
            or (root / name).stat().st_size != record["bytes"]
        ):
            raise ValueError(f"production file size mismatch: {name}")
        if weight:
            total += record["bytes"]
    if (
        type(data["weight_bytes"]) is not int
        or total != data["weight_bytes"]
        or total > MAX_WEIGHTS
    ):
        raise ValueError("production weight budget mismatch or exceeds 2GB")
    for name, record in records.items():
        if file_sha256(root / name) != record["sha256"]:
            raise ValueError(f"production file SHA mismatch: {name}")
    cfg = _provenance(root, data)
    return dict(
        manifest_sha256=manifest_sha256,
        weight_bytes=total,
        files=records,
        scales=[c["similarity_std"] for c in cfg["components"]],
        kind=data["kind"],
        recipe=RECIPE,
        provenance_statements_verified=True,
        historical_training_verified=False,
        final_acceptance=False,
        gpu_acceptance="pending",
        quality_acceptance="pending",
        verifier_gallery_cache=False,
    )


class ProductionGraphScorer:
    """Only constructed after inspect_bundle; exact math of the unchanged A head."""

    def __init__(self, root, report):
        from reid.infer.list_ranker import _read_pinned

        raw = _read_pinned(
            root / "head/weights.safetensors",
            report["files"]["head/weights.safetensors"]["sha256"],
        )
        self.model = CandidateGraphRanker(features=13, hidden=64, depth=2).cpu().eval()
        self.model.load_state_dict(load(raw), strict=True)

    def score_lists(self, nodes, edges, base):
        if (
            nodes.ndim != 3
            or nodes.shape[1:] != (10, 13)
            or len(nodes) == 0
            or edges.shape != (len(nodes), 10, 10, 2)
            or base.shape != (len(nodes), 10)
            or any(
                v.dtype != np.float32 or not np.isfinite(v).all()
                for v in (nodes, edges, base)
            )
        ):
            raise ValueError(
                "graph expects finite float32 nodes[N,10,13], edges[N,10,10,2], base[N,10]"
            )
        with torch.inference_mode():
            return (
                self.model.score_lists(
                    *(torch.from_numpy(v) for v in (nodes, edges, base))
                )
                .cpu()
                .numpy()
            )


@dataclass(frozen=True)
class ProductionRuntime:
    model_dir: Path
    identity: dict
    scorer: ProductionGraphScorer

    @property
    def verifier_path(self):
        return self.model_dir / "verifier.pt"

    def prepare(self, components, gallery_rows):
        return PreparedRankerGallery(
            components,
            [r.image_id for r in gallery_rows],
            [(r.image_path, r.bbox) for r in gallery_rows],
            tuple(self.identity["scales"]),
            ranker=self.scorer,
            residual_weight=0.5,
        )


def load_production_runtime(root: Path, pin: str, verifier_path: Path | None = None):
    report = inspect_bundle(root, pin)
    root = root.absolute()
    if verifier_path is not None:
        given = (
            verifier_path / "verifier.pt" if verifier_path.is_dir() else verifier_path
        )
        if given.is_symlink() or given.resolve() != root / "verifier.pt":
            raise ValueError("production must use its pinned verifier")
    # Restricted weights_only loading, not a sandbox for hostile pickle files:
    # the independently trusted A pin is mandatory. Check internal metadata too;
    # matching sidecars alone must not promote an OOF verifier.
    payload = torch.load(root / "verifier.pt", map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or any(
        not _equal(payload.get(k), v) for k, v in VERIFIER_CONFIG.items()
    ):
        raise ValueError("verifier payload contradicts all-data config/receipt")
    if not isinstance(payload.get("state"), dict) or not payload["state"]:
        raise ValueError("verifier checkpoint must contain a state dictionary")
    from safetensors import safe_open

    for i in (0, 1):
        # Validate the real tensor container without instantiating both large
        # encoders; strict architecture/state matching happens in Extractor.
        with safe_open(
            root / f"components/{i}/weights.safetensors", framework="pt", device="cpu"
        ) as weights:
            if not list(weights.keys()):
                raise ValueError("encoder safetensors must not be empty")
    return ProductionRuntime(root, report, ProductionGraphScorer(root, report))
