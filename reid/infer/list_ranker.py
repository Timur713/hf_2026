"""Pinned CPU adapter for A's OOF D-082 graph heads; not release approval.

This intentionally rejects fold2/sealed/all-data heads. Supporting them needs
the separate frozen-policy control/release gates. No existing runtime imports
this module automatically and there is no default/model manifest change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
from safetensors.torch import load
import torch

from reid.model.list_ranker import CandidateGraphRanker


def _read_pinned(path: Path, digest: str) -> bytes:
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("a full lowercase SHA-256 pin is required")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"SHA-256 mismatch: {path.name}")
    return data


def _json(data: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    result = json.loads(data, object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError("ranker metadata must be an object")
    return result


def _folds_equal(value: object, expected: list[int]) -> bool:
    return (
        isinstance(value, list)
        and all(type(v) is int for v in value)
        and value == expected
    )


def _digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class DevelopmentGraphScorer:
    """Use unchanged A network, CPU float32 and single-query neural operations.

    Explicit pins must originate from the independently verified handoff
    manifest, not hashes silently computed from arbitrary local files.
    Provenance checks validate the receipt, not the historical execution itself.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        outer_fold: int,
        weights_sha256: str,
        config_sha256: str,
        receipt_sha256: str,
    ):
        if type(outer_fold) is not int or outer_fold not in (0, 1, 3, 4):
            raise ValueError("only OOF development heads0/1/3/4; fold2 is sealed")
        directory = Path(directory).resolve()
        paths = [
            directory / name
            for name in (
                "weights.safetensors",
                "model_config.json",
                "training_receipt.json",
            )
        ]
        if any(not p.resolve().is_relative_to(directory) for p in paths):
            raise ValueError("head files must remain inside the artifact directory")
        weight_bytes = _read_pinned(paths[0], weights_sha256)
        cfg = _json(_read_pinned(paths[1], config_sha256))
        receipt = _json(_read_pinned(paths[2], receipt_sha256))
        if any(
            type(cfg.get(k)) is not int or cfg[k] != v
            for k, v in {"features": 13, "hidden": 64, "depth": 2}.items()
        ):
            raise ValueError(
                "expected frozen graph architecture features13/hidden64/depth2"
            )
        allowed = [f for f in (0, 1, 3, 4) if f != outer_fold]
        excluded = sorted([outer_fold, 2])
        if (
            type(cfg.get("outer")) is not int
            or cfg["outer"] != outer_fold
            or not _folds_equal(
                cfg.get("actual_supervised_folds", cfg.get("actual_train_folds")),
                allowed,
            )
            or not _folds_equal(cfg.get("excluded_folds"), excluded)
            or not _folds_equal(receipt.get("actual_train_folds"), allowed)
            or not _folds_equal(receipt.get("excluded_folds"), excluded)
            or receipt.get("weights_sha256") != weights_sha256
            or receipt.get("config_sha256") != config_sha256
        ):
            raise ValueError("head config/receipt has inconsistent OOF provenance")
        nested = receipt.get("nested_supervision", [])
        if (
            not isinstance(nested, list)
            or any(
                not isinstance(v, dict) or type(v.get("view")) is not int
                for v in nested
            )
            or sorted(v["view"] for v in nested) != allowed
        ):
            raise ValueError(
                "nested supervision views must match actual training folds"
            )
        for entry in nested:
            view = entry["view"]
            parents = entry.get("provenance", [])
            if len(parents) != 2:
                raise ValueError(
                    "each nested view needs two encoder provenance records"
                )
            expected = [f for f in allowed if f != view]
            for parent in parents:
                record, export = parent.get("receipt", {}), parent.get("export", {})
                if (
                    not _folds_equal(record.get("actual_train_folds"), expected)
                    or not _folds_equal(export.get("actual_train_folds"), expected)
                    or not _folds_equal(
                        record.get("excluded_folds"), sorted([outer_fold, view, 2])
                    )
                    or not _digest(record.get("weights_sha256"))
                    or not _digest(record.get("config_sha256"))
                    or record.get("weights_sha256") != export.get("weights_sha256")
                    or record.get("config_sha256") != export.get("config_sha256")
                ):
                    raise ValueError(
                        "nested encoder receipt/export violates OOF provenance"
                    )
        model = CandidateGraphRanker(features=13, hidden=64, depth=2).cpu().eval()
        # Deserialize the exact pinned bytes (no pickle, no second path read).
        model.load_state_dict(load(weight_bytes), strict=True)
        self.model = model
        self.identity = {
            "weights_sha256": weights_sha256,
            "config_sha256": config_sha256,
            "receipt_sha256": receipt_sha256,
            "outer_fold": outer_fold,
            "scope": "OOF development adapter; no release acceptance",
        }

    def score_lists(
        self, nodes: np.ndarray, edges: np.ndarray, base: np.ndarray
    ) -> np.ndarray:
        if (
            nodes.ndim != 3
            or nodes.shape[1:] != (10, 13)
            or edges.shape != (len(nodes), 10, 10, 2)
            or base.shape != (len(nodes), 10)
            or len(nodes) == 0
        ):
            raise ValueError("expected nodes[N,10,13], edges[N,10,10,2], base[N,10]")
        if any(
            v.dtype != np.float32 or not np.isfinite(v).all()
            for v in (nodes, edges, base)
        ):
            raise ValueError("head inputs must be finite float32")
        with torch.inference_mode():
            return (
                self.model.score_lists(
                    *(torch.from_numpy(v) for v in (nodes, edges, base))
                )
                .cpu()
                .numpy()
            )
