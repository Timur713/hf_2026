"""Strict, offline fusion artifact metadata (D-052).

Similarity scales are fixed artifact constants, never fitted from query inputs.
Every component remains a normal C3 model artifact below the fusion directory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate fusion manifest key: {key}")
        result[key] = value
    return result


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def load_fusion_config(model_dir: str | Path) -> dict:
    """Validate schema v1, its fixed scales, and contained C3 artifact paths."""
    root = Path(model_dir).resolve()
    manifest = root / "fusion_config.json"
    if not manifest.resolve().is_relative_to(root):
        raise ValueError("fusion_config.json must remain inside the fusion root")
    if (root / "model_config.yaml").exists():
        raise ValueError("ambiguous artifact: both fusion_config.json and model_config.yaml")
    try:
        cfg = json.loads(manifest.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid fusion_config.json: {exc}") from exc
    keys = {"schema_version", "name", "components", "candidate_policy",
            "candidate_threshold", "rerank_mode"}
    if not isinstance(cfg, dict) or not keys <= set(cfg) or set(cfg) - keys - {"rerank_weight"}:
        raise ValueError(f"fusion manifest requires exactly {sorted(keys)} plus optional rerank_weight")
    if type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1:
        raise ValueError("unsupported fusion schema_version; expected 1")
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("fusion name must be a non-empty string")
    if cfg["candidate_policy"] != "gallery_z":
        raise ValueError("fusion candidate_policy must be gallery_z")
    if cfg["rerank_mode"] != "standardized":
        raise ValueError("fusion rerank_mode must be standardized")
    cfg["candidate_threshold"] = _positive_number(cfg["candidate_threshold"], "candidate_threshold")
    if "rerank_weight" in cfg:
        from reid.post.policy import rerank_weight_for

        cfg["rerank_weight"] = rerank_weight_for(cfg)
    components = cfg["components"]
    if not isinstance(components, list) or len(components) < 2:
        raise ValueError("fusion requires at least two components")
    seen = set()
    for component in components:
        if not isinstance(component, dict) or set(component) != {"path", "similarity_std"}:
            raise ValueError("fusion component must contain exactly path and similarity_std")
        value = component["path"]
        if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
            raise ValueError("fusion component path must be a non-empty relative directory")
        path = (root / value).resolve()
        if path == root or not path.is_relative_to(root):
            raise ValueError("fusion component path must remain inside the fusion root")
        if path in seen:
            raise ValueError("duplicate fusion component path")
        seen.add(path)
        if (path / "fusion_config.json").exists():
            raise ValueError("nested fusion components are not supported")
        for name in ("model_config.yaml", "weights.safetensors"):
            asset = path / name
            if not asset.resolve().is_relative_to(root):
                raise ValueError(f"component {name} must remain inside the fusion root")
            if not asset.is_file():
                raise ValueError(f"missing C3 component file: {asset}")
        component["similarity_std"] = _positive_number(component["similarity_std"], "similarity_std")
    return cfg
