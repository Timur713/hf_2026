"""Frozen candidate policy for the final recipe (A-028, A-036, D-044)."""

from __future__ import annotations

import math
from typing import Mapping


# D-044 uses 0.583 in the delivered submission. The OOF grid selected 0.58333…
# within the broad 0.550–0.633 plateau. Do not retune this value on test data.
DEFAULT_CANDIDATE_THRESHOLD = 0.583
LEGACY_MODEL_V1_NAME = "vitb14_f0_clsavg_in280_ep100"
LEGACY_MODEL_V1_THRESHOLD = 0.600
FUSION_GALLERY_THRESHOLD = 6.0  # D-057: provisional two-fold calibration, not cosine.


def rerank_weight_for(cfg: Mapping, override: float | None = None) -> float:
    """Keep the selected verifier weight with its artifact, not an implicit CLI default."""
    value = override if override is not None else cfg.get("rerank_weight", 0.2)
    if isinstance(value, bool):
        raise ValueError("rerank weight must be finite and non-negative")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("rerank weight must be finite and non-negative")
    return result


def candidate_threshold_for(cfg: Mapping, override: float | None = None) -> float:
    """CLI override > artifact config > gallery-z / legacy / single-model default.

    model-v1 predates plate-band training and has its own five-fold calibration.
    Preserve its published threshold without modifying its checksum-pinned YAML.
    """

    policy = cfg.get("candidate_policy", "cosine")
    if policy not in ("cosine", "gallery_z"):
        raise ValueError(f"unknown candidate policy: {policy!r}")
    if override is not None:
        value = override
    elif "candidate_threshold" in cfg:
        value = cfg["candidate_threshold"]
    elif policy == "gallery_z":
        value = FUSION_GALLERY_THRESHOLD
    elif cfg.get("name") == LEGACY_MODEL_V1_NAME:
        value = LEGACY_MODEL_V1_THRESHOLD
    else:
        value = DEFAULT_CANDIDATE_THRESHOLD
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("candidate threshold must be finite")
    return result
