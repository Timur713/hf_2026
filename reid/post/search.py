"""Deterministic cosine search and candidate rejection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    ranking: list[list[str]]
    top_ids: list[str]
    top_scores: np.ndarray


def cosine_search(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    gallery_ids: Sequence[str],
    top_k: int = 10,
) -> SearchResult:
    """Return stable top-k rankings; ties keep gallery CSV order."""

    q = np.asarray(query_emb, dtype=np.float32)
    g = np.asarray(gallery_emb, dtype=np.float32)
    if q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1]:
        raise ValueError("query/gallery embeddings must be 2D with equal dimensions")
    if g.shape[0] != len(gallery_ids):
        raise ValueError("gallery_ids length does not match gallery embeddings")
    if len(gallery_ids) < top_k:
        raise ValueError(f"gallery has {len(gallery_ids)} rows, need at least top_k={top_k}")
    if not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError("embeddings contain NaN or infinity")

    scores = q @ g.T
    order = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    ranking = [[str(gallery_ids[j]) for j in row] for row in order]
    top_scores = scores[np.arange(q.shape[0]), order[:, 0]].astype(np.float32, copy=False)
    top_ids = [row[0] for row in ranking]
    return SearchResult(ranking=ranking, top_ids=top_ids, top_scores=top_scores)

