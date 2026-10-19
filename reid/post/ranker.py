"""Opt-in D-082 ranking preparation; no image I/O, model loading or defaults.

Unlike Extractor's fused-only output, this API requires the original two
component embeddings, in DINO/CLIP order. Recovering them from a normalized
concatenation is deliberately unsupported. Scales must be fixed artifact
constants; they are never fitted from queries here.
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence

import numpy as np

from reid.post.pipeline import PreparedGallery
from reid.post.search import SearchResult
from reid.post.verifier import Item, PairScorer


class ListScorer(Protocol):
    def score_lists(
        self, nodes: np.ndarray, edges: np.ndarray, base: np.ndarray
    ) -> np.ndarray: ...


def _z(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-9
    )


def _component(value: np.ndarray, rows: int | None = None) -> np.ndarray:
    value = np.asarray(value)
    if (
        value.dtype != np.float32
        or value.ndim != 2
        or value.shape[1] == 0
        or not np.isfinite(value).all()
    ):
        raise ValueError("component embeddings must be finite float32 matrices")
    if rows is not None and len(value) != rows:
        raise ValueError("component row counts differ")
    if not np.allclose(np.linalg.norm(value, axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("original component embeddings must be L2-normalized")
    return value


class PreparedRankerGallery:
    """Static gallery plus opt-in ranker, preserving the exact hub top-10 set.

    ``ranker=None`` or ``residual_weight=0`` delegates to the unchanged
    PreparedGallery verifier path. The new path retains the frozen D-082
    .5 residual + .15 exact-window verifier recipe. Threshold decisions still
    belong to the normal output writer and need separate calibration.
    """

    def __init__(
        self,
        gallery_components: Sequence[np.ndarray],
        gallery_ids: Sequence[str],
        gallery_items: Sequence[Item],
        similarity_scales: Sequence[float],
        *,
        ranker: ListScorer | None = None,
        residual_weight: float = 0.5,
    ):
        if len(gallery_components) != 2 or len(similarity_scales) != 2:
            raise ValueError("D-082 requires exactly DINO and CLIP components/scales")
        if isinstance(residual_weight, bool) or residual_weight not in (0, 0.5):
            raise ValueError(
                "only frozen D-082 residual .5 or explicit parity control 0 allowed"
            )
        if any(
            isinstance(s, (bool, str)) or not np.isfinite(s) or s <= 0
            for s in similarity_scales
        ):
            raise ValueError(
                "similarity scales must be positive finite artifact constants"
            )
        scales = tuple(float(s) for s in similarity_scales)
        reference = math.sqrt(min(scales))
        self.coefficients = tuple(reference / math.sqrt(s) for s in scales)
        if min(self.coefficients) < np.finfo(np.float32).tiny:
            raise ValueError("similarity scale range is too wide for float32")
        self.components = tuple(
            _component(g, len(gallery_ids)).copy() for g in gallery_components
        )
        if len(gallery_ids) < 10:
            raise ValueError("D-082 requires at least 10 gallery entries")
        self.gallery = PreparedGallery(
            self._fuse(self.components),
            gallery_ids,
            gallery_items,
            {
                "candidate_policy": "gallery_z",
                "rerank_mode": "standardized",
                "rerank_weight": 0.15,
            },
        )
        self.ranker = ranker
        self.residual_weight = float(residual_weight)
        self.component_gg = []
        self.component_hub = []
        for component in self.components:
            gg = component @ component.T
            neighbours = gg.copy()
            np.fill_diagonal(neighbours, -np.inf)
            hub = np.sort(neighbours, axis=1)[:, -3:].mean(1)
            hub = (hub - hub.mean()) / (hub.std() + 1e-9)
            component.setflags(write=False)
            gg.setflags(write=False)
            hub.setflags(write=False)
            self.component_gg.append(gg)
            self.component_hub.append(hub)

    def _fuse(self, components: Sequence[np.ndarray]) -> np.ndarray:
        # Same arithmetic and dtype as A's replay fuse_components; no query
        # statistics, refitting, reconstructed blocks or extra normalization.
        fused = np.concatenate(
            [c * coefficient for c, coefficient in zip(components, self.coefficients)],
            axis=1,
        )
        return fused / np.linalg.norm(fused, axis=1, keepdims=True)

    def _query_components(self, query_components: Sequence[np.ndarray]):
        if len(query_components) != 2:
            raise ValueError("two original query components are required")
        components = tuple(_component(q) for q in query_components)
        n_query = len(components[0])
        if n_query == 0:
            raise ValueError("at least one query is required")
        if any(
            len(q) != n_query or q.shape[1] != g.shape[1]
            for q, g in zip(components, self.components)
        ):
            raise ValueError("query/component shapes differ from static gallery")
        return components

    def fuse_queries(self, query_components: Sequence[np.ndarray]) -> np.ndarray:
        """C6 embeddings with the same fixed-scale arithmetic as ranking."""
        return self._fuse(self._query_components(query_components))

    def features(self, query_components: Sequence[np.ndarray]) -> dict:
        components = self._query_components(query_components)
        n_query = len(components[0])
        q = self._fuse(components)
        full = self.gallery.policy.ranking_scores(q)
        indices = np.argsort(-full, axis=1, kind="stable")[:, :10]
        raw_hub = np.take_along_axis(full, indices, axis=1)
        base = _z(raw_hub).astype(np.float32)
        row = np.arange(n_query)[:, None]
        parts, edges = [], []
        for qc, gc, gg, hub in zip(
            components, self.components, self.component_gg, self.component_hub
        ):
            raw = np.concatenate([qc[i : i + 1] @ gc.T for i in range(n_query)])
            selected = raw[row, indices]
            ranks = np.argsort(
                np.argsort(-raw, axis=1, kind="stable"), axis=1, kind="stable"
            )
            parts.extend(
                [
                    selected,
                    _z(raw)[row, indices] / 5,
                    _z(selected),
                    np.log1p(ranks[row, indices]) / np.log1p(len(gc)),
                    hub[indices] / 5,
                    raw.max(1, keepdims=True) - selected,
                ]
            )
            edges.append(gg[indices[:, :, None], indices[:, None, :]])
        return {
            "nodes": np.stack([*parts, base], axis=-1).astype(np.float32),
            "edges": np.stack(edges, axis=-1).astype(np.float32),
            "base": base,
            "indices": indices,
            "hub": raw_hub,
            "q": q,
            "candidates": np.asarray(self.gallery.ids)[indices],
        }

    def rank(
        self,
        query_components: Sequence[np.ndarray],
        query_items: Sequence[Item],
        verifier: PairScorer,
    ) -> SearchResult:
        feat = self.features(query_components)
        if len(query_items) != len(feat["q"]):
            raise ValueError("one image reference per query is required")
        if verifier is None:
            raise ValueError("D-082 requires the exact-window verifier")
        if self.ranker is None or self.residual_weight == 0:
            return self.gallery.rank(feat["q"], query_items, verifier)
        head = np.asarray(
            self.ranker.score_lists(feat["nodes"], feat["edges"], feat["base"])
        )
        if (
            head.shape != feat["base"].shape
            or head.dtype != np.float32
            or not np.isfinite(head).all()
        ):
            raise ValueError("ranker must return finite float32 [N_query,10] scores")
        logits = []
        for item, indices in zip(query_items, feat["indices"]):
            values = np.asarray(
                verifier.score_pairs(item, [self.gallery.items[i] for i in indices]),
                dtype=np.float32,
            )
            if values.shape != (10,) or not np.isfinite(values).all():
                raise ValueError("verifier must return ten finite exact-window logits")
            logits.append(values)
        scores = (
            _z(feat["hub"])
            + 0.15 * _z(np.stack(logits))
            + self.residual_weight * (head - feat["base"])
        )
        order = np.argsort(-scores, axis=1, kind="stable")
        ranking = np.take_along_axis(feat["candidates"], order, axis=1).tolist()
        top = [row[0] for row in ranking]
        return SearchResult(
            ranking, top, self.gallery.policy.confidence_for(feat["q"], top)
        )
