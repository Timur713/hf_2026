"""Gallery-only hub correction (D-055) and calibrated confidence (D-057)."""

from __future__ import annotations

from numbers import Integral
from typing import Sequence

import numpy as np

from reid.post.search import SearchResult


class GalleryPolicy:
    """Search already L2-normalized C4 embeddings using a fixed gallery.

    Ranking uses a per-query z score minus the standardized gallery-neighbour
    hub score. Confidence is a *different* quantity: the selected candidate's
    raw cosine standardized with off-diagonal gallery--gallery statistics.
    No query contributes to any other query's scores or calibration.

    The gallery matrix is materialized once (O(N_gallery**2) storage); this is
    the exact-search policy, not the separate large-scale ANN demonstration.
    Query search itself retains only one score row at a time.
    """

    def __init__(
        self,
        gallery_emb: np.ndarray,
        gallery_ids: Sequence[str],
        hub_k: int = 3,
        hub_weight: float = 0.5,
    ) -> None:
        g = np.array(gallery_emb, dtype=np.float32, copy=True)
        if g.ndim != 2 or g.shape[1] == 0:
            raise ValueError("gallery embeddings must be 2D with a nonempty feature dimension")
        if not np.isfinite(g).all():
            raise ValueError("gallery embeddings contain NaN or infinity")
        if isinstance(gallery_ids, (str, bytes)):
            raise ValueError("gallery_ids must be a sequence of nonempty strings")
        ids = tuple(gallery_ids)
        if len(ids) != g.shape[0]:
            raise ValueError("gallery_ids length does not match gallery embeddings")
        if any(not isinstance(value, str) or not value.strip() for value in ids):
            raise ValueError("gallery_ids must be nonempty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("gallery_ids must be unique")
        if (
            isinstance(hub_k, bool)
            or not isinstance(hub_k, Integral)
            or not 1 <= hub_k < len(ids)
        ):
            raise ValueError("hub_k must be an integer from 1 to gallery size minus 1")
        if not np.isscalar(hub_weight) or isinstance(hub_weight, (bool, str, bytes)):
            raise ValueError("hub_weight must be a finite nonnegative number")
        weight = float(hub_weight)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("hub_weight must be a finite nonnegative number")

        gg = g @ g.T
        off = gg[~np.eye(len(ids), dtype=bool)]
        # Match calibrate_fusion_threshold.py: population std, no diagonal,
        # and 1e-12 (NOT the ranking z-score's 1e-9) in the denominator.
        self.gallery_mean = off.mean()
        self.gallery_std = off.std(ddof=0)
        if (
            not np.isfinite(self.gallery_std)
            or self.gallery_std <= 1e-12
            or off.min() == off.max()
        ):
            raise ValueError("gallery off-diagonal confidence standard deviation is degenerate")
        np.fill_diagonal(gg, -np.inf)
        hub = np.sort(gg, axis=1)[:, -int(hub_k):].mean(axis=1)
        self.hub = (hub - hub.mean()) / (hub.std(ddof=0) + 1e-9)
        g.setflags(write=False)
        self.hub.setflags(write=False)
        self.gallery_emb = g
        self.gallery_ids = ids
        self.hub_k = int(hub_k)
        self.hub_weight = weight
        self._id_to_index = {value: i for i, value in enumerate(ids)}

    def _queries(self, query_emb: np.ndarray) -> np.ndarray:
        q = np.asarray(query_emb, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != self.gallery_emb.shape[1]:
            raise ValueError("query/gallery embeddings must be 2D with equal dimensions")
        if not np.isfinite(q).all():
            raise ValueError("query embeddings contain NaN or infinity")
        return q

    def _raw_row(self, query: np.ndarray) -> np.ndarray:
        # Always execute a single-query operation. In addition to semantic
        # independence this avoids BLAS selecting a different reduction kernel
        # depending on how many unrelated queries happened to arrive together.
        return query[None, :] @ self.gallery_emb.T

    def _rank_row(self, raw: np.ndarray) -> np.ndarray:
        z = (raw - raw.mean(axis=1, keepdims=True)) / (
            raw.std(axis=1, keepdims=True, ddof=0) + 1e-9
        )
        return z - self.hub_weight * self.hub[None, :]

    def _confidence(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self.gallery_mean) / (self.gallery_std + 1e-12)

    def ranking_scores(self, query_emb: np.ndarray) -> np.ndarray:
        """Return [N_query, N_gallery] corrected scores in gallery CSV order."""
        q = self._queries(query_emb)
        scores = np.empty((len(q), len(self.gallery_ids)), dtype=np.float32)
        for i, query in enumerate(q):
            scores[i] = self._rank_row(self._raw_row(query))[0]
        return scores

    def search(self, query_emb: np.ndarray, top_k: int = 10) -> SearchResult:
        """Return stable rankings and D-057 confidence for the selected top-1.

        ``top_scores`` are confidence values, not the hub-corrected ranking
        scores. Equal ranking scores retain the gallery CSV's original order.
        """
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, Integral)
            or not 1 <= top_k <= len(self.gallery_ids)
        ):
            raise ValueError("top_k must be an integer from 1 to gallery size")
        q = self._queries(query_emb)
        ranking: list[list[str]] = []
        confidence = np.empty(len(q), dtype=np.float32)
        for i, query in enumerate(q):
            raw = self._raw_row(query)
            order = np.argsort(-self._rank_row(raw)[0], kind="stable")[:top_k]
            ranking.append([self.gallery_ids[j] for j in order])
            confidence[i] = self._confidence(raw[:, order[0]])[0]
        return SearchResult(
            ranking=ranking,
            top_ids=[row[0] for row in ranking],
            top_scores=confidence,
        )

    def confidence_for(
        self, query_emb: np.ndarray, selected_ids: Sequence[str]
    ) -> np.ndarray:
        """Compute D-057 confidence for one selected gallery ID per query.

        This also supports a top-1 changed by a subsequent per-query verifier.
        The verifier's ranking score is never mistaken for confidence.
        """
        q = self._queries(query_emb)
        if isinstance(selected_ids, (str, bytes)):
            raise ValueError("selected_ids must contain one gallery ID per query")
        ids = tuple(selected_ids)
        if len(ids) != len(q):
            raise ValueError("selected_ids must contain one gallery ID per query")
        if any(not isinstance(value, str) or value not in self._id_to_index for value in ids):
            raise ValueError("selected_ids contains an unknown gallery ID")
        confidence = np.empty(len(q), dtype=np.float32)
        for i, (query, value) in enumerate(zip(q, ids)):
            raw = self._raw_row(query)[:, self._id_to_index[value]]
            confidence[i] = self._confidence(raw)[0]
        return confidence
