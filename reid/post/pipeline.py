"""Reusable static-gallery ranking state, with no query cache or image I/O."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from reid.post.gallery_policy import GalleryPolicy
from reid.post.policy import rerank_weight_for
from reid.post.search import SearchResult, cosine_search
from reid.post.verifier import Item, PairScorer, rerank_topk


class PreparedGallery:
    """Bind embeddings, IDs and opaque image references once for serving.

    All mutable caller inputs are copied. Only static gallery statistics persist;
    each rank call receives one or more independent queries and stores none.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        ids: Sequence[str],
        items: Sequence[Item],
        cfg: dict,
    ):
        self.ids = tuple(ids)
        self.items = tuple((path, tuple(bbox)) for path, bbox in items)
        self.embeddings = np.array(embeddings, copy=True)
        if len(self.items) != len(self.ids) or len(self.ids) != len(self.embeddings):
            raise ValueError(
                "gallery items, ids and embeddings must have equal lengths"
            )
        self.embeddings.setflags(write=False)
        self.policy_name = cfg.get("candidate_policy", "cosine")
        if self.policy_name not in ("cosine", "gallery_z"):
            raise ValueError(f"unknown candidate policy: {self.policy_name!r}")
        self.default_mode = cfg.get("rerank_mode", "raw")
        self.default_weight = cfg.get("rerank_weight", 0.2)
        self.policy = (
            GalleryPolicy(self.embeddings, self.ids)
            if self.policy_name == "gallery_z"
            else None
        )

    def rank(
        self,
        query_emb: np.ndarray,
        query_items: Sequence[Item],
        verifier: PairScorer | None = None,
        rerank_weight: float | None = None,
        rerank_mode: str | None = None,
    ) -> SearchResult:
        mode = rerank_mode or self.default_mode
        if mode not in ("raw", "standardized"):
            raise ValueError(f"unknown rerank mode: {mode!r}")
        weight = rerank_weight_for(
            {"rerank_weight": self.default_weight}, rerank_weight
        )
        result = (
            self.policy.search(query_emb)
            if self.policy is not None
            else cosine_search(query_emb, self.embeddings, self.ids)
        )
        if verifier is not None:
            result = rerank_topk(
                query_items,
                self.items,
                query_emb,
                self.embeddings,
                self.ids,
                result,
                verifier,
                weight,
                score_mode=mode,
                ranking_scores=(
                    self.policy.ranking_scores(query_emb)
                    if self.policy is not None
                    else None
                ),
            )
            if self.policy is not None:
                result = SearchResult(
                    ranking=result.ranking,
                    top_ids=result.top_ids,
                    top_scores=self.policy.confidence_for(query_emb, result.top_ids),
                )
        return result
