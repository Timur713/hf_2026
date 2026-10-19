"""Deterministic top-k reranking, independent of model and image I/O (D-039).

The verifier only sees one query and its own embedding top-k gallery candidates.
It never consumes another query, camera_id, batch statistics, or test labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from reid.post.search import SearchResult


Item = tuple[str | Path, tuple[float, float, float, float]]


class PairScorer(Protocol):
    def score_pairs(self, query: Item, candidates: Sequence[Item]) -> np.ndarray: ...


def __getattr__(name: str):
    # Compatibility for published callers. Production imports the adapter from
    # infer directly; importing the ranking functions alone never loads torch.
    if name == "PairVerifier":
        from reid.infer.verifier import PairVerifier

        return PairVerifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def rerank_topk(
    query_items: Sequence[Item],
    gallery_items: Sequence[Item],
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    gallery_ids: Sequence[str],
    base: SearchResult,
    verifier: PairScorer,
    weight: float = 0.2,
    *,
    score_mode: str = "raw",
    ranking_scores: np.ndarray | None = None,
) -> SearchResult:
    """Rerank each query independently; default preserves the model-v1 formula.

    ``raw`` uses ``cosine + weight*sigmoid(logit)``. The opt-in D-053 mode
    ``standardized`` uses ``z(base_score) + weight*z(logit)`` with population
    standard deviation plus 1e-9, only over this query's candidate list. Its
    base score comes from ``ranking_scores[N_query, N_gallery]`` when supplied,
    otherwise cosine. ``ranking_scores`` does not change the legacy raw mode.

    ``top_scores`` deliberately stays on the cosine scale. D-037 calibrated the
    candidate rejection threshold on cosine, not on the mixed ranking score.
    Callers using a different rejection policy must recalibrate it afterwards.
    At zero weight the base candidate order is preserved without calling the
    verifier, including when the base was produced by a non-cosine search.
    """

    if score_mode not in ("raw", "standardized"):
        raise ValueError("score_mode must be 'raw' or 'standardized'")
    q, g = np.asarray(query_emb), np.asarray(gallery_emb)
    if q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1]:
        raise ValueError("query/gallery embeddings must be 2D with equal dimensions")
    for matrix in (q, g):
        if not np.issubdtype(matrix.dtype, np.number) or not np.isrealobj(matrix):
            raise ValueError("embeddings must contain real numbers")
        if not np.isfinite(matrix).all():
            raise ValueError("embeddings contain NaN or infinity")
    if len(query_items) != len(base.ranking) or len(query_items) != len(q):
        raise ValueError("query inputs and base ranking lengths differ")
    if len(gallery_items) != len(gallery_ids) or len(gallery_items) != len(g):
        raise ValueError("gallery inputs, ids, and embeddings lengths differ")
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("rerank weight must be finite and non-negative")
    if ranking_scores is not None:
        ranking_scores = np.asarray(ranking_scores)
        if ranking_scores.shape != (len(q), len(g)):
            raise ValueError("ranking_scores must have shape (N_query, N_gallery)")
        if (
            not np.issubdtype(ranking_scores.dtype, np.number)
            or not np.isrealobj(ranking_scores)
            or not np.isfinite(ranking_scores).all()
        ):
            raise ValueError("ranking_scores must contain finite real numbers")
    gallery_index = {str(image_id): index for index, image_id in enumerate(gallery_ids)}
    if len(gallery_index) != len(gallery_ids):
        raise ValueError("gallery ids must be unique")

    # Validate every row before any image decoding or verifier computation.
    selected_indices = []
    for candidate_ids in base.ranking:
        if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(
                "base ranking must contain nonempty lists of unique gallery ids"
            )
        try:
            selected_indices.append(
                np.asarray(
                    [gallery_index[image_id] for image_id in candidate_ids],
                    dtype=np.intp,
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"base ranking contains unknown gallery id: {exc.args[0]}"
            ) from exc

    ranking: list[list[str]] = []
    top_ids: list[str] = []
    top_scores: list[float] = []
    for query_index, (query_item, candidate_ids, candidate_indices) in enumerate(
        zip(query_items, base.ranking, selected_indices)
    ):
        cosine = q[query_index] @ g[candidate_indices].T
        if weight == 0:
            order = np.arange(len(candidate_ids))
        else:
            candidate_items = [gallery_items[index] for index in candidate_indices]
            logits = np.asarray(
                verifier.score_pairs(query_item, candidate_items), dtype=np.float32
            )
            if logits.shape != (len(candidate_ids),) or not np.isfinite(logits).all():
                raise ValueError(
                    "verifier scores must be one finite logit per candidate"
                )

            if score_mode == "standardized":
                base_scores = (
                    cosine
                    if ranking_scores is None
                    else ranking_scores[query_index, candidate_indices]
                )

                # Float64 avoids overflow/underflow of variance for finite
                # float32 scores. No statistic contains another query.
                def z(values: np.ndarray) -> np.ndarray:
                    values = values.astype(np.float64, copy=False)
                    return (values - values.mean()) / (values.std(ddof=0) + 1e-9)

                mixed = z(base_scores) + weight * z(logits)
            else:
                probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
                mixed = cosine + weight * probability
            order = np.argsort(-mixed, kind="stable")
        reranked = [candidate_ids[index] for index in order]
        ranking.append(reranked)
        top_ids.append(reranked[0])
        top_scores.append(float(cosine[order[0]]))

    return SearchResult(
        ranking=ranking,
        top_ids=top_ids,
        top_scores=np.asarray(top_scores, dtype=np.float32),
    )
