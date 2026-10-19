"""Camera-free features using one query and its fixed gallery only."""

import numpy as np
from reid.post.gallery_policy import GalleryPolicy
from scripts.replay_fusion import fuse_components


def z(x):
    x = x.astype(np.float64)
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-9)


def build_features(arrays, gallery_ids, top_k=10):
    if type(top_k) is not int or not 2 <= top_k <= len(gallery_ids):
        raise ValueError("top_k must be2..gallery size")
    q, g, scales = fuse_components(arrays)
    policy = GalleryPolicy(g, gallery_ids)
    full = policy.ranking_scores(q)
    indices = np.argsort(-full, axis=1, kind="stable")[:, :top_k]
    baseline = z(np.take_along_axis(full, indices, axis=1)).astype(np.float32)
    row = np.arange(len(q))[:, None]
    node_parts = []
    edge_parts = []
    for qc, gc in arrays:
        # Fixed one-query dot-product shape; no query-batch statistics.
        raw = np.concatenate([qc[i : i + 1] @ gc.T for i in range(len(qc))])
        selected = raw[row, indices]
        ranks = np.argsort(
            np.argsort(-raw, axis=1, kind="stable"), axis=1, kind="stable"
        )
        gg = gc @ gc.T
        neighbours = gg.copy()
        np.fill_diagonal(neighbours, -np.inf)
        hub = np.sort(neighbours, axis=1)[:, -3:].mean(1)
        hub = (hub - hub.mean()) / (hub.std() + 1e-9)
        node_parts.extend(
            [
                selected,
                z(raw)[row, indices] / 5,
                z(selected),
                np.log1p(ranks[row, indices]) / np.log1p(len(gc)),
                hub[indices] / 5,
                raw.max(1, keepdims=True) - selected,
            ]
        )
        edge_parts.append(gg[indices[:, :, None], indices[:, None, :]])
    nodes = np.stack([*node_parts, baseline], axis=-1).astype(np.float32)
    edges = np.stack(edge_parts, axis=-1).astype(np.float32)
    assert nodes.shape == (len(q), top_k, 13) and edges.shape == (
        len(q),
        top_k,
        top_k,
        2,
    )
    assert np.isfinite(nodes).all() and np.isfinite(edges).all()
    return dict(
        nodes=nodes,
        edges=edges,
        base=baseline,
        indices=indices,
        candidates=np.array(gallery_ids)[indices],
        q=q,
        g=g,
        policy=policy,
        scales=scales,
    )
