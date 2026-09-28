"""Снимает логиты верификатора на кандидатах фузии один раз и кладёт в npz.

Зачем отдельный шаг. Логиты не зависят от способа смешивания, а считаются на GPU минутами. Сняв
их однажды, любой вопрос про смешивание — вес, выравнивание шкал, гейт по уверенности базы —
превращается в секундный numpy-перебор на ноутбуке. Именно перебор на разных базах и породил
ошибочные выводы A-050 и A-054: числа брались из разных прогонов.

В файле лежит ровно то, что нужно для оценки: порядок кандидатов от фузии, её сходства и логиты.

    PYTHONPATH=. python scripts/analysis/rerank_dump.py --fold 0 \\
        --emb artifacts/emb/pool_cls_top16_f0_cls_top16 artifacts/emb/div_clip_f0_cls_top16 \\
              artifacts/emb/div_siglip_f0_top16 \\
        --verifier artifacts/models/v_cvnet_f0/verifier.pt --cache artifacts/crops1024 \\
        --out artifacts/rerank/v_cvnet_f0_fuse3_f0.npz

Для точного production replay окно надо передать готовым: оно уже содержит top-10
после fusion и hub penalty и не должно пересчитываться старой pre-hub формулой::

    PYTHONPATH=. python scripts/analysis/rerank_dump.py --fold 0 \\
        --window artifacts/rerank/window_f0.npz \\
        --verifier artifacts/models/v_frz_4k/verifier.pt --cache artifacts/crops1024 \\
        --out artifacts/rerank/v_frz_4k_window_f0.npz
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch

from reid.data.crop_cache import cached_box, load_manifest, read_crop
from reid.model.verifier import load_verifier, prep_crop
from scripts.analysis.fusion_headroom import fused_similarity


def load_candidate_window(path: str, fold: int):
    """Load an exact production top-10 without pickle or silent reshaping."""
    with np.load(path, allow_pickle=False) as data:
        required = {"fold", "query_ids", "candidates", "sim"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"окно кандидатов не содержит ключи: {sorted(missing)}")
        stored_fold = data["fold"]
        if stored_fold.shape != () or int(stored_fold.item()) != fold:
            raise ValueError(f"окно кандидатов относится не к fold {fold}")
        query_ids = data["query_ids"]
        candidates = data["candidates"]
        similarities = data["sim"]
        if query_ids.ndim != 1 or candidates.ndim != 2 or similarities.shape != candidates.shape:
            raise ValueError("неверная форма query_ids/candidates/sim в окне")
        if candidates.shape != (len(query_ids), 10):
            raise ValueError("production-окно должно иметь форму (N_query, 10)")
        if similarities.dtype != np.float32 or not np.isfinite(similarities).all():
            raise ValueError("sim в окне должен быть конечным float32")
        q_ids = query_ids.astype(str).tolist()
        cand = candidates.astype(str)
        if len(set(q_ids)) != len(q_ids) or any(len(set(row)) != 10 for row in cand.tolist()):
            raise ValueError("ID query и десять кандидатов каждого query должны быть уникальны")
        emb = data["emb"].astype(str) if "emb" in data.files else np.array([], dtype=str)
        return q_ids, cand, similarities.copy(), emb


def main():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--emb", nargs="+", help="дампы фузии, обученные без этого фолда")
    source.add_argument("--window", help="готовое production top-10 после hub penalty")
    ap.add_argument("--verifier", required=True)
    ap.add_argument("--cache", default="artifacts/crops1024")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--topk", type=int, default=30, help="окно кандидатов, которое снимаем")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.window:
        q_ids, cand, sim, emb = load_candidate_window(args.window, args.fold)
    else:
        S, q_ids, g_ids = fused_similarity(args.emb, args.fold)
        order = np.argsort(-S, axis=1)[:, : args.topk]
        cand = np.array([[g_ids[j] for j in row] for row in order], dtype=str)
        sim = np.take_along_axis(S, order, axis=1).astype(np.float32)
        emb = np.array(args.emb)

    model, cfg = load_verifier(args.verifier, args.device)
    man = load_manifest(args.cache, ("train",)).set_index("image_id")

    def prep(image_id):
        row = SimpleNamespace(image_id=image_id, **man.loc[image_id].to_dict())
        x0, y0, x1, y1 = cached_box(row, 0.0)
        return prep_crop(read_crop(args.cache, image_id)[:, y0:y1, x0:x1], cfg)

    need = sorted(set(q_ids) | {g for row in cand for g in row})
    prepared = {i: prep(i) for i in need}
    print(f"кропов подготовлено: {len(prepared)}, query: {len(q_ids)}, окно: {cand.shape[1]}", flush=True)

    logits = np.zeros(cand.shape, dtype=np.float32)
    with torch.inference_mode():
        for i, q in enumerate(q_ids):
            xq = prepared[q].to(args.device)
            xc = torch.stack([prepared[g] for g in cand[i]]).to(args.device)
            logits[i] = model.score_pairs(xq, xc, batch=args.batch).float().cpu().numpy()
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(q_ids)}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, query_ids=np.array(q_ids), candidates=cand.astype(str),
                        sim=sim.astype(np.float32), logits=logits,
                        emb=emb, verifier=np.array(args.verifier),
                        fold=args.fold)
    print(f"сохранено {args.out}")


if __name__ == "__main__":
    main()
