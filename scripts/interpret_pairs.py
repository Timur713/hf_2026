"""Картинка для защиты: по каким деталям сошлись две машины (§12 ТЗ, интерпретируемость).

Берёт с фолда пары трёх видов — верное совпадение, ошибочный первый кандидат и истинный ответ,
который модель поставила ниже, — и рисует карты соответствия патчей.

    PYTHONPATH=. python scripts/interpret_pairs.py --model artifacts/models/final_all_data
"""
import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from reid.data.crop_cache import cached_box, load_manifest, read_crop
from reid.data.preprocess import to_model_input
from reid.eval.eval_local import FoldTruth, ranking_from_embeddings
from reid.model import load, load_config
from reid.model.interpret import correspondence, overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--emb", default="artifacts/emb_final/final_oof_f0_cls_avg/fold0.npz")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cache", default="artifacts/crops")
    ap.add_argument("--out", default="docs/errors/interpretability.jpg")
    ap.add_argument("--rows", type=int, default=4)
    args = ap.parse_args()

    cfg, model = load_config(args.model), load(args.model, "cpu")
    man = load_manifest(args.cache, ("train",)).set_index("image_id")
    truth = FoldTruth.load(args.fold)
    d = np.load(args.emb)
    q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()
    ranking = ranking_from_embeddings(d["query_emb"], d["gallery_emb"], g_ids, q_ids)
    n_pos = truth.n_pos()

    def crop_of(image_id):
        row = SimpleNamespace(image_id=image_id, **man.loc[image_id].to_dict())
        x0, y0, x1, y1 = cached_box(row, cfg.get("bbox_pad", 0.0))
        return read_crop(args.cache, image_id)[:, y0:y1, x0:x1]

    cases = []
    for q in q_ids:
        if n_pos[q] == 0:
            continue
        kept = [g for g in ranking[q] if not truth.is_junk(q, g)]
        if not kept:
            continue
        if truth.is_pos(q, kept[0]):
            cases.append((q, kept[0], "верное совпадение"))
        else:
            pos = next((g for g in kept if truth.is_pos(q, g)), None)
            if pos:
                cases.append((q, kept[0], "ошибка: чужая машина"))
                cases.append((q, pos, "она же, верный ответ ниже"))
        if len(cases) >= args.rows:
            break

    # Посторонний кадр для поправки: берём машину другой личности
    ref_id = next(g for g in g_ids if truth.vid[g] != truth.vid[cases[0][0]])
    C = 220
    canvas = Image.new("RGB", (C * 4 + 30, C * len(cases) + 20), "white")
    for r, (q, g, label) in enumerate(cases):
        ca, cb = crop_of(q), crop_of(g)
        xa = to_model_input(ca, cfg["input_size"], cfg["mean"], cfg["std"])
        xb = to_model_input(cb, cfg["input_size"], cfg["mean"], cfg["std"])
        ref = to_model_input(crop_of(ref_id), cfg["input_size"], cfg["mean"], cfg["std"])
        ha, hb, score = correspondence(model, xa, xb, ref)
        tiles = [np.asarray(Image.fromarray(ca.permute(1, 2, 0).numpy()).resize((C, C))) / 255.0,
                 overlay(torch.from_numpy(np.asarray(Image.fromarray(
                     ca.permute(1, 2, 0).numpy()).resize((C, C)))).permute(2, 0, 1), ha),
                 np.asarray(Image.fromarray(cb.permute(1, 2, 0).numpy()).resize((C, C))) / 255.0,
                 overlay(torch.from_numpy(np.asarray(Image.fromarray(
                     cb.permute(1, 2, 0).numpy()).resize((C, C)))).permute(2, 0, 1), hb)]
        for c, tile in enumerate(tiles):
            canvas.paste(Image.fromarray((tile * 255).astype(np.uint8)), (c * C + 10, r * C + 10))
        print(f"строка {r + 1}: {label}, сила соответствия {score:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    canvas.save(args.out, quality=88)
    print(f"сохранено {args.out}: запрос, его карта, кандидат, его карта")


if __name__ == "__main__":
    main()
