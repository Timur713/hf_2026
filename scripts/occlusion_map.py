"""Карта чувствительности эмбеддинга к закрытию участков кадра (задача A4.2, решение D-027).

По очереди закрываем клетки сетки и смотрим, насколько меняется эмбеддинг. Получается карта
«на что модель опирается». Нужна как доказательство для защиты: номерная зона не должна быть
среди влиятельных участков (§2.3 ТЗ, контрольный прогон жюри с залитыми номерами).

Детектор номеров для этого не нужен — мы не ищем номер, а смотрим, куда смотрит модель.

    PYTHONPATH=. python scripts/occlusion_map.py --model artifacts/models/<name> --n 40 --grid 6
"""
import argparse
import os

import numpy as np
import torch

from reid.data.crop_cache import cached_box, load_manifest, read_crop
from reid.data.preprocess import to_model_input
from reid.model import load, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", default="artifacts/crops")
    ap.add_argument("--n", type=int, default=40, help="сколько кадров усреднять")
    ap.add_argument("--grid", type=int, default=6)
    ap.add_argument("--out", default="docs/errors/occlusion.txt")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, model = load_config(args.model), load(args.model, device)
    rows = load_manifest(args.cache, ("train",)).sample(args.n, random_state=0)

    g = args.grid
    heat = np.zeros((g, g), dtype=np.float64)
    with torch.inference_mode():
        for row in rows.itertuples():
            x0, y0, x1, y1 = cached_box(row, cfg.get("bbox_pad", 0.0))
            crop = read_crop(args.cache, row.image_id)[:, y0:y1, x0:x1]
            base_input = to_model_input(crop, cfg["input_size"], cfg["mean"], cfg["std"])
            base = model(base_input.unsqueeze(0).to(device))
            H, W = base_input.shape[-2:]
            batch = []
            for i in range(g):
                for j in range(g):
                    x = base_input.clone()
                    x[:, i * H // g:(i + 1) * H // g, j * W // g:(j + 1) * W // g] = 0.0
                    batch.append(x)
            emb = model(torch.stack(batch).to(device))
            # чем сильнее падает косинус с исходным эмбеддингом, тем важнее участок
            drop = (1.0 - (emb @ base.T).squeeze(1)).float().cpu().numpy()
            heat += drop.reshape(g, g)
    heat /= len(rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lines = ["Карта чувствительности: насколько падает косинус с исходным эмбеддингом",
             f"при закрытии клетки сетки {g}×{g}. Модель: {os.path.basename(args.model)}, "
             f"усреднено по {len(rows)} кадрам.", ""]
    for i in range(g):
        lines.append("  ".join(f"{v:6.4f}" for v in heat[i]))
    hottest = np.unravel_index(np.argmax(heat), heat.shape)
    lines += ["", f"Самая влиятельная клетка: строка {hottest[0] + 1} из {g}, столбец {hottest[1] + 1} из {g}.",
              f"Нижняя треть кадра (где обычно номер) даёт {heat[int(g * 2 / 3):].mean() / heat.mean():.2f} "
              f"от средней важности по кадру."]
    text = "\n".join(lines)
    print(text)
    with open(args.out, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
