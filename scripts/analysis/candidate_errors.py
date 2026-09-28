"""Разбор ошибок режима кандидатов для защиты (задача A7.2).

`top1_errors.jpg` показывает ошибки ранжирования. Здесь — ошибки решения «отвечать или отказаться»:
ложные отказы, ложные ответы на query без пары и ответы не той машиной. Считается на фолдах из
`splits/`, тестовые данные не используются (правило 4).

    PYTHONPATH=. python scripts/analysis/candidate_errors.py --emb artifacts/emb_final/final_oof_f{k}_cls_avg
"""
import argparse
import os
from collections import Counter
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from reid.data.crop_cache import cached_box, load_manifest, read_crop
from reid.eval.eval_local import FoldTruth
from reid.post.policy import DEFAULT_CANDIDATE_THRESHOLD

# Категории в порядке, в котором они интересны жюри
FALSE_REFUSAL = "ложный отказ"
OPEN_TWIN = "ответ на query без пары: в галерее есть двойник"
OPEN_OTHER = "ответ на query без пары: спутали с чужой машиной"
WRONG_ANSWER = "ответили, но не той машиной"


def collect(emb_pattern: str, threshold: float):
    """Классифицирует каждый query фолдов по исходу режима кандидатов."""
    cases, stats = [], Counter()
    for k in range(5):
        path = f"{emb_pattern.format(k=k)}/fold{k}.npz"
        if not os.path.exists(path):
            continue
        d = np.load(path)
        q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()
        truth = FoldTruth.load(k)
        n_pos = truth.n_pos()
        sim = d["query_emb"] @ d["gallery_emb"].T
        best = np.argmax(sim, axis=1)
        for i, q in enumerate(q_ids):
            top, score = g_ids[best[i]], float(sim[i, best[i]])
            has_pair, answered = n_pos[q] > 0, score >= threshold
            stats["всего query"] += 1
            if has_pair and not answered:
                kind = FALSE_REFUSAL
            elif not has_pair and answered:
                twin = any(truth.is_junk(q, g) for g in g_ids)
                kind = OPEN_TWIN if twin else OPEN_OTHER
            elif has_pair and answered and truth.vid[top] != truth.vid[q]:
                kind = WRONG_ANSWER
            else:
                stats["верно" if (has_pair == answered) else "прочее"] += 1
                continue
            stats[kind] += 1
            cases.append(SimpleNamespace(fold=k, q=q, top=top, score=score, kind=kind))
    return cases, stats


# Шрифт по умолчанию в PIL не содержит кириллицы — подписи выходят квадратами.
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",              # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",           # Debian/Ubuntu
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",                    # RHEL
)


def cyrillic_font(size: int):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    raise SystemExit("не нашёл шрифт с кириллицей, добавь путь в FONT_CANDIDATES")


def render(cases, cache: str, out: str, per_kind: int):
    """Рисует по несколько примеров каждой категории: query слева, наш ответ справа."""
    man = load_manifest(cache, ("train",)).set_index("image_id")

    def crop_of(image_id):
        row = SimpleNamespace(image_id=image_id, **man.loc[image_id].to_dict())
        x0, y0, x1, y1 = cached_box(row, 0.0)
        arr = read_crop(cache, image_id)[:, y0:y1, x0:x1]
        return Image.fromarray(arr.permute(1, 2, 0).numpy()).resize((C, C))

    C, PAD, LABEL = 200, 8, 30
    font = cyrillic_font(15)
    chosen = []
    for kind in (FALSE_REFUSAL, OPEN_TWIN, OPEN_OTHER, WRONG_ANSWER):
        same = [c for c in cases if c.kind == kind]
        # берём самые показательные: отказ — с наибольшим счётом, ошибочный ответ — с наименьшим,
        # то есть те, что ближе всего к порогу и объясняют, почему порог стоит именно там
        same.sort(key=lambda c: -c.score if kind == FALSE_REFUSAL else c.score)
        chosen += same[:per_kind]
    if not chosen:
        raise SystemExit("нет случаев для отрисовки")

    W = max(C * 2 + PAD * 3, 560)
    canvas = Image.new("RGB", (W, (C + LABEL) * len(chosen) + PAD), "white")
    draw = ImageDraw.Draw(canvas)
    for r, c in enumerate(chosen):
        y = r * (C + LABEL) + PAD
        canvas.paste(crop_of(c.q), (PAD, y + LABEL))
        canvas.paste(crop_of(c.top), (C + PAD * 2, y + LABEL))
        draw.text((PAD, y + 8), f"{c.kind} | фолд {c.fold} | косинус {c.score:.3f}",
                  fill="black", font=font)
        # красная рамка на том кадре, который делает исход ошибкой
        draw.rectangle([C + PAD * 2, y + LABEL, C * 2 + PAD * 2 - 1, y + LABEL + C - 1],
                       outline="red" if c.kind != FALSE_REFUSAL else "green", width=3)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    canvas.save(out, quality=88)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="artifacts/emb_final/final_oof_f{k}_cls_avg")
    ap.add_argument("--threshold", type=float, default=DEFAULT_CANDIDATE_THRESHOLD)
    ap.add_argument("--cache", default="artifacts/crops")
    ap.add_argument("--out", default="docs/errors/candidate_errors.jpg")
    ap.add_argument("--per-kind", type=int, default=2)
    args = ap.parse_args()

    cases, stats = collect(args.emb, args.threshold)
    total = stats["всего query"]
    print(f"порог {args.threshold}, всего query {total}\n")
    for kind in (FALSE_REFUSAL, OPEN_TWIN, OPEN_OTHER, WRONG_ANSWER):
        print(f"{kind:52s} {stats[kind]:4d}  ({stats[kind] / total * 100:.2f}%)")

    chosen = render(cases, args.cache, args.out, args.per_kind)
    print(f"\nсохранено {args.out}: {len(chosen)} примеров, слева query, справа наш ответ")


if __name__ == "__main__":
    main()
