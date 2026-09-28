"""Интерпретируемость: по каким частям машин модель решила, что это один и тот же автомобиль.

Grad-CAM показывает, куда модель смотрит на одном кадре. Нам нужно другое: задача парная, и
оператору важно, **чем именно** два кадра похожи. Поэтому считаем соответствие патчей: для каждого
кусочка одной картинки ищем самый похожий кусочек на второй.

Побочная польза: средняя сила соответствия — это готовая оценка пары, не требующая обучения.
В отличие от обучаемого верификатора, здесь нет параметров, которые нужно подгонять.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.inference_mode()
def patch_tokens(model, x: torch.Tensor) -> torch.Tensor:
    """Патч-токены без служебных: [B, N, D], нормированные."""
    t = model.tokens(x)
    return F.normalize(t[:, model.n_prefix:], dim=-1)


@torch.inference_mode()
def correspondence(model, xa: torch.Tensor, xb: torch.Tensor, xref: torch.Tensor | None = None):
    """Соответствие патчей двух кадров.

    Если передан xref — посторонний кадр, — карта показывает не сходство само по себе, а
    **превосходство над ним**: насколько патч похож на этот кандидат сильнее, чем на случайную
    другую машину. Без такой поправки карта бесполезна: фон и кузов находят себе пару где угодно,
    и подсвечивается всё подряд.

    Возвращает heat_a, heat_b (карты [h, w]) и score — силу соответствия по лучшей половине патчей.
    """
    ta, tb = patch_tokens(model, xa.unsqueeze(0))[0], patch_tokens(model, xb.unsqueeze(0))[0]
    sim = ta @ tb.T                      # [Na, Nb]
    best_a = sim.max(dim=1).values
    best_b = sim.max(dim=0).values
    if xref is not None:
        tr = patch_tokens(model, xref.unsqueeze(0))[0]
        best_a = best_a - (ta @ tr.T).max(dim=1).values
        best_b = best_b - (tb @ tr.T).max(dim=1).values
    side_a = int(round(len(best_a) ** 0.5))
    side_b = int(round(len(best_b) ** 0.5))
    top = torch.sort(best_a, descending=True).values[: max(1, len(best_a) // 2)]
    return (best_a.reshape(side_a, side_a).cpu().numpy(),
            best_b.reshape(side_b, side_b).cpu().numpy(),
            float(top.mean()))


def overlay(crop: torch.Tensor, heat: np.ndarray, top_share: float = 0.25) -> np.ndarray:
    """Подсвечивает красным только самые информативные патчи, остальное оставляет как есть.

    Подсвечивать всё бессмысленно: глазу нужна не заливка, а указание на конкретные детали.
    """
    img = crop.permute(1, 2, 0).cpu().numpy().astype(np.float32) / 255.0
    h = torch.tensor(heat.astype(np.float32))[None, None]
    h = F.interpolate(h, size=img.shape[:2], mode="bilinear", align_corners=False)[0, 0].numpy()
    thr = np.quantile(h, 1.0 - top_share)
    mask = np.clip((h - thr) / (h.max() - thr + 1e-6), 0, 1) ** 0.7
    red = np.zeros_like(img)
    red[..., 0] = 1.0
    return np.clip(img * (1 - 0.6 * mask[..., None]) + red * (0.6 * mask[..., None]), 0, 1)


@torch.inference_mode()
def pair_scores(tok_q: torch.Tensor, tok_c: torch.Tensor, top_share: float = 0.5) -> torch.Tensor:
    """Сила соответствия query против нескольких кандидатов: [P, D] и [C, P, D] → [C].

    Симметрично по обеим сторонам: иначе кадр, где машина снята крупно, всегда выигрывал бы у
    кадра, где она мелкая, — у крупного просто больше патчей, которым есть к чему прицепиться.

    Берём лучшую половину патчей, а не все: у любой пары кадров часть патчей — фон и ровные
    панели, они находят себе пару всегда и сигнал только размывают.
    """
    sim = torch.einsum("pd,cqd->cpq", tok_q, tok_c)

    def top_mean(x: torch.Tensor) -> torch.Tensor:
        k = max(1, int(x.shape[1] * top_share))
        return x.topk(k, dim=1).values.mean(dim=1)

    return 0.5 * (top_mean(sim.max(dim=2).values) + top_mean(sim.max(dim=1).values))


@torch.inference_mode()
def tokens_for(model, cache: str, image_ids: list[str], cfg: dict, device: str,
               batch: int = 16) -> torch.Tensor:
    """Патч-токены для списка кадров: [N, P, D] в float16, чтобы фолд помещался в память."""
    from types import SimpleNamespace

    from reid.data.crop_cache import cached_box, load_manifest, read_crop
    from reid.data.preprocess import to_model_input

    man = load_manifest(cache, ("train",)).set_index("image_id")
    out = []
    for start in range(0, len(image_ids), batch):
        xs = []
        for image_id in image_ids[start:start + batch]:
            row = SimpleNamespace(image_id=image_id, **man.loc[image_id].to_dict())
            x0, y0, x1, y1 = cached_box(row, cfg.get("bbox_pad", 0.0))
            crop = read_crop(cache, image_id)[:, y0:y1, x0:x1]
            xs.append(to_model_input(crop, cfg["input_size"], cfg["mean"], cfg["std"]))
        out.append(patch_tokens(model, torch.stack(xs).to(device)).half().cpu())
    return torch.cat(out)
