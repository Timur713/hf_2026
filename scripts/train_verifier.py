"""Верификатор пар: отдельная модель, сравнивающая два кропа (задача A3.3, схема D-032).

Отличия от провалившейся первой версии (A-009):
  * своя модель со своим backbone, а не голова над токенами эмбеддера;
  * трудные негативы добыты OOF-моделями (scripts/mine_pairs.py) — это настоящие ошибки;
  * оценка на фолде, незнакомом и эмбеддеру, и верификатору.

Итоговый порядок: cosine + weight * sigmoid(logit), то есть при weight=0 порядок прежний.

Оценка идёт поверх фузии и по протоколу сдачи, см. eval_rerank.

    PYTHONPATH=. python scripts/train_verifier.py --val-fold 0 --steps 4000 \
        --emb artifacts/emb/pool_cls_top16_f0_cls_top16 \
              artifacts/emb/div_clip_f0_cls_top16 artifacts/emb/div_siglip_f0_top16
"""
from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from reid.data.augment import TrainAugment
from reid.data.crop_cache import cached_box, load_manifest, read_crop
from reid.data.preprocess import to_model_input
from reid.eval.eval_local import FoldTruth, ranking_metrics
from reid.model.verifier import PairVerifier


class PairDataset(torch.utils.data.Dataset):
    """Пары из обучающих фолдов: позитивы по меткам, негативы — добытые OOF-моделями.

    Датасет, а не генератор в главном потоке: вчера подготовка картинок съедала 13 ядер и
    тормозила обучение сильнее, чем сама модель (A-019).
    """

    def __init__(self, pairs_dir, folds, cache, cfg, aug, seed=2026, length=100000,
                 curriculum=False, hide_seek=0.0, overfit=0):
        """curriculum — наращивать трудность негативов по ходу обучения; hide_seek — доля
        выбиваемых патчей входа.

        Оба приёма взяты из CVNet (CVPR 2022): в их абляции подбор трудных негативов с
        расписанием даёт +9,2 mAP, тогда как смена архитектуры на корреляционную — только +2,9,
        а Hide-and-Seek +1,2. У нас негативы брались равномерно из двадцати заранее добытых,
        без расписания вовсе.
        """
        self.cache, self.cfg, self.aug, self.seed, self.length = cache, cfg, aug, seed, length
        self.curriculum, self.hide_seek = curriculum, hide_seek
        self.overfit = overfit
        man = load_manifest(cache, ("train",)).set_index("image_id")
        self.rows, self.labels, self.cams, self.neg = [], [], [], []
        base = 0
        for k in folds:
            d = np.load(os.path.join(pairs_dir, f"fold{k}.npz"))
            ids = d["image_ids"].tolist()
            self.rows += [SimpleNamespace(image_id=i, **man.loc[i].to_dict()) for i in ids]
            self.labels += [f"{k}_{v}" for v in d["vehicle_id"].tolist()]
            self.cams += d["camera_id"].tolist()
            self.neg += [[base + j for j in row if j >= 0] for row in d["negatives"]]
            base += len(ids)
        by_label = {}
        for i, lab in enumerate(self.labels):
            by_label.setdefault(lab, []).append(i)
        self.pos = [[j for j in by_label[lab] if j != i and self.cams[j] != self.cams[i]]
                    or [j for j in by_label[lab] if j != i] for i, lab in enumerate(self.labels)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.overfit:
            # Контроль обучаемости: гоняем одни и те же N пар. Пара и её аугментация зависят
            # только от idx, поэтому остаток по модулю даёт буквально фиксированную выборку.
            # Если модель не запоминает сотню пар, дело не в объёме данных и не в негативах.
            idx = idx % self.overfit
        rng = random.Random(self.seed * 1000003 + idx)
        want_pos = idx % 2 == 0
        # Загрузчик идёт по индексам подряд, поэтому доля пройденного обучения = idx / length
        progress = idx / max(self.length - 1, 1)
        while True:
            i = rng.randrange(len(self.rows))
            pool = self.pos[i] if want_pos else self.neg[i]
            if not pool:
                continue
            if not want_pos and self.curriculum:
                # Как в CVNet: доля трудных негативов растёт с 0.2 до 1.0. Остальное время берём
                # СЛУЧАЙНЫЙ кадр другой машины — лёгкий негатив. Без лёгкой фазы голова стартует
                # сразу на предельных случаях, и loss застревает у ln 2: именно это мы и видели.
                hard_ratio = 0.2 + 0.8 * progress
                if rng.random() >= hard_ratio:
                    j = rng.randrange(len(self.rows))
                    while self.labels[j] == self.labels[i]:
                        j = rng.randrange(len(self.rows))
                    return (self.image(i, rng, progress), self.image(j, rng, progress), 0.0)
            return (self.image(i, rng, progress), self.image(rng.choice(pool), rng, progress),
                    float(want_pos))

    def image(self, i, rng=None, progress=0.0):
        row = self.rows[i]
        crop = read_crop(self.cache, row.image_id)
        if rng is not None:
            random.setstate(rng.getstate())  # аугментации детерминированы по индексу примера
        x0, y0, x1, y1 = self.aug.jitter_box(cached_box(row, 0.0), crop.shape[-2:], 0.0)
        x = to_model_input(self.aug(crop[:, y0:y1, x0:x1]), self.cfg["input_size"],
                           self.cfg["mean"], self.cfg["std"])
        if self.hide_seek and rng is not None:
            # Выбиваем случайные клетки сетки 8x8: модель перестаёт опираться на одну деталь
            p = self.hide_seek * progress
            g = 8
            h, w = x.shape[-2] // g, x.shape[-1] // g
            for gy in range(g):
                for gx in range(g):
                    if rng.random() < p:
                        x[:, gy * h:(gy + 1) * h, gx * w:(gx + 1) * w] = 0.0
        return x


class ListDataset(PairDataset):
    """Списки вместо независимых пар: один query и K кандидатов из его реальной выдачи.

    Зачем. У двух третей query ровно один валидный позитив, поэтому AP@10 = 1/ранг позитива, и
    метрика — это ранжирование внутри списка. BCE по независимым парам оптимизирует калибровку
    каждой пары в отдельности и про порядок ничего не знает. Softmax по списку — прямой суррогат:
    он штрафует ровно за то, что позитив не первый. Так устроены LCE (Gao et al., ECIR 2021) и
    RankT5; в обоих listwise устойчиво бьёт pointwise.

    Негативы берутся из тех же добытых трудных кандидатов, что и раньше, так что сравнение с
    прежним обучением честное: меняется только функция потерь.
    """

    def __init__(self, *a, k: int = 16, **kw):
        super().__init__(*a, **kw)
        self.k = k
        self.anchors = [i for i in range(len(self.rows)) if self.pos[i] and self.neg[i]]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = random.Random(self.seed * 7919 + idx)
        progress = idx / max(self.length - 1, 1)
        i = self.anchors[rng.randrange(len(self.anchors))]
        pos = rng.choice(self.pos[i])
        negs = self.neg[i][:]
        rng.shuffle(negs)
        negs = negs[: self.k - 1]
        while len(negs) < self.k - 1:                       # редкий случай короткого списка
            j = rng.randrange(len(self.rows))
            if self.labels[j] != self.labels[i]:
                negs.append(j)
        cand = [pos] + negs
        target = rng.randrange(len(cand))
        cand[0], cand[target] = cand[target], cand[0]       # позитив на случайную позицию
        imgs = torch.stack([self.image(j, rng, progress) for j in cand])
        return self.image(i, rng, progress), imgs, target


@torch.inference_mode()
def eval_rerank(model, emb_dirs, fold, cache, cfg, device, weights, topk=10, max_queries=None):
    """Оценка переранжирования на той базе и по тому протоколу, что уйдут в сдачу.

    Три вещи, каждая из которых раньше давала ложный вывод:

    * **База — фузия, а не одна модель.** Верификатор ставится поверх фузии (D-052), и чинить ему
      придётся только те ошибки, которые фузия не починила сама. Прирост, измеренный поверх одной
      модели, к сдаче отношения не имеет — на этом обжёгся A-054.
    * **Смешивание по выровненным шкалам.** `cos + w·sigmoid(logit)` складывает величины разных
      масштабов; после стандартизации обеих по списку кандидатов вклад верификатора вырос втрое
      (D-053, A-060). Сырой вариант печатаем рядом — как напоминание, а не как результат.
    * **Ровно 10 кандидатов в оценку.** `ranking_metrics` выбрасывает двойника из поданного списка
      и добирает недостающее из хвоста. Подашь 30 — получишь завышенное число, потому что в сдаче
      добора не будет: двойник просто съест место.
    """
    S, q_ids, g_ids = None, None, None
    for path in emb_dirs:
        d = np.load(f"{path}/fold{fold}.npz")
        ids = (d["query_ids"].tolist(), d["gallery_ids"].tolist())
        assert q_ids is None or ids == (q_ids, g_ids), f"{path}: другой порядок query/gallery"
        q_ids, g_ids = ids
        unit = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        s = unit(d["query_emb"]) @ unit(d["gallery_emb"]).T
        s = (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-9)
        S = s if S is None else S + s

    truth = FoldTruth.load(fold)
    if max_queries and max_queries < len(q_ids):  # быстрая промежуточная оценка на подвыборке
        keep = np.linspace(0, len(q_ids) - 1, max_queries).astype(int)
        S, q_ids = S[keep], [q_ids[i] for i in keep]
        # Метрика считается по truth.query_ids, и пропущенный query получает AP = 0. Без сужения
        # разметки подвыборка просто делила бы число на долю оставленных query.
        truth = replace(truth, query_ids=q_ids)

    order = np.argsort(-S, axis=1)[:, :topk]
    cand = {q: [g_ids[j] for j in order[i]] for i, q in enumerate(q_ids)}
    base = ranking_metrics(truth, {q: c[:10] for q, c in cand.items()})["map10"]

    man = load_manifest(cache, ("train",)).set_index("image_id")

    def prep(image_id):
        row = SimpleNamespace(image_id=image_id, **man.loc[image_id].to_dict())
        crop = read_crop(cache, image_id)
        x0, y0, x1, y1 = cached_box(row, 0.0)
        return to_model_input(crop[:, y0:y1, x0:x1], cfg["input_size"], cfg["mean"], cfg["std"])

    model.eval()
    # Каждый кроп готовим ровно один раз: раньше prep() вызывался на каждую пару (A-019).
    prepared = {i: prep(i) for i in sorted(set(q_ids) | {g for q in q_ids for g in cand[q]})}
    scores = {}
    for i, q in enumerate(q_ids):
        xa = prepared[q].unsqueeze(0).expand(len(cand[q]), -1, -1, -1)
        xb = torch.stack([prepared[g] for g in cand[q]])
        lg = model(xa.to(device), xb.to(device)).float().cpu().numpy()
        scores[q] = (S[i, order[i]], lg)
    model.train()

    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    out = {}
    for w in weights:
        aligned, raw = {}, {}
        for q in q_ids:
            sim, lg = scores[q]
            aligned[q] = [cand[q][j] for j in np.argsort(-(z(sim) + w * z(lg)))][:10]
            raw[q] = [cand[q][j] for j in np.argsort(-(sim + w / (1 + np.exp(-lg))))][:10]
        out[w] = ranking_metrics(truth, aligned)["map10"]
        out[f"raw{w}"] = ranking_metrics(truth, raw)["map10"]
    return base, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--emb", nargs="+", default=[],
                help="дампы эмбеддеров, обученных без этого фолда; несколько — фузия, та же\n                     база, что уйдёт в сдачу")
    ap.add_argument("--pairs", default="artifacts/pairs")
    ap.add_argument("--cache", default="artifacts/crops")
    ap.add_argument("--arch", default="dinov2_vits14")
    ap.add_argument("--input", type=int, default=168)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="скорость обучения головы")
    ap.add_argument("--lr-backbone", type=float, default=1e-5, help="как при дообучении эмбеддера")
    ap.add_argument("--freeze-backbone", action="store_true", help="учить только голову: быстро и устойчиво")
    ap.add_argument("--unfreeze-after", type=int, default=0,
                    help="разморозить backbone после N шагов: голова к этому моменту уже осмысленна")
    ap.add_argument("--overfit", type=int, default=0,
                    help="контроль обучаемости: гонять только N фиксированных пар (0 = выкл)")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-queries", type=int, default=None, help="оценивать на подвыборке query")
    ap.add_argument("--name", help="имя артефакта; по умолчанию verifier_f<fold>")
    ap.add_argument("--loss", choices=("bce", "listwise"), default="bce",
                    help="listwise — softmax по списку кандидатов вместо BCE по парам")
    ap.add_argument("--list-k", type=int, default=16, help="кандидатов в списке при --loss listwise")
    ap.add_argument("--head", choices=("concat", "corr"), default="concat",
                    help="concat — склейка токенов и self-attention (прежняя); "
                         "corr — явные признаки соответствия патчей")
    ap.add_argument("--curriculum", action="store_true",
                    help="наращивать трудность негативов по ходу обучения (CVNet: +9,2 mAP)")
    ap.add_argument("--hide-seek", type=float, default=0.0,
                    help="итоговая доля выбиваемых клеток сетки 8x8 (CVNet: +1,2 mAP)")
    ap.add_argument("--symmetric", action="store_true",
                    help="усреднять логит по обоим порядкам пары (CVNet считает симметричный loss)")
    ap.add_argument("--vehicleid", default="",
                    help="каталог ужатого VehicleID: предобучение на 113 тыс. кадров с негативами "
                         "той же модели и того же цвета вместо наших 7,6 тыс.")
    ap.add_argument("--plate-band", type=float, default=-1.0,
                    help="вероятность заливки номерной зоны; по умолчанию 0,5 на наших данных и "
                         "1,0 на внешних, где номера не размыты (правило 3)")
    ap.add_argument("--hard-ratio", type=float, default=0.8,
                    help="доля негативов из группы «та же модель, тот же цвет» при --vehicleid")
    ap.add_argument("--init-verifier", default="",
                    help="чекпоинт верификатора: продолжить с предобученного вместо случайной головы")
    ap.add_argument("--init-backbone", default="",
                    help="каталог нашего эмбеддера: взять его backbone вместо универсального")
    ap.add_argument("--train-folds", default="",
                    help="через запятую; по умолчанию все, кроме отложенного. Нужно, когда трудные "
                         "пары добыты не для всех фолдов")
    ap.add_argument("--topk", type=int, default=10,
                    help="сколько кандидатов переранжируем; в файл сдачи всё равно идут 10")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {"input_size": [args.input, args.input], "mean": [0.485, 0.456, 0.406],
           "std": [0.229, 0.224, 0.225], "resize": "stretch", "bbox_pad": 0.0}
    weights = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    # val_fold < 0 → берём все фолды: так готовится верификатор для сдачи, оценивать его негде
    if args.train_folds:
        folds = [int(k) for k in args.train_folds.split(",")]
        assert args.val_fold not in folds, "отложенный фолд не может быть в обучающих"
    else:
        folds = list(range(5)) if args.val_fold < 0 else [k for k in range(5) if k != args.val_fold]
    # Номерная зона заливается и при обучении верификатора. На наших кадрах номер размыт
    # организаторами, но правило 3 запрещает и остаточные признаки размытия; на внешних данных
    # номер виден целиком, поэтому там заливаем всегда — иначе модель выучит читать его и
    # провалит контрольный прогон жюри.
    plate = args.plate_band if args.plate_band >= 0 else (1.0 if args.vehicleid else 0.5)
    aug = TrainAugment(0.05, 0.05, color=0.2, plate_band_p=plate)
    print(f"заливка номерной зоны с вероятностью {plate}", flush=True)
    if args.vehicleid:
        from reid.data.vehicleid import VehicleIDPairs
        dataset = VehicleIDPairs(args.vehicleid, cfg, aug, mode=args.loss, k=args.list_k,
                                 length=args.steps * args.batch, hard_ratio=args.hard_ratio,
                                 curriculum=args.curriculum)
        print(dataset.stats(), flush=True)
    elif args.loss == "listwise":
        dataset = ListDataset(args.pairs, folds, args.cache, cfg, aug,
                              length=args.steps * args.batch, curriculum=args.curriculum,
                              hide_seek=args.hide_seek, k=args.list_k)
    else:
        dataset = PairDataset(args.pairs, folds, args.cache, cfg, aug,
                              length=args.steps * args.batch, curriculum=args.curriculum,
                              hide_seek=args.hide_seek, overfit=args.overfit)
    if not args.vehicleid:
        print(f"пар из фолдов {folds}: {len(dataset.rows)} кадров, обучение {args.steps} шагов",
              flush=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, num_workers=args.workers,
                                         drop_last=True, persistent_workers=args.workers > 0)
    if args.head == "corr":
        from reid.model.corr_verifier import CorrVerifier
        model = CorrVerifier(args.arch, tuple(cfg["input_size"]), pretrained=True,
                             init_from=args.init_backbone or None).to(device)
    else:
        model = PairVerifier(args.arch, tuple(cfg["input_size"]), pretrained=True,
                             init_from=args.init_backbone or None).to(device)
    if args.init_verifier:
        # Дообучение с предобученного: голова уже умеет сравнивать, случайный старт её бы стёр
        ck = torch.load(args.init_verifier, map_location="cpu")
        model.load_state_dict(ck["state"])
        print(f"голова взята из {args.init_verifier}", flush=True)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    back = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    # Разморозка с первого шага трижды схлопывала обучение (A-018, A-032): случайная голова
    # посылает в предобученный backbone разрушительные градиенты. Поэтому либо держим его
    # замороженным, либо размораживаем, когда голова уже что-то умеет.
    if args.freeze_backbone or args.unfreeze_after:
        for p in back:
            p.requires_grad_(False)
    # Backbone нельзя учить со скоростью головы: шаг на порядок больше разрушает предобученные
    # признаки, и loss застревает на ln 2 — так провалилась третья попытка (A-018).
    # Backbone кладём в оптимизатор всегда, даже когда он заморожен: у замороженного параметра
    # нет градиента, и AdamW его пропускает, а вот OneCycleLR группу, добавленную на ходу,
    # не переживает — падает с KeyError('max_lr'). Именно так сорвалось предобучение на VehicleID.
    groups = [{"params": head, "lr": args.lr}, {"params": back, "lr": args.lr_backbone}]
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups],
                                                total_steps=args.steps, pct_start=0.1)
    t0 = time.time()
    model.train()
    for step, batch in enumerate(loader, start=1):
        if args.unfreeze_after and step == args.unfreeze_after:
            for p in back:
                p.requires_grad_(True)
            print(f"шаг {step}: backbone разморожен, lr {args.lr_backbone}", flush=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            if args.loss == "listwise":
                xq, xc, target = batch
                n, k = xc.shape[:2]
                q = xq.to(device).unsqueeze(1).expand(-1, k, -1, -1, -1).reshape(n * k, *xq.shape[1:])
                logit = model(q, xc.to(device).reshape(n * k, *xc.shape[2:])).float().view(n, k)
                loss = F.cross_entropy(logit, target.to(device))
            else:
                xa, xb, y = batch
                a, b, target = xa.to(device), xb.to(device), y.to(device)
                logit = model(a, b).float()
                if args.symmetric:
                    # «та же машина» — отношение симметричное, а голова с сегментными
                    # эмбеддингами об этом не знает: иначе она учит два разных правила
                    logit = 0.5 * (logit + model(b, a).float())
                loss = F.binary_cross_entropy_with_logits(logit, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if args.val_fold < 0 and (step % args.eval_every == 0 or step == args.steps):
            # Без отложенного фолда оценки нет, но loss печатать обязательно: иначе застревание
            # у ln 2 = 0.693 остаётся невидимым (так и вышло с v_fus_all и v_fus_all392).
            print(f"шаг {step}/{args.steps}  loss {loss.item():.4f}", flush=True)
        if args.val_fold >= 0 and (step % args.eval_every == 0 or step == args.steps):
            base, res = eval_rerank(model, args.emb, args.val_fold, args.cache, cfg, device, weights,
                                    topk=args.topk, max_queries=args.eval_queries)
            aligned = {w: v for w, v in res.items() if not isinstance(w, str)}
            best_w = max(aligned, key=aligned.get)
            raw_best = max(v for k, v in res.items() if isinstance(k, str))
            print(f"шаг {step}/{args.steps}  loss {loss.item():.4f}  база {base:.4f}  "
                  + "  ".join(f"w={w}: {v:.4f}" for w, v in aligned.items())
                  + f"  | лучший {best_w}: {aligned[best_w] - base:+.4f}"
                  + f"  (сырое смешивание {raw_best:.4f})", flush=True)
    out = f"artifacts/models/{args.name or f'verifier_f{args.val_fold}'}"
    os.makedirs(out, exist_ok=True)
    torch.save({"state": model.state_dict(), "arch": args.arch, "input": args.input,
                "head": args.head, "weight": 0.2, "trained_on": f"folds {folds}",
                "emb": args.emb}, f"{out}/verifier.pt")
    print(f"saved {out}  ({(time.time() - t0) / 60:.1f} мин)")


if __name__ == "__main__":
    main()
