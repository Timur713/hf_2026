"""Дообучение эмбеддера (задача A2.1).

Обучение идёт на всех кадрах фолдов, кроме val_fold; валидация — query/gallery этого фолда по
протоколу организаторов (reid.eval.eval_local). На выходе — артефакт модели по контракту C3.

    PYTHONPATH=. python scripts/train.py --config configs/vitb14_224.yaml --name vitb14_f0_v1
    ... --set train.epochs=2 train.p=4 train.pretrained=false   # быстрый прогон механики
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from reid.data.augment import RandomErasing, TrainAugment
from reid.data.dataset import EvalDataset, TrainDataset
from reid.data.external import ExternalReIDDataset
from reid.data.sampler import PKSampler
from reid.eval.eval_local import evaluate, ranking_from_embeddings
from reid.model import Embedder, build_backbone, save
from reid.model.losses import ReIDLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def param_groups(model: Embedder, tr: dict) -> list[dict]:
    """Послойное уменьшение lr: чем ближе к входу, тем меньше шаг."""
    blocks = list(getattr(model.backbone, "blocks", []))
    n = len(blocks) + 1

    def layer_id(name: str) -> int:
        if name.startswith("backbone.patch_embed") or name in ("backbone.cls_token", "backbone.pos_embed"):
            return 0
        if name.startswith("backbone.blocks."):
            return int(name.split(".")[2]) + 1
        return n  # norm, neck и всё остальное — верхний уровень

    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = layer_id(name)
        head = not name.startswith("backbone.")
        lr = tr["lr_head"] if head else tr["lr_backbone"] * (tr["llrd"] ** (n - lid))
        wd = 0.0 if p.ndim == 1 else tr["weight_decay"]
        groups.setdefault((lr, wd), {"params": [], "lr": lr, "weight_decay": wd})["params"].append(p)
    return list(groups.values())


def lr_scale(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + np.cos(np.pi * t))


@torch.no_grad()
def embed_split(model, cache, split_csv, mcfg, device, batch, workers):
    ds = EvalDataset(cache, split_csv, mcfg)
    loader = DataLoader(ds, batch_size=batch, num_workers=workers)
    out = [model(x.to(device)).float().cpu().numpy() for x in loader]
    return np.concatenate(out), ds.ids


def validate(model, cache, fold, mcfg, device, batch, workers):
    model.eval()
    q, q_ids = embed_split(model, cache, f"splits/fold{fold}_query.csv", mcfg, device, batch, workers)
    g, g_ids = embed_split(model, cache, f"splits/fold{fold}_gallery.csv", mcfg, device, batch, workers)
    res = evaluate(fold, ranking_from_embeddings(q, g, g_ids, q_ids), emb=np.concatenate([q, g]))
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vitb14_224.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--cache", default="artifacts/crops")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--set", nargs="*", default=[], help="переопределения: train.epochs=2 model.pool=avg")
    ap.add_argument("--external", help="каталог внешнего датасета для предобучения (A4.1)")
    ap.add_argument("--external-list", help="файл «имя номер_машины» рядом с кадрами (VehicleID)")
    ap.add_argument("--init-from", help="артефакт модели, с весов которой начинаем")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    for item in args.set:
        key, value = item.split("=", 1)
        section, field = key.split(".")
        cfg[section][field] = yaml.safe_load(value)
    mcfg, tr = cfg["model"], cfg["train"]
    set_seed(tr["seed"])
    device = args.device

    aug = TrainAugment(tr["box_jitter"], tr["pad_jitter"], color=tr["color"],
                       plate_band_p=tr.get("plate_band_p", 0.0))
    erasing = RandomErasing(tr["erasing_p"])
    if args.external:
        # Предобучение на внешнем датасете. Валидация всё равно на нашем фолде: так сразу видно,
        # насколько внешние данные переносятся на конкурсные.
        train_ds = ExternalReIDDataset(args.external, mcfg, aug=aug, erasing=erasing,
                                       list_file=args.external_list)
    else:
        # val_fold = null → обучение на всех фолдах: так готовится финальная модель для сдачи
        train_ds = TrainDataset(args.cache, "splits/folds.csv", tr["val_fold"], mcfg, aug, erasing)
    neighbors = None
    if tr.get("hard_ids"):
        # Файл хранит соседей по vehicle_id, сэмплеру нужны индексы личностей
        d = np.load(tr["hard_ids"])
        idx = train_ds.vid_index
        neighbors = {idx[int(v)]: [idx[int(n)] for n in row if int(n) in idx]
                     for v, row in zip(d["vids"], d["neighbors"]) if int(v) in idx}
        print(f"путающиеся машины: {tr['hard_ids']}, доля батча {tr.get('hard_frac', 0.5)}",
              flush=True)
    sampler = PKSampler(train_ds.labels, train_ds.cams, tr["p"], tr["k"], tr["seed"],
                        views=train_ds.views if tr.get("view_spread", False) else None,
                        neighbors=neighbors, hard_frac=tr.get("hard_frac", 0.5) if neighbors else 0.0)
    loader = DataLoader(train_ds, batch_size=tr["p"] * tr["k"], sampler=sampler,
                        num_workers=tr["workers"], drop_last=True, persistent_workers=tr["workers"] > 0)

    if args.init_from:
        # Постепенное увеличение разрешения: модель, обученная на меньшем входе, — хорошая
        # инициализация для большего. Позиционные эмбеддинги интерполируются при загрузке.
        from reid.model import load as load_artifact
        model = load_artifact(args.init_from, device, input_size=mcfg["input_size"]).train()
    else:
        model = Embedder(build_backbone(mcfg["arch"], mcfg["input_size"],
                                        pretrained=tr.get("pretrained", True)),
                         mcfg["pool"], mcfg["neck"], parts=int(mcfg.get("parts", 0))).to(device)
    emb_dim = (model.neck.num_features if mcfg["neck"] == "bnneck"
               else model.backbone.num_features * (2 if mcfg["pool"] == "cls_avg" else 1))
    if tr.get("head", "linear") == "arcface":
        from reid.model.losses import ArcFace
        classifier = ArcFace(emb_dim, train_ds.num_classes, tr.get("arc_scale", 30.0),
                             tr.get("arc_margin", 0.3)).to(device)
    else:
        classifier = torch.nn.Linear(emb_dim, train_ds.num_classes, bias=False).to(device)
    # У каждой полосы свой классификатор: именно это заставляет ветку кодировать свою область,
    # а не повторять глобальный вектор (схема MGN)
    part_classifiers = torch.nn.ModuleList(
        [torch.nn.Linear(d, train_ds.num_classes, bias=False) for d in model.part_dims()]).to(device)
    part_ce = torch.nn.CrossEntropyLoss(label_smoothing=tr["label_smoothing"])
    part_w = tr.get("part_weight", 1.0)
    criterion = ReIDLoss(tr["label_smoothing"], tr["triplet_margin"], tr["triplet_weight"])
    head_params = list(classifier.parameters()) + list(part_classifiers.parameters())
    groups = param_groups(model, tr) + [{"params": head_params,
                                         "lr": tr["lr_head"], "weight_decay": tr["weight_decay"]}]
    opt = torch.optim.AdamW(groups)
    base_lrs = [g["lr"] for g in opt.param_groups]
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(tr["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=tr["amp"] == "fp16" and device == "cuda")

    steps_per_epoch = len(loader)
    total, warmup = steps_per_epoch * tr["epochs"], steps_per_epoch * tr["warmup_epochs"]
    print(f"train: {len(train_ds)} кадров, {train_ds.num_classes} машин, "
          f"{steps_per_epoch} шагов/эпоху, батч {tr['p'] * tr['k']}, устройство {device}", flush=True)

    step, history, t0 = 0, [], time.time()
    model.train()
    for epoch in range(tr["epochs"]):
        sampler.set_epoch(epoch)
        losses = []
        for x, y, _ in loader:
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * lr_scale(step, total, warmup)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                g, ps = model.branches(x)
                neck = model.neck(g)
                logits = classifier(neck, y) if tr.get("head", "linear") == "arcface" else classifier(neck)
                feat = g if not ps else torch.cat(
                    [torch.nn.functional.normalize(g, dim=1)]
                    + [torch.nn.functional.normalize(p, dim=1) for p in ps], dim=1)
                loss, parts = criterion(logits, feat.float(), y)
                for bn, cl, p in zip(model.part_necks, part_classifiers, ps):
                    loss = loss + part_w * part_ce(cl(bn(p)), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append((loss.item(), parts["ce"], parts["triplet"]))
            step += 1
        m = np.mean(losses, axis=0)
        line = (f"epoch {epoch + 1}/{tr['epochs']}  loss {m[0]:.3f} (ce {m[1]:.3f}, tri {m[2]:.3f})  "
                f"{(time.time() - t0) / 60:.1f} мин")
        if tr["val_fold"] is not None and ((epoch + 1) % tr["eval_every"] == 0 or epoch + 1 == tr["epochs"]):
            res = validate(model, args.cache, tr["val_fold"], mcfg, device, tr["p"] * tr["k"], tr["workers"])
            history.append({"epoch": epoch + 1, **{k: res[k] for k in ("map10", "rank1", "full_map")}})
            line += f"  | val mAP@10 {res['map10']:.4f}  R1 {res['rank1']:.4f}"
        print(line, flush=True)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out_cfg = dict(mcfg, name=args.name, embed_dim=int(model(torch.zeros(2, 3, *mcfg["input_size"]).to(device)).shape[1]),
                   train={"commit": commit, "config": args.config,
                          "folds": args.external or ("all" if tr["val_fold"] is None
                                                     else f"all except {tr['val_fold']}"),
                          "init_from": args.init_from,
                          "recipe": tr},
                   val={f"fold{tr['val_fold']}": history[-1] if history else None} if history else {})
    save(model, out_cfg, f"artifacts/models/{args.name}")
    with open(f"artifacts/models/{args.name}/history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved artifacts/models/{args.name}  ({(time.time() - t0) / 60:.1f} мин)")


if __name__ == "__main__":
    main()
