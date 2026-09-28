"""Строит фолды локальной валидации (контракт C1 в docs/contracts.md).

Устройство повторяет открытый тест (docs/findings.md §2):
  * в gallery — ровно один кадр на пару (vehicle_id, camera_id), остальные кадры группы — в query;
  * около 20% query фолда принадлежат машинам без пары: их gallery-кадры помечаются `drop`.

Фолды коммитятся и после Д1 не пересоздаются. Запуск:
    DATA_ROOT=./dataset python scripts/make_folds.py
"""
import argparse
import collections
import os
import random

import pandas as pd

N_FOLDS = 5
SEED = 2026
OPENSET_QUERY_SHARE = 0.20


def assign_folds(df: pd.DataFrame, rng: random.Random) -> dict:
    """Раскидывает vehicle_id по фолдам со стратификацией по (число кадров, число камер)."""
    stats = df.groupby("vehicle_id").agg(n_img=("image_id", "size"), n_cam=("camera_id", "nunique"))
    strata = collections.defaultdict(list)
    for vid, row in stats.iterrows():
        strata[(row.n_img, min(row.n_cam, 4))].append(vid)
    order = []
    for key in sorted(strata):
        vids = sorted(strata[key])
        rng.shuffle(vids)
        order.extend(vids)
    # Сквозная раздача по кругу держит размеры фолдов и страт ровными.
    return {vid: i % N_FOLDS for i, vid in enumerate(order)}


def assign_roles(fold_df: pd.DataFrame, rng: random.Random) -> dict:
    """Возвращает роль каждого кадра фолда: query / gallery / drop."""
    roles = {}
    for _, grp in fold_df.groupby(["vehicle_id", "camera_id"], sort=True):
        ids = sorted(grp.image_id)
        g = rng.choice(ids)
        for iid in ids:
            roles[iid] = "gallery" if iid == g else "query"

    # Машины без пары: набираем случайные машины, пока их query не составят ~20% query фолда.
    n_query = sum(r == "query" for r in roles.values())
    q_per_vid = fold_df[fold_df.image_id.map(roles) == "query"].groupby("vehicle_id").size()
    vids = sorted(q_per_vid.index)
    rng.shuffle(vids)
    target, taken, open_vids = OPENSET_QUERY_SHARE * n_query, 0, set()
    for vid in vids:
        if taken >= target:
            break
        # Не перелетаем цель сильнее, чем недолетаем.
        if taken + q_per_vid[vid] - target > target - taken:
            continue
        open_vids.add(vid)
        taken += q_per_vid[vid]
    for iid, vid in zip(fold_df.image_id, fold_df.vehicle_id):
        if vid in open_vids and roles[iid] == "gallery":
            roles[iid] = "drop"
    return roles


def check_fold(f: pd.DataFrame, k: int) -> None:
    """Печатает статистику фолда и проверяет инварианты контракта C1."""
    q, g = f[f.role == "query"], f[f.role == "gallery"]
    gal = collections.defaultdict(list)
    for vid, cam in zip(g.vehicle_id, g.camera_id):
        gal[vid].append(cam)
    n_open = 0
    for vid, cam in zip(q.vehicle_id, q.camera_id):
        cams = gal.get(vid, [])
        if not cams:
            n_open += 1
            continue
        assert sum(c == cam for c in cams) == 1, f"fold {k}: query without exactly one junk sibling"
        assert any(c != cam for c in cams), f"fold {k}: paired query without cross-camera positive"
    share = n_open / len(q)
    print(f"fold {k}: query {len(q):5d}  gallery {len(g):4d}  drop {(f.role == 'drop').sum():3d}  "
          f"vehicles {f.vehicle_id.nunique():3d}  open-set query {n_open:4d} ({share:.1%})")
    assert abs(share - OPENSET_QUERY_SHARE) <= 0.02, f"fold {k}: open-set share {share:.3f} out of 20% ± 2%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "dataset"))
    ap.add_argument("--out", default="splits")
    args = ap.parse_args()

    train = pd.read_csv(os.path.join(args.data_root, "train.csv"), dtype={"image_id": str})
    rng = random.Random(SEED)
    fold_of = assign_folds(train, rng)
    train["fold"] = train.vehicle_id.map(fold_of)

    roles = {}
    for k in range(N_FOLDS):
        roles.update(assign_roles(train[train.fold == k], rng))
    train["role"] = train.image_id.map(roles)

    os.makedirs(args.out, exist_ok=True)
    train[["image_id", "vehicle_id", "camera_id", "fold", "role"]].to_csv(
        os.path.join(args.out, "folds.csv"), index=False)
    for k in range(N_FOLDS):
        f = train[train.fold == k]
        check_fold(f, k)
        for role in ("query", "gallery"):
            f[f.role == role][["image_id", "x", "y", "w", "h"]].to_csv(
                os.path.join(args.out, f"fold{k}_{role}.csv"), index=False)


if __name__ == "__main__":
    main()
