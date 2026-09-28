"""Воспроизводит числа из docs/findings.md: статистику train и структуру открытого теста.

Только стандартная библиотека Python, картинки не читает (работает по CSV).
Запуск:  DATA_ROOT=./dataset python3 scripts/analysis/data_structure.py
"""
import collections
import csv
import os

DATA_ROOT = os.environ.get("DATA_ROOT", "dataset")
NEAR_PX = 10  # «двойник»: все четыре координаты BBox отличаются не более чем на 10 px


def read(name):
    with open(os.path.join(DATA_ROOT, name), newline="") as f:
        return list(csv.DictReader(f))


def box(row):
    return tuple(int(row[k]) for k in "xywh")


def near(a, b):
    return all(abs(p - q) <= NEAR_PX for p, q in zip(a, b))


def main():
    train, query, gallery = read("train.csv"), read("test_query.csv"), read("test_gallery.csv")

    # --- train: identities, cameras, burst groups -------------------------------------------
    by_vid = collections.defaultdict(list)
    for r in train:
        by_vid[r["vehicle_id"]].append(r)
    groups = collections.Counter((r["vehicle_id"], r["camera_id"]) for r in train)
    cams = {v: {r["camera_id"] for r in rs} for v, rs in by_vid.items()}
    print(f"train: {len(train)} images, {len(by_vid)} vehicle_id, "
          f"{len({r['camera_id'] for r in train})} camera_id")
    print("  images per vehicle:", sorted(collections.Counter(len(v) for v in by_vid.values()).items()))
    print("  cameras per vehicle:", sorted(collections.Counter(len(c) for c in cams.values()).items()))
    print("  images per (vehicle, camera):", sorted(collections.Counter(groups.values()).items()))

    # --- near-duplicate BBox => same camera? (same-vehicle pairs) ---------------------------
    same_cam = diff_cam = same_cam_pairs = 0
    for rs in by_vid.values():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                sc = rs[i]["camera_id"] == rs[j]["camera_id"]
                same_cam_pairs += sc
                if near(box(rs[i]), box(rs[j])):
                    same_cam += sc
                    diff_cam += not sc
    p_near = same_cam / same_cam_pairs
    print(f"  same-vehicle near-dup pairs: same camera {same_cam}, different camera {diff_cam}")
    print(f"  share of same-vehicle same-camera pairs that are near-dup: {p_near:.3f}")

    # --- open test structure --------------------------------------------------------------
    qb, gb = [box(r) for r in query], [box(r) for r in gallery]
    per_q = collections.Counter(sum(near(a, b) for b in gb) for a in qb)
    gg = sum(near(gb[i], gb[j]) for i in range(len(gb)) for j in range(i + 1, len(gb)))
    qq = sum(near(qb[i], qb[j]) for i in range(len(qb)) for j in range(i + 1, len(qb)))
    print(f"\ntest: {len(query)} query, {len(gallery)} gallery")
    print("  near-dup gallery items per query:", sorted(per_q.items()))
    print(f"  near-dup pairs inside gallery: {gg}; inside query: {qq}")

    # --- hypothesis: gallery = 1 frame per (vehicle, camera), query = the rest ------------
    scale = (len(query) + len(gallery)) / len(train)
    n_groups3 = sum(1 for v in groups.values() if v == 3)
    print("\nhypothesis 'gallery = one frame per (vehicle, camera)': predicted vs observed")
    print(f"  gallery size        {len(groups) * scale:7.0f} vs {len(gallery)}")
    print(f"  query size          {(len(train) - len(groups)) * scale:7.0f} vs {len(query)}")
    print(f"  queries with twin   {len(query) * p_near:7.0f} vs {per_q.get(1, 0)}")
    print(f"  query-query twins   {n_groups3 * scale * p_near:7.0f} vs {qq}")


if __name__ == "__main__":
    main()
