"""Ужимает обучающую часть VehicleID до 320 px по длинной стороне (ставка 3, D-043).

Исходник — 7,8 ГБ при 222 тыс. кадров, из них обучающая часть 113 346 кадров и 13 164 машины.
Наш вход 280, поэтому 320 достаточно, а объём падает втрое: это разница между часами заливки
на сервер и десятком минут.

    python scripts/repack_vehicleid.py --src VehicleID_V1.0 --dst /tmp/vehicleid_small
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

from PIL import Image

SIDE = 320
SRC = DST = None


def init(src, dst):
    global SRC, DST
    SRC, DST = src, dst


def work(name):
    im = Image.open(f"{SRC}/image/{name}.jpg").convert("RGB")
    scale = min(1.0, SIDE / max(im.size))
    if scale < 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BILINEAR)
    im.save(f"{DST}/image/{name}.jpg", quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="VehicleID_V1.0")
    ap.add_argument("--dst", default="/tmp/vehicleid_small")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(f"{args.dst}/image", exist_ok=True)

    pairs = [l.split() for l in open(f"{args.src}/train_test_split/train_list.txt")]
    print(f"{len(pairs)} кадров, {len({v for _, v in pairs})} машин", flush=True)
    t = time.time()
    with ProcessPoolExecutor(args.workers, initializer=init, initargs=(args.src, args.dst)) as ex:
        list(ex.map(work, [n for n, _ in pairs], chunksize=64))
    with open(f"{args.dst}/train_list.txt", "w") as f:
        f.writelines(f"{n} {v}\n" for n, v in pairs)
    print(f"готово за {time.time() - t:.0f} с", flush=True)


if __name__ == "__main__":
    main()
