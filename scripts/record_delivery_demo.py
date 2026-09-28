"""Record actual Docker commands/results; no fabricated or replayed PASS output.

Run from a clean checkout on Linux with Docker/NVIDIA configured. The small
dataset is derived from fixed train-fold CSVs; no input images are published.
Only the three output files and a timed JSON transcript are presentation assets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time


def prepare_demo(data: Path, dest: Path, root: Path, count: int = 25,
                 gallery_count: int | None = None) -> dict:
    gallery_count = count if gallery_count is None else gallery_count
    if gallery_count < 10 or count < 1:
        raise ValueError("at least 10 gallery objects required")
    if dest.exists():
        raise FileExistsError("use a new demo directory")
    data = data.resolve()
    if dest.resolve().is_relative_to(data):
        raise ValueError("never write into the organizer dataset")
    selected = {}
    fields = ["image_id", "x", "y", "w", "h"]
    for role in ("query", "gallery"):
        size = count if role == "query" else gallery_count
        with (root / f"splits/fold0_{role}.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))[:size]
        if len(rows) != size:
            raise ValueError("not enough fixed train-fold rows")
        selected[role] = [{key: row[key] for key in fields} for row in rows]
    paths = {}
    for rows in selected.values():
        for row in rows:
            image_id = row["image_id"]
            if Path(image_id).name != image_id:
                raise ValueError("unsafe image ID")
            candidates = [data / "images" / (image_id + ext) for ext in ("", ".jpg", ".png")]
            found = next((p for p in candidates if p.is_file()), None)
            if found is None:
                raise FileNotFoundError("missing train image")
            paths[found.name] = found
    (dest / "images").mkdir(parents=True)
    for name, path in paths.items():
        shutil.copyfile(path, dest / "images" / name)
    for role, rows in selected.items():
        with (dest / f"test_{role}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return {"source": "fixed train fold0 CSV prefix; not test selection", "query": count,
            "gallery": gallery_count, "images": len(paths), "images_published": False}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--image", default="falcon-reid:public-d065")
    p.add_argument("--query-count", type=int, default=5)
    p.add_argument("--gallery-count", type=int, default=636)
    args = p.parse_args()
    root = Path.cwd()
    if args.out.exists():
        p.error("output must be new; never overwrite accepted artifacts")
    args.out.mkdir(parents=True)
    demo = args.out / "demo-input"
    metadata = prepare_demo(args.data_root, demo, root, args.query_count, args.gallery_count)
    output = args.out / "outputs"
    output.mkdir()
    events = []
    started = time.monotonic()
    transcript = args.out / "terminal.json"
    def save():
        transcript.write_text(json.dumps({"schema": 1, "kind": "actual_command_recording",
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "demo": metadata, "events": events}, ensure_ascii=False, indent=2) + "\n")
    def run(title, command):
        event = {"title": title, "command": shlex.join(command),
                 "started_seconds": time.monotonic() - started, "chunks": []}
        events.append(event)
        print("$ " + event["command"], flush=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, env=env) as proc:
            for line in proc.stdout:
                event["chunks"].append({"seconds": time.monotonic() - started, "text": line})
                print(line, end="", flush=True)
            event["returncode"] = proc.wait()
        event["ended_seconds"] = time.monotonic() - started
        save()
        if event["returncode"]:
            raise RuntimeError("recorded command failed; do not publish as PASS")
    run("Версия кода и образ", ["docker", "image", "inspect", args.image,
        "--format", "{{.Id}}"])
    common = ["docker", "run", "--rm", "--network", "none",
        "--mount", f"type=bind,src={demo.resolve()},dst=/data,readonly",
        "--mount", f"type=bind,src={output.resolve()},dst=/out"]
    run(f"Docker: офлайн · {metadata['query']} query + {metadata['gallery']} gallery", common + ["--gpus", "all", args.image])
    run("Три выходных файла", common + ["--entrypoint", "python3", args.image, "-c",
        "from pathlib import Path; import numpy as np; "
        "[print(p.name, p.stat().st_size, 'bytes') for p in sorted(Path('/out').iterdir()) if p.is_file()]; "
        "a=np.load('/out/embeddings.npy',allow_pickle=False); print('embeddings:',a.shape,a.dtype)"])
    run("Проверка формата C6", common + ["--entrypoint", "python3", args.image,
        "-m", "reid.eval.validate_outputs", "--query", "/data/test_query.csv",
        "--gallery", "/data/test_gallery.csv", "--submission", "/out/submission.csv",
        "--embeddings", "/out/embeddings.npy", "--candidates", "/out/candidates.csv"])
    records = {path.name: {"bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in output.iterdir() if path.is_file()}
    (args.out / "files.json").write_text(json.dumps(records, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
