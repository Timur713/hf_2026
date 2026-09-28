"""Train-fold-only C3/cache/precision diagnostic; no verifier or parameter search.

Writes independent C4 dumps and strict C6 replay for fixed OOF checkpoints.
Output must not exist. Test CSVs are never read; weights/configs remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from reid.data.crop_cache import cached_model_input
from reid.eval.eval_local import FoldTruth
from reid.infer.extractor import Extractor
from reid.infer.run import extract_rows, read_inputs
from reid.model import load_config
from scripts.audit_release import file_record
from scripts.build_crop_cache import work as cache_work
from scripts.prepare_fusion_artifact import gallery_similarity_std
from scripts.replay_fusion import replay

VARIANTS = ("full_fp16", "full_fp32", "cache512_fp32", "cache1024_fp32")


def validate_components(components, fold):
    if type(fold) is not int or fold not in range(5):
        raise ValueError("Only train folds 0..4 are allowed")
    if len(components) < 2 or len({path.resolve() for path in components}) != len(components):
        raise ValueError("Provide at least two distinct OOF component directories")
    records = []
    for path in components:
        cfg = load_config(str(path))
        if cfg.get("train", {}).get("folds") != f"all except {fold}":
            raise ValueError(f"Component {path} must declare train.folds='all except {fold}'")
        if cfg.get("tta_flip", False) or cfg.get("bbox_pad", 0) > 0.15:
            raise ValueError("Diagnostic requires no TTA and bbox_pad <= CACHE_PAD 0.15")
        records.append({"path": str(path), "model_config": file_record(path / "model_config.yaml"),
                        "checkpoint": file_record(path / "weights.safetensors"), "config": cfg})
    return records


def train_rows(data_root, splits, fold):
    if type(fold) is not int or fold not in range(5):
        raise ValueError("Only train folds 0..4 are allowed")
    q = read_inputs(splits / f"fold{fold}_query.csv", data_root / "images")
    g = read_inputs(splits / f"fold{fold}_gallery.csv", data_root / "images")
    truth = FoldTruth.load(fold, str(splits / "folds.csv"))
    if [r.image_id for r in q] != truth.query_ids or [r.image_id for r in g] != truth.gallery_ids:
        raise ValueError("Fold CSV IDs/order disagree with committed fold truth")
    with (data_root / "train.csv").open(newline="") as handle:
        training = {r["image_id"]: tuple(float(r[k]) for k in ("x", "y", "w", "h"))
                    for r in csv.DictReader(handle)}
    for row in q + g:
        if row.image_id not in training or training[row.image_id] != row.bbox:
            raise ValueError("Only exact train.csv IDs and bounding boxes may be evaluated")
        if row.image_path.name != row.image_id + ".jpg":
            raise ValueError("Existing cache builder requires train <image_id>.jpg")
    if len({r.image_id for r in q + g}) != len(q) + len(g):
        raise ValueError("Query and gallery must be disjoint train image IDs")
    return q, g


def build_cache(rows, data_root, out, max_side, workers):
    (out / "img").mkdir(parents=True, exist_ok=False)
    tasks = [(r.image_id, *r.bbox, "train", str(data_root), str(out), max_side) for r in rows]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(cache_work, tasks))
    manifest = out / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    # Read back like the training extractor, including CSV numeric conversion.
    return [SimpleNamespace(**r) for r in pd.read_csv(manifest, dtype={"image_id": str}).to_dict("records")]


def cache_embeddings(extractor, cache, rows, batch_size, workers):
    blocks = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            tensors = list(pool.map(lambda row: cached_model_input(str(cache), row, extractor.cfg), batch_rows))
            batch = torch.stack(tensors).to(extractor.device)
            with torch.inference_mode(), torch.autocast(device_type=extractor.device.type, enabled=False):
                features = extractor.model.features(batch)
            blocks.append(torch.nn.functional.normalize(features.float(), dim=1).cpu().numpy())
    result = np.concatenate(blocks).astype(np.float32)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if not np.isfinite(result).all() or np.any(norms <= 0):
        raise ValueError("Invalid cache embeddings")
    return (result / norms).astype(np.float32)


def run(components, variants, fold, data_root, splits, out, device="cuda", batch=32,
        workers=8, scales=None):
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {out}")
    if not variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError("Variants must be a nonempty unique subset of supported paths")
    if batch <= 0 or workers <= 0:
        raise ValueError("Batch size and workers must be positive")
    if "full_fp16" in variants and (torch.device(device).type != "cuda" or not torch.cuda.is_available()):
        raise ValueError("full_fp16 requires a CUDA device; CPU cannot certify FP16")
    if scales is not None and (len(scales) != len(components) or any(not np.isfinite(s) or s <= 0 for s in scales)):
        raise ValueError("Need one positive finite fixed scale per component")
    sources = validate_components(components, fold)
    query, gallery = train_rows(data_root, splits, fold)
    rows = query + gallery
    out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(2026)
    caches = {}
    for variant in variants:
        (out / variant).mkdir()
        if variant.startswith("cache"):
            side = int(variant.removeprefix("cache").split("_")[0])
            cache = out / f"cache_{side}"
            caches[variant] = (cache, build_cache(rows, data_root, cache, side, workers))
    paths = {v: [] for v in variants}
    extraction = []
    for index, component in enumerate(components):
        extractor = Extractor(component, device=device, preprocess_workers=workers)
        extractor.model.float().eval().requires_grad_(False)
        try:
            for variant in variants:
                start = time.monotonic()
                if variant.startswith("full"):
                    extractor.amp_dtype = torch.float16 if variant == "full_fp16" else None
                    embeddings = extract_rows(extractor, rows, batch)
                else:
                    cache, cache_rows = caches[variant]
                    embeddings = cache_embeddings(extractor, cache, cache_rows, batch, workers)
                if embeddings.shape != (len(rows), int(extractor.cfg["embed_dim"])):
                    raise ValueError("Component output shape disagrees with C3 config")
                path = out / variant / f"component{index}_fold{fold}.npz"
                np.savez_compressed(path, query_ids=np.array([r.image_id for r in query]),
                                    gallery_ids=np.array([r.image_id for r in gallery]),
                                    query_emb=embeddings[:len(query)], gallery_emb=embeddings[len(query):])
                paths[variant].append(path)
                extraction.append({"component": index, "variant": variant, "seconds": time.monotonic() - start,
                                   "dump": str(path.relative_to(out)), **file_record(path)})
                print(f"component={index} variant={variant}: C4 saved", flush=True)
        finally:
            if extractor._preprocess_pool is not None:
                extractor._preprocess_pool.shutdown()
            del extractor
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
    if scales is None:
        scales = []
        for path in paths[variants[0]]:
            with np.load(path, allow_pickle=False) as data:
                scales.append(gallery_similarity_std(data["gallery_emb"]))
        scale_source = f"static train-fold gallery of first variant: {variants[0]}"
    else:
        scale_source = "explicit caller-provided fixed constants"
    result = {"kind": "train_fold_c3_gap_diagnostic", "fold": fold, "components": sources,
              "variants": variants, "device": device, "batch": batch, "workers": workers,
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "tf32": False, "similarity_std": scales, "scale_source": scale_source,
              "extraction": extraction, "results": {}, "training_provenance_verified": False,
              "scope": "Fixed-weight train-fold ablation; no verifier, threshold tuning, test CSVs, production edits or speed claim"}
    # The offline production image intentionally has no git binary or .git.
    try:
        result["code_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        result["code_commit"] = "unavailable; see host session source revision"
    result["diagnostic_script"] = file_record(Path(__file__))
    result["inputs"] = {str(p): file_record(p) for p in (data_root / "train.csv", splits / "folds.csv",
                        splits / f"fold{fold}_query.csv", splits / f"fold{fold}_gallery.csv")}
    for variant in variants:
        evaluation = replay(paths[variant], fold, out / variant / "replay", splits=splits,
                            scales=scales, rerank_weight=0, candidate_threshold=6.3)
        result["results"][variant] = {stage: record["metrics"] for stage, record in evaluation["stages"].items()}
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", type=Path, action="append", required=True)
    parser.add_argument("--variant", choices=VARIANTS, action="append")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--splits", type=Path, default=Path("splits"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--similarity-std", type=float, action="append")
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    run(args.component, args.variant or list(VARIANTS), args.fold, args.data_root, args.splits,
        args.out, args.device, args.batch, args.workers, args.similarity_std)


if __name__ == "__main__":
    main()
