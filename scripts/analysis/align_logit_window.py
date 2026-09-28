"""Align per-pair verifier logits to an authoritative production candidate order.

The operation is valid only when every query has exactly the same ten candidate
IDs in the source cache and target window. A changed candidate set is rejected:
missing pairs must be scored on GPU instead of being filled or approximated.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def align_logits(source: Path, target: Path):
    with np.load(source, allow_pickle=False) as src, np.load(target, allow_pickle=False) as dst:
        for key in ("fold", "query_ids", "candidates", "logits"):
            if key not in src.files:
                raise ValueError(f"source cache missing {key}")
        for key in ("fold", "query_ids", "candidates"):
            if key not in dst.files:
                raise ValueError(f"target window missing {key}")
        if src["fold"].shape != () or dst["fold"].shape != ():
            raise ValueError("fold must be scalar")
        if src["fold"].item() != dst["fold"].item():
            raise ValueError("source and target belong to different folds")
        source_qids = src["query_ids"].astype(str)
        target_qids = dst["query_ids"].astype(str)
        if source_qids.ndim != 1 or source_qids.tolist() != target_qids.tolist():
            raise ValueError("query IDs/order differ")
        source_candidates = src["candidates"].astype(str)
        target_candidates = dst["candidates"].astype(str)
        logits = src["logits"]
        expected = (len(source_qids), 10)
        if source_candidates.shape != expected or target_candidates.shape != expected:
            raise ValueError("both windows must have shape (N_query, 10)")
        if logits.shape != expected or logits.dtype != np.float32 or not np.isfinite(logits).all():
            raise ValueError("source logits must be finite float32 with shape (N_query, 10)")

        aligned = np.empty_like(logits)
        changed_rows = 0
        for index, (old, new) in enumerate(zip(source_candidates, target_candidates)):
            if len(set(old)) != 10 or len(set(new)) != 10:
                raise ValueError(f"duplicate candidate ID in row {index}")
            if set(old) != set(new):
                raise ValueError(f"candidate set differs in row {index}; rescore it on GPU")
            positions = {candidate: position for position, candidate in enumerate(old)}
            aligned[index] = logits[index, [positions[candidate] for candidate in new]]
            changed_rows += int(not np.array_equal(old, new))
        return int(src["fold"].item()), target_qids, target_candidates, aligned, changed_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error(f"refusing to overwrite {args.out}")
    try:
        fold, query_ids, candidates, logits, changed = align_logits(args.source, args.target)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"ALIGN FAILED: {exc}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, fold=np.array(fold), query_ids=query_ids,
                        candidates=candidates, logits=logits,
                        source_cache=np.array(str(args.source)),
                        target_window=np.array(str(args.target)),
                        reordered_rows=np.array(changed))
    print(f"fold {fold}: {len(query_ids)} rows, reordered {changed}; wrote {args.out}")


if __name__ == "__main__":
    main()
