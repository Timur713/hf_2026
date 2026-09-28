"""Train-only audit of the frozen A-091 recipe, not an inference implementation.

Run from the repository root with PYTHONPATH=.; no parameter search, GPU, images,
or verifier. All reported ranking metrics are delegated to eval_local.evaluate.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from reid.eval.eval_local import FoldTruth, evaluate, read_submission
from reid.post.gallery_policy import GalleryPolicy
from scripts.analysis.hub_penalty import concat_parts
from scripts.analysis.calibrate_fusion_threshold import fused as calibration_fused
from scripts.audit_release import file_record
from scripts.replay_fusion import fuse_components, load_components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, action="append", required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if any(path.name != "fold0.npz" for path in args.dump):
        raise ValueError("this fixed audit accepts fold0 dumps only")
    truth = FoldTruth.load(0)
    qids, gids = truth.query_ids, truth.gallery_ids
    q, g, scales = fuse_components(load_components(args.dump, qids, gids))
    b_policy = GalleryPolicy(g, gids)
    b_matrix = b_policy.ranking_scores(q)
    b_rankings = b_policy.search(q).ranking
    serialized = read_submission(str(args.replay / "hub/submission.csv"))
    if {qid: row for qid, row in zip(qids, b_rankings)} != serialized:
        raise ValueError("computed production rankings disagree with serialized replay")
    a_matrix, hub, aqids, agids = concat_parts([str(p.parent) for p in args.dump], 0, 3)
    if (aqids, agids) != (qids, gids):
        raise ValueError("A concat_parts IDs differ")
    a_matrix = a_matrix - np.float32(0.5) * hub[None, :]
    a_order = np.argsort(-a_matrix, axis=1)
    a_full = {qid: [gids[j] for j in a_order[i]] for i, qid in enumerate(qids)}
    a_strict = {qid: row[:10] for qid, row in a_full.items()}
    a_refilled = {qid: [gid for gid in row if not truth.is_junk(qid, gid)][:10]
                  for qid, row in a_full.items()}
    npos = truth.n_pos()
    c_score, c_conf, cqids, cgids = calibration_fused([str(p.parent) for p in args.dump], 0, 3)
    if (cqids, cgids) != (qids, gids):
        raise ValueError("A calibration function IDs differ")
    c_top = np.argmax(c_score, axis=1)
    c_candidates = [(qid, gids[c_top[i]], float(c_conf[i, c_top[i]]))
                    for i, qid in enumerate(qids) if c_conf[i, c_top[i]] >= 6.0]

    def summarize(ranking):
        # Labels below construct a non-deployable upper bound / coverage only.
        # They never change actual replay outputs or inference policy.
        evaluable = [qid for qid in qids if npos[qid] > 0]
        covered = sum(any(truth.is_pos(qid, gid) for gid in ranking[qid])
                      for qid in evaluable)
        oracle = {qid: sorted(row, key=lambda gid: not truth.is_pos(qid, gid))
                  for qid, row in ranking.items()}
        return {
            "metrics": evaluate(0, ranking),
            "oracle_reordering_only": evaluate(0, oracle),
            "n_evaluable": len(evaluable),
            "n_with_positive_in_window": covered,
            "positive_window_coverage": covered / len(evaluable),
        }

    # Independently reproduce the counted failure population of rank_causes.py.
    strict_fail, refilled_fail = set(), set()
    for qid in qids:
        if not npos[qid]:
            continue
        for source, target in ((a_strict, strict_fail), (a_refilled, refilled_fail)):
            kept = [gid for gid in source[qid] if not truth.is_junk(qid, gid)]
            if not truth.is_pos(qid, kept[0]) and any(truth.is_pos(qid, gid) for gid in kept):
                target.add(qid)

    report = {
        "kind": "A091_formula_and_window_audit_train_only",
        "fold": 0,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "audit_script": file_record(Path(__file__)),
        "source_functions": {
            name: file_record(Path(name)) for name in (
                "scripts/analysis/hub_penalty.py", "scripts/analysis/rank_causes.py",
                "scripts/analysis/calibrate_fusion_threshold.py",
                "reid/post/gallery_policy.py", "reid/eval/eval_local.py")},
        "dumps": [{"path": str(path), **file_record(path)} for path in args.dump],
        "replay_report": file_record(args.replay / "report.json"),
        "similarity_std": scales,
        "formula": "per-component fixed gallery std; concat+L2; per-query z; one gallery-only hub, k=3, lambda=0.5",
        "matrix_max_abs_A_vs_B": float(np.max(np.abs(a_matrix - b_matrix))),
        "top1_differing_queries_A_vs_B": sum(a_strict[qid][0] != serialized[qid][0] for qid in qids),
        "top10_differing_queries_A_vs_B": sum(a_strict[qid] != serialized[qid] for qid in qids),
        "production_C6": summarize(serialized),
        "A_concat_strict_C6": summarize(a_strict),
        "A_rank_causes_refilled_window_NOT_C6": summarize(a_refilled),
        "A_fixed_threshold_6_no_search": evaluate(0, a_strict, c_candidates),
        "failure_population": {
            "strict_C6": len(strict_fail), "rank_causes_refilled": len(refilled_fail),
            "added_by_refill": len(refilled_fail - strict_fail)},
        "limitations": [
            "No verifier logits were supplied or evaluated.",
            "No BBox correction was implemented, selected, or evaluated.",
            "Coverage/oracle use train labels only and are not deployable rankings.",
            "This is not independent certification of C4 training/extraction provenance.",
            "No test analysis, full-frame embedding extraction, plate control, or GPU benchmark."],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"A": report["A_concat_strict_C6"]["metrics"],
                      "B": report["production_C6"]["metrics"],
                      "difference": report["matrix_max_abs_A_vs_B"],
                      "windows": report["failure_population"]}, indent=2))


if __name__ == "__main__":
    main()
