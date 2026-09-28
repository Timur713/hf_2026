"""Тесты локальной оценки. Протокол сверен с third_party/official/evaluate.py (A1.5)."""
import numpy as np
import pandas as pd
import pytest

from reid.eval.eval_local import FoldTruth, evaluate, ranking_from_embeddings, read_submission

# Игрушечный фолд:
#   v1: q1 (кам 1), gallery g1a (кам 1, junk для q1), g1b (кам 2, позитив)
#   v2: q2 (кам 3), gallery g2a (кам 3, junk), g2b (кам 1, позитив), g2c (кам 4, позитив)
#   v3: q3 (кам 5) — без пары, его gallery-кадр помечен drop
#   v4..v11: по одному кадру-дистрактору в gallery
ROWS = [
    ("q1", 1, 1, "query"), ("g1a", 1, 1, "gallery"), ("g1b", 1, 2, "gallery"),
    ("q2", 2, 3, "query"), ("g2a", 2, 3, "gallery"), ("g2b", 2, 1, "gallery"), ("g2c", 2, 4, "gallery"),
    ("q3", 3, 5, "query"), ("g3", 3, 5, "drop"),
] + [(f"x{i}", 3 + i, 1, "gallery") for i in range(1, 9)]
DISTRACTORS = [f"x{i}" for i in range(1, 9)]


@pytest.fixture
def folds(tmp_path):
    path = tmp_path / "folds.csv"
    pd.DataFrame(ROWS, columns=["image_id", "vehicle_id", "camera_id", "role"]).assign(fold=0).to_csv(
        path, index=False)
    return str(path)


def base_ranking():
    return {"q1": ["g1b"] + DISTRACTORS + ["g2b"],
            "q2": ["g2b"] + DISTRACTORS + ["g1b"],
            "q3": DISTRACTORS + ["g1b", "g2b"]}


def test_n_pos_counts_cross_camera_only(folds):
    assert FoldTruth.load(0, folds).n_pos() == {"q1": 1, "q2": 2, "q3": 0}


def test_junk_removed_before_truncation_without_backfill(folds):
    ranking = base_ranking()
    ranking["q1"] = ["g1a"] + DISTRACTORS + ["g1b"]  # junk на 1-м месте, позитив на 10-м
    res = evaluate(0, ranking, folds_path=folds)
    # После удаления junk позитив на 9-м месте из 9 оставшихся: AP(q1) = 1/9.
    # q2: позитив на 1-м месте, n_pos = 2 → AP = 1/2.
    assert res["map10"] == pytest.approx((1 / 9 + 1 / 2) / 2)


def test_ap_normalized_by_min_npos_10(folds):
    ranking = base_ranking()
    ranking["q2"] = ["g2b", "x1", "g2c"] + DISTRACTORS[1:]
    res = evaluate(0, ranking, folds_path=folds)
    assert res["map10"] == pytest.approx((1.0 + (1 + 2 / 3) / 2) / 2)


def test_openset_query_excluded_from_map(folds):
    res = evaluate(0, base_ranking(), folds_path=folds)
    assert res["n_query_eval"] == 2 and res["n_query_openset"] == 1
    assert res["rank1"] == 1.0


def test_candidate_modes(folds):
    cands = [("q1", "g1a", 0.9), ("q1", "g1b", 0.8),  # верхний — junk-двойник
             ("q3", "x1", 0.5)]                         # ответ на query без пары
    res = evaluate(0, base_ranking(), cands, folds_path=folds)
    # official: junk считается верным ответом (совпал vehicle_id) → q1 TP, q2 FN, q3 FP_open.
    off = res["official"]
    assert (off["tp"], off["fp"], off["fn"], off["tn"]) == (1, 1, 1, 0)
    assert off["f1"] == pytest.approx(0.5) and off["tnr"] == 0.0
    # cross_cam — диагностика: верхний junk не засчитывается.
    assert (res["cross_cam"]["tp"], res["cross_cam"]["fp"]) == (0, 2)
    assert res["candidate_score"] == pytest.approx(0.7 * 0.5)


def test_junk_stripped_from_full_list_then_truncated(folds):
    """Эталон снимает junk со всего поданного списка и только потом берёт первые 10."""
    ranking = base_ranking()
    ranking["q1"] = ["g1a"] + DISTRACTORS + ["g2b", "g1b"]  # 11 кандидатов, junk первым
    res = evaluate(0, ranking, folds_path=folds)
    assert res["map10"] == pytest.approx((1 / 10 + 1 / 2) / 2)  # позитив на 10-м месте после снятия junk


def test_query_missing_from_submission_gets_zero_ap(folds):
    ranking = base_ranking()
    del ranking["q1"]
    res = evaluate(0, ranking, folds_path=folds)
    assert res["map10"] == pytest.approx(0.5 / 2) and res["n_query_eval"] == 2


def test_candidate_tie_resolved_by_file_order_and_refusal_is_tn(folds):
    cands = [("q1", "x1", 0.9), ("q1", "g1b", 0.9)]
    off = evaluate(0, base_ranking(), cands, folds_path=folds)["official"]
    assert off["fp"] == 1 and off["tp"] == 0  # из равных взят первый — x1
    assert off["tn"] == 1 and off["tnr"] == 1.0  # q3 без строк — корректный отказ


def test_embedding_metrics_perfect(folds):
    truth = FoldTruth.load(0, folds)
    ids = truth.query_ids + truth.gallery_ids
    emb = np.stack([np.eye(32)[truth.vid[i]] for i in ids])
    res = evaluate(0, base_ranking(), emb=emb, folds_path=folds)
    assert res["full_map"] == pytest.approx(1.0) and res["minp"] == pytest.approx(1.0)


def test_real_fold_sanity_random_vs_oracle():
    """Случайные эмбеддинги дают mAP@10 около 0, оракул — ровно 1."""
    truth = FoldTruth.load(0)
    rng = np.random.default_rng(0)
    q, g = rng.normal(size=(len(truth.query_ids), 64)), rng.normal(size=(len(truth.gallery_ids), 64))
    rnd = evaluate(0, ranking_from_embeddings(q, g, truth.gallery_ids, truth.query_ids))
    assert rnd["map10"] < 0.05

    oracle = {}
    for qid in truth.query_ids:
        pos = [x for x in truth.gallery_ids if truth.is_pos(qid, x)]
        rest = [x for x in truth.gallery_ids if x not in pos]
        oracle[qid] = (pos + rest)[:10]
    assert evaluate(0, oracle)["map10"] == pytest.approx(1.0)


def test_read_submission_accepts_file_with_and_without_header(tmp_path):
    body = "q1,g1,g2\nq2,g2,g1\n"
    plain, with_header = tmp_path / "a.csv", tmp_path / "b.csv"
    plain.write_text(body)
    with_header.write_text("query_id,gallery_id_1,gallery_id_2\n" + body)
    assert read_submission(str(plain)) == read_submission(str(with_header)) == {
        "q1": ["g1", "g2"], "q2": ["g2", "g1"]}
