"""Инварианты закоммиченных фолдов (контракт C1). Фолды не пересоздаются после Д1."""
import pandas as pd


def test_folds_invariants():
    df = pd.read_csv("splits/folds.csv", dtype={"image_id": str})
    assert len(df) == 9556 and df.image_id.is_unique
    assert set(df.fold) == {0, 1, 2, 3, 4}
    assert (df.groupby("vehicle_id").fold.nunique() == 1).all(), "машина попала в несколько фолдов"
    kept = df[df.role != "drop"]
    # В gallery максимум один кадр на пару (машина, камера).
    assert (kept[kept.role == "gallery"].groupby(["vehicle_id", "camera_id"]).size() == 1).all()
    for k, f in df.groupby("fold"):
        q = f[f.role == "query"]
        g_vids = set(f[f.role == "gallery"].vehicle_id)
        share = (~q.vehicle_id.isin(g_vids)).mean()
        assert 0.18 <= share <= 0.22, f"fold {k}: доля query без пары {share:.3f}"
