"""Пересчёт всех материалов защиты одной командой.

Модель ещё будет меняться, а числа в `docs/defense_notes.md` и картинки в `docs/errors/` должны
оставаться верными. Скрипт прогоняет весь набор замеров на текущих дампах и печатает сводку,
которую остаётся перенести в конспект защиты.

Предполагается, что эмбеддинги уже извлечены:
    PYTHONPATH=. python scripts/extract_embeddings.py --model <модель> --pools cls_avg

    PYTHONPATH=. python scripts/analysis/refresh_defense.py --model artifacts/models/final_all_data
"""
import argparse
import subprocess
import sys
import time

import numpy as np

from reid.eval.eval_local import evaluate, ranking_from_embeddings

METRICS = ("map10", "rank1", "rank5", "full_map", "minp")


def fold_metrics(emb: str) -> None:
    """Сводка по пяти фолдам. `eval_dumps.py` ждёт все фолды в одном каталоге, а OOF-дампы
    разложены по каталогу на фолд, поэтому считаем здесь."""
    per_fold = []
    for k in range(5):
        d = np.load(f"{emb.format(k=k)}/fold{k}.npz")
        q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()
        ranking = ranking_from_embeddings(d["query_emb"], d["gallery_emb"], g_ids, q_ids)
        sim = d["query_emb"] @ d["gallery_emb"].T
        cands = [(q, ranking[q][0], float(sim[i].max())) for i, q in enumerate(q_ids)]
        per_fold.append(evaluate(k, ranking, cands, np.concatenate([d["query_emb"], d["gallery_emb"]])))
    for m in METRICS:
        vals = [r[m] for r in per_fold]
        print(f"{m:9s} {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    score = [0.7 * r["official"]["f1"] + 0.3 * r["official"]["tnr"] for r in per_fold]
    print(f"{'кандидаты':9s} {np.mean(score):.4f} ± {np.std(score):.4f}  (наивная политика top-1)")

# Шаг: (заголовок, аргументы, обязателен ли). Необязательные шаги требуют модель на диске
# и пропускаются, если её нет, — числовая часть от них не зависит.
def steps(model: str, emb: str) -> list[tuple[str, list[str], bool]]:
    return [

        ("Ошибки режима кандидатов + картинка",
         ["scripts/analysis/candidate_errors.py", "--emb", emb], True),
        ("Утечка фона в выдаче",
         ["scripts/analysis/bg_leak.py", emb], True),
        ("Стресс-тест открытого множества",
         ["scripts/analysis/openset_stress.py", "--emb", emb], True),
        ("Сравнение политик уверенности",
         ["scripts/analysis/policy_compare.py"], True),
        ("Карта соответствия патчей",
         ["scripts/interpret_pairs.py", "--model", model, "--rows", "4"], False),
        ("Карта чувствительности",
         ["scripts/occlusion_map.py", "--model", model], False),
    ]


def run(title: str, argv: list[str], required: bool) -> bool:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, *argv], capture_output=True, text=True)
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-3:]
        print("НЕ ВЫПОЛНЕНО:", " | ".join(tail))
        return not required
    # печатаем только содержательные строки, без предупреждений библиотек
    for line in proc.stdout.splitlines():
        if line.strip() and "Warning" not in line:
            print(line)
    print(f"[{time.perf_counter() - t0:.0f} с]")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/models/final_all_data")
    ap.add_argument("--emb", default="artifacts/emb_final/final_oof_f{k}_cls_avg")
    args = ap.parse_args()

    print(f"{'=' * 72}\nМетрики на пяти фолдах\n{'=' * 72}", flush=True)
    fold_metrics(args.emb)

    failed = [t for t, argv, req in steps(args.model, args.emb) if not run(t, argv, req)]
    print(f"\n{'=' * 72}")
    if failed:
        print("Незакрытые шаги:", ", ".join(failed))
        return 1
    print("Все замеры пройдены. Перенести числа в docs/defense_notes.md и docs/errors/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
