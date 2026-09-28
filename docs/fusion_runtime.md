# D7–D8: интеграция новых решений A в трек B

Дополнение 25.09: критический аудит A до `b41902f` и команды CPU replay —
[в отдельном отчёте](map10_growth_review_20260925.md). В `45106dc` добавлен
необязательный `rerank_weight`: CLI override → config → legacy 0,2.
Builder принимает `--rerank-weight` и записывает значение в пакет. Для рецепта
A-087 можно явно передать 0,15, но выбор не переносится автоматически на другие
модели. `check_determinism` и `run_plate_mask_control` также уважают config.
242 теста + C6 + real-image smoke пройдены; новые реальные веса не приняты.
Ниже исторический статус 24.09 и основной формат пакета.

Статус 24.09.2026, проверенный исходный код A — `51ecaa1`, код B — `7eab137`.
Перед публикацией также приняты свежие A-коммиты до `d5fde18` (part-based ветки,
эксперименты с selector и шумовой контроль D-058); они не содержат финальных весов
и не меняют выбранный здесь opt-in путь. Запись инженерного решения B — D-059.
`gh release list --repo Timur713/hack_falcon_2026` по-прежнему показывает только
`model-v1`. Новые исследовательские решения и загрузчики моделей добавлены, но
финальных файлов fusion/all-data в Release нет. SSH-псевдоним `lct-gpu` из
`handoff.md` на ноутбуке B не настроен; доступ к серверу A не предполагается.

## Что готово

- `Extractor` принимает обычный C3-каталог или явно выбранный fusion-пакет.
  JPEG/PNG декодируется **один раз на объект**, затем каждый компонент получает
  свою точную C3-предобработку. Признаки — `features()` до BNNeck. Результат:
  `L2(concat(L2(e_i) / sqrt(sigma_i)))`, размерность — сумма размерностей моделей.
  Коэффициенты фиксированы в пакете; query не изменяет их и не влияет на других query.
- D-055: gallery-only hub penalty, `k=3`, `lambda=0.5` для **concat-пути**.
  Сортировка — `z(cosine по gallery для одного query) − 0.5*z(hub)`.
- D-053: опциональный verifier сортирует свой top-10 по
  `z(base_score) + weight*z(raw_logit)`; обе нормировки только внутри top-10 одного query.
  Legacy weight=0.2; с 25.09 можно закрепить выбранное значение в пакете.
  Поддержаны checkpoint-головы `concat` и `corr`. Без verifier остаётся hub-порядок.
- D-057: confidence фактически выбранного кандидата — `(cos − mu_gg)/sigma_gg`,
  где статистика gallery–gallery **без диагонали**. Порог 6.0 — предварительный,
  по двум фолдам A. Это не вероятность и не прежний cosine-порог 0.600/0.583.
- Контрольная заливка номера применяется к общему кадру до всех компонентов.
  Верификатор контролируется своим существующим masked-путём.
- Сборщик пакета копирует только выбранные C3-файлы и optional verifier, фиксирует
  SHA-256/размеры, проверяет лимит 2 GB и metadata all-data по запросу. Повторная
  проверка выполняется по **скопированным** файлам. Неполный пакет не публикуется.
- `audit_release.py weights` принимает и fusion, считает все веса внутри `--root`,
  включая лишние экспериментальные файлы, и проверяет metadata каждого компонента.

Старые `model-v1`, Docker SHA/URL и raw-rerank по умолчанию **не заменены**.
Существующий Dockerfile/`release_acceptance.sh` пока упаковывают одиночный релиз,
не новый fusion-пакет. Их успешная приёмка B-015 не переносится на fusion.

## Формат пакета

```text
fusion/
  fusion_config.json
  manifest.json                 # отдельный аудит/происхождение, не runtime-конфиг
  components/0/{model_config.yaml,weights.safetensors}
  components/1/{model_config.yaml,weights.safetensors}
  components/2/{model_config.yaml,weights.safetensors}  # только для тройки
  verifier.pt                   # если включён явно
```

Runtime-схема строгая (единственный необязательный ключ — `rerank_weight`):
лишние/повторяющиеся JSON-ключи, внешние пути, вложенные
fusion, дубли путей, неверные коэффициенты или `tta_flip=True` завершаются ошибкой.
Поддержка TTA не заявляется. Пример двух моделей, **числа sigma здесь условные**:

```json
{
  "schema_version": 1,
  "name": "fusion2",
  "components": [
    {"path": "components/0", "similarity_std": 0.12},
    {"path": "components/1", "similarity_std": 0.08}
  ],
  "candidate_policy": "gallery_z",
  "candidate_threshold": 6.0,
  "rerank_mode": "standardized",
  "rerank_weight": 0.2
}
```

Реальные sigma создаёт сборщик по указанной статической gallery, **с диагональю**,
как в D-052. Это другая статистика, чем confidence D-057. Масштабы не переоцениваются
по запросам. A в экспериментах считает их на gallery каждого фолда; замороженные
масштабы пакета не обязаны давать те же числа на другой gallery. Перед финальным
релизом нужно согласовать происхождение фиксированных масштабов и проверить
переносимость именно этого пути. Сборка пакета не выбирает гиперпараметры и не
считает метрики теста.

## Команды после получения весов

Ниже `member0`, `member1`, `member2` — **реальные переданные C3-каталоги**, не имена
уже доступных Release-артефактов. Для честной оценки fold 0 все модели и verifier
должны быть обучены без fold 0. Сначала подготовить двойку; для тройки добавить
`--component artifacts/models/member2` и изменить `--out`/`--name`.

```bash
PYTHONPATH=. python scripts/prepare_fusion_artifact.py \
  --component artifacts/models/member0 --component artifacts/models/member1 \
  --images dataset/images --gallery splits/fold0_gallery.csv \
  --out artifacts/models/fusion2 --name fusion2 --device cuda \
  --verifier artifacts/models/verifier_f0/verifier.pt

PYTHONPATH=. python -m reid.infer.run \
  --model artifacts/models/fusion2 --verifier artifacts/models/fusion2/verifier.pt \
  --images dataset/images --query splits/fold0_query.csv \
  --gallery splits/fold0_gallery.csv --out outputs/fusion2-fold0 --device cuda
PYTHONPATH=. python -m reid.eval.eval_local --fold 0 \
  --submission outputs/fusion2-fold0/submission.csv \
  --candidates outputs/fusion2-fold0/candidates.csv \
  --embeddings outputs/fusion2-fold0/embeddings.npy --out outputs/fusion2-fold0/metrics.json

# Одинаковая машина A5000 и условия для двойки и тройки, один GPU cuda:0.
PYTHONPATH=. python scripts/bench_extract.py --model artifacts/models/fusion2 \
  --images dataset/images --csv splits/fold0_query.csv --device cuda:0 \
  --warmup 50 --iterations 300 --batch-sizes 1,8,16,32 --throughput-seconds 10 \
  --require-cuda --run-label a5000-fusion2 --json-out outputs/a5000-fusion2.json
```

Для финальных all-data компонентов использовать `--require-all-data` и отдельный
каталог. Нельзя оценивать их на train-фолдах как OOF. Аудит пакета с verifier:

```bash
PYTHONPATH=. python scripts/audit_release.py weights \
  --root artifacts/models/fusion-final --model artifacts/models/fusion-final \
  --verifier artifacts/models/fusion-final/verifier.pt --require-all-data \
  --report outputs/fusion-final-weights.json
```

После упаковки повторить аудит по **всему `/app`**, не только выбранному подкаталогу.
Metadata-проверка читает заявленное происхождение; она не доказывает качество обучения.

## Что осталось для B7.1/B7.2

1. Получить два/три выбранных C3 checkpoint+YAML для speed-проверки. Зафиксировать
   состав только после реального C5-замера на A5000: ≤40 ms batch=1 и ≥100 FPS.
   Старые 241 FPS одиночной модели не являются скоростью ансамбля.
2. Получить окончательные all-data веса каждого выбранного участника и verifier
   с правильным происхождением. Нужны закреплённые URL/SHA-256 и новые Release assets.
3. Подтвердить OOF5 и порог **полной цепочки**. Калибровка D-057 сейчас двухфолдовая
   и без verifier; код пересчитывает confidence после изменения top-1 verifier,
   но перенос прежнего score на этот вариант не утверждается.
4. Упаковать именно выбранный пакет в Docker (без прежних лишних весов), закрепить
   SHA/URL всех компонентов. Повторить offline build/run, полный C6/детерминизм,
   masked-control, общий runtime/VRAM и генерацию сдачи. Проверить ограничение 2 GB.
5. После этого закрыть B7.1/B7.2; B8.2 остаётся условным и пока отложен.

## Проверено в B, без оплачиваемого GPU

`make check PYTHON=.venv/bin/python` на `7eab137`: **217 tests passed**, self-test C6
и сквозной smoke на 50 реальных кадрах с доступным model-v1. Новая fusion-ветка
отдельно проходит реальный PNG/JPEG decode, синтетические C3 safetensors,
masked/unmasked pipeline → gallery policy → verifier → C6. Тесты покрывают формулы,
одиночный/батчевый/переставленный query, отрицательные входы, лимит весов и SHA.
Это инженерные проверки, **не новая mAP-оценка и не GPU/offline-Docker приёмка fusion**.
