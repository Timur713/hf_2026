# C3: precision и JPEG-кэш — измерено на A5000, 27.09.2026

**Итог:** в данном OOF fold0 сравнении перевод полного пути FP16→FP32 почти
не меняет mAP@10. Основная наблюдаемая разница связана с **путём JPEG-кэша**, а
не с FP16. Это диагностический результат, не принятие нового preprocessing.

## Результаты

Одинаковые три OOF checkpoint, pooling, batch32, фиксированные fusion scales;
полный train fold0. Для mAP учитываются 910 query с cross-camera позитивом из
1138 query (228 open-set), gallery — 636 кадров. Verifier отсутствует.

| Входной путь | Fusion mAP@10 | Fusion + hub mAP@10 | Δ hub к full FP32 |
|---|---:|---:|---:|
| Full-frame FP16 | 0.84434857 | **0.85818093** | −0.00001832 |
| Full-frame FP32 | 0.84445279 | **0.85819924** | 0 |
| JPEG cache512, FP32 | 0.84692360 | **0.86401479** | **+0.00581555** |
| JPEG cache1024, FP32 | 0.84511089 | **0.85948304** | **+0.00128380** |

Подтверждается более узкий вывод: **для этой тройки, fold0 и строгого top-10
падение около 0.005 не объясняется арифметикой FP16**. Это не утверждение о
побитной эквивалентности представлений или стабильности всех отдельных query.

Старый replay опубликованных C4 давал **0.8628516556195127**. Вновь построенный
cache512 даёт **0.8640147901040758**, то есть отличается на **+0.0011631345**.
Следовательно, старые C4 **не воспроизведены точно**. Причина остатка не
установлена: нельзя приписывать её версии библиотек, старому кэшу, pooling или
модели без дополнительного контроля. В старом replay автоматически вычисленные
scales также слегка отличаются численно от явно переданных констант этой
диагностики; это зафиксированное отличие, а не доказанная причина остатка.

Для сопоставления с [предыдущим полнокадровым контролем](../fusion_replay_20260925/README.md):
его FP16 baseline с исходными production scales был 0.85812729; при подстановке
C4 scales — 0.85818093. В настоящей таблице используются фиксированные C4-origin
константы, поэтому сравнивать FP16 следует со **вторым**, а не первым числом.

## Что именно сравнивалось

- DINOv2-B/14 input392 `cls_top16` (D=1536), CLIP-B/16 input288 `cls_top16`
  (D=1536), SigLIP2-B/16 input288 `top16` (D=768), сумма D=3840.
- Каждый YAML декларирует `train.folds: all except 0`. SHA checkpoint и полные
  конфиги сохранены в [raw/report.json](raw/report.json); all-data в этом запуске
  не принимались. Истинная история обучения/init/mining независимо не проверена
  (`training_provenance_verified: false`), metadata не объявляется доказательством.
- Full-frame: действующий `Extractor` с `model.features()` до BNNeck и L2.
  В варианте FP32 autocast выключен; TF32 выключен для всех вариантов.
- Cache: существующий `scripts/build_crop_cache.work`, `CACHE_PAD=0.15`,
  ограничение длинной стороны 512/1024, PIL bilinear, JPEG quality95; затем
  существующий `cached_model_input` и те же features/L2. Кэши построены заново
  только для query/gallery выбранного train fold, а не взяты из старого архива.
- Поэтому сравнение cache/full **совместно меняет** предварительное уменьшение,
  JPEG-перекодирование и преобразование координат кропа. Влияние каждой операции
  отдельно не изолировано. Нельзя назвать результат эффектом одного resize.
- Fusion scales одинаковы для всех четырёх вариантов:
  `0.05278994732695534`, `0.07144717974866033`, `0.002510766842444479`.
  Они не пересчитывались отдельно для выгодного варианта и не оценивались по test.
- Hub: top-3 gallery neighbours, weight0.5; query обрабатываются независимо.
  Выход каждого этапа проходит C6, затем `reid/eval/eval_local.py`: junk удаляется
  без добора десятки. Этот локальный оценщик ранее сверен с эталоном организаторов;
  отдельный запуск `third_party/official/evaluate.py` здесь не заявляется.

## Среда и воспроизведение

Сессия: NVIDIA RTX A5000, одна GPU, `cuda:0`; snapshot хоста показывает драйвер
535.54.03 / поддерживаемую драйвером CUDA12.2. В контейнере Torch `2.5.1+cu121`, NumPy `1.26.4`,
batch32, decode workers8, PyTorch CPU threads1. Base image `falcon-runtime:20260927`
имел ID `sha256:fd37c6b13ce3ef4e3ba33e92ea7fe9e6433219f1f6e469e92783572bcd97677e`.
Это **диагностический контейнер** с read-only source/artifact mounts, не
self-contained финальная Docker-приёмка. Время отдельных extraction в JSON —
не C5 benchmark, не FPS-оценка и не проверка общего тайм-аута.

Wrapper: [`scripts/fusion_c3_diagnostic_session.sh`](../../../scripts/fusion_c3_diagnostic_session.sh),
версия в `bc7b280`. Приготовить свежий session directory: `src/` с кодом этой
версии, `artifacts/release-cache/fusion-oof-v0/` с тремя OOF C3-парами и проверенным
`SHA256SUMS.txt`, `outputs/`; данные монтируются read-only. Затем:

```bash
SESSION_ROOT=/absolute/path/to/fresh-session \
DATA_ROOT=/absolute/path/to/dataset \
FUSION_BASE_IMAGE=falcon-runtime:20260927 \
bash scripts/fusion_c3_diagnostic_session.sh
```

Wrapper проверяет checksum исходных release-файлов, создаёт отдельные OOF
component directories и запускает `--network none` с явными параметрами:

```bash
PYTHONPATH=. python scripts/diagnose_c3_gap.py \
  --component /artifacts/oof_models/hires392 \
  --component /artifacts/oof_models/div_clip \
  --component /artifacts/oof_models/div_siglip \
  --data-root /data --fold 0 --out /out/c3_gap --device cuda:0 \
  --batch 32 --workers 8 \
  --similarity-std 0.05278994732695534 \
  --similarity-std 0.07144717974866033 \
  --similarity-std 0.002510766842444479
```

Wrapper и diagnostic отказываются перезаписывать существующие каталоги. В этом
документационном проходе GPU-прогон не повторялся: сохранены уже полученные JSON.
Git внутри контейнера отсутствовал, поэтому runtime `code_commit` указан как
unavailable. Вместо выдуманного SHA код связывается через записанный SHA скрипта
`7d6dad3a81af0c15e500a1f0542ec7964e8d3a8f3d03a4b730504456ae5dea21`, который
совпадает с `scripts/diagnose_c3_gap.py` в `bc7b280`. Это не утверждение, что
каждый файл смонтированного дерева побайтно совпадал с полным commit.

## Candidate threshold и границы решения

Для всех диагностических C6 применён фиксированный **threshold6.3**, verifier
отключён (`rerank_weight=0`, `logits=null`). Этот порог передан A для другого,
полного стека; здесь он **не калибровался**. CandidateScore в raw — сопутствующая
диагностика, не оценка финального all-data стека и не причина выбирать cache512.
Порог не влияет на `submission.csv` и mAP@10.

Разрыв cache512/full-frame всё ещё превышает контрактный допуск C3 0.002.
Контроль локализовал проблему до пути подготовки изображения, но **не закрыл
этот gate** и не разрешает самовольно менять shared preprocessing. Если команда
решит воспроизводить кэш на инференсе, нужны согласование A+B, matched-fold
проверка качества, повторная калибровка полного verifier-стека и новый замер
скорости. Одного fold0 недостаточно для утверждения общего выигрыша 0.0058.
Masked-control, финальные веса/три файла, query-independence и аппаратная
приёмка выбранного 2/3-model состава остаются отдельными проверками.

## Сохранённые свидетельства

Только небольшие исходные JSON, без массивов, изображений и dataset:

- [Общий отчёт, C3/config/checkpoint/input/C4 SHA](raw/report.json).
- Строгий C6 replay: [full FP16](raw/full_fp16.json), [full FP32](raw/full_fp32.json),
  [cache512 FP32](raw/cache512_fp32.json), [cache1024 FP32](raw/cache1024_fp32.json).

Каждый replay JSON включает метрики и SHA трёх C6-файлов для fusion/hub. Полные
дампы и выходы остаются в архиве сессии команды; JSON сами по себе не заменяют
их при независимом пересчёте.
