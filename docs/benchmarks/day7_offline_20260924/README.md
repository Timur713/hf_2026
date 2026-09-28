# B7/B8: офлайн-приёмка после A-053 / D-051, 24.09.2026

**PASS для опубликованной model-v1 + verifier_f0**, код `357920c`.
Это техническая приёмка по разрешённому в PLAN B7.1 сценарию, **не приёмка
неопубликованных final_all_data/verifier_final** и не H9.1 на A5000/R535.

## Происхождение и воспроизведение

- 1× NVIDIA A10, 8 vCPU, 32 GB RAM, драйвер 615.71.09.
- PyTorch 2.5.1+cu121, CUDA runtime 12.1, 8 preprocess workers.
- Полный source SHA: `source_commit.txt`; образ: `image_inspect.json`.
  Поле `commit: null` внутри JSON ожидаемо: `.git` не копируется в образ.
- `compose_resolved.yml`: `network_mode: none`, dataset read-only.
- `pip_freeze.txt`: фактически установленные версии; lock в корне репозитория.
- Веса и конфиг: `weights_audit.json`, фиксированный Release `model-v1`.
  Модель обучена без fold 0, verifier — на folds 1–4; порог **0,600**.
- `Extractor` использует features + L2 **до BNNeck** (D-051).

```bash
DATA_ROOT=./dataset OUTPUT_ROOT=./outputs/release-aligned \
  bash scripts/release_acceptance.sh
OUTPUT_ROOT=./outputs/release-aligned \
  QUERY_CSV=/app/splits/fold0_query.csv \
  GALLERY_CSV=/app/splits/fold0_gallery.csv OUT_DIR=/out/fold0 \
  docker compose run --rm reid
OUTPUT_ROOT=./outputs/release-aligned docker compose run --rm \
  --entrypoint python3 --env PYTHONPATH=/app reid -m reid.eval.eval_local \
  --fold 0 --submission /out/fold0/submission.csv \
  --candidates /out/fold0/candidates.csv --embeddings /out/fold0/embeddings.npy \
  --out /out/fold0/metrics.json
```

Выходной каталог создаётся скриптом и не должен уже существовать. Полный лог —
`acceptance.log`; оболочка завершилась с кодом 0. После Docker build прошли
**93 pytest-теста, C6 self-test и smoke на 50 реальных изображениях**.
При обкатке исправлены отсутствие `make` в образе (`cd1a47f`) и сброс счётчиков
CUDA до её ленивой инициализации (`7031f39`); окончательный прогон успешен.

## Скорость и память

| Измерение | Результат |
|---|---:|
| Latency extract, batch=1, p50 / p90 | 16,865 / 18,719 ms |
| FPS, batch=8 / 16 / 32 | 195,798 / 207,495 / 218,542 |
| Загрузка эмбеддера в benchmark | 1,103 s |
| Peak allocated VRAM эмбеддера | 834,903 MiB |
| Peak allocated / reserved всей связки | 958,359 / 1022 MiB |
| Полный pipeline в двух повторных прогонах | 77,251 / 76,270 s |
| Все файлы весов внутри `/app` | 442 853 672 bytes < 2 GB |

Benchmark: `bench_extract.json`, 50 прогревов, 300 замеров batch=1,
не менее 10 s на каждый batch. В `extract` включены чтение полного кадра,
decode/crop/resize/forward/L2; поиск и verifier измеряются полным pipeline,
а не включаются в latency extract. Память связки из `report.json` — счётчики
Torch allocator, **не вся память процесса/драйвера**. A10 не подменяет A5000;
эти цифры не являются новым замером на эталонном GPU или R535.

## Полные входы и детерминизм

`format_audit.json` относится к **новому GPU-run model-v1**: 1110 query,
750 gallery, embeddings `(1860, 1536)` float32. Все три файла проходят C6.
`report.json` подтверждает побайтное равенство двух полных повторных прогонов
с повторной загрузкой обеих моделей **в одном Python-процессе**. SHA этих
трёх файлов совпадают также с отдельным обычным production-run из
`format_audit.json`. Это не обещание равенства CPU/GPU и разных версий.

`submission_format.json` — **отдельный** аудит файлов трека A из `submission/`
на момент проверки (984 ответа, 126 отказов). Они не заменены выходами model-v1.
Все проверки test — только inference/формат/скорость/целостность; никаких
метрик test, анализа ошибок, подбора весов или порога на этих данных нет.

## Контрольная метрика на train-fold 0

`fold0_metrics.json` получен только через `reid.eval.eval_local` на фиксированном
train-фолде, после полного GPU-инференса из кадров и с verifier:

| Метрика | Результат |
|---|---:|
| mAP@10 по reranked top-10 | 0,799708 |
| Rank-1 / Rank-5 | 0,769231 / 0,938462 |
| Candidate F1 / TNR (official) | 0,963214 / 0,973684 |
| Candidate score (official) | 0,966355 |

Это **один fold, опубликованные старые веса**, не среднее по пяти фолдам,
не all-data модель и не повтор A-053 на `final_oof_f0`. Порог не подбирался.
Диагностический cross-camera F1 всего 0,025862: по официальной трактовке
same-camera двойник засчитывается верным кандидатом, а по cross-camera — нет.
Это известное ограничение D-023/D-048/D-050; три строки кандидатов не доказывают
устойчивость к иной политике оценщика. `full_map`/`minp` в JSON вычислены по
эмбеддингам без verifier, в отличие от mAP@10 по submission.

## Бэкап и оставшиеся условия

Полные выходы, включая embeddings и логи промежуточных прогонов, сохранены
локально в игнорируемом `outputs/day7_offline_20260924/`; в git только отчёты.
Архив `falcon-day7-day8-20260924-results.tar.gz` проверен на обеих сторонах:
SHA-256 `58cccbe3e4f72249f64f726edee647ddcd38487beef9da32c9af91588af394dd`.

Для финального B7.1/B7.2 остаются три артефакта выбранного A запуска,
их новые SHA/Release/URL, `--require-all-data` и повторная приёмка/генерация
сдачи тем же production-кодом. Детали — [release_handoff.md](../../release_handoff.md).
Gradio B8.2 отложен согласно условию PLAN «только если всё выше готово».
