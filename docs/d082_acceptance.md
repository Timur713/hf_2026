# D082: интеграция A/B и статус приёмки

**NOT_ACCEPTED.** Этот PR переносит финальный D082 в `hf_2026`, исправляет
совместимость вспомогательных SHA-функций с Python 3.10 и подключает новый
production smoke. Он не объявляет успешной повторную GPU-приёмку.

## Проверенная передача

Изучены полный отчёт B, commit `52b0a4e`, machine-readable summary,
три benchmark JSON, engineering, pipeline и plate-evidence audit.
Выбранные исходные JSON без редактирования сохранены в
[benchmarks/final_d082_acceptance_B_20261019/](benchmarks/final_d082_acceptance_B_20261019/).

- [Полный отчёт B](https://github.com/Timur713/hack_falcon_2026/blob/52b0a4e/docs/final_d082_acceptance_B_20261019.md).
- [Архив всех измерений B](https://github.com/Timur713/hack_falcon_2026/releases/tag/b-final-d082-acceptance-20261019):
  102983340 байт, SHA-256 `77f5278ac7458caced7d73b516390ad530bf2dea994731f3d23ff0eb92932865`.
- Исходный runtime A: `4f1bd6de87ac6dfd0b9cfa481142433fdedcabf3`,
  tree `59e887d7a0fbb20d789dc747da73f5c6a7b7118c`.
- Исходный образ B: `sha256:7a2226323af6d84382ef207a15f2ae5f3d2a515b6f135b720c4f1333d5e2c596`.
- Manifest: `94125e2665558df43fb6701f7f0ca6b5c1cf5289cbfd48c83c03d1ee0180305c`.

Исходные runtime-числа не переименовываются в результаты этого PR.
Перенесены только production-модули и необходимые инструменты/тесты;
очереди исследований, временные пути JH и экспериментальные модели исключены.
Форматирование приведено к Ruff, импорты проверены. Общие preprocess и splits
не менялись. Кэш verifier и projection precast остаются выключенными.

## Что подтвердил B на настоящей A5000

Одна RTX A5000, 8 vCPU, frozen train-fold0: 1138 query/636 gallery.
Все запуски контейнера с `--network none`. Это инженерный набор, не новая
OOF-оценка all-data весов. Fold 2 и качество test не анализировались.

| Gate | Результат исходного пакета |
|---|---|
| SHA/source, manifest/provenance, весовой бюджет | PASS с ограничением исторического аудита |
| Docker build + фактический offline | PASS |
| Три файла, C6/D3072, два одинаковых полных запуска | PASS |
| Обратный порядок всех query и выбранные singleton/neighbours/permutation | PASS, точные confidence/embedding на проверенных вариантах |
| Реальный smoke50 | PASS |
| Mean extraction batch=1 ≤40 мс | PASS: 38,7971 мс |
| Максимальный throughput ≥100 FPS | FAIL: 93,2278 FPS |
| Linux exact-runtime `make check` | FAIL: 15 failed, 824 passed, 6 skipped, 3 errors |

Все 18 problem cases Linux-проверки вызваны `hashlib.file_digest`, которого нет
в Python 3.10.12. В перенесённых инструментах используется streaming helper
`scripts/hashing.py`; тест удаляет API Python 3.11 и проверяет в том числе
поток больше двух чанков. Повторная проверка именно Linux/CUDA-образа ещё нужна.

| Команда B | Mean batch=1, мс | Max FPS | Результат |
|---|---:|---:|---|
| Threads8/workers8 | 38,7971 | 93,2278 | FAIL throughput |
| Повтор threads8/workers8 | 38,8604 | 93,0985 | FAIL throughput |
| Threads1/workers2 | 46,1419 | 59,2960 | FAIL latency + throughput |

C5: диск/JPEG → crop → preprocessing → оба encoder → production L2/fusion.
Модельная загрузка, поиск, graph и verifier исключены.
Полный JPEG→кандидат: p50 **197,873 мс**, p95 **211,435 мс**;
verifier mean144,647 мс, extraction mean48,268 мс, прочий post mean4,680 мс.
Ускорение verifier само по себе не устранит недобор FPS в C5.
Дополнительная команда имеет полный JSON, но пустой stdout из-за разрыва SSH;
этот неуспех не скрыт. Арендованная A5000 удалена B, повтор требует нового стенда.

## Качество и ограничения

Историческое matched development сравнение на folds0/1/3/4:
mAP .8459950570→.8521834625; Rank1 .8159600361→.8234031549;
Rank5 .9494968909→.9492200272. Это не оценка финальных all-data весов.
Цель Rank1 .90 не достигнута. Порог6.0 не улучшил независимую проверку3/4.

Однократный закрытый контроль D082+D141 уже выполнен и отклонил новую политику
отказов по candidate-quality. D082+gallery-z6.6 — явно разрешённый fallback,
а не CONTROL_PASS. Повторять fold2 quality или подбирать по его результатам
новый порог запрещено. All-data обучение после контроля разрешено отдельно.

Plate evidence: 40 аннотаций, 8 query/32 gallery, 18 контрастов,
0 смен selected-ID/acceptance, но 53 наблюдения изменения top10 и max confidence
delta .2468166351. Данные all-data/in-sample; distractors размечены не полностью,
human_attested=false; возможно захватывание пикселей кузова.
B подтвердил привязку и арифметику предоставленного отчёта, а не самостоятельно
повторил масочный GPU-контроль. Универсальная независимость от номера не доказана.

## Повторная приёмка этого PR

1. Из чистой ветки собрать pinned D082 образ, проверить bundle SHA и весовой бюджет.
2. В QA-среде Python3.10/CUDA с Git выполнить pytest и `make check`, затем
   настоящий `smoke-strict` на 50 изображениях. Тесты создают временные Git-репозитории:
   QA должен содержать Git, production runtime не требует его для инференса.
3. На одной видимой A5000 повторить C5 тем же batch-планом1/8/16/32,
   сохранить все результаты. Выход0 benchmark означает завершённое измерение;
   целевой PASS находится в отдельном `a5000_target.actual_a5000_target_pass`.
4. Повторить полный offline/C6, детерминизм и независимость query, сохранить SHA.
5. Обновить статус и согласовать merge/замену D065. Этот PR остаётся draft до решения.

Пример C5 из собранного образа, с train-only `bench_query.csv` во входном root:

```bash
docker compose run --rm --entrypoint python3 reid scripts/bench_final_candidate.py \
  --images /data/images --csv /data/bench_query.csv \
  --run-label d082-integration-r2-a5000 --torch-threads 8 --preprocess-workers 8 \
  --json-out /out/d082-integration-r2-c5.json
```

Benchmark перенесён из SHA-проверенного wrapper A
(`1a19c7c5f49f72e10d2c7407e784624d564e42b6107197af701ee65758ba00db`).
Production-арифметика, fixed4, таймеры и границы измерения сохранены;
исходная source binding заменена SHA перенесённых файлов этой ветки.
Это новый benchmark source snapshot, не byte-identical wrapper исходной приёмки.
Новые числа скорости локально на CPU не заявляются.

Сентябрьские `submission/` и материалы защиты не перегенерированы.
Не следует отправлять их как результат D082; новые выходы должен сформировать
только код после финального решения команды. Слияние автоматически не выполняется.

## Проверки этой интеграции

Локальный Python3.10.22/macOS ARM64: `make check` exit0, **693 passed, 5 skipped**,
реальные веса D082 + 25query/25gallery синтетических JPEG, C6/D3072 PASS.
Это synthetic engineering smoke, не quality: все25query отклонены.
Linux/CUDA и скорость повторно не проверены. [Лог и receipt](benchmarks/d082_integration_20261019/).
AST-сравнение13 production-модулей с upstream совпало; в run удалён только
неиспользуемый импорт. Smoke изменён намеренно. AST-проверка не заменяет
численную GPU-приёмку.
