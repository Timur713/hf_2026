# D082: измерения и проверка поставки

Пользователь выбрал для сдачи D082 с gallery-z 6,6 и ускоренным extraction.
В этом snapshot ускорение подключено к штатному CLI, одиночному `extract()`
и production-бенчмарку через один `D082Extractor`.

Результаты разделены: **исходный ускоренный runtime измерен на A5000**;
**перенос в этот production snapshot проверен на H100 и локальном Python 3.10**.
Новое измерение именно собранного snapshot на A5000 и повторный Docker build
в этой сессии не выполнялись: соответствующих стендов здесь нет.
Скорость на закрытом тесте определят измерения организаторов.

## Скорость на настоящей A5000

Независимая проверка B: [отчёт bf11495](https://github.com/Timur713/hack_falcon_2026/blob/bf11495/docs/singleton_latency_acceptance_B_20261020.md),
точный измеренный runtime `57734078b52b783efd9c546dd0704e66343e74ac`.
RTX A5000 24 ГБ, 8 vCPU Xeon Gold 5315Y, driver 535.54.03,
Torch 2.5.1+cu121, Python 3.10; 8 intra-op threads / 8 preprocessing workers.
Контейнер без сети; deterministic algorithms ON, TF32 OFF.

Полный C5 с исходным `bench_extract.py:main` и source-bound адаптером,
1138 query / 636 gallery как инженерный набор:

| Повтор singleton | p50 batch=1, мс | Mean, мс | Batch32 FPS |
|---|---:|---:|---:|
| 1 | 38,740 | 39,115 | 124,040 |
| 2 | 38,880 | 39,130 | 121,611 |
| 3 | 39,950 | 40,141 | 122,467 |

[Raw 1](benchmarks/a5000_singleton_20261020/full-c5-singleton-1.json) ·
[Raw 2](benchmarks/a5000_singleton_20261020/full-c5-singleton-2.json) ·
[Raw 3](benchmarks/a5000_singleton_20261020/full-c5-singleton-3.json).
Baseline member-graph до/после: p50 40,604/40,611 мс,
122,703/122,548 FPS; throughput сохраняется, ускоряется singleton.
На train128 отдельные три C5-процесса дали p50 39,069–39,551 мс,
122,232–122,983 FPS.

Протокол ТЗ: 50 warmup / 300 batch1 замеров, синхронизация CUDA до/после,
медиана; batch 1/8/16/32 по ≥10 секунд, лучший FPS.
Таймер включает чтение JPEG, decode, crop, preprocessing, forward, L2/fusion.
Поиск, graph и verifier исключены. OS page cache не очищался после warmup.

**Ограничение:** другой experimental harness с предварительной проверкой
паритета дал p50 **41,006–41,321 мс** для того же исходного runtime.
[Исходный JSON](benchmarks/a5000_singleton_20261020/singleton.json).
Оба harness используют одну функцию измерения; причина зависимости от
подготовки процесса не доказана. Из всех результатов нельзя выбирать только
выгодные. Худший исходный C5 имеет всего 0,050 мс запаса до 40 мс.
Прохождение этих замеров не гарантирует тот же результат на стенде жюри.

## Сохранение результата и перенос

### Исходный A5000 runtime

- Native graph-pipeline и singleton: три файла побитно совпали на
  1138 query / 636 gallery; [сравнение SHA](benchmarks/a5000_singleton_20261020/stock-cli-parity.json).
- Reverse всех query, accepted/refused singleton, changed neighbours и
  permutation: exact на проверенных случаях; C6/D3072 и smoke50 PASS.
  [Инженерный отчёт](benchmarks/a5000_singleton_20261020/engineering.json).
- Linux/Python 3.10: 898 passed / 6 skipped. Legacy smoke SKIP;
  настоящий D082 smoke выполнен отдельно. Offline Docker PASS.

Полный архив B, без JPEG и весов:
[Release](https://github.com/Timur713/hack_falcon_2026/releases/tag/b-singleton-a5000-20261020),
127735064 байт, SHA256
`99d0999f6ecd4f2e125c1331fb9f6b663c2fbfe373f514d0c3781d0764a0622e`.
Выбранные raw JSON перенесены без редактирования; их происхождение и SHA:
[provenance.json](benchmarks/a5000_singleton_20261020/provenance.json).

### Текущий production snapshot

Убран внешний experimental adapter: штатный pipeline и C5 используют
`reid/infer/d082_extractor.py`. В нём оставлены принятые member CUDA Graphs,
bounded pipeline, deferred checks и singleton transfer. Неудавшийся
whole-fusion graph и исследовательские очереди не перенесены.

- H100: **76 targeted tests PASS**, включая сравнение реальных CUDA graphs
  с reference, partial batches, singleton capture, NaN/Inf/zero и recovery.
- Штатный `python -m reid.infer.d082_entrypoint`, 32 query / 625 gallery:
  три файла побитно равны reference и повтору; 7 accepted / 25 refused —
  только покрытие исполнения, не метрики качества.
- Reverse, accepted singleton, refused singleton и changed neighbours:
  embeddings query/gallery, ranking и confidence точны; C6 и D3072 PASS.
- Настоящий H100 `smoke-strict`: 25 query / 25 gallery, C6 PASS.
- Текущий source-bound C5 на H100 завершён; `actual_a5000_target_pass=false`,
  как и должно быть для другого оборудования. Скорость H100 не переносится
  коэффициентом в A5000.
- Локальный Python 3.10.22/macOS: `make check` exit 0,
  **715 passed / 12 skipped**, затем настоящий D082 smoke50 с имеющимися
  весами и синтетическими JPEG: C6 PASS. CUDA-тесты локально пропущены.

[Receipt локального make check](benchmarks/d082_singleton_integration_20261020/local-validation.json) ·
[Receipt H100](benchmarks/d082_singleton_integration_20261020/integration.json) ·
[Source-bound C5 H100](benchmarks/d082_singleton_integration_20261020/production-c5-h100.json).
Receipt содержит SHA проверенных исходников и трёх файлов.

Полный pytest на серверной файловой системе: 671 passed / 5 skipped,
3 failed / 48 fixture errors. Причина — `RENAME_NOREPLACE` не поддерживается
файловой системой в старом development publisher; runtime-инференс этой
операцией не пользуется. Ограничение не скрыто и не объявлено GPU QA PASS.
Полный набор на локальной поддерживаемой FS прошёл, целевые CUDA-тесты и
реальный H100 pipeline прошли отдельно. Docker отсутствует в текущей среде;
Docker-рецепт ранее проходил offline-проверку B, новый image не выдаётся
за повторно собранный и принятый на A5000.

## Воспроизведение текущего кода

Из собранного образа с SHA-проверенными весами:

```bash
docker compose run --rm --entrypoint python3 reid scripts/bench_final_candidate.py \
  --images /data/images --csv /data/test_query.csv \
  --run-label d082-singleton --json-out /out/c5.json
```

Исходники runtime, Docker-рецепт, lock-файлы и manifest проверяются по SHA.
При несовпадении выполнение останавливается; прежний source binding не
отключён, а обновлён на этот snapshot. C5 вызывает тот же production-класс,
который выбран штатным запуском. Лимит latency проверяется по `p50_ms`,
а не по среднему. Недостаточные warmup/iterations/duration не получают PASS.

Возврат 0 означает завершение измерения. Аппаратный gate:
`a5000_target.actual_a5000_target_pass`; полный отчёт сохранит и mean/p90.
Источник исходных таймеров — `scripts/bench_extract.py`, он не изменён.
Проверка три файла/C6 и всего pipeline выполняется отдельно от C5.

## Качество, отказы и номерные зоны

Matched development folds 0/1/3/4: mAP@10 0,8459950570 → 0,8521834625;
Rank-1 0,8159600361 → 0,8234031549;
Rank-5 0,9494968909 → 0,9492200272. Это результат OOF-моделей,
не независимая оценка финальных all-data весов. Цель Rank-1 0,90 не достигнута.

Порог 6,0 не улучшил независимую проверку 3/4. Закрытый однократный контроль
новой политики D141 завершился отказом от неё; оставлен явно разрешённый
fallback D082/gallery-z6,6. Он не обозначается CONTROL_PASS.
В этой интеграции обучение, настройка порогов и повторная quality-оценка
fold 2/test не выполнялись. All-data обучение после контроля разрешено отдельно.

Plate evidence: 40 аннотаций, 8 query / 32 gallery, 18 контрастов;
0 смен selected-ID/acceptance, 53 наблюдения изменения top-10,
max confidence delta 0,2468166351. Это in-sample all-data контроль;
разметка distractors неполная, human_attested=false, заливка могла задевать кузов.
Универсальная независимость от номера не доказана. B проверил привязку и
арифметику отчёта, но не повторял масочный GPU-контроль самостоятельно.

Состав двух выборок жюри и окончательное поведение evaluator для same-camera
не раскрыты. Локальные измерения не подменяют результаты закрытого теста.

## Неизменные артефакты и история

Manifest:
`94125e2665558df43fb6701f7f0ca6b5c1cf5289cbfd48c83c03d1ee0180305c`.
All-data bundle SHA:
`0920e24524e9b32215323bbe3f8e47ae6e19d1f036b14eacfa596507b83d95d5`.
788873516 байт весов; рецепт DINOv2-B392 + CLIP-B288, FP16 preBN,
fixed4, graph0,5/verifier0,15/gallery-z6,6, cache OFF, precast OFF.
Проверка provenance подтверждает предоставленные receipts и SHA,
а не независимо восстанавливает исторический процесс обучения.

Исходный неускоренный D082 19 октября: 93,228 FPS, повтор 93,098 FPS;
38,797 мс были **mean**, не p50. Его Python 3.10 issue с
`hashlib.file_digest` исправлен streaming helper и regression-тестом.
Исторические JSON сохранены в
[final_d082_acceptance_B_20261019](benchmarks/final_d082_acceptance_B_20261019/).
Полный JPEG→кандидат того запуска: p50 197,873 / p95 211,435 мс;
это другая зона таймера, не 100 FPS extraction.

Сентябрьские `submission/`, презентация и видео остаются материалами D065.
Новые submission/candidates/embeddings генерирует только текущий код из
входов организаторов; сохранённые D065 файлы не выдаются за результат D082.
