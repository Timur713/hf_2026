# Самостоятельный D-063 Docker на A5000, 27.09.2026

**Технический результат:** образ без сети и без host-source/model mounts
создал все три файла, прошёл C6 и два побайтно одинаковых полных прогона.
Рабочий input392 verifier включён. Однако общий сценарий завершился с
ненулевым кодом на **строгом сравнении confidence между разными batch shapes**;
не объявлять все acceptance gates зелёными.

## Точное происхождение

- Чистое tracked-source дерево из `git archive`
  `f47f8a7669c924616ee699fbbcd2f2cbbe887049`.
- Image `falcon-fusion:d063-rc1`, локальный Docker imageID
  `sha256:8d2a53783e391b5c3f98fca375db7a0bb56bc3d5a5d52ab5e52da745e81caaa9`.
  Это imageID, не опубликованный registry manifest digest.
- Archive SHA `ee364eb7bb965ff308b50975fab3bd74827d348a5659f49d1c44297a4be9d2f4`;
  manifestSHA `b47761be7a7ab03163e9d9f76f029abe9a4a9c0ac9a1f3c1634f57086c4eaf7a`.
- GitHub private prerelease `fusion-d063-a5000-20260927-rc1`: asset size и
  API digest независимо совпали с локально скачанным/серверным архивом.
- 1× RTX A5000, R535.54.03, CUDA runtime12.1/Torch2.5.1+cu121,
  CPU8vCPU/45GBRAM. Прямой R535 smoke теперь выполнен.

## Пройдено

| Проверка | Результат |
|---|---|
| Pytest внутри минимального image | 397passed,2skipped; пропущены только Git-history fixtures, поскольку git не включён в runtime |
| Локальный `make check` | 407passed +C6 selftest +реальный smoke50; все Git-history fixtures пройдены |
| Разница числа tests | 8 CPU replay tests живут в docs/, не входят в Docker COPY allowlist; это не скрытый пропуск runtime tests |
| Fusion+workingverifier smoke50 | PASS, все три C6 файла |
| Полные test inputs | 1110query/750gallery; только формат/детерминизм, без качества/подбора/анализа ошибок |
| Два independent reload/run | 132.216/132.305s pipeline; все3SHA совпадают |
| C6 | embeddings1860×3840float32, 1031answered/79refused,3093candidate rows; эти количества не оценка качества |
| All-data metadata/weight budget | PASS; 1,160,606,944bytes,metadata каждого embedder/verifier all-data |
| Train geometric masked-control | Все1138query/636gallery завершились, C6PASS, одна маска наembedder+verifier; all-data метрики не считались |
| Готовый image C5 | p50=31.599ms,p90=34.238ms; best81.445FPS,batch32; peak1667.67MiB,load4.064s |
| Полный pipeline peak | max1,878,423,552bytes ≈1791.41MiB allocated; это не totalprocessVRAM |

Сами raw JSON в этой директории неизменены. В контейнере нетgit, поэтому
runtime commit=null; связь с исходниками задаёт зафиксированный build выше.
Выполненный build использовал проверенный локальный private archive cache.
Зависимости/base допускают сеть при build. Попытка `--network none` для всей
сборки не смогла выполнить apt в новом network-cache key; это не запрет
инференса офлайн. Анонимная загрузка private Release намеренно недоступна.

## Query independence: точный результат, не завышенный PASS

Собственная query-группа24trainкадра,8добавочных, статическаяgallery636.
Gallery извлекалась один раз; batch1 сравнивался сbatch32, reverse и
interleaved extras. Во всех сравнениях:

- **Top10 IDs, порядок candidate IDs и решения принять/отказать совпали.**
- Confidence CSV отличается у17из24query; maxabsdifference **0.00142097**.
- Maxabs embeddingdifference **0.0002609342**.
- Строгое поле `passed=false` сохранено, exit1 не замаскирован.

Результат совместим с FP16 batch-shape rounding, но конкретный вычислительный
источник ещё не изолирован контролем. Кодовый аудит не нашёл cross-query
статистик, а входcamera_id отвергается. Это не доказательство нарушения правил,
но и не обещание побайтной независимости на всех данных: около порога/ничьих
такой drift потенциально может изменить решение. Не округлять confidence для
маскировки расхождения. Для окончательной приёмки согласовать критерий
численного допуска/уточнение организаторов и расширить контроль; тест24 не
заменяет проверку всех возможных входов.

Дополнительное попарное сравнение уже сохранённых данных уточняет источник:
**batch32 против reverse и против extraqueries полностью совпадает**, включая
эмбеддинги (maxabs0) и confidence (maxabs0). Отличается только singleton путь
против многокадрового. Меткаbatch32 означает максимальныйbatch: исходная группа
содержит24изображения, расширенная32. Это сильнее, чем одинаковыеIDs, но не
универсальное доказательство. Reproduction без новогоGPU:

```bash
PYTHONPATH=. python docs/benchmarks/fusion_a5000_20260927/acceptance/compare_batch_variants.py \
  --root /path/to/outputs/offline_d063/query_independence \
  --images dataset/images --out /path/to/new-comparison.json
```

Результат и хеши конкретных входов: `batch_variant_comparison.json`.

## C6 SHA-256

| Файл | SHA-256 |
|---|---|
| submission.csv | `9ed62e1a270daac8f61fa762c04c3b7a080035228499cbab68adc81d89fce7bc` |
| embeddings.npy | `66760947b2dc7a8daf23342e086b0267aa34e7f128082c1b13aeeaacdcc140cf` |
| candidates.csv | `17e6ddef2ad79515198999d4a908ba54ebee17934cd9cc8504e7bda7c1913905` |

Это выходы интеграционного кандидата. Канонический `submission/` не заменён,
пока не принят состав/качество. Все3файла сохраняются в архиве этой сессии.
Ни весовой all-data gate, ни проверкаформата не доказывают training provenance.

## Воспроизведение

Код и источник весов зафиксировать как выше; подготовить local archive cache
по `docs/fusion_docker.md`, существующий read-only dataset и новыйoutputroot:

```bash
BUILD_CONTEXT=/absolute/path/checkout DATA_ROOT=/absolute/path/dataset \
OUTPUT_ROOT=/absolute/path/new-output FUSION_IMAGE=falcon-fusion:d063-rc1 \
bash scripts/fusion_rc_acceptance_session.sh
```

Внутри: build →pytest→smoke50→двойнойrun/C6→train plain/masked→all-data
audit→C5→query independence. При confidence drift ожидается ненулевой выход,
rawreport следует прочитать, не переписать. A-quality/provenance/pooling,
реальное maskcoverage, публичный release/default и презентация остаются
отдельными пунктами в `docs/final_delivery_checklist.md`.
