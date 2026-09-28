# Финальная сдача: проверяемые ворота, а не наличие файлов

Срез **27.09.2026 после финальной A10-приёмки**: handoff A D-065/D-066,
build source`6a49664`,image`cf90a88d…fa89`,независимый Linux replay`02133e9`.
**B7.1/B7.2 технически закрыты по D-067; strict cross-shape numeric FAIL
сохранён, не назван PASS.** Это дополнение к PLAN, а не разрешение автоматически
публиковать или отправлять решение. [Фактические receipts](benchmarks/final_pair_acceptance_20260927/README.md).

## 1. Выбранный стек и что уже подтверждено

**D-065: DINO392 + CLIP288, D=3072, verifier input392, `rerank_weight=0.15`,
`candidate_threshold=6.6`, без SigLIP.** Порог 6.3 от тройки не переносится.

| Проверка | Подтверждённый факт | Граница подтверждения |
|---|---|---|
| Рецепт и калибровка A | A-104: matched C4 folds 0/1, mAP@10 0.852541, CandidateScore 0.960470 | Не OOF5, не метрика полного Docker-пути и не оценка all-data модели на train |
| Независимый replay B | `02133e9`: оба strict Linux/x86 replay с исходными окнами/логитами прошли, все метрики совпали с A | ARM near-tie отказ сохранён в истории; cross-platform побитовый confidence не обещается |
| Handoff данных | `fusion-pair-oof-v1` содержит C4, exact windows/logits и OOF verifier; необходимые дампы/логиты скачаны и сверены по SHA | OOF verifier checkpoint не пересчитывался в CPU-аудите B; происхождение обучения не доказывается одним названием файла |
| Финальные веса | Независимо сверены SHA DINO392, CLIP288 и рабочего input392 verifier; **788647632 bytes** суммарно | Это сумма выбранных весов; после упаковки повторить аудит всего `/app`, чтобы исключить лишние checkpoints |
| Исходные публичные веса | D-066: авторы, immutable URL/revision, SHA, byte-check A; B повторил HF metadata/registry consistency check | B не скачивал повторно 779 MB исходных tensors; metadata не является независимой реконструкцией истории обучения |
| Metadata-only C3 и sidecar | `97aa71b`: verifier provenance включается в SHA-манифест и строгую установку; C3 изменения только в `pretrained` | Веса и preprocessing не менялись; новый manifest/archive SHA должен соответствовать новому пакету |
| Default Docker / Compose | `3769e4f`, `0e9de2a`: default pair; image из`6a49664` реально собран и проверен offline | Обычный build без--target и два свежихdefault-run подтверждены; anonymous download ещё требует открытия repo |
| Локальная сборка пакета | `0e9de2a`: D3072,788647632 bytes,w0.15,t6.6; строгая USTAR-установка/all-data audit PASS,466tests+C6+реальный smoke50 PASS | Не Docker/GPU-приёмка; результаты в `benchmarks/final_pair_release_20260927/` |
| Канонический `submission/` | Финальная параD3072,четыре полных запуска дали одинаковыеSHA; C6PASS | СтарыеD1536 сохранены вGit history; ни тройкаD3840,ни ручная правка не использовались |

Проверяемые основания:

- [Калибровка пары A](benchmarks/fusion_pair_exact_20260927/README.md).
- [Исходные веса и исправленные YAML/sidecar](benchmarks/final_pair_provenance_20260927/README.md).
- [Независимый replay/provenance-аудит B](benchmarks/final_pair_replay_b_20260927/README.md).

**Отсутствие финальных весов, собственной калибровки пары и исходного provenance
больше не блокеры A.** Не требуем исправлять SigLIP `full_avg`/`top16`: SigLIP
не входит в выбранный состав. Остаточные проверки ниже относятся к
воспроизведению и приёмке выбранной пары, а не к ожиданию прежнего handoff.

## 2. Историческая A5000-сессия: что можно использовать, а что нельзя

- [x] `89c9e2e`: прямой all-data C5 на A5000/R535 — пара **102.674 FPS**,
  тройка 79.59–82.16 FPS. Это реальный замер JPEG→L2, не прогноз и не весь
  pipeline с verifier. Вместе с A-104 он обосновал выбор пары в D-065.
- [x] `754f8a7`: самодостаточная **тройка D-063** прошла offline C6 и два
  полных запуска 132.216/132.305 s с одинаковыми SHA трёх файлов. Это не
  приёмка нового pair image и не его каноническая сдача.
- [x] `754f8a7`, `e7a85b5`: на 24 train-query тройки top-10/кандидаты/отказы
  совпали; reverse/extras были побитно равны batch32. Singleton confidence
  отличался до 0.001421: **strict numeric gate FAIL**, не скрывать округлением.
- [x] `71e36e2`: C3-контроль отделил малый вклад FP16 от существенного
  cache/full-frame расхождения. Общая предобработка не изменялась.
- [x] `c56c138`: train mask QA показал реальные промахи геометрической маски.
  Технический masked-run не доказывает покрытие настоящих номерных областей.
- [x] `ca28020`: после backup и подтверждения пользователя прежняя A5000
  удалена. Новая A10-сессия отдельно согласована≤2h/85₽ и завершена27.09
  в20:36МСК сSHELVED_OFFLOADED,≈32мин/22₽. Обе GPU-сессии остановлены.

Сырые данные: [A5000](benchmarks/fusion_a5000_20260927/README.md),
[C3-контроль](benchmarks/c3_precision_cache_20260927/README.md),
[mask QA](benchmarks/train_plate_mask_qa_20260927/README.md),
[аудит query isolation](query_independence_code_audit_20260927.md).

## 3. Порядок закрытия B

### Пакет, Release и настройки

- [x] Принять D-065/D-066 и проверить реальные SHA финальных weights и C4
  (`4c734b3`), а не только сообщение коллеги.
- [x] Включить verifier provenance sidecar в строгую упаковку (`97aa71b`).
- [x] Переключить default Docker/Compose и smoke на fusion (`3769e4f`),
  сохранив явный `--target legacy`. Порог/вес берутся из pinned артефакта.
- [x] `0e9de2a`: собрать отдельный pair bundle и закрепить archive URL/SHA,
  manifest SHA и Release URL в Dockerfile. Model-v1/trio RC не перезаписаны.
- [x] 27.09 19:58МСК: pair archive опубликован, API digest совпал с локальным
  SHA; [receipt](benchmarks/final_pair_release_20260927/github_release.json).
- [x] `0e9de2a`: синхронизировать README с составом, pins и источниками всех
  трёх выбранных моделей. После открытия ещё проверить анонимный clean build.

### Реальная приёмка выбранного образа

- [x] `make check`:482image tests+3host-Gitskips,493local tests;
  required pair/verifier smoke50 PASS без пропуска модели.
- [x] Весь`/app`:all-data metadata/SHA audit PASS,788647632bytes,D3072,input392.
- [x] Четыре полныхoffline-run:дваharness+двасвежихdefault-контейнера,
  `--network none`,безhost code/weights; SHA всех3файлов равны,C6PASS.
- [x] Query train-control выполнен:замена24соседей/перестановкаbatch32
  сохраняет8якорей побитово. На24query singleton/batched IDs/отказы совпали,
  confidenceΔ≤0.000393; **strictFAIL сохранён**,никакого ослабления теста.
  Это bounded information-isolation evidence,не универсальная cross-shape побитовость.
- [x] Trainmasked-control1138/636 выполнен тем же стеком. Его PASS технический;
  реальное покрытие номерных зон остаётся ограничением,не доказано этим run.
- [x] C5 A10/R615:24.415ms/107.288FPS; fullpipeline127.768/121.619s,
  peak1436.692MiB. A5000/R535102.674FPS — отдельный прежний замер.
- [x] Канонический`submission/` заменён: `(1860,3072)`,float32,C6,
  SHA точно совпадают с final image и четырьмя полными прогонами.
- [x] Commit,image identity,manifest,input/output SHA,команды/логи сохранены
  в`benchmarks/final_pair_acceptance_20260927/`. Registry image не публиковался.
- [x] Backup SHA проверен,27.09 около20:36МСК A10SHELVED_OFFLOADED;
  GPU освобождён,108₽/месяц за сохраняемые ресурсы остаются.
- [x] B7.1/B7.2 закрыты по фактическим результатам и оговоркамD-067.
  Это не утверждение «все диагностические gates PASS». GradioB8.2 необязателен.

### Отдельные ограничения достоверности качества

- [x] Strict Linux/x86 replay обоихфолдов:исходные логиты приняты без
  выравнивания/подмены, все метрики совпали;`02133e9`.
- [ ] Проверить C4→full-frame parity и применимость выбранной калибровки на
  полном train-fold пути с теми же весами. Прежний разрыв тройки не является
  измерением пары, но его нельзя считать автоматически устранённым.
- [x] Сохранить фактические ограничения по номерной маске и численному drift,
  если они воспроизводятся. Не выдавать C4 twofold качество или успешный C6
  формат за доказательство качества полного GPU pipeline.

## 4. Люди совместно — доступ, шаблон, отправка

- [ ] При необходимости согласовать изменения общих контрактов; их
  [предложение](contracts_update_proposal.md) ничего автоматически не меняет.
- [ ] Сверить условия внешних ресурсов и возможность доступа жюри.
  Разрешение research/gated источников не даёт коммерческой лицензии.
- [ ] Проверить публикуемые историю, refs, releases и материалы на ключи,
  токены, dataset credentials и приватные вложения. История ещё содержит
  старые credential assignments; удаляющий commit не очищает прежние значения.
- [ ] Согласовать безопасную публикацию с сохранением приватного архива.
  Не открывать репозиторий, не переписывать общую историю и не публиковать
  архивы исходных данных автоматически.
- [ ] **Презентация — следующим этапом:** PDF/PPTX, слайды 7–11 строго в
  шаблоне, фактические метрики с fold/стеком/оборудованием; различать candidates
  и cross-camera retrieval. Twofold C4 не называть OOF5/full GPU quality.
- [ ] Скринкаст рабочего алгоритма, документация и ссылки; Gradio не обязателен.
- [ ] Уточнить у организаторов отсутствующий `extractor.py`, команду speed-test,
  доступные RAM/диск. Предварительный C5 интерфейс не выдавать за полученный файл.
- [ ] Перед freeze проверить анонимное чтение README/кода/weights и воспроизведение.
  По PLAN открытие — вечер 29.09, дедлайн **29.09 23:59 МСК**; отправить заранее
  и сохранить подтверждение приёма. После freeze не коммитить в сданную ветку.

## 5. Воспроизводимый CPU-контроль пары

После настоящего GPU-прогона и замены канонической сдачи:

```bash
PYTHONPATH=. python scripts/check_delivery_readiness.py \
  --query dataset/test_query.csv --gallery dataset/test_gallery.csv \
  --submission-dir submission --expected-dim 3072 \
  --bundle artifacts/fusion_bundle \
  --acceptance-c6 docs/benchmarks/final_pair_acceptance_20260927/c6.json \
  --scan-history --report outputs/delivery_readiness.json
```

`--acceptance-c6` — реальный результат `scripts/audit_release.py submission`
на выходах финального Docker-run, не вручную составленный JSON. Checker сверяет
SHA/размеры трёх файлов и входных CSV, но сам не доказывает их происхождение из
выбранного image, историю обучения или лицензии. Существующий report не
перезаписывается. Коды: `0` — нет blocked/missing автоматических проверок;
`1` — есть открытые проверки; `2` — ошибка запуска. Ручные review остаются.
`final_acceptance` всегда `false`: этот небольшой checker не заменяет всю приёмку.

[Исторический CPU-срез `479ecd7`](benchmarks/delivery_readiness_20260927/report.json)
сохранён для аудита, но его legacy/SigLIP/provenance blockers не являются текущим
статусом пары. Secret scanner показывает только path/type и явно перечисляет
пропуски; он не гарантирует отсутствие всех секретов.
