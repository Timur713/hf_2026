# Day 7: формат сдачи проверен, GPU-приёмка ожидает веса

Это исторический отчёт за 23.09 для прежних файлов. Актуальная структурная
проверка после D-048/D-050 — [24.09](../day7_offline_20260924/submission_format.json).
По обновлённому PLAN B7.1 техническая GPU-приёмка разрешена на model-v1;
публикация финальных весов и сверка OOF/production (BNNeck) остаются отдельными пунктами.

Исходные `submission/` взяты из коммита трека A `c5fa7d7`, без изменений.
Проверка B выполнена кодом `6a9ed5f`:

```bash
PYTHONPATH=. .venv/bin/python scripts/audit_release.py submission \
  --query dataset/test_query.csv --gallery dataset/test_gallery.csv \
  --submission-dir submission --report outputs/final-format-report.json
```

Результат в `format_report.json`: 1110 query, 750 gallery, 970 ответов,
140 отказов, embeddings `(1860, 1536)` float32. Сохранены размеры и SHA-256
каждого файла и входных CSV. Это проверка формата и целостности, не оценка
точности на test и не доказательство воспроизводимости инференса.

## Что требуется для завершения B7.1/B7.2

На 23.09.2026 в Release `model-v1` только старые артефакты:
`vitb14_in280_ep100.safetensors`, его YAML и `verifier_f0.pt`.
Локальные `artifacts/models/final` содержат именно этот старый артефакт:
`name: vitb14_f0_clsavg_in280_ep100`, `train.folds: all except 0`.
Название каталога `final` не означает финальную модель D-044.

Нужны три файла от трека A, из одного финального запуска D-044:

- `artifacts/models/final_all_data/weights.safetensors`;
- `artifacts/models/final_all_data/model_config.yaml`;
- `artifacts/models/verifier_final/verifier.pt`.

Получив файлы, трек B проверяет их SHA-256 и metadata, публикует новый фиксированный
Release, обновляет URL/хеши в Dockerfile и команды скачивания. **Старый релиз не
перезаписывается.** Затем — офлайн Docker-run на A10/A5000, контроль размера
всех весов ≤2 ГБ, VRAM, времени, двух повторных запусков и сравнение с `submission/`.

Команда проверки весов для подготовленного каталога решения:

```bash
PYTHONPATH=. .venv/bin/python scripts/audit_release.py weights \
  --root artifacts/models \
  --model artifacts/models/final_all_data \
  --verifier artifacts/models/verifier_final/verifier.pt \
  --require-all-data --report outputs/final-weights-report.json
```

Для проверки именно состава Docker-образа запускать внутри контейнера с
`--root /app` и путями реальных артефактов `/app/...`. Подсчитываются **все**
расширения весов из ТЗ, включая лишние экспериментальные файлы; перед сборкой
в образе должны остаться только веса инференса. Текущий Dockerfile ещё закреплён
за model-v1 и не является финальным образом D-044.

## Выполненные проверки

- Все три сдаваемых файла прошли валидатор C6.
- `make check PYTHON=.venv/bin/python`: 79 тестов и e2e smoke на 50 кадрах.
  Smoke использует доступную model-v1; финальная модель им не проверена.
- `audit_release.py weights --require-all-data` отклоняет старую model-v1 с
  причиной `all except 0`, что предотвращает ошибочную приёмку старых весов.
- Финальные веса, их размер, VRAM и Docker-воспроизводимость пока **не проверены**.
- Платные GPU в этой сессии не запускались.
