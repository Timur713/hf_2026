# Falcon ReID

**DeepTech AI · ЛЦТ 2026 · задача №7 «Фалькон Тех»**

Поиск автомобиля между камерами: JPEG и BBox → эмбеддинг → top-10 и кандидаты
с confidence либо отказ. Инференс работает офлайн в Docker.

Эта ветка интегрирует **D082, кандидат на замену D065**. Финальная приёмка
**не пройдена**: на A5000 получено 93,23 FPS при цели ≥100.
Исправления Python 3.10 и перенос в этот репозиторий требуют повторной Linux/GPU-приёмки.
[Проверки и ограничения](docs/d082_acceptance.md) · [План](docs/final_sprint_20261016.md).

## Запуск кандидата

Нужны Linux x86-64, NVIDIA GPU, NVIDIA Container Toolkit, Docker с BuildKit
и Compose ≥2.39.4. CUDA 12.1 и зависимости закреплены в Dockerfile и lock-файле.
Репозиторий и Release должны быть доступны сборщику; пока репозиторий private,
используйте локальный кэш из авторизованного скачивания ниже.

```bash
git clone https://github.com/Timur713/hf_2026.git
cd hf_2026
# Распакуйте предоставленные данные в dataset/.
mkdir -p outputs
docker compose run --build --rm reid
```

На этапе build скачивается фиксированный архив с проверкой SHA-256.
Веса уже находятся в образе при запуске, `network_mode: none` отключает сеть.
Для private Release скачайте кэш через авторизованный GitHub CLI:

```bash
mkdir -p artifacts
gh release download d082-candidate-20261019-r2 --repo Timur713/hf_2026 \
  --pattern falcon-D082-legacy66-base-bundle.tar.gz --dir artifacts
mv artifacts/falcon-D082-legacy66-base-bundle.tar.gz artifacts/d082_production_bundle.tar.gz
docker compose run --build --rm reid
```

Архив: `732784145` байт, SHA-256
`0920e24524e9b32215323bbe3f8e47ae6e19d1f036b14eacfa596507b83d95d5`.
[Release](https://github.com/Timur713/hf_2026/releases/tag/d082-candidate-20261019-r2).
Сборка отклоняет несовпадающие SHA, лишние файлы и ссылки в архиве.

```text
dataset/
├── images/
├── test_query.csv
└── test_gallery.csv
```

CSV: `image_id,x,y,w,h`, координаты в пикселях полного кадра.
Gallery содержит минимум 10 объектов; ID уникальны внутри каждого CSV.
`camera_id` и `vehicle_id` на инференсе не используются. Вход монтируется read-only.

```bash
DATA_ROOT=/absolute/path/dataset OUTPUT_ROOT=/absolute/path/results \
  QUERY_CSV=/data/query.csv GALLERY_CSV=/data/gallery.csv \
  docker compose run --rm reid
```

Выходной каталог должен существовать. Повторный запуск перезаписывает результаты.
Модель, manifest, порог и коэффициенты запечены в образ и не переопределяются CLI.
Уровень логирования задаётся `--log-level DEBUG|INFO|WARNING|ERROR`;
служебные сообщения идут в stderr.

## Выходы

| Файл | Содержимое |
|---|---|
| `submission.csv` | Без заголовка: query ID и 10 различных gallery ID |
| `embeddings.npy` | float32 `(N_query + N_gallery, 3072)`, query затем gallery, L2 |
| `candidates.csv` | `query_id,gallery_id,confidence`; три строки для принятого query, при отказе строк нет |

Confidence — gallery-z, не вероятность. Порог верхнего кандидата — 6,6.
Файлы в `submission/`, презентация и видео относятся к прежнему D065;
они сохранены как история и не выдаются за новые результаты D082.

## Рецепт D082

DINOv2-B392 + CLIP-B288, FP16 preBN, исходные component L2 и фиксированная
взвешенная конкатенация. Gallery hub correction → camera-free graph residual
0,5 внутри top-10 → стандартизированный verifier DINOv2-S392 с весом 0,15
→ gallery-z 6,6. Внутренний batch фиксирован на 4, последний дополняется
повтором собственного изображения. Кэш verifier и projection precast выключены.

Каждый query использует только себя и статическую gallery. Нет query expansion,
кластеризации query или статистик по query-batch. Номерные знаки не распознаются;
ограниченный контроль заливки описан в отчёте и не доказывает универсальную
независимость от остаточных признаков номера.

Финальные encoder и verifier обучены на всех пяти train-фолдах.
Все веса инференса: **788873516 байт**, включая graph, при лимите 2 ГБ.
All-data веса не используются для заявления OOF-качества.

## Результаты и статус

| Проверка | Результат | Область |
|---|---:|---|
| mAP@10, matched development folds 0/1/3/4 | 84,60% → 85,22% | Исторические OOF-сравнения, не финальные all-data веса |
| Rank-1, те же фолды | 81,60% → 82,34% | Статистическая неопределённость сохраняется |
| Rank-5, те же фолды | 94,95% → 94,92% | Цель Rank-1 90% не достигнута |
| A5000, extraction mean batch=1 | 38,797 мс | JPEG→L2; ≤40 мс пройдено |
| A5000, лучший throughput | 93,228 FPS | Внешний batch=32; ≥100 FPS не пройдено |
| Полный JPEG→кандидат | p50 197,873 / p95 211,435 мс | Gallery 636; включает verifier |

Замеры A5000 относятся к исходному пакету, до переноса в эту ветку.
Повтор throughput — 93,098 FPS. Поиск, graph и verifier не входят в extraction C5.
Офлайн-Docker, три файла/C6, повторяемость и проверенные варианты независимости
query прошли в исходном пакете. Его `make check` не прошёл из-за API Python 3.11
в контейнере Python 3.10. Эта ветка использует совместимый streaming SHA helper.
Это не закрытые метрики жюри и не разрешение на автоматическую замену решения.

## Проверки

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. make check DATA_ROOT=/path/to/dataset
PYTHONPATH=. make smoke-strict DATA_ROOT=/path/to/dataset
```

Поддерживаемый контейнерный Python — 3.10. Обычный smoke может сообщить SKIP
при отсутствии весов; `smoke-strict` требует фактического инференса.
При установленном D082 smoke проверяет именно его, без fallback на D065.
Данные закрытого fold 2 повторно не оцениваются.

[Приёмка и команды A5000](docs/d082_acceptance.md) ·
[Источники и SHA](docs/sources.md) · [Обучение](docs/training.md).

Исходные pretrained: Meta AI DINOv2-B/S и LAION/OpenCLIP ViT-B/16,
timm-конверсии. Автор дообученных весов и graph — DeepTech AI.

| Автор / модель | Immutable revision URL | Исходный SHA-256 |
|---|---|---|
| Meta AI, DINOv2-B | [4685c99dabffe5affac90bd99dbffd25801ae58d](https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m/tree/4685c99dabffe5affac90bd99dbffd25801ae58d) | `55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b` |
| LAION/OpenCLIP, CLIP-B | [fbadfdd1c0e903c58855b70ae04740b22da9d305](https://huggingface.co/timm/vit_base_patch16_clip_224.laion2b/tree/fbadfdd1c0e903c58855b70ae04740b22da9d305) | `35fcf20089626ce1534aa3d4d8d97a02ca79e171a1735d0c8d58e150eded9ad5` |
| Meta AI, DINOv2-S verifier | [4610ca143709d58a633b6397a74412c2c3842454](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/tree/4610ca143709d58a633b6397a74412c2c3842454) | `04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081` |
