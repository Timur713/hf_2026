# Falcon ReID

**DeepTech AI · ЛЦТ 2026 · задача №7 «Фалькон Тех»**

Поиск автомобиля между камерами: JPEG и BBox → эмбеддинг → top-10 и кандидаты
с confidence либо отказ. Инференс работает офлайн в Docker.

Версия **D082** объединяет два визуальных encoder, графовое ранжирование
и парную проверку кандидатов. Модели дообучены на всех предоставленных train-фолдах.
[Результаты проверок](docs/d082_acceptance.md) · [План](docs/final_sprint_20261016.md).

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

## Результаты измерений

| Проверка | Результат | Область |
|---|---:|---|
| mAP@10, matched development folds 0/1/3/4 | 84,60% → 85,22% | Исторические OOF-сравнения, не финальные all-data веса |
| Rank-1, те же фолды | 81,60% → 82,34% | Статистическая неопределённость сохраняется |
| Rank-5, те же фолды | 94,95% → 94,92% | Исторические OOF-сравнения |
| D082, A5000, extraction mean batch=1 | 38,797 мс | JPEG→L2, внутренний batch=4 |
| D082, A5000, throughput | 93,228 FPS | Внешний batch=32, внутренний batch=4 |
| D065, A5000, throughput | 102,674 FPS | Исторический замер, batch=32 ([исходный JSON](docs/evidence/a5000.json)) |
| Полный JPEG→кандидат | p50 197,873 / p95 211,435 мс | Gallery 636; включает verifier |

Замеры скорости подписаны по версиям: D065 и D082 используют разные режимы
внутреннего batch, поэтому 102,674 FPS относятся к D065. Поиск, graph и verifier
не входят в extraction C5. Замеры D082 выполнены до переноса в эту ветку;
повтор throughput дал 93,098 FPS.

В исходном D082 проверены офлайн-Docker, три файла/C6, повторяемость
и независимость query на проверенных вариантах. После переноса локальный
проверки на Python 3.10: **700 тестов прошли, 5 пропущены**, smoke50
с реальными весами и синтетическими JPEG — PASS.
Протокол, статус повторной Linux/GPU-приёмки и ограничения приведены
в [отчёте](docs/d082_acceptance.md). Это локальные измерения команды;
результаты закрытого теста жюри здесь не заявляются.

## Доработки по обратной связи жюри

| Замечание от 16 октября | Что сделано | Подтверждённый результат |
|---|---|---|
| **1. Режим отказов** | Проверены шкала confidence, применяемый порог и ошибки по стадиям; калибровка на folds0/1 с независимой проверкой3/4 | Снижение порога до6,0 не дало прироста на3/4; в поставке сохранён gallery-z6,6. Улучшение recall закрытого теста не заявляется |
| **2. Порядок кандидатов** | Добавлен обученный camera-free graph residual к поиску и verifier; финальные визуальные модели переобучены на всех train-фолдах | В matched OOF-сравнении folds0/1/3/4: Rank-1 +0,74 п.п., mAP@10 +0,62 п.п.; Rank-5 практически прежний. Это результат OOF-моделей, не оценка финальных all-data весов |
| **5. Логирование и зависимости слоёв** | Стандартный logging в stderr с уровнями, фактическими настройками, SHA артефактов и таймерами; JPEG/нейросетевой adapter verifier перенесён из post в infer | Добавлены тесты логирования и границы импортов; функции ранжирования импортируются без torch/JPEG I/O. Конкретный импорт жюри не называло — исправлена найденная нами зависимость |
| **6. Одиночный запрос и скорость** | Измерен полный JPEG→кандидат при batch=1, отдельно extraction и verifier; подготовлено состояние статической gallery | A5000: полный ответ p50 197,873 / p95 211,435 мс. Основная часть — verifier (mean144,647 мс). Профиль получен; ускорение полного пути относительно D065 не подтверждено |

Изменение порога и оптимизация verifier проверялись отдельно от ранжирования.
Экспериментальный кэш verifier оставлен выключенным: точное совпадение ответов
не подтвердилось. Обработка одного запроса не требует ожидания соседних запросов;
внутреннее дополнение до4 выполняется собственным изображением.

Состав двух выборок жюри и окончательное поведение evaluator для same-camera
не раскрыты; локальные результаты не подменяют эти сведения. В локальной оценке
сохранены зафиксированные правила junk и кандидатов. Контроль номерных зон
и его ограничения описаны в [отчёте](docs/d082_acceptance.md).

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
