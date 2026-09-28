# Контракты между треками

Эти интерфейсы позволяют треку A (модель) и треку B (инференс) работать параллельно, не дожидаясь друг друга. **Менять их можно только с согласия обоих людей.** Каждое изменение записывается в DECISIONS.md.

## C1. Фолды

Владелец — A. Скрипт `scripts/make_folds.py` создаёт `splits/folds.csv`:

```text
image_id,vehicle_id,camera_id,fold,role
```

- `fold` ∈ {0..4}. Все кадры одной машины попадают в один фолд.
- `role` ∈ {`query`, `gallery`, `drop`} — роль кадра в тот момент, когда его фолд служит валидацией. При валидации на фолде k обучение идёт на всех кадрах с `fold != k`, независимо от роли.

**Построение повторяет устройство открытого теста** (доказательство в docs/findings.md):
1. Раскидать `vehicle_id` по 5 фолдам, стратифицируя по числу кадров и числу камер. Seed 2026.
2. Внутри фолда в каждой группе (`vehicle_id`, `camera_id`) один случайный кадр идёт в `gallery`, остальные — в `query`. Если в группе один кадр, он уходит в `gallery`.
3. **Query без пары.** В каждом фолде выбрать машины так, чтобы их query составили около 20% query фолда. Все их gallery-кадры помечаются `drop`. У таких query в gallery нет ни пары, ни junk-брата.
4. Скрипт печатает проверки по каждому фолду:
   - число query и gallery;
   - доля query без пары, цель 20% ± 2%;
   - у каждого парного query есть хотя бы один позитив с другой камеры;
   - у каждого парного query ровно один junk-брат.

Производные файлы для прогона настоящего пайплайна на фолде — в формате тестовых CSV, в порядке `folds.csv`:
- `splits/fold{k}_query.csv` — `image_id,x,y,w,h`;
- `splits/fold{k}_gallery.csv` — `image_id,x,y,w,h`.

Фолды коммитятся в git и после Д1 **не пересоздаются**. Иначе все сравнения экспериментов теряют смысл.

## C2. Локальная оценка `eval_local.py`

Владелец — A. Запуск:

```bash
python -m reid.eval.eval_local --fold K \
    --submission S.csv [--candidates C.csv] [--embeddings E.npy] [--out metrics.json]
```

Функция на Python: `evaluate(fold: int, ranking: dict[str, list[str]], candidates: list[tuple[str, str, float]] | None, emb: np.ndarray | None) -> dict`.

Входные файлы — в формате сдачи (контракт C6) по query и gallery фолда K.

**Протокол** (ТЗ §5–6 и ответы организаторов в Q&A):
- **Junk** — кадр gallery с тем же `vehicle_id` **и** тем же `camera_id`, что у query. Удаляется до обрезки до 10 позиций, **без добора**.
- **AP@10** нормируется на `min(n_pos, 10)`, где `n_pos` — число валидных позитивов после удаления junk. Query с `n_pos = 0` исключаются из mAP и Rank. Усреднение по query (macro).
- **F1 и TNR** считаются по суммарным счётчикам (micro) по верхнему кандидату, то есть с максимальным confidence. Кандидаты с равным confidence разрешаются в пользу первого в файле, пока `evaluate.py` не скажет иначе.
- **Кандидаты:** junk из кандидатов не удаляется, верхний по confidence считается верным при совпадении `vehicle_id` — двойник с той же камеры тоже верный ответ (D-023). Это режим `official`, по нему и принимаем решения. Режим `cross_cam` — диагностика: показывает, сколько F1 держится на двойниках.
- **Полный mAP и mINP** — справочно, по полному косинусному ранжированию из `embeddings.npy`, с тем же junk-фильтром.

Выход — JSON с полями: `map10, rank1, rank5`; `f1, tnr, precision, recall, pr_auc, candidate_score` для режимов `official` и `cross_cam`; `candidate_score, full_map, minp, n_query, n_query_openset`.

**Сверено 20.09** (A1.5): `scripts/crosscheck_official.py` сравнивает 16 метрик с `third_party/official/evaluate.py` на одинаковом входе, все совпали до 1e-6. Прогоняйте его после любого изменения `eval_local.py`: при расхождении прав эталон.

## C3. Артефакт модели

Владелец — A. Каталог `artifacts/models/<name>/`:
- `weights.safetensors`
- `model_config.yaml`:

```yaml
name: dinov2_vitb14_ft_v1
arch: dinov2_vitb14              # ключ в reid.model.registry
pretrained:                      # исходные публичные веса
  url: "https://..."
  revision: "<tag или commit>"
  sha256: "<...>"
input_size: [256, 256]           # H, W
bbox_pad: 0.0                    # доля расширения BBox перед кропом
resize: stretch                  # stretch | letterbox
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]
pool: cls                        # cls | avg | cls_avg — как токены ViT сводятся в вектор (D-021)
embed_dim: 768
infer_dtype: fp16
tta_flip: false
train: {commit: "<sha>", config: "configs/<...>.yaml", folds: "all | 1,2,3,4"}
val: {map10_fold0: null}
```

- Загрузка: `reid.model.load(model_dir, device) -> torch.nn.Module`.
- `forward(x)`: на входе float-тензор `[B,3,H,W]`, уже нормализованный по mean и std. На выходе `[B, embed_dim]`, L2-нормированный.
- **Предобработка задаётся только конфигом и одной общей функцией в `reid/data/preprocess.py`:**
  - BBox обрезается до границ кадра;
  - расширяется на `bbox_pad`;
  - делается crop;
  - resize через `torch.nn.functional.interpolate(mode="bilinear", antialias=True)`;
  - нормализация по mean и std.

  Обучение и инференс вызывают **одну и ту же** функцию. Любая другая реализация будет расходиться с обучением.
- Если B меняет декодер (`draft`, ROI, nvJPEG) и пиксели при этом меняются, это проверяется на fold 0. Допустимое падение mAP@10 — не больше 0,002.

## C4. Дампы эмбеддингов

Владелец — A, потребитель — B. Файлы:
- `artifacts/emb/<model_name>/fold{k}.npz` — модель, обученная **без** фолда k (out-of-fold, OOF);
- `artifacts/emb/<model_name>/test.npz` — открытый тест.

Ключи внутри npz:
- `query_ids`, `gallery_ids` — массивы строк;
- `query_emb` `[Nq, D]`, `gallery_emb` `[Ng, D]` — float32, L2-нормированные.

Порядок строк — как в `splits/fold{k}_query.csv` и `fold{k}_gallery.csv` (для теста — как во входных CSV). Когда в `artifacts/emb/` кладётся новая модель, в EXPERIMENTS.md пишется её имя.

## Где взять артефакты

Модели и эмбеддинги в git не хранятся (`artifacts/` в `.gitignore`). Имя актуальной лучшей модели смотрите в последних строках трека A в EXPERIMENTS.md.

**Zero-shot модель** `artifacts/models/dinov2_vitb14_zeroshot_224_cls` (A-001) пересобирается одной командой, ревизия весов на Hugging Face зафиксирована:

```bash
PYTHONPATH=. python scripts/make_pretrained_artifact.py --arch dinov2_vitb14 --input 224 --pool cls
```

**Эмбеддинги** `artifacts/emb/<model_name>/` весят около 33 МБ на модель. Есть два способа их получить:
- скачать из общей папки команды (docs/infra.md, раздел «Обмен артефактами»);
- пересчитать самому. На CPU это около 28 минут, на GPU — минуты. Для пересчёта нужен кэш кропов:

```bash
PYTHONPATH=. python scripts/build_crop_cache.py --out artifacts/crops
PYTHONPATH=. python scripts/extract_embeddings.py --model artifacts/models/<model_name>
```

**Дообученные модели** трек A кладёт в общую папку сразу после записи в EXPERIMENTS.md.

## C5. Пайплайн и Docker

Владелец — B. Запуск пайплайна:

```bash
python -m reid.infer.run --images DIR --query Q.csv --gallery G.csv --out OUT_DIR \
    [--model artifacts/models/final] [--twins on|off]
```

На выходе в `OUT_DIR` лежат `submission.csv`, `embeddings.npy`, `candidates.csv`.

Запуск через Docker одной командой:

```bash
docker compose run --rm reid
```

- `./dataset` монтируется в `/data` только для чтения, `./outputs` — в `/out`;
- в compose выставлено `network_mode: none`;
- пути к CSV переопределяются через env.

`Extractor` — предварительный интерфейс. Когда организаторы пришлют `extractor.py`, приводим к нему.

```python
class Extractor:
    def __init__(self, model_dir: str, device: str = "cuda"): ...
    def extract(self, image_path: str, bbox: tuple[int, int, int, int]) -> np.ndarray: ...   # [D] float32, L2
    def extract_batch(self, items: list[tuple[str, tuple[int, int, int, int]]]) -> np.ndarray: ...  # [B, D]
```

**Зона замера:** чтение с диска → декодирование → crop → preprocessing → forward → postprocessing → L2. Загрузка модели и построение индекса в эту зону не входят.

## C7. Парный верификатор (переранжирование top-10)

Владелец — A, потребитель — B. Модуль `reid/model/verifier.py`, веса — `verifier.pt` из GitHub Release.

```python
from reid.model.verifier import load_verifier, prep_crop, rerank_order
ver, vcfg = load_verifier("artifacts/models/verifier_f0/verifier.pt", device)
logits = ver.score_pairs(prep_crop(query_crop, vcfg),
                         torch.stack([prep_crop(c, vcfg) for c in cand_crops]))
order = rerank_order(cosine_top10, logits, vcfg["weight"])   # индексы кандидатов в новом порядке
```

- **Вход:** те же uint8-кропы, что идут в эмбеддер; свой размер входа берётся из `vcfg`.
- **Формула порядка:** `cosine + weight · sigmoid(logit)`, вес по умолчанию 0,2. При weight = 0 порядок в точности прежний, поэтому включение верификатора не может ухудшить выдачу сильнее, чем на свой вес.
- **Что даёт:** +0,018 к mAP@10 в среднем по пяти фолдам (A-020).
- **Бюджет:** около 90 МБ весов; переранжирование **не входит** в замер `extract()`, только в общий тайм-аут около 4 минут. На 1100 query это 11 тысяч пар.
- **Пока весов нет:** интеграцию можно писать и замерять по времени со случайно инициализированной моделью — `PairVerifier(arch, (input, input))` без загрузки состояния.

## C6. Выходные файлы и валидатор

Владелец — B, файл `reid/eval/validate_outputs.py`.

**`submission.csv`**
- **Без заголовка** (D-025): эталонный `evaluate.py` читает файл как данные с первой строки.
- Одна строка на каждый query из входного CSV, в том же порядке.
- В каждой строке ровно 10 **различных** `gallery_id`, и все они есть во входной gallery. Пустых значений нет.

**`embeddings.npy`**
- Форма `(N_query + N_gallery, D)`, dtype float32.
- Все значения конечны (без NaN и inf).
- Порядок: сначала все query, затем вся gallery, как во входных CSV.

**`candidates.csv`**
- Заголовок: `query_id,gallery_id,confidence`.
- Все ID есть во входных файлах, confidence — конечное число.
- Отказ — отсутствие строк этого query в файле.
- Политика команды: кадр, распознанный как двойник, никогда не становится верхним кандидатом.

Валидатор завершается с кодом ≠ 0 при любом нарушении. Он входит в `make check`.

## C7. Таблица скоростей

Владелец — B, файл `docs/speed_table.md`, по одной строке на замер:

```text
| дата | gpu | драйвер | backbone | вход | dtype | декодер | latency p50, мс (batch=1) | лучший FPS (batch) | пик VRAM, МБ | веса, МБ | коммит |
```

Замеры ведутся по протоколу C5 и ТЗ §7.1. Пороги для выбора модели на A10 и 4090 пересчитываются по коэффициентам из сессии A5000 №1.
