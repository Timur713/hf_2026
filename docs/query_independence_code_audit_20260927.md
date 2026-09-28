# Структурная независимость query: аудит кода 27.09.2026

Прочитан рабочий код трека B в commit
`71e36e2aff60743931906500840718d2f1dc5de7`. Это **read-only аудит структуры**, не
результат GPU-теста. Производственный код и общие контракты не менялись.
Практическая проверка batch1/32/reverse/extra-query сохраняется отдельным отчётом.

**Вывод:** в выбранном production fusion-пути не найдено вычислений, использующих
другие query для статистик, ранжирования или порога текущего query. Входа
`camera_id` тоже не найдено. Это обоснование по коду, а не гарантия для всех
моделей, всех батчей, всех внешних реализаций scorer и всех численных режимов.

## Свидетельства

| Узел | Код в проверенном commit | Почему другие query не участвуют |
|---|---|---|
| Fusion scales | [`fusion.py:44`](https://github.com/Timur713/hack_falcon_2026/blob/71e36e2aff60743931906500840718d2f1dc5de7/reid/infer/fusion.py#L44), 68–88 | `similarity_std` читаются из фиксированного artifact JSON, валидируются; никаких входных query этот loader не принимает |
| Preprocessing и L2 | [`extractor.py:181`](https://github.com/Timur713/hack_falcon_2026/blob/71e36e2aff60743931906500840718d2f1dc5de7/reid/infer/extractor.py#L181), 186–209 | Decode/preprocessing выполняются по одному изображению; компоненты склеиваются по оси признаков, нормы считаются `dim=1` отдельно для каждого объекта. `ThreadPoolExecutor.map` сохраняет порядок |
| Eval, BNNeck | [`extractor.py:152`](https://github.com/Timur713/hack_falcon_2026/blob/71e36e2aff60743931906500840718d2f1dc5de7/reid/infer/extractor.py#L152), 155, 200; [`model/__init__.py:152`](https://github.com/Timur713/hack_falcon_2026/blob/71e36e2aff60743931906500840718d2f1dc5de7/reid/model/__init__.py#L152), 158–163, 217 | Fusion-модели загружаются `.eval().requires_grad_(False)`. Используется `features()`, обходящий обучающий BNNeck; обычный model loader тоже возвращает eval. Pooling работает по токенам отдельного изображения, не по batch |
| Gallery statistics | [`gallery_policy.py:59`](https://github.com/Timur713/hack_falcon_2026/blob/71e36e2aff60743931906500840718d2f1dc5de7/reid/post/gallery_policy.py#L59), 60–79 | Mean/std confidence и hub получают только `G @ G.T`; диагональ исключается для confidence и соседей. Gallery/hub сохраняются read-only |
| Поиск и confidence | [`gallery_policy.py:90`](https://github.com/Timur713/hack_falcon_2026/blob/71e36e2aff60743931906500840718d2f1dc5de7/reid/post/gallery_policy.py#L90), 96–110, 128–132, 156–158 | Даже матричное умножение вызывается по одной query-строке. Z-score ранжирования считается по gallery этой строки; confidence выбранного кандидата использует только её cosine и фиксированные gallery statistics |
| Pair verifier | [`post/verifier.py:64`](https://github.com/Timur713/hack_falcon_2026/blob/71e36e2aff60743931906500840718d2f1dc5de7/reid/post/verifier.py#L64), 88–103, 183–205 | Verifier в eval; один вызов получает один query и его top-10. Стандартизация logits/base производится внутри этой десятки, в цикле по query |
| Attention verifier | [`model/verifier.py:52`](https://github.com/Timur713/hack_falcon_2026/blob/71e36e2aff60743931906500840718d2f1dc5de7/reid/model/verifier.py#L52), 76–81 | Transformer задан `batch_first=True`; после backbone токены собираются в отдельную последовательность для каждой пары, внимание не объединяет разные query. Eval отключает обучающий dropout |
| Связывание стадий | [`run.py:194`](https://github.com/Timur713/hack_falcon_2026/blob/71e36e2aff60743931906500840718d2f1dc5de7/reid/infer/run.py#L194), 200–228 | `GalleryPolicy` строится только по gallery; reranker получает соответствующую query-строку. После перестановки confidence пересчитывается для нового top-1 того же query |
| Порог и запись | [`policy.py:17`](https://github.com/Timur713/hack_falcon_2026/blob/71e36e2aff60743931906500840718d2f1dc5de7/reid/post/policy.py#L17), 28–50; [`run.py:127`](https://github.com/Timur713/hack_falcon_2026/blob/71e36e2aff60743931906500840718d2f1dc5de7/reid/infer/run.py#L127), 138–141 | Порог/вес берутся из config или явного override, не из query-batch. Отказ и confidence запасных кандидатов рассчитываются отдельно для текущей строки |

В [run.py:28–36,61–80](https://github.com/Timur713/hack_falcon_2026/blob/71e36e2aff60743931906500840718d2f1dc5de7/reid/infer/run.py#L28)
входной CSV строго ограничен `image_id,x,y,w,h`; `InputRow` содержит ID, путь и
BBox. В Extractor/verifier передаются только путь изображения и BBox
(`run.py:88,206–207`), не camera/vehicle labels. В shared preprocessing
нормализация использует константы mean/std из config, а не статистики батча
([preprocess.py:60–74](https://github.com/Timur713/hack_falcon_2026/blob/71e36e2aff60743931906500840718d2f1dc5de7/reid/data/preprocess.py#L60)).

## Реальные риски и пределы вывода

1. **Численная зависимость от формы batch возможна.** Extractor действительно
   выполняет forward батчами с autocast (`extractor.py:195–200,228–238`). Даже
   при отсутствии обмена информацией между образцами CUDA kernels могут дать
   различные младшие разряды. Вблизи равенства scores это способно менять
   порядок, а вблизи порога — решение об отказе. Stable sort разрешает точные
   равенства, но не устраняет почти равные числа. Candidate CSV округляет
   confidence до восьми знаков (`run.py:141`), не гарантируя эквивалентности.
   Поэтому GPU-сравнение необходимо; ULP-drift сам по себе не доказывает
   запрещённые query statistics и не является основанием объявлять дисквалификацию.
2. **Legacy cosine path слабее по численной стабильности.** В
   [search.py:37–40](https://github.com/Timur713/hack_falcon_2026/blob/71e36e2aff60743931906500840718d2f1dc5de7/reid/post/search.py#L37)
   выполняется целое `Q @ G.T`; форма Q может менять BLAS kernel. Математически
   строки независимы, но побитное равенство не обещается. Выбранный fusion
   `gallery_z` идёт по single-query `_raw_row`, а не по этой ветке.
3. **Runtime-код не удостоверяет происхождение констант и обучения.** Фиксированный
   scale/threshold может быть структурно независим на инференсе, но допустимость
   данных его калибровки доказывается отдельным provenance/OOF-аудитом. Здесь
   история обучения, mining teachers, plate reliance и all-data качество не
   подтверждались. Положительная проверка JSON не заменяет эти свидетельства.
4. **Проверена текущая production-реализация.** Вывод нельзя переносить на будущий
   backbone с иным поведением нормализации, внешний `PairScorer`, ручное
   переключение модели в train-mode или изменённые preprocessing hooks. Опциональный
   projection-precast (`extractor.py:23–57,88–99`) меняет только веса Linear/Conv
   при загрузке eval/frozen модели и сам не вводит query statistics, но требует
   своей численной приёмки; default flag — false.

Общие проверки конечности массива могут аварийно остановить весь batch при
испорченном элементе (`extractor.py:205–207`, `gallery_policy.py:86–87`). Это
fail-fast обработка ошибочного входа, а не использование других query для
улучшения ответа на корректный вход. Область вывода выше — валидные входы,
неизменные веса/config и одна статическая gallery.

При расхождении GPU-проверки следует отдельно показывать top-10 IDs, candidate
IDs, accept/refuse и величину confidence/embedding drift; не подменять строгий
FAIL общим PASS и не настраивать порог на test. См.
[описание практической проверки](query_independence_check.md).
