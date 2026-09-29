# Обучение и локальная оценка

Команды выполняются из корня с `PYTHONPATH=.`. Обучение требует исходного датасета организаторов и доступа к публичным pretrained-весам. Для Linux/GPU используются зависимости основного `requirements.txt`; `requirements-train.txt` сохраняет проверенное CPU-окружение Intel macOS.

## Данные

Train содержит 9556 кадров и 1541 автомобиль. Фиксированные разбиения лежат в `splits/`: пять фолдов по `vehicle_id`, seed 2026. В локальной gallery оставлен один кадр на пару автомобиль–камера, около 20% query не имеют пары. Для сопоставимых результатов фолды не пересоздаются.

```bash
PYTHONPATH=. python scripts/build_crop_cache.py --data-root dataset --out artifacts/crops
```

Сборщик может подготовить кропы всех входных CSV, но обучающие датасеты читают только записи `split=train`. Тестовые данные не используются для настройки или анализа ошибок. `camera_id` применяется при sampling и оценке, но не подаётся в модель.

## Эмбеддеры

Точные параметры финальных артефактов сохранены в [DINO392](evidence/hires392_all.yaml) и [CLIP288](evidence/div_clip_all.yaml). Оба encoder: 100 эпох, AdamW, CE с label smoothing 0.1 и triplet margin 0.3, PK-batch 16×4, layer-wise LR decay 0.75. Аугментации: рамка, цвет, random erasing и геометрическая заливка номерной области.

Пример повторного обучения DINO на всех train-фолдах:

```bash
PYTHONPATH=. python scripts/train.py --config configs/vitb14_224.yaml \
  --name hires392_all --set model.input_size='[392,392]' model.pool=cls_top16 \
  train.val_fold=null train.epochs=100 train.erasing_p=0.7
```

Для CLIP замените имя на `div_clip_all` и добавьте `model.arch=clip_vitb16 model.input_size='[288,288]'`. Для OOF-модели замените `train.val_fold=null` на номер исключённого фолда и используйте отдельное имя артефакта. All-data веса не дают честной оценки на обучающих фолдах.

Извлечение OOF-признаков, пример для fold 0:

```bash
PYTHONPATH=. python scripts/extract_embeddings.py \
  --model artifacts/models/hires392_f0 --folds 0 --no-test
```

При retrieval используются `model.features()` и L2, без BNNeck. Подробности формата входа модели и сохранения артефакта — в `reid/data/preprocess.py` и `reid/model/__init__.py`.

## Verifier

[Конфигурация и происхождение](evidence/verifier.provenance.yaml): DINOv2-S/14, вход 392, backbone заморожен, 4000 шагов обучения головы, `plate_band_p=0.5`.

Трудные негативы готовятся OOF-эмбеддерами через `scripts/mine_pairs.py` либо `scripts/analysis/mine_fusion_pairs.py`. Для каждого обучающего фолда нужны пары, найденные моделью, которая не видела этот фолд. `scripts/train_verifier.py` принимает каталог пар, кэш и исключённый фолд; для all-data используется `--val-fold -1`.

```bash
PYTHONPATH=. python scripts/train_verifier.py --pairs artifacts/pairs \
  --cache artifacts/crops --arch dinov2_vits14 --input 392 \
  --steps 4000 --freeze-backbone --val-fold -1 --name verifier_all
```

Это порядок повторного обучения, а не обещание побайтно идентичных весов на другом оборудовании. Для воспроизведения инференса используйте опубликованный архив с фиксированным SHA.

## Оценка

Порядок top-10 строится текущим поиском и передаётся в `scripts/analysis/rerank_dump.py` через `--window`. Нельзя смешивать логиты и кандидатов из разных прогонов. `scripts/replay_fusion.py` применяет ту же политику поиска и отказа, что и инференс, и вызывает локальный evaluator.

```bash
PYTHONPATH=. python scripts/replay_fusion.py \
  --dump artifacts/emb/hires392_f0_cls_top16/fold0.npz \
  --dump artifacts/emb/div_clip_f0_cls_top16/fold0.npz \
  --fold 0 --logits artifacts/rerank/pair_logits_f0.npz \
  --rerank-weight 0.15 --candidate-threshold 6.6 --out outputs/replay-fold0
```

Пути здесь обозначают результаты предыдущих шагов; OOF-дампы и логиты не поставляются с финальными all-data весами. Повторите для fold 1 с его независимыми моделями. Для готовых CSV:

```bash
PYTHONPATH=. python -m reid.eval.eval_local --fold 0 \
  --submission outputs/fold0/submission.csv --candidates outputs/fold0/candidates.csv
```

[Результаты и ограничения](results.md) · [Источники весов](sources.md)
