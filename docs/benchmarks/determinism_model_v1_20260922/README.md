# B5.2: детерминизм model-v1

Контрольный smoke-прогон на 5 реальных test query и 10 test gallery. Он только выполняет инференс: не читает test labels, не обучается и не подбирает параметры.

Команда:

```bash
PYTHONPATH=. .venv/bin/python scripts/check_determinism.py \
  --images dataset/images \
  --query dataset/test_query.csv \
  --gallery dataset/test_gallery.csv \
  --model artifacts/models/final \
  --verifier artifacts/models/verifier_f0/verifier.pt \
  --device cpu --batch-size 8 \
  --query-limit 5 --gallery-limit 10 \
  --out outputs/day5-determinism
```

Каждый прогон заново инициализирует model-v1 и verifier-v1. Seed равен 2026; детермированные алгоритмы Torch включены. Три C6-артефакта двух запусков побайтово совпали. Точные времена, версии и SHA-256 находятся в `report.json`.

Это smoke-приёмка инструмента. Полный двойной прогон по всему открытому test в Docker на A5000 запланирован в H9.1.
