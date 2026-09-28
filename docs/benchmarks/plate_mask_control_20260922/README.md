# B5.3: контрольный прогон с залитым номером

Smoke-прогон model-v1 + verifier-v1 на 5 test query и 10 test gallery. Это только инференс без test labels, обучения или подбора параметров. Три C6-файла созданы и прошли штатный валидатор.

Команда:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_plate_mask_control.py \
  --images dataset/images \
  --query dataset/test_query.csv \
  --gallery dataset/test_gallery.csv \
  --model artifacts/models/final \
  --verifier artifacts/models/verifier_f0/verifier.pt \
  --device cpu --batch-size 8 \
  --query-limit 5 --gallery-limit 10 \
  --out outputs/day5-plate-mask \
  --report outputs/day5-plate-mask-report.json
```

Маска создаётся в памяти до общего C3-препроцессинга и не изменяет датасет. В `report.json` зафиксированы геометрия маски и SHA-256 результа.
